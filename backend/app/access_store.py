import hashlib
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from config import get_access_db_path

logger = logging.getLogger(__name__)


class AccessStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or get_access_db_path()
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        db_file = Path(self.db_path)
        if db_file.parent and str(db_file.parent) not in ("", "."):
            db_file.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    owner_id TEXT,
                    secret_hash TEXT NOT NULL UNIQUE,
                    revoked INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS protected_endpoints (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    method TEXT NOT NULL,
                    model_pattern TEXT NOT NULL DEFAULT '*'
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_protected_endpoint_unique
                    ON protected_endpoints(path, method, model_pattern);

                CREATE TABLE IF NOT EXISTS budgets (
                    id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    scope TEXT,
                    window TEXT NOT NULL,
                    budget_amount REAL NOT NULL,
                    created_at INTEGER NOT NULL,
                    UNIQUE(entity_type, entity_id, scope, window)
                );

                CREATE TABLE IF NOT EXISTS budget_usage (
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    scope TEXT,
                    window TEXT NOT NULL,
                    window_bucket TEXT NOT NULL,
                    cost REAL NOT NULL,
                    PRIMARY KEY(entity_type, entity_id, scope, window, window_bucket)
                );

                CREATE TABLE IF NOT EXISTS request_logs (
                    id TEXT PRIMARY KEY,
                    api_key_id TEXT,
                    timestamp INTEGER NOT NULL,
                    requested_model TEXT,
                    target_model_name TEXT,
                    provider TEXT,
                    scope TEXT,
                    usage REAL,
                    usage_unit TEXT,
                    price REAL,
                    price_per_unit REAL,
                    cost REAL,
                    FOREIGN KEY(api_key_id) REFERENCES api_keys(id)
                );
                """
            )

    @staticmethod
    def _hash_secret(secret: str) -> str:
        return hashlib.sha256(secret.encode("utf-8")).hexdigest()

    @staticmethod
    def _mask_secret(secret: str) -> str:
        if len(secret) < 8:
            return "*" * len(secret)
        return f"{secret[:4]}...{secret[-4:]}"

    @staticmethod
    def _new_secret() -> str:
        raw_uuid = str(uuid.uuid4())
        uuid_hash = hashlib.sha256(raw_uuid.encode("utf-8")).hexdigest()
        return f"foap-{uuid_hash}"

    def _secret_hash_exists(self, conn: sqlite3.Connection, secret_hash: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM api_keys WHERE secret_hash = ? LIMIT 1",
            (secret_hash,),
        ).fetchone()
        return row is not None

    def create_api_key(self, name: str, owner_id: Optional[str]) -> dict:
        now = int(time.time())

        with self._lock:
            with self._connect() as conn:
                while True:
                    key_id = str(uuid.uuid4())
                    secret = self._new_secret()
                    secret_hash = self._hash_secret(secret)

                    if self._secret_hash_exists(conn, secret_hash):
                        continue

                    try:
                        conn.execute(
                            """
                            INSERT INTO api_keys (id, name, owner_id, secret_hash, revoked, created_at)
                            VALUES (?, ?, ?, ?, 0, ?)
                            """,
                            (key_id, name, owner_id, secret_hash, now),
                        )
                        break
                    except sqlite3.IntegrityError:
                        continue

        return {
            "id": key_id,
            "name": name,
            "owner_id": owner_id,
            "masked_key": self._mask_secret(secret),
            "api_key": secret,
        }

    def list_api_keys(self, owner_id: Optional[str] = None) -> list[dict]:
        query = "SELECT id, name, owner_id, secret_hash FROM api_keys WHERE revoked = 0"
        params: tuple = ()
        if owner_id is not None:
            query += " AND owner_id = ?"
            params = (owner_id,)
        query += " ORDER BY created_at DESC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        def _masked_from_hash(secret_hash: str) -> str:
            if not secret_hash or len(secret_hash) < 12:
                return "foap-****"
            return f"foap-{secret_hash[:8]}…{secret_hash[-4:]}"

        keys: list[dict] = []
        for row in rows:
            keys.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "owner_id": row["owner_id"],
                    "masked_key": _masked_from_hash(row["secret_hash"]),
                }
            )
        return keys

    def revoke_api_key(self, key_id: str, owner_id: Optional[str] = None) -> bool:
        query = "UPDATE api_keys SET revoked = 1 WHERE id = ? AND revoked = 0"
        params: tuple = (key_id,)
        if owner_id is not None:
            query += " AND owner_id = ?"
            params = (key_id, owner_id)

        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(query, params)
                return cursor.rowcount > 0

    def verify_api_key(self, secret: str) -> Optional[dict]:
        secret_hash = self._hash_secret(secret)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, name, owner_id
                FROM api_keys
                WHERE secret_hash = ? AND revoked = 0
                """,
                (secret_hash,),
            ).fetchone()

        if row is None:
            return None

        return {"id": row["id"], "name": row["name"], "owner_id": row["owner_id"]}

    def get_api_key(self, key_id: str, owner_id: Optional[str] = None) -> Optional[dict]:
        query = "SELECT id, name, owner_id, revoked FROM api_keys WHERE id = ?"
        params: tuple = (key_id,)
        if owner_id is not None:
            query += " AND owner_id = ?"
            params = (key_id, owner_id)

        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()

        if row is None or int(row["revoked"]) == 1:
            return None

        return {"id": row["id"], "name": row["name"], "owner_id": row["owner_id"]}

    def key_exists(self, key_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM api_keys WHERE id = ? AND revoked = 0 LIMIT 1",
                (key_id,),
            ).fetchone()
        return row is not None

    @staticmethod
    def _normalize_method(method: str) -> str:
        return method.strip().upper()

    @staticmethod
    def _normalize_path(path: str) -> str:
        value = path.strip()
        if not value.startswith("/"):
            value = f"/{value}"
        if len(value) > 1 and value.endswith("/"):
            value = value[:-1]
        return value

    @staticmethod
    def _normalize_model(model: str) -> str:
        return model.strip()

    def list_protected_endpoints(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, path, method, model_pattern FROM protected_endpoints ORDER BY method, path, model_pattern"
            ).fetchall()
        return [
            {
                "id": row["id"],
                "path": row["path"],
                "method": row["method"],
                "model_pattern": row["model_pattern"] if "model_pattern" in row.keys() else "*",
            }
            for row in rows
        ]

    def create_protected_endpoint(self, path: str, method: str, model_pattern: str = '*') -> dict:
        endpoint_id = str(uuid.uuid4())
        normalized_path = self._normalize_path(path)
        normalized_method = self._normalize_method(method)

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO protected_endpoints (id, path, method, model_pattern)
                    VALUES (?, ?, ?, ?)
                    """,
                    (endpoint_id, normalized_path, normalized_method, model_pattern),
                )

        return {"id": endpoint_id, "path": normalized_path, "method": normalized_method, "model_pattern": model_pattern}

    def delete_protected_endpoint(self, endpoint_id: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    "DELETE FROM protected_endpoints WHERE id = ?",
                    (endpoint_id,),
                )
                return cursor.rowcount > 0

    def is_endpoint_protected(self, path: str, method: str, model: Optional[str] = None) -> bool:
        import fnmatch
        normalized_path = self._normalize_path(path)
        normalized_method = self._normalize_method(method)
        normalized_model = self._normalize_model(model) if model else None

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT path, method, model_pattern
                FROM protected_endpoints
                """
            ).fetchall()
            
            for row in rows:
                p_pattern = row["path"]
                m_pattern = row["method"]
                mod_pattern = row["model_pattern"] if "model_pattern" in row.keys() else "*"
                
                path_match = fnmatch.fnmatch(normalized_path, p_pattern) or normalized_path == p_pattern
                method_match = m_pattern == "*" or m_pattern == normalized_method
                model_match = True
                
                if normalized_model and mod_pattern != "*":
                    model_match = fnmatch.fnmatch(normalized_model, mod_pattern)
                elif not normalized_model and mod_pattern != "*":
                    model_match = False
                    
                if path_match and method_match and model_match:
                    return True
                    
        return False

    # --- BUDGETS ---
    
    def list_budgets(self, entity_type: Optional[str] = None, entity_id: Optional[str] = None) -> list[dict]:
        query = "SELECT * FROM budgets WHERE 1=1"
        params = []
        if entity_type:
            query += " AND entity_type = ?"
            params.append(entity_type)
        if entity_id:
            query += " AND entity_id = ?"
            params.append(entity_id)
        
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            return [dict(row) for row in rows]
            
    def get_budget(self, budget_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM budgets WHERE id = ?", (budget_id,)).fetchone()
            return dict(row) if row else None

    def create_budget(self, entity_type: str, entity_id: str, window: str, budget_amount: float, scope: Optional[str] = None) -> dict:
        budget_id = str(uuid.uuid4())
        now = int(time.time())
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO budgets (id, entity_type, entity_id, scope, window, budget_amount, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (budget_id, entity_type, entity_id, scope, window, budget_amount, now)
                )
        return self.get_budget(budget_id)

    def update_budget(self, budget_id: str, budget_amount: float) -> Optional[dict]:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute("UPDATE budgets SET budget_amount = ? WHERE id = ?", (budget_amount, budget_id))
                if cursor.rowcount == 0:
                    return None
        return self.get_budget(budget_id)

    def delete_budget(self, budget_id: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute("DELETE FROM budgets WHERE id = ?", (budget_id,))
                return cursor.rowcount > 0

    # --- Async Worker / Usage Logs ---

    def log_request(self, api_key_id: Optional[str], timestamp: int, requested_model: Optional[str], 
                    target_model_name: Optional[str], provider: Optional[str], scope: Optional[str], 
                    usage: Optional[float], usage_unit: Optional[str], price: Optional[float], 
                    price_per_unit: Optional[float], cost: Optional[float]) -> str:
        log_id = str(uuid.uuid4())
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO request_logs (id, api_key_id, timestamp, requested_model, target_model_name, provider, scope, usage, usage_unit, price, price_per_unit, cost)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (log_id, api_key_id, timestamp, requested_model, target_model_name, provider, scope, usage, usage_unit, price, price_per_unit, cost)
                )
        return log_id

    def add_budget_usage(self, entity_type: str, entity_id: str, window: str, window_bucket: str, cost: float, scope: Optional[str] = None) -> None:
        if cost == 0.0:
            return
            
        with self._lock:
            with self._connect() as conn:
                # Upsert budget usage
                # We normalize scope to empty string if None for the unique constraint since NULL behaves differently in some unique indexes.
                mod_scope = scope if scope is not None else ""
                
                row = conn.execute(
                    """
                    SELECT cost FROM budget_usage
                    WHERE entity_type = ? AND entity_id = ? AND scope = ? AND window = ? AND window_bucket = ?
                    """,
                    (entity_type, entity_id, mod_scope, window, window_bucket)
                ).fetchone()
                
                if row:
                    conn.execute(
                        """
                        UPDATE budget_usage
                        SET cost = cost + ?
                        WHERE entity_type = ? AND entity_id = ? AND scope = ? AND window = ? AND window_bucket = ?
                        """,
                        (cost, entity_type, entity_id, mod_scope, window, window_bucket)
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO budget_usage (entity_type, entity_id, scope, window, window_bucket, cost)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (entity_type, entity_id, mod_scope, window, window_bucket, cost)
                    )

    def get_all_budget_usage(self, entity_type: str, entity_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM budget_usage WHERE entity_type = ? AND entity_id = ?",
                (entity_type, entity_id)
            ).fetchall()
            return [dict(row) for row in rows]

store = AccessStore()
