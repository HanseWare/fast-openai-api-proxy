import hashlib
import secrets
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from config import get_access_db_path


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
                    method TEXT NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_protected_endpoint_unique
                    ON protected_endpoints(path, method);

                CREATE TABLE IF NOT EXISTS api_key_quotas (
                    api_key_id TEXT PRIMARY KEY,
                    requests_per_minute INTEGER NOT NULL,
                    FOREIGN KEY(api_key_id) REFERENCES api_keys(id)
                );

                CREATE TABLE IF NOT EXISTS api_key_usage (
                    api_key_id TEXT NOT NULL,
                    window_minute INTEGER NOT NULL,
                    request_count INTEGER NOT NULL,
                    PRIMARY KEY(api_key_id, window_minute)
                );

                CREATE TABLE IF NOT EXISTS quota_policies (
                    id TEXT PRIMARY KEY,
                    api_path TEXT NOT NULL,
                    model TEXT NOT NULL,
                    window_type TEXT NOT NULL,
                    request_limit INTEGER NOT NULL,
                    enforce_per_user INTEGER NOT NULL DEFAULT 1,
                    created_at INTEGER NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_quota_policy_unique
                    ON quota_policies(api_path, model);

                CREATE TABLE IF NOT EXISTS quota_policy_usage (
                    policy_id TEXT NOT NULL,
                    bucket_key TEXT NOT NULL,
                    window_bucket INTEGER NOT NULL,
                    request_count INTEGER NOT NULL,
                    PRIMARY KEY(policy_id, bucket_key, window_bucket)
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
        return f"foap_{secrets.token_urlsafe(24)}"

    def create_api_key(self, name: str, owner_id: Optional[str]) -> dict:
        secret = self._new_secret()
        now = int(time.time())
        key_id = str(uuid.uuid4())

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO api_keys (id, name, owner_id, secret_hash, revoked, created_at)
                    VALUES (?, ?, ?, ?, 0, ?)
                    """,
                    (key_id, name, owner_id, self._hash_secret(secret), now),
                )

        return {
            "id": key_id,
            "name": name,
            "owner_id": owner_id,
            "masked_key": self._mask_secret(secret),
            "api_key": secret,
        }

    def list_api_keys(self, owner_id: Optional[str] = None) -> list[dict]:
        query = "SELECT id, name, owner_id FROM api_keys WHERE revoked = 0"
        params: tuple = ()
        if owner_id is not None:
            query += " AND owner_id = ?"
            params = (owner_id,)
        query += " ORDER BY created_at DESC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        keys: list[dict] = []
        for row in rows:
            keys.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "owner_id": row["owner_id"],
                    "masked_key": "hidden",
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

    def key_exists(self, key_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM api_keys WHERE id = ? AND revoked = 0 LIMIT 1",
                (key_id,),
            ).fetchone()
        return row is not None

    def set_quota(self, api_key_id: str, requests_per_minute: int) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO api_key_quotas (api_key_id, requests_per_minute)
                    VALUES (?, ?)
                    ON CONFLICT(api_key_id)
                    DO UPDATE SET requests_per_minute = excluded.requests_per_minute
                    """,
                    (api_key_id, requests_per_minute),
                )

    def get_quota(self, api_key_id: str) -> Optional[int]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT requests_per_minute FROM api_key_quotas WHERE api_key_id = ?",
                (api_key_id,),
            ).fetchone()
        return None if row is None else int(row["requests_per_minute"])

    def consume_quota(self, api_key_id: str) -> tuple[bool, Optional[int]]:
        quota = self.get_quota(api_key_id)
        if quota is None:
            return True, None

        window_minute = int(time.time() // 60)
        retry_after = 60 - int(time.time() % 60)

        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT request_count
                    FROM api_key_usage
                    WHERE api_key_id = ? AND window_minute = ?
                    """,
                    (api_key_id, window_minute),
                ).fetchone()

                current_count = 0 if row is None else int(row["request_count"])
                if current_count >= quota:
                    return False, retry_after

                if row is None:
                    conn.execute(
                        """
                        INSERT INTO api_key_usage (api_key_id, window_minute, request_count)
                        VALUES (?, ?, 1)
                        """,
                        (api_key_id, window_minute),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE api_key_usage
                        SET request_count = request_count + 1
                        WHERE api_key_id = ? AND window_minute = ?
                        """,
                        (api_key_id, window_minute),
                    )

        return True, None

    def get_quota_usage(self, api_key_id: str) -> Optional[dict]:
        quota = self.get_quota(api_key_id)
        if quota is None:
            return None

        now = int(time.time())
        window_minute = now // 60
        reset_in_seconds = 60 - (now % 60)

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT request_count
                FROM api_key_usage
                WHERE api_key_id = ? AND window_minute = ?
                """,
                (api_key_id, window_minute),
            ).fetchone()

        used = 0 if row is None else int(row["request_count"])
        remaining = max(quota - used, 0)
        return {
            "api_key_id": api_key_id,
            "requests_per_minute": quota,
            "used": used,
            "remaining": remaining,
            "reset_in_seconds": reset_in_seconds,
        }

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

    @staticmethod
    def _window_bucket(now: int, window_type: str) -> tuple[int, int]:
        if window_type == "minute":
            return now // 60, 60 - (now % 60)
        if window_type == "hour":
            return now // 3600, 3600 - (now % 3600)
        if window_type == "day":
            return now // 86400, 86400 - (now % 86400)
        raise ValueError(f"Unsupported window_type: {window_type}")

    def list_protected_endpoints(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, path, method FROM protected_endpoints ORDER BY method, path"
            ).fetchall()
        return [{"id": row["id"], "path": row["path"], "method": row["method"]} for row in rows]

    def create_protected_endpoint(self, path: str, method: str) -> dict:
        endpoint_id = str(uuid.uuid4())
        normalized_path = self._normalize_path(path)
        normalized_method = self._normalize_method(method)

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO protected_endpoints (id, path, method)
                    VALUES (?, ?, ?)
                    """,
                    (endpoint_id, normalized_path, normalized_method),
                )

        return {"id": endpoint_id, "path": normalized_path, "method": normalized_method}

    def delete_protected_endpoint(self, endpoint_id: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    "DELETE FROM protected_endpoints WHERE id = ?",
                    (endpoint_id,),
                )
                return cursor.rowcount > 0

    def is_endpoint_protected(self, path: str, method: str) -> bool:
        normalized_path = self._normalize_path(path)
        normalized_method = self._normalize_method(method)

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM protected_endpoints
                WHERE path = ? AND method = ?
                LIMIT 1
                """,
                (normalized_path, normalized_method),
            ).fetchone()
        return row is not None

    def list_quota_policies(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, api_path, model, window_type, request_limit, enforce_per_user
                FROM quota_policies
                ORDER BY api_path, model
                """
            ).fetchall()
        return [
            {
                "id": row["id"],
                "api_path": row["api_path"],
                "model": row["model"],
                "window_type": row["window_type"],
                "request_limit": int(row["request_limit"]),
                "enforce_per_user": bool(row["enforce_per_user"]),
            }
            for row in rows
        ]

    def create_quota_policy(
        self,
        api_path: str,
        model: str,
        window_type: str,
        request_limit: int,
        enforce_per_user: bool,
    ) -> dict:
        policy_id = str(uuid.uuid4())
        normalized_path = self._normalize_path(api_path)
        normalized_model = self._normalize_model(model)
        now = int(time.time())

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO quota_policies (id, api_path, model, window_type, request_limit, enforce_per_user, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        policy_id,
                        normalized_path,
                        normalized_model,
                        window_type,
                        request_limit,
                        1 if enforce_per_user else 0,
                        now,
                    ),
                )

        return {
            "id": policy_id,
            "api_path": normalized_path,
            "model": normalized_model,
            "window_type": window_type,
            "request_limit": request_limit,
            "enforce_per_user": enforce_per_user,
        }

    def update_quota_policy(
        self,
        policy_id: str,
        api_path: str,
        model: str,
        window_type: str,
        request_limit: int,
        enforce_per_user: bool,
    ) -> Optional[dict]:
        normalized_path = self._normalize_path(api_path)
        normalized_model = self._normalize_model(model)

        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    UPDATE quota_policies
                    SET api_path = ?, model = ?, window_type = ?, request_limit = ?, enforce_per_user = ?
                    WHERE id = ?
                    """,
                    (
                        normalized_path,
                        normalized_model,
                        window_type,
                        request_limit,
                        1 if enforce_per_user else 0,
                        policy_id,
                    ),
                )
                if cursor.rowcount == 0:
                    return None

        return {
            "id": policy_id,
            "api_path": normalized_path,
            "model": normalized_model,
            "window_type": window_type,
            "request_limit": request_limit,
            "enforce_per_user": enforce_per_user,
        }

    def delete_quota_policy(self, policy_id: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM quota_policy_usage WHERE policy_id = ?", (policy_id,))
                cursor = conn.execute("DELETE FROM quota_policies WHERE id = ?", (policy_id,))
                return cursor.rowcount > 0

    def find_quota_policy(self, api_path: str, model: str) -> Optional[dict]:
        normalized_path = self._normalize_path(api_path)
        normalized_model = self._normalize_model(model)

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, api_path, model, window_type, request_limit, enforce_per_user
                FROM quota_policies
                WHERE api_path = ? AND model = ?
                LIMIT 1
                """,
                (normalized_path, normalized_model),
            ).fetchone()

        if row is None:
            return None

        return {
            "id": row["id"],
            "api_path": row["api_path"],
            "model": row["model"],
            "window_type": row["window_type"],
            "request_limit": int(row["request_limit"]),
            "enforce_per_user": bool(row["enforce_per_user"]),
        }

    def consume_quota_policy(self, policy_id: str, bucket_key: str, window_type: str, request_limit: int) -> tuple[bool, int]:
        now = int(time.time())
        window_bucket, retry_after = self._window_bucket(now, window_type)

        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT request_count
                    FROM quota_policy_usage
                    WHERE policy_id = ? AND bucket_key = ? AND window_bucket = ?
                    """,
                    (policy_id, bucket_key, window_bucket),
                ).fetchone()

                current_count = 0 if row is None else int(row["request_count"])
                if current_count >= request_limit:
                    return False, retry_after

                if row is None:
                    conn.execute(
                        """
                        INSERT INTO quota_policy_usage (policy_id, bucket_key, window_bucket, request_count)
                        VALUES (?, ?, ?, 1)
                        """,
                        (policy_id, bucket_key, window_bucket),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE quota_policy_usage
                        SET request_count = request_count + 1
                        WHERE policy_id = ? AND bucket_key = ? AND window_bucket = ?
                        """,
                        (policy_id, bucket_key, window_bucket),
                    )

        return True, retry_after


store = AccessStore()



