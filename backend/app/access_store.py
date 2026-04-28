import hashlib
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

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
                    method TEXT NOT NULL,
                    model_pattern TEXT NOT NULL DEFAULT '*'
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_protected_endpoint_unique
                    ON protected_endpoints(path, method, model_pattern);

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

                CREATE TABLE IF NOT EXISTS usage_counters (
                    owner_id TEXT NOT NULL,
                    api_path TEXT NOT NULL,
                    model TEXT NOT NULL,
                    window_type TEXT NOT NULL,
                    window_bucket INTEGER NOT NULL,
                    request_count INTEGER NOT NULL,
                    PRIMARY KEY(owner_id, api_path, model, window_type, window_bucket)
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

                CREATE TABLE IF NOT EXISTS quota_overrides (
                    id TEXT PRIMARY KEY,
                    api_path TEXT NOT NULL,
                    model TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    window_type TEXT NOT NULL,
                    request_limit INTEGER NOT NULL,
                    exempt INTEGER NOT NULL DEFAULT 0,
                    starts_at INTEGER,
                    ends_at INTEGER,
                    created_at INTEGER NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_quota_override_unique
                    ON quota_overrides(api_path, model, owner_id);

                CREATE TABLE IF NOT EXISTS quota_override_usage (
                    override_id TEXT NOT NULL,
                    bucket_key TEXT NOT NULL,
                    window_bucket INTEGER NOT NULL,
                    request_count INTEGER NOT NULL,
                    PRIMARY KEY(override_id, bucket_key, window_bucket)
                );

                CREATE TABLE IF NOT EXISTS providers (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    api_key_variable TEXT,
                    prefix TEXT NOT NULL DEFAULT '',
                    default_base_url TEXT,
                    default_request_timeout INTEGER,
                    default_health_timeout INTEGER,
                    max_upstream_retry_seconds INTEGER DEFAULT 0,
                    sync_provider_ratelimits BOOLEAN DEFAULT 0,
                    route_fallbacks TEXT DEFAULT '{}',
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS provider_models (
                    id TEXT PRIMARY KEY,
                    provider_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    owned_by TEXT DEFAULT 'FOAP',
                    hide_on_models_endpoint BOOLEAN DEFAULT 0,
                    FOREIGN KEY(provider_id) REFERENCES providers(id),
                    UNIQUE(provider_id, name)
                );

                CREATE TABLE IF NOT EXISTS provider_model_endpoints (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    target_model_name TEXT NOT NULL,
                    target_base_url TEXT,
                    request_timeout INTEGER,
                    health_timeout INTEGER,
                    fallback_model_name TEXT,
                    FOREIGN KEY(model_id) REFERENCES provider_models(id),
                    UNIQUE(model_id, path)
                );

                CREATE TABLE IF NOT EXISTS model_aliases (
                    id TEXT PRIMARY KEY,
                    alias_name TEXT NOT NULL UNIQUE,
                    target_model_name TEXT NOT NULL,
                    owned_by TEXT DEFAULT 'FOAP',
                    hide_on_models_endpoint BOOLEAN DEFAULT 0,
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS provider_ratelimits (
                    provider_name TEXT PRIMARY KEY,
                    limit_minute INTEGER,
                    remaining_minute INTEGER,
                    limit_hour INTEGER,
                    remaining_hour INTEGER,
                    limit_day INTEGER,
                    remaining_day INTEGER,
                    updated_at INTEGER NOT NULL
                );
                """
            )

            try:
                with conn:
                    conn.execute("ALTER TABLE protected_endpoints ADD COLUMN model_pattern TEXT NOT NULL DEFAULT '*'")
            except sqlite3.OperationalError:
                pass
            
            try:
                with conn:
                    conn.execute("ALTER TABLE providers ADD COLUMN max_upstream_retry_seconds INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass

            try:
                with conn:
                    conn.execute("ALTER TABLE providers ADD COLUMN sync_provider_ratelimits BOOLEAN DEFAULT 0")
            except sqlite3.OperationalError:
                pass

            try:
                with conn:
                    conn.execute("ALTER TABLE provider_model_endpoints ADD COLUMN fallback_model_name TEXT")
            except sqlite3.OperationalError:
                pass
            
            try:
                with conn:
                    conn.execute("ALTER TABLE providers ADD COLUMN route_fallbacks TEXT DEFAULT '{}'")
            except sqlite3.OperationalError:
                pass

            # provider_models: owned_by + hide_on_models_endpoint
            try:
                with conn:
                    conn.execute("ALTER TABLE provider_models ADD COLUMN owned_by TEXT DEFAULT 'FOAP'")
            except sqlite3.OperationalError:
                pass
            try:
                with conn:
                    conn.execute("ALTER TABLE provider_models ADD COLUMN hide_on_models_endpoint BOOLEAN DEFAULT 0")
            except sqlite3.OperationalError:
                pass

            # model_aliases: owned_by + hide_on_models_endpoint
            try:
                with conn:
                    conn.execute("ALTER TABLE model_aliases ADD COLUMN owned_by TEXT DEFAULT 'FOAP'")
            except sqlite3.OperationalError:
                pass
            try:
                with conn:
                    conn.execute("ALTER TABLE model_aliases ADD COLUMN hide_on_models_endpoint BOOLEAN DEFAULT 0")
            except sqlite3.OperationalError:
                pass

    def get_provider_ratelimits(self, provider_name: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT provider_name, limit_minute, remaining_minute, limit_hour, remaining_hour, limit_day, remaining_day, updated_at
                FROM provider_ratelimits
                WHERE provider_name = ?
                """,
                (provider_name,),
            ).fetchone()

        return dict(row) if row is not None else None

    def sync_provider_ratelimits(self, provider_name: str, limits: dict[str, Any]) -> dict:
        now = int(time.time())
        existing = self.get_provider_ratelimits(provider_name) or {}
        merged = {
            "provider_name": provider_name,
            "limit_minute": existing.get("limit_minute"),
            "remaining_minute": existing.get("remaining_minute"),
            "limit_hour": existing.get("limit_hour"),
            "remaining_hour": existing.get("remaining_hour"),
            "limit_day": existing.get("limit_day"),
            "remaining_day": existing.get("remaining_day"),
        }

        for field in ("limit_minute", "remaining_minute", "limit_hour", "remaining_hour", "limit_day", "remaining_day"):
            if field in limits and limits[field] is not None:
                merged[field] = int(limits[field])

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO provider_ratelimits (
                        provider_name,
                        limit_minute,
                        remaining_minute,
                        limit_hour,
                        remaining_hour,
                        limit_day,
                        remaining_day,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(provider_name)
                    DO UPDATE SET
                        limit_minute = excluded.limit_minute,
                        remaining_minute = excluded.remaining_minute,
                        limit_hour = excluded.limit_hour,
                        remaining_hour = excluded.remaining_hour,
                        limit_day = excluded.limit_day,
                        remaining_day = excluded.remaining_day,
                        updated_at = excluded.updated_at
                    """,
                    (
                        provider_name,
                        merged["limit_minute"],
                        merged["remaining_minute"],
                        merged["limit_hour"],
                        merged["remaining_hour"],
                        merged["limit_day"],
                        merged["remaining_day"],
                        now,
                    ),
                )

        merged["updated_at"] = now
        return merged

    def get_exhausted_provider_ratelimit_windows(self, provider_name: str) -> list[str]:
        snapshot = self.get_provider_ratelimits(provider_name)
        if not snapshot:
            return []

        exhausted: list[str] = []
        for window in ("minute", "hour", "day"):
            remaining = snapshot.get(f"remaining_{window}")
            if isinstance(remaining, int) and remaining <= 0:
                exhausted.append(window)
        return exhausted

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
            # Show a compact, non-reversible masked label with foap- prefix (does not reveal the secret)
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

    def record_usage(self, owner_id: str, api_path: str, model: str) -> None:
        normalized_owner = self._normalize_owner_id(owner_id)
        normalized_path = self._normalize_path(api_path)
        normalized_model = self._normalize_model(model)
        now = int(time.time())

        with self._lock:
            with self._connect() as conn:
                for window_type in ("minute", "hour", "day"):
                    window_bucket, _ = self._window_bucket(now, window_type)
                    conn.execute(
                        """
                        INSERT INTO usage_counters (owner_id, api_path, model, window_type, window_bucket, request_count)
                        VALUES (?, ?, ?, ?, ?, 1)
                        ON CONFLICT(owner_id, api_path, model, window_type, window_bucket)
                        DO UPDATE SET request_count = request_count + 1
                        """,
                        (normalized_owner, normalized_path, normalized_model, window_type, window_bucket),
                    )

    def get_usage_summary(
        self,
        owner_id: str,
        model: Optional[str] = None,
        api_path: Optional[str] = None,
        window_size: int = 6,
    ) -> dict:
        normalized_owner = self._normalize_owner_id(owner_id)
        normalized_model = self._normalize_model(model) if model is not None else None
        normalized_path = self._normalize_path(api_path) if api_path is not None else None
        window_size = max(1, min(int(window_size), 96))
        now = int(time.time())

        windows: dict[str, Any] = {"minute": [], "hour": [], "day": []}
        totals: dict[str, int] = {"minute": 0, "hour": 0, "day": 0}

        with self._connect() as conn:
            for window_type in ("minute", "hour", "day"):
                window_bucket, reset_in_seconds = self._window_bucket(now, window_type)

                clauses = ["owner_id = ?", "window_type = ?", "window_bucket = ?"]
                params: list[object] = [normalized_owner, window_type, window_bucket]

                if normalized_model is not None:
                    clauses.append("model = ?")
                    params.append(normalized_model)
                if normalized_path is not None:
                    clauses.append("api_path = ?")
                    params.append(normalized_path)

                where_sql = " AND ".join(clauses)

                total_row = conn.execute(
                    f"SELECT COALESCE(SUM(request_count), 0) AS total FROM usage_counters WHERE {where_sql}",
                    params,
                ).fetchone()
                totals[window_type] = int(total_row["total"]) if total_row is not None else 0

                rows = conn.execute(
                    f"""
                    SELECT api_path, model, request_count
                    FROM usage_counters
                    WHERE {where_sql}
                    ORDER BY request_count DESC, model, api_path
                    """,
                    params,
                ).fetchall()

                windows[window_type] = [
                    {
                        "api_path": row["api_path"],
                        "model": row["model"],
                        "request_count": int(row["request_count"]),
                    }
                    for row in rows
                ]

                trend_rows = conn.execute(
                    f"""
                    SELECT window_bucket, COALESCE(SUM(request_count), 0) AS total
                    FROM usage_counters
                    WHERE owner_id = ?
                      AND window_type = ?
                      AND window_bucket BETWEEN ? AND ?
                      {"AND model = ?" if normalized_model is not None else ""}
                      {"AND api_path = ?" if normalized_path is not None else ""}
                    GROUP BY window_bucket
                    ORDER BY window_bucket ASC
                    """,
                    (
                        [
                            normalized_owner,
                            window_type,
                            window_bucket - window_size + 1,
                            window_bucket,
                        ]
                        + ([normalized_model] if normalized_model is not None else [])
                        + ([normalized_path] if normalized_path is not None else [])
                    ),
                ).fetchall()

                trend_map = {int(row["window_bucket"]): int(row["total"]) for row in trend_rows}
                windows[window_type + "_trend"] = [
                    {
                        "window_bucket": bucket,
                        "request_count": trend_map.get(bucket, 0),
                    }
                    for bucket in range(window_bucket - window_size + 1, window_bucket + 1)
                ]

                windows[window_type + "_meta"] = {
                    "window_bucket": window_bucket,
                    "reset_in_seconds": reset_in_seconds,
                }

        return {
            "owner_id": normalized_owner,
            "generated_at": now,
            "filters": {
                "model": normalized_model,
                "api_path": normalized_path,
                "window_size": window_size,
            },
            "totals": totals,
            "windows": windows,
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
    def _normalize_owner_id(owner_id: str) -> str:
        return owner_id.strip()

    @staticmethod
    def _quota_override_state(starts_at: Optional[int], ends_at: Optional[int], now: Optional[int] = None) -> tuple[bool, str]:
        now = int(time.time()) if now is None else now
        active_now = (starts_at is None or starts_at <= now) and (ends_at is None or ends_at > now)
        if active_now:
            return True, "active"
        if starts_at is not None and starts_at > now:
            return False, "scheduled"
        return False, "expired"

    @staticmethod
    def _build_quota_override_read(row: sqlite3.Row, now: Optional[int] = None) -> dict:
        active_now, window_state = AccessStore._quota_override_state(row["starts_at"], row["ends_at"], now=now)
        return {
            "id": row["id"],
            "api_path": row["api_path"],
            "model": row["model"],
            "owner_id": row["owner_id"],
            "window_type": row["window_type"],
            "request_limit": int(row["request_limit"]),
            "exempt": bool(row["exempt"]),
            "starts_at": row["starts_at"],
            "ends_at": row["ends_at"],
            "created_at": int(row["created_at"]),
            "active_now": active_now,
            "window_state": window_state,
        }

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

    def list_quota_policies(
        self,
        api_path: Optional[str] = None,
        model: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> tuple[list[dict], int]:
        clauses: list[str] = []
        params: list[object] = []

        if api_path is not None:
            clauses.append("api_path = ?")
            params.append(self._normalize_path(api_path))
        if model is not None:
            clauses.append("model = ?")
            params.append(self._normalize_model(model))

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        count_sql = f"SELECT COUNT(*) AS total FROM quota_policies {where_sql}"
        data_sql = f"""
            SELECT id, api_path, model, window_type, request_limit, enforce_per_user
            FROM quota_policies
            {where_sql}
            ORDER BY api_path, model
        """

        if limit is not None:
            data_sql += " LIMIT ?"
            params_with_pagination = [*params, limit]
            if offset is not None:
                data_sql += " OFFSET ?"
                params_with_pagination.append(offset)
        else:
            params_with_pagination = params

        with self._connect() as conn:
            total_row = conn.execute(count_sql, params).fetchone()
            rows = conn.execute(data_sql, params_with_pagination).fetchall()

        total = 0 if total_row is None else int(total_row["total"])
        
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
        ], total

    def get_quota_policy(self, policy_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, api_path, model, window_type, request_limit, enforce_per_user
                FROM quota_policies
                WHERE id = ?
                """,
                (policy_id,),
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

    def list_quota_overrides(
        self,
        owner_id: Optional[str] = None,
        api_path: Optional[str] = None,
        model: Optional[str] = None,
        exempt: Optional[bool] = None,
        active_only: bool = False,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> tuple[list[dict], int]:
        clauses: list[str] = []
        params: list[object] = []
        now = int(time.time())

        if owner_id is not None:
            clauses.append("owner_id = ?")
            params.append(self._normalize_owner_id(owner_id))
        if api_path is not None:
            clauses.append("api_path = ?")
            params.append(self._normalize_path(api_path))
        if model is not None:
            clauses.append("model = ?")
            params.append(self._normalize_model(model))
        if exempt is not None:
            clauses.append("exempt = ?")
            params.append(1 if exempt else 0)
        if active_only:
            clauses.append("(starts_at IS NULL OR starts_at <= ?)")
            clauses.append("(ends_at IS NULL OR ends_at > ?)")
            params.extend([now, now])

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        count_sql = f"SELECT COUNT(*) AS total FROM quota_overrides {where_sql}"
        data_sql = f"""
            SELECT id, api_path, model, owner_id, window_type, request_limit, exempt, starts_at, ends_at, created_at
            FROM quota_overrides
            {where_sql}
            ORDER BY created_at DESC, api_path, model, owner_id
        """

        if limit is not None:
            data_sql += " LIMIT ?"
            params_with_pagination = [*params, limit]
            if offset is not None:
                data_sql += " OFFSET ?"
                params_with_pagination.append(offset)
        else:
            params_with_pagination = params

        with self._connect() as conn:
            total_row = conn.execute(count_sql, params).fetchone()
            rows = conn.execute(data_sql, params_with_pagination).fetchall()

        total = 0 if total_row is None else int(total_row["total"])
        return [self._build_quota_override_read(row, now=now) for row in rows], total

    def get_quota_override(self, override_id: str) -> Optional[dict]:
        now = int(time.time())
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, api_path, model, owner_id, window_type, request_limit, exempt, starts_at, ends_at, created_at
                FROM quota_overrides
                WHERE id = ?
                """,
                (override_id,),
            ).fetchone()
            
        if row is None:
            return None
            
        return self._build_quota_override_read(row, now=now)

    def create_quota_override(
        self,
        api_path: str,
        model: str,
        owner_id: str,
        window_type: str,
        request_limit: int,
        exempt: bool,
        starts_at: Optional[int],
        ends_at: Optional[int],
    ) -> dict:
        override_id = str(uuid.uuid4())
        normalized_path = self._normalize_path(api_path)
        normalized_model = self._normalize_model(model)
        normalized_owner = self._normalize_owner_id(owner_id)
        now = int(time.time())

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO quota_overrides
                    (id, api_path, model, owner_id, window_type, request_limit, exempt, starts_at, ends_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        override_id,
                        normalized_path,
                        normalized_model,
                        normalized_owner,
                        window_type,
                        request_limit,
                        1 if exempt else 0,
                        starts_at,
                        ends_at,
                        now,
                    ),
                )

        return {
            "id": override_id,
            "api_path": normalized_path,
            "model": normalized_model,
            "owner_id": normalized_owner,
            "window_type": window_type,
            "request_limit": request_limit,
            "exempt": exempt,
            "starts_at": starts_at,
            "ends_at": ends_at,
            "created_at": now,
            "active_now": self._quota_override_state(starts_at, ends_at, now=now)[0],
            "window_state": self._quota_override_state(starts_at, ends_at, now=now)[1],
        }

    def update_quota_override(
        self,
        override_id: str,
        api_path: str,
        model: str,
        owner_id: str,
        window_type: str,
        request_limit: int,
        exempt: bool,
        starts_at: Optional[int],
        ends_at: Optional[int],
    ) -> Optional[dict]:
        normalized_path = self._normalize_path(api_path)
        normalized_model = self._normalize_model(model)
        normalized_owner = self._normalize_owner_id(owner_id)

        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    UPDATE quota_overrides
                    SET api_path = ?, model = ?, owner_id = ?, window_type = ?, request_limit = ?, exempt = ?, starts_at = ?, ends_at = ?
                    WHERE id = ?
                    """,
                    (
                        normalized_path,
                        normalized_model,
                        normalized_owner,
                        window_type,
                        request_limit,
                        1 if exempt else 0,
                        starts_at,
                        ends_at,
                        override_id,
                    ),
                )
                if cursor.rowcount == 0:
                    return None

                row = conn.execute(
                    """
                    SELECT created_at
                    FROM quota_overrides
                    WHERE id = ?
                    """,
                    (override_id,),
                ).fetchone()

        return {
            "id": override_id,
            "api_path": normalized_path,
            "model": normalized_model,
            "owner_id": normalized_owner,
            "window_type": window_type,
            "request_limit": request_limit,
            "exempt": exempt,
            "starts_at": starts_at,
            "ends_at": ends_at,
            "created_at": int(row["created_at"]) if row is not None else int(time.time()),
            "active_now": self._quota_override_state(starts_at, ends_at)[0],
            "window_state": self._quota_override_state(starts_at, ends_at)[1],
        }

    def delete_quota_override(self, override_id: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM quota_override_usage WHERE override_id = ?", (override_id,))
                cursor = conn.execute("DELETE FROM quota_overrides WHERE id = ?", (override_id,))
                return cursor.rowcount > 0

    def find_active_quota_override(self, api_path: str, model: str, owner_id: str) -> Optional[dict]:
        normalized_path = self._normalize_path(api_path)
        normalized_model = self._normalize_model(model)
        normalized_owner = self._normalize_owner_id(owner_id)
        now = int(time.time())

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, api_path, model, owner_id, window_type, request_limit, exempt, starts_at, ends_at, created_at
                FROM quota_overrides
                WHERE api_path = ?
                  AND model = ?
                  AND owner_id = ?
                  AND (starts_at IS NULL OR starts_at <= ?)
                  AND (ends_at IS NULL OR ends_at > ?)
                LIMIT 1
                """,
                (normalized_path, normalized_model, normalized_owner, now, now),
            ).fetchone()

        if row is None:
            return None

        return {
            "id": row["id"],
            "api_path": row["api_path"],
            "model": row["model"],
            "owner_id": row["owner_id"],
            "window_type": row["window_type"],
            "request_limit": int(row["request_limit"]),
            "exempt": bool(row["exempt"]),
            "starts_at": row["starts_at"],
            "ends_at": row["ends_at"],
            "created_at": int(row["created_at"]),
            "active_now": self._quota_override_state(row["starts_at"], row["ends_at"], now=now)[0],
            "window_state": self._quota_override_state(row["starts_at"], row["ends_at"], now=now)[1],
        }

    def consume_quota_override(
        self, override_id: str, bucket_key: str, window_type: str, request_limit: int
    ) -> tuple[bool, int]:
        now = int(time.time())
        window_bucket, retry_after = self._window_bucket(now, window_type)

        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT request_count
                    FROM quota_override_usage
                    WHERE override_id = ? AND bucket_key = ? AND window_bucket = ?
                    """,
                    (override_id, bucket_key, window_bucket),
                ).fetchone()

                current_count = 0 if row is None else int(row["request_count"])
                if current_count >= request_limit:
                    return False, retry_after

                if row is None:
                    conn.execute(
                        """
                        INSERT INTO quota_override_usage (override_id, bucket_key, window_bucket, request_count)
                        VALUES (?, ?, ?, 1)
                        """,
                        (override_id, bucket_key, window_bucket),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE quota_override_usage
                        SET request_count = request_count + 1
                        WHERE override_id = ? AND bucket_key = ? AND window_bucket = ?
                        """,
                        (override_id, bucket_key, window_bucket),
                    )

        return True, retry_after


store = AccessStore()

