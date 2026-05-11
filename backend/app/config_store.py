import sqlite3
import threading
import time
import uuid
from typing import Optional, List, Dict, Any
from config import get_access_db_path

class ConfigStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or get_access_db_path()
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS providers (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    api_key_variable TEXT,
                    prefix TEXT NOT NULL DEFAULT '',
                    base_url TEXT,
                    request_timeout INTEGER,
                    health_timeout INTEGER,
                    max_upstream_retry_seconds INTEGER DEFAULT 0,
                    sync_provider_ratelimits BOOLEAN DEFAULT 0,
                    route_fallbacks TEXT DEFAULT '{}',
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS provider_models (
                    id TEXT PRIMARY KEY,
                    provider_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL DEFAULT 'llm',
                    target_model_name TEXT NOT NULL,
                    target_base_url TEXT,
                    fallback_model_name TEXT,
                    supported_endpoints TEXT DEFAULT '[]',
                    price_per_unit REAL DEFAULT 0.0,
                    min_credits_per_request REAL DEFAULT 0.0,
                    owned_by TEXT DEFAULT 'FOAP',
                    hide_on_models_endpoint BOOLEAN DEFAULT 0,
                    FOREIGN KEY(provider_id) REFERENCES providers(id),
                    UNIQUE(provider_id, name)
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
                    limit_second INTEGER,
                    remaining_second INTEGER,
                    limit_minute INTEGER,
                    remaining_minute INTEGER,
                    limit_hour INTEGER,
                    remaining_hour INTEGER,
                    limit_day INTEGER,
                    remaining_day INTEGER,
                    limit_month INTEGER,
                    remaining_month INTEGER,
                    ratelimit_limit INTEGER,
                    ratelimit_remaining INTEGER,
                    ratelimit_reset INTEGER,
                    ratelimit_retry_after INTEGER,
                    current_limiting_window TEXT DEFAULT 'second',
                    updated_at INTEGER NOT NULL
                );
                """
            )
            
            # Simple migration for existing tables
            try:
                conn.execute("ALTER TABLE provider_ratelimits ADD COLUMN limit_month INTEGER")
                conn.execute("ALTER TABLE provider_ratelimits ADD COLUMN remaining_month INTEGER")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE provider_ratelimits ADD COLUMN limit_second INTEGER")
                conn.execute("ALTER TABLE provider_ratelimits ADD COLUMN remaining_second INTEGER")
                conn.execute("ALTER TABLE provider_ratelimits ADD COLUMN ratelimit_limit INTEGER")
                conn.execute("ALTER TABLE provider_ratelimits ADD COLUMN ratelimit_remaining INTEGER")
                conn.execute("ALTER TABLE provider_ratelimits ADD COLUMN ratelimit_reset INTEGER")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE provider_ratelimits ADD COLUMN current_limiting_window TEXT")
                conn.execute("ALTER TABLE provider_ratelimits ADD COLUMN ratelimit_retry_after INTEGER")
            except sqlite3.OperationalError:
                pass

    # ---------------- Provider Ratelimits ----------------

    def get_provider_ratelimits(self, provider_name: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM provider_ratelimits
                WHERE provider_name = ?
                """,
                (provider_name,),
            ).fetchone()

        return dict(row) if row is not None else None

    def sync_provider_ratelimits(self, provider_name: str, limits: dict[str, Any]) -> dict:
        now = int(time.time())
        existing = self.get_provider_ratelimits(provider_name) or {}
        fields = ("limit_second", "remaining_second","limit_minute", "remaining_minute", "limit_hour", "remaining_hour", "limit_day", "remaining_day", "limit_month", "remaining_month", "ratelimit_limit", "ratelimit_remaining", "ratelimit_reset", "ratelimit_retry_after", "current_limiting_window", "updated_at")
        merged = {
            "provider_name": provider_name,
        }

        for field in fields:
            merged[field] = existing.get(field)
            if field in limits and limits[field] is not None:
                merged[field] = int(limits[field])

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO provider_ratelimits (
                        provider_name,
                        limit_second,
                        remaining_second,
                        limit_minute,
                        remaining_minute,
                        limit_hour,
                        remaining_hour,
                        limit_day,
                        remaining_day,
                        limit_month,
                        remaining_month,
                        ratelimit_limit,
                        ratelimit_remaining,
                        ratelimit_reset,
                        ratelimit_retry_after,
                        current_limiting_window,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(provider_name)
                    DO UPDATE SET
                        limit_second = excluded.limit_second,
                        remaining_second = excluded.remaining_second,
                        limit_minute = excluded.limit_minute,
                        remaining_minute = excluded.remaining_minute,
                        limit_hour = excluded.limit_hour,
                        remaining_hour = excluded.remaining_hour,
                        limit_day = excluded.limit_day,
                        remaining_day = excluded.remaining_day,
                        limit_month = excluded.limit_month,
                        remaining_month = excluded.remaining_month,
                        ratelimit_limit = excluded.ratelimit_limit,
                        ratelimit_remaining = excluded.ratelimit_remaining,
                        ratelimit_reset = excluded.ratelimit_reset,
                        ratelimit_retry_after = excluded.ratelimit_retry_after,
                        current_limiting_window = excluded.current_limiting_window,
                        updated_at = excluded.updated_at
                    """,
                    (
                        provider_name,
                        merged["limit_second"],
                        merged["remaining_second"],
                        merged["limit_minute"],
                        merged["remaining_minute"],
                        merged["limit_hour"],
                        merged["remaining_hour"],
                        merged["limit_day"],
                        merged["remaining_day"],
                        merged["limit_month"],
                        merged["remaining_month"],
                        merged["ratelimit_limit"],
                        merged["ratelimit_remaining"],
                        merged["ratelimit_reset"],
                        merged["ratelimit_retry_after"],
                        merged["current_limiting_window"],
                        now,
                    ),
                )

        merged["updated_at"] = now
        return merged

    def get_exhausted_provider_ratelimit_window(self, provider_name: str) -> Optional[str]:
        snapshot = self.get_provider_ratelimits(provider_name)
        if not snapshot:
            return None
        window = snapshot.get("current_limiting_window")
        remaining = snapshot.get("ratelimit_remaining")
        retry_after = snapshot.get("ratelimit_retry_after")
        updated_at = snapshot.get("updated_at")
        if remaining is not None and isinstance(remaining, int) and remaining <= 0 and retry_after is not None and isinstance(retry_after, int) and updated_at is not None and isinstance(updated_at, int):
            time_to_reset = updated_at + retry_after - int(time.time())
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Time to reset {provider_name} ratelimit window: {time_to_reset} seconds")
            if time_to_reset > 0:
                return window
        return None

    # ---------------- Providers ----------------

    def list_providers(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM providers ORDER BY created_at DESC").fetchall()
        
        result = []
        for row in rows:
            d = dict(row)
            if 'route_fallbacks' in d and isinstance(d['route_fallbacks'], str):
                import json
                try:
                    d['route_fallbacks'] = json.loads(d['route_fallbacks'])
                except:
                    d['route_fallbacks'] = {}
            result.append(d)
        return result

    def get_provider(self, provider_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM providers WHERE id = ?", (provider_id,)).fetchone()
        if not row: return None
        d = dict(row)
        if 'route_fallbacks' in d and isinstance(d['route_fallbacks'], str):
            import json
            try:
                d['route_fallbacks'] = json.loads(d['route_fallbacks'])
            except:
                d['route_fallbacks'] = {}
        return d
        
    def get_provider_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM providers WHERE name = ?", (name,)).fetchone()
        if not row: return None
        d = dict(row)
        if 'route_fallbacks' in d and isinstance(d['route_fallbacks'], str):
            import json
            try:
                d['route_fallbacks'] = json.loads(d['route_fallbacks'])
            except:
                d['route_fallbacks'] = {}
        return d

    def create_provider(self, name: str, api_key_variable: Optional[str] = None, prefix: str = '',
                        base_url: Optional[str] = None,
                        request_timeout: Optional[int] = None,
                        health_timeout: Optional[int] = None,
                        max_upstream_retry_seconds: int = 0,
                        sync_provider_ratelimits: bool = False,
                        route_fallbacks: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        provider_id = str(uuid.uuid4())
        now = int(time.time())
        import json
        rf_str = json.dumps(route_fallbacks) if route_fallbacks else '{}'
        
        safe_api_key_var = api_key_variable if api_key_variable is not None else ""

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO providers (id, name, api_key_variable, prefix, base_url, request_timeout, health_timeout, max_upstream_retry_seconds, sync_provider_ratelimits, route_fallbacks, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (provider_id, name, safe_api_key_var, prefix, base_url, request_timeout, health_timeout, max_upstream_retry_seconds, int(sync_provider_ratelimits), rf_str, now)
                )
        return self.get_provider(provider_id)

    def update_provider(self, provider_id: str, name: str, api_key_variable: Optional[str], prefix: str,
                        base_url: Optional[str], request_timeout: Optional[int], health_timeout: Optional[int],
                        max_upstream_retry_seconds: int = 0, sync_provider_ratelimits: bool = False,
                        route_fallbacks: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        import json
        rf_str = json.dumps(route_fallbacks) if route_fallbacks is not None else '{}'
        
        safe_api_key_var = api_key_variable if api_key_variable is not None else ""

        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    UPDATE providers SET name=?, api_key_variable=?, prefix=?, base_url=?, request_timeout=?, health_timeout=?, max_upstream_retry_seconds=?, sync_provider_ratelimits=?, route_fallbacks=?
                    WHERE id=?
                    """,
                    (name, safe_api_key_var, prefix, base_url, request_timeout, health_timeout, max_upstream_retry_seconds, int(sync_provider_ratelimits), rf_str, provider_id)
                )
                if cursor.rowcount == 0:
                    return None
        return self.get_provider(provider_id)

    def delete_provider(self, provider_id: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM provider_models WHERE provider_id = ?", (provider_id,))
                cursor = conn.execute("DELETE FROM providers WHERE id = ?", (provider_id,))
                return cursor.rowcount > 0

    # ---------------- Models ----------------

    def list_models_for_provider(self, provider_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM provider_models WHERE provider_id = ? ORDER BY name", (provider_id,)).fetchall()
        
        result = []
        for row in rows:
            d = dict(row)
            if 'supported_endpoints' in d and isinstance(d['supported_endpoints'], str):
                import json
                try:
                    d['supported_endpoints'] = json.loads(d['supported_endpoints'])
                except:
                    d['supported_endpoints'] = []
            result.append(d)
        return result

    def get_model_by_name(self, provider_id: str, name: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM provider_models WHERE provider_id = ? AND name = ?", (provider_id, name)).fetchone()
        if not row: return None
        d = dict(row)
        if 'supported_endpoints' in d and isinstance(d['supported_endpoints'], str):
            import json
            try:
                d['supported_endpoints'] = json.loads(d['supported_endpoints'])
            except:
                d['supported_endpoints'] = []
        return d

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM provider_models WHERE id = ?", (model_id,)).fetchone()
        if not row: return None
        d = dict(row)
        if 'supported_endpoints' in d and isinstance(d['supported_endpoints'], str):
            import json
            try:
                d['supported_endpoints'] = json.loads(d['supported_endpoints'])
            except:
                d['supported_endpoints'] = []
        return d

    def create_model(self, provider_id: str, name: str, type: str, target_model_name: str,
                     target_base_url: Optional[str] = None, fallback_model_name: Optional[str] = None,
                     supported_endpoints: Optional[List[str]] = None, price_per_unit: float = 0.0,
                     min_credits_per_request: float = 0.0, owned_by: str = 'FOAP', hide_on_models_endpoint: bool = False) -> Dict[str, Any]:
        model_id = str(uuid.uuid4())
        import json
        endpoints_str = json.dumps(supported_endpoints) if supported_endpoints else '[]'

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO provider_models (id, provider_id, name, type, target_model_name, target_base_url, fallback_model_name, supported_endpoints, price_per_unit, min_credits_per_request, owned_by, hide_on_models_endpoint)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (model_id, provider_id, name, type, target_model_name, target_base_url, fallback_model_name, endpoints_str, price_per_unit, min_credits_per_request, owned_by or 'FOAP', int(hide_on_models_endpoint))
                )
        return self.get_model(model_id)

    def update_model(self, model_id: str, name: str, type: str, target_model_name: str,
                     target_base_url: Optional[str] = None, fallback_model_name: Optional[str] = None,
                     supported_endpoints: Optional[List[str]] = None, price_per_unit: float = 0.0,
                     min_credits_per_request: float = 0.0,
                     owned_by: Optional[str] = None, hide_on_models_endpoint: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        import json
        endpoints_str = json.dumps(supported_endpoints) if supported_endpoints else '[]'

        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    UPDATE provider_models SET name = ?, type = ?, target_model_name = ?, target_base_url = ?, fallback_model_name = ?,
                    supported_endpoints = ?, price_per_unit = ?, min_credits_per_request = ?, owned_by = COALESCE(?, owned_by),
                    hide_on_models_endpoint = COALESCE(?, hide_on_models_endpoint) WHERE id = ?
                    """,
                    (name, type, target_model_name, target_base_url, fallback_model_name, endpoints_str, price_per_unit, min_credits_per_request, owned_by, int(hide_on_models_endpoint) if hide_on_models_endpoint is not None else None, model_id)
                )
                if cursor.rowcount == 0:
                    return None
        return self.get_model(model_id)

    def delete_model(self, model_id: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute("DELETE FROM provider_models WHERE id = ?", (model_id,))
                return cursor.rowcount > 0

    # ---------------- Aliases ----------------

    def list_aliases(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM model_aliases ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]
        
    def get_alias_by_name(self, alias_name: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM model_aliases WHERE alias_name = ?", (alias_name,)).fetchone()
        return dict(row) if row else None

    def create_alias(self, alias_name: str, target_model_name: str, owned_by: str = 'FOAP', hide_on_models_endpoint: bool = False) -> Dict[str, Any]:
        alias_id = str(uuid.uuid4())
        now = int(time.time())
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO model_aliases (id, alias_name, target_model_name, owned_by, hide_on_models_endpoint, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (alias_id, alias_name, target_model_name, owned_by or 'FOAP', int(hide_on_models_endpoint), now)
                )
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM model_aliases WHERE id = ?", (alias_id,)).fetchone()
        return dict(row)

    def update_alias(self, alias_id: str, alias_name: str, target_model_name: str, owned_by: Optional[str] = None, hide_on_models_endpoint: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    "UPDATE model_aliases SET alias_name = ?, target_model_name = ?, owned_by = COALESCE(?, owned_by), hide_on_models_endpoint = COALESCE(?, hide_on_models_endpoint) WHERE id = ?",
                    (alias_name, target_model_name, owned_by, int(hide_on_models_endpoint) if hide_on_models_endpoint is not None else None, alias_id)
                )
                if cursor.rowcount == 0:
                    return None
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM model_aliases WHERE id = ?", (alias_id,)).fetchone()
        return dict(row)

    def delete_alias(self, alias_id: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute("DELETE FROM model_aliases WHERE id = ?", (alias_id,))
                return cursor.rowcount > 0

config_store = ConfigStore()
