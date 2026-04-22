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

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ---------------- Providers ----------------

    def list_providers(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM providers ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def get_provider(self, provider_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM providers WHERE id = ?", (provider_id,)).fetchone()
        return dict(row) if row else None
        
    def get_provider_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM providers WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None

    def create_provider(self, name: str, api_key_variable: str, prefix: str = '',
                        default_base_url: Optional[str] = None,
                        default_request_timeout: Optional[int] = None,
                        default_health_timeout: Optional[int] = None) -> Dict[str, Any]:
        provider_id = str(uuid.uuid4())
        now = int(time.time())
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO providers (id, name, api_key_variable, prefix, default_base_url, default_request_timeout, default_health_timeout, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (provider_id, name, api_key_variable, prefix, default_base_url, default_request_timeout, default_health_timeout, now)
                )
        return self.get_provider(provider_id)

    def update_provider(self, provider_id: str, name: str, api_key_variable: str, prefix: str,
                        default_base_url: Optional[str], default_request_timeout: Optional[int], default_health_timeout: Optional[int]) -> Optional[Dict[str, Any]]:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    UPDATE providers SET name=?, api_key_variable=?, prefix=?, default_base_url=?, default_request_timeout=?, default_health_timeout=?
                    WHERE id=?
                    """,
                    (name, api_key_variable, prefix, default_base_url, default_request_timeout, default_health_timeout, provider_id)
                )
                if cursor.rowcount == 0:
                    return None
        return self.get_provider(provider_id)

    def delete_provider(self, provider_id: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                # Get models to delete their endpoints
                models = conn.execute("SELECT id FROM provider_models WHERE provider_id = ?", (provider_id,)).fetchall()
                for model in models:
                    conn.execute("DELETE FROM provider_model_endpoints WHERE model_id = ?", (model['id'],))
                conn.execute("DELETE FROM provider_models WHERE provider_id = ?", (provider_id,))
                cursor = conn.execute("DELETE FROM providers WHERE id = ?", (provider_id,))
                return cursor.rowcount > 0

    # ---------------- Models ----------------

    def list_models_for_provider(self, provider_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM provider_models WHERE provider_id = ? ORDER BY name", (provider_id,)).fetchall()
        return [dict(row) for row in rows]

    def get_model_by_name(self, provider_id: str, name: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM provider_models WHERE provider_id = ? AND name = ?", (provider_id, name)).fetchone()
        return dict(row) if row else None

    def create_model(self, provider_id: str, name: str) -> Dict[str, Any]:
        model_id = str(uuid.uuid4())
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO provider_models (id, provider_id, name) VALUES (?, ?, ?)",
                    (model_id, provider_id, name)
                )
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM provider_models WHERE id = ?", (model_id,)).fetchone()
        return dict(row)

    def delete_model(self, model_id: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM provider_model_endpoints WHERE model_id = ?", (model_id,))
                cursor = conn.execute("DELETE FROM provider_models WHERE id = ?", (model_id,))
                return cursor.rowcount > 0

    # ---------------- Endpoints ----------------

    def list_endpoints_for_model(self, model_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM provider_model_endpoints WHERE model_id = ? ORDER BY path", (model_id,)).fetchall()
        return [dict(row) for row in rows]

    def get_endpoint_by_path(self, model_id: str, path: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM provider_model_endpoints WHERE model_id = ? AND path = ?", (model_id, path)).fetchone()
        return dict(row) if row else None

    def create_endpoint(self, model_id: str, path: str, target_model_name: str, 
                        target_base_url: Optional[str] = None, request_timeout: Optional[int] = None, health_timeout: Optional[int] = None) -> Dict[str, Any]:
        endpoint_id = str(uuid.uuid4())
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO provider_model_endpoints (id, model_id, path, target_model_name, target_base_url, request_timeout, health_timeout)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (endpoint_id, model_id, path, target_model_name, target_base_url, request_timeout, health_timeout)
                )
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM provider_model_endpoints WHERE id = ?", (endpoint_id,)).fetchone()
        return dict(row)

    def delete_endpoint(self, endpoint_id: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute("DELETE FROM provider_model_endpoints WHERE id = ?", (endpoint_id,))
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

    def create_alias(self, alias_name: str, target_model_name: str) -> Dict[str, Any]:
        alias_id = str(uuid.uuid4())
        now = int(time.time())
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO model_aliases (id, alias_name, target_model_name, created_at) VALUES (?, ?, ?, ?)",
                    (alias_id, alias_name, target_model_name, now)
                )
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM model_aliases WHERE id = ?", (alias_id,)).fetchone()
        return dict(row)

    def update_alias(self, alias_id: str, alias_name: str, target_model_name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    "UPDATE model_aliases SET alias_name = ?, target_model_name = ? WHERE id = ?",
                    (alias_name, target_model_name, alias_id)
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
