import os
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent
APP_DIR = str((BASE_DIR / ".." / "app").resolve())
CONFIG_DIR = str((BASE_DIR / ".." / "configs").resolve())
TEST_DB = str(BASE_DIR / "provider-ratelimit-test.db")

os.environ.setdefault("FOAP_CONFIG_DIR", CONFIG_DIR)
os.environ["FOAP_ACCESS_DB_PATH"] = TEST_DB
os.environ["FOAP_ADMIN_TOKEN"] = "admin-secret"

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from access_store import store  # noqa: E402

with store._connect() as conn:
    conn.executescript(
        """
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

from config_store import config_store  # noqa: E402
from routers.admin_config import router as admin_config_router  # noqa: E402
import utils  # noqa: E402


def _reset_store() -> None:
    store._init_db()
    with store._connect() as conn:
        conn.executescript(
            """
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

            DELETE FROM provider_ratelimits;
            DELETE FROM provider_model_endpoints;
            DELETE FROM provider_models;
            DELETE FROM model_aliases;
            DELETE FROM providers;
            """
        )


def test_provider_ratelimit_sync_is_persisted_and_admin_visible():
    _reset_store()

    provider_name = f"openai-{uuid.uuid4().hex[:8]}"
    provider = config_store.create_provider(name=provider_name, api_key_variable="OPENAI_API_TOKEN")

    initial = store.sync_provider_ratelimits(
        provider_name,
        {
            "limit_minute": 60,
            "remaining_minute": 0,
            "limit_hour": 1000,
            "remaining_hour": 500,
        },
    )
    assert initial["limit_minute"] == 60
    assert initial["remaining_minute"] == 0
    assert initial["limit_hour"] == 1000
    assert initial["remaining_hour"] == 500

    merged = store.sync_provider_ratelimits(provider_name, {"remaining_minute": 12})
    assert merged["limit_minute"] == 60
    assert merged["remaining_minute"] == 12
    assert merged["limit_hour"] == 1000
    assert merged["remaining_hour"] == 500

    exhausted_window = store.get_exhausted_provider_ratelimit_window(provider_name)
    assert exhausted_window == None

    app = FastAPI()
    app.include_router(admin_config_router)
    client = TestClient(app)

    response = client.get(
        f"/api/admin/config/providers/{provider['id']}/ratelimits",
        headers={"Authorization": "Bearer admin-secret"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["provider_name"] == provider_name
    assert body["remaining_minute"] == 12
    assert body["remaining_hour"] == 500


class _ExplodingAsyncClient:
    def __init__(self, *args, **kwargs):
        raise AssertionError("Upstream client should not be created when provider rate limits are exhausted")


def test_handle_request_short_circuits_on_exhausted_provider_ratelimit(monkeypatch):
    _reset_store()
    store.sync_provider_ratelimits(
        "openai",
        {
            "limit_minute": 60,
            "remaining_minute": 0,
        },
    )

    monkeypatch.setattr(utils, "can_request", lambda model, token: True)
    monkeypatch.setattr(
        utils.models,
        "get_model_data",
        lambda model, api_path: {
            "provider": "openai",
            "model_requested": model,
            "target_model_name": "gpt-4o",
            "target_base_url": "https://backend.local",
            "api_key": "backend-token",
            "request_timeout": 5,
            "sync_provider_ratelimits": True,
        },
    )
    monkeypatch.setattr(utils.httpx, "AsyncClient", _ExplodingAsyncClient)

    app = FastAPI()

    @app.post("/proxy")
    async def proxy(request: Request):
        response = await utils.handle_request(request, "v1/chat/completions")
        return utils.process_completion_response(response)

    client = TestClient(app)
    response = client.post(
        "/proxy",
        headers={"Authorization": "Bearer client-token"},
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 429
    assert response.json()["detail"] == "Upstream provider rate limit exhausted for openai (minute)"

