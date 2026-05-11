import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent
APP_DIR = str((BASE_DIR / ".." / "app").resolve())
CONFIG_DIR = str((BASE_DIR / ".." / "configs").resolve())
TEST_DB = str(BASE_DIR / "access-test.db")

os.environ.setdefault("FOAP_CONFIG_DIR", CONFIG_DIR)
os.environ["FOAP_ACCESS_DB_PATH"] = TEST_DB
os.environ["FOAP_ADMIN_TOKEN"] = "admin-secret"

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from middleware.access_control import AccessControlMiddleware  # noqa: E402
from routers.admin import router as admin_router  # noqa: E402
from routers.self_service import router as self_service_router  # noqa: E402


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(AccessControlMiddleware)  # type: ignore[arg-type]
    app.include_router(admin_router)
    app.include_router(self_service_router)

    @app.post("/v1/chat/completions")
    async def protected_completion():
        return {"status": "ok"}

    return app


def _reset_access_store():
    from access_store import store  # noqa: E402

    store._init_db()
    with store._connect() as conn:
        conn.executescript(
            """
            DELETE FROM budget_usage;
            DELETE FROM budgets;
            DELETE FROM request_logs;
            DELETE FROM protected_endpoints;
            DELETE FROM api_keys;
            """
        )


def test_access_key_lifecycle_and_budget_api():
    _reset_access_store()

    client = TestClient(_build_app())

    # Create self-service key
    create_self = client.post(
        "/api/keys",
        headers={"Authorization": "Bearer user-seed-token"},
        json={"name": "my-key"},
    )
    assert create_self.status_code == 200
    created_key = create_self.json()
    created_id = created_key["id"]
    created_owner_id = created_key["owner_id"]
    created_secret = created_key["api_key"]
    assert created_secret.startswith("foap-")
    assert len(created_secret) == 69

    # Admin list keys
    admin_list = client.get("/api/admin/keys", headers={"Authorization": "Bearer admin-secret"})
    assert admin_list.status_code == 200
    assert any(item["id"] == created_id for item in admin_list.json())

    # Admin Create Budget
    budget_create = client.post(
        "/api/admin/budgets",
        headers={"Authorization": "Bearer admin-secret"},
        json={
            "entity_type": "user",
            "entity_id": created_owner_id,
            "window": "daily",
            "budget_amount": 100.0,
            "scope": "llm"
        },
    )
    assert budget_create.status_code == 200
    budget_id = budget_create.json()["id"]

    # Self-service list budgets
    own_budgets = client.get(
        "/api/budgets",
        headers={"Authorization": "Bearer user-seed-token"}
    )
    assert own_budgets.status_code == 200
    assert len(own_budgets.json()) == 1
    assert own_budgets.json()[0]["id"] == budget_id
    assert own_budgets.json()[0]["budget_amount"] == 100.0

    # Self-service budget usage (should be empty initially)
    own_usage = client.get(
        "/api/budgets/usage",
        headers={"Authorization": "Bearer user-seed-token"}
    )
    assert own_usage.status_code == 200
    assert len(own_usage.json()) == 0

    # Admin Update Budget
    budget_update = client.put(
        f"/api/admin/budgets/{budget_id}",
        headers={"Authorization": "Bearer admin-secret"},
        json={
            "budget_amount": 200.0
        },
    )
    assert budget_update.status_code == 200
    assert budget_update.json()["budget_amount"] == 200.0

    # Admin Delete Budget
    budget_delete = client.delete(
        f"/api/admin/budgets/{budget_id}",
        headers={"Authorization": "Bearer admin-secret"}
    )
    assert budget_delete.status_code == 200

    # Ensure it's gone
    own_budgets_after = client.get(
        "/api/budgets",
        headers={"Authorization": "Bearer user-seed-token"}
    )
    assert len(own_budgets_after.json()) == 0


def test_self_service_oidc_callback_reads_session_cookie_and_sets_auth_cookie(monkeypatch):
    from session_store import store as oidc_session_store  # noqa: E402

    oidc_session_store._sessions.clear()

    app = FastAPI()
    app.include_router(self_service_router)

    captured = {}

    def fake_build_authorization_uri(redirect_uri, state, code_verifier):
        captured["redirect_uri"] = redirect_uri
        captured["state"] = state
        return "https://auth.example.com/authorize"

    monkeypatch.setattr("routers.self_service.is_oidc_auth_enabled", lambda: True)
    monkeypatch.setattr("routers.self_service.get_oidc_client_id", lambda: "client-id")
    monkeypatch.setattr("routers.self_service.get_oidc_client_secret", lambda: "client-secret")
    monkeypatch.setattr("routers.self_service.build_authorization_uri", fake_build_authorization_uri)
    monkeypatch.setattr("routers.self_service.exchange_code_for_token", lambda code, redirect_uri, code_verifier: {"access_token": "access-token"})
    monkeypatch.setattr("routers.self_service.get_oidc_claims", lambda token: {"sub": "user-1"})
    monkeypatch.setattr("routers.self_service.has_self_service_access", lambda claims: True)
    monkeypatch.setattr("routers.self_service.get_owner_id_from_claims", lambda claims: "user-1")
    monkeypatch.setattr("routers.self_service.get_base_url_from_request", lambda request: "https://ai-api.example.com")

    client = TestClient(app, base_url="https://ai-api.example.com")

    login = client.get("/api/oidc/login")
    assert login.status_code == 200
    assert login.json()["authorization_uri"] == "https://auth.example.com/authorize"
    assert captured["redirect_uri"] == "https://ai-api.example.com/api/oidc/callback"

    callback = client.get(
        "/api/oidc/callback",
        params={"code": "auth-code", "state": captured["state"]},
        follow_redirects=False,
    )
    assert callback.status_code == 302
    assert callback.headers["location"] == "/account"
    assert "foap_session=" in callback.headers.get("set-cookie", "")
