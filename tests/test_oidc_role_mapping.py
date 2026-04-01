import os
import sys

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
TEST_DB = os.path.abspath(os.path.join(os.path.dirname(__file__), "oidc-access-test.db"))
os.environ.setdefault("FOAP_CONFIG_DIR", CONFIG_DIR)
os.environ.setdefault("FOAP_ACCESS_DB_PATH", TEST_DB)

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from fastapi import FastAPI
from fastapi.testclient import TestClient

import auth
from access_store import store
from routers.admin import router as admin_router
from routers.self_service import router as self_service_router


def _reset_store_tables() -> None:
    with store._connect() as conn:
        conn.execute("DELETE FROM api_key_usage")
        conn.execute("DELETE FROM api_key_quotas")
        conn.execute("DELETE FROM protected_endpoints")
        conn.execute("DELETE FROM api_keys")


def test_admin_route_accepts_oidc_admin_claim(monkeypatch):
    _reset_store_tables()
    monkeypatch.setenv("FOAP_ENABLE_OIDC_AUTH", "1")
    monkeypatch.delenv("FOAP_ADMIN_TOKEN", raising=False)

    app = FastAPI()
    app.include_router(admin_router)
    client = TestClient(app)

    monkeypatch.setattr("routers.admin.get_oidc_claims", lambda token: {"roles": ["foap-admin"], "sub": "user-1"})
    monkeypatch.setattr("routers.admin.has_admin_access", lambda claims: True)

    response = client.get("/api/admin/health", headers={"Authorization": "Bearer oidc-token"})
    assert response.status_code == 200


def test_self_service_uses_oidc_subject_as_owner(monkeypatch):
    _reset_store_tables()
    monkeypatch.setenv("FOAP_ENABLE_OIDC_AUTH", "1")

    app = FastAPI()
    app.include_router(self_service_router)
    client = TestClient(app)

    monkeypatch.setattr("routers.self_service.get_oidc_owner_id", lambda token: "oidc:subject-123")

    created = client.post(
        "/api/keys",
        headers={"Authorization": "Bearer oidc-token"},
        json={"name": "oidc-key"},
    )
    assert created.status_code == 200

    listed = client.get("/api/keys", headers={"Authorization": "Bearer oidc-token"})
    assert listed.status_code == 200
    assert len(listed.json()) == 1
    assert listed.json()[0]["owner_id"] == "oidc:subject-123"


def test_can_request_allows_oidc_when_access_control_enabled(monkeypatch):
    _reset_store_tables()
    monkeypatch.setenv("FOAP_ENABLE_ACCESS_CONTROL", "1")
    monkeypatch.setenv("FOAP_ENABLE_OIDC_AUTH", "1")

    monkeypatch.setattr(auth, "get_api_key_context", lambda token: None)
    monkeypatch.setattr(auth, "get_oidc_claims", lambda token: {"roles": ["member"], "sub": "abc"})
    monkeypatch.setattr(auth, "has_self_service_access", lambda claims: True)

    assert auth.can_request("gpt-4o", "oidc-token") is True




