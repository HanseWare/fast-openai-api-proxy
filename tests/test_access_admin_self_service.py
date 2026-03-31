import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
TEST_DB = os.path.abspath(os.path.join(os.path.dirname(__file__), "access-test.db"))

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
    app.add_middleware(AccessControlMiddleware)
    app.include_router(admin_router)
    app.include_router(self_service_router)

    @app.post("/v1/chat/completions")
    async def protected_completion():
        return {"status": "ok"}

    return app


def test_access_key_lifecycle_and_quota_enforcement():
    db_file = Path(TEST_DB)
    if db_file.exists():
        db_file.unlink()

    # Import after cleaning DB path to re-create schema for this test run.
    from access_store import store  # noqa: E402

    store._init_db()

    client = TestClient(_build_app())

    create_self = client.post(
        "/api/keys",
        headers={"Authorization": "Bearer user-seed-token"},
        json={"name": "my-key"},
    )
    assert create_self.status_code == 200
    created_key = create_self.json()
    created_id = created_key["id"]
    created_secret = created_key["api_key"]

    self_list = client.get("/api/keys", headers={"Authorization": "Bearer user-seed-token"})
    assert self_list.status_code == 200
    assert len(self_list.json()) == 1
    assert self_list.json()[0]["id"] == created_id

    admin_list = client.get("/api/admin/keys", headers={"Authorization": "Bearer admin-secret"})
    assert admin_list.status_code == 200
    assert any(item["id"] == created_id for item in admin_list.json())

    protect_resp = client.post(
        "/api/admin/protected-endpoints",
        headers={"Authorization": "Bearer admin-secret"},
        json={"path": "/v1/chat/completions", "method": "POST"},
    )
    assert protect_resp.status_code == 200

    protect_duplicate = client.post(
        "/api/admin/protected-endpoints",
        headers={"Authorization": "Bearer admin-secret"},
        json={"path": "/v1/chat/completions", "method": "POST"},
    )
    assert protect_duplicate.status_code == 409

    quota_resp = client.put(
        f"/api/admin/keys/{created_id}/quota",
        headers={"Authorization": "Bearer admin-secret"},
        json={"requests_per_minute": 1},
    )
    assert quota_resp.status_code == 200

    first_call = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {created_secret}"},
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert first_call.status_code == 200

    second_call = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {created_secret}"},
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi again"}]},
    )
    assert second_call.status_code == 429

    usage_resp = client.get(
        f"/api/admin/keys/{created_id}/usage",
        headers={"Authorization": "Bearer admin-secret"},
    )
    assert usage_resp.status_code == 200
    usage_payload = usage_resp.json()
    assert usage_payload["requests_per_minute"] == 1
    assert usage_payload["used"] >= 1
    assert usage_payload["remaining"] == 0
    assert usage_payload["reset_in_seconds"] >= 1

    no_auth_call = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert no_auth_call.status_code == 401


