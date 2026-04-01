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

    create_self_2 = client.post(
        "/api/keys",
        headers={"Authorization": "Bearer user-seed-token-2"},
        json={"name": "my-key-2"},
    )
    assert create_self_2.status_code == 200
    created_secret_2 = create_self_2.json()["api_key"]

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

    policy_create = client.post(
        "/api/admin/quota-policies",
        headers={"Authorization": "Bearer admin-secret"},
        json={
            "api_path": "/v1/chat/completions",
            "model": "gpt-4o",
            "window_type": "minute",
            "request_limit": 1,
            "enforce_per_user": True,
        },
    )
    assert policy_create.status_code == 200
    policy_id = policy_create.json()["id"]

    policy_duplicate = client.post(
        "/api/admin/quota-policies",
        headers={"Authorization": "Bearer admin-secret"},
        json={
            "api_path": "/v1/chat/completions",
            "model": "gpt-4o",
            "window_type": "minute",
            "request_limit": 2,
            "enforce_per_user": True,
        },
    )
    assert policy_duplicate.status_code == 409

    policy_list = client.get(
        "/api/admin/quota-policies",
        headers={"Authorization": "Bearer admin-secret"},
    )
    assert policy_list.status_code == 200
    assert any(item["id"] == policy_id for item in policy_list.json())

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

    other_user_call = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {created_secret_2}"},
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "separate user"}]},
    )
    assert other_user_call.status_code == 200

    policy_update = client.put(
        f"/api/admin/quota-policies/{policy_id}",
        headers={"Authorization": "Bearer admin-secret"},
        json={
            "api_path": "/v1/chat/completions",
            "model": "gpt-4o",
            "window_type": "minute",
            "request_limit": 1,
            "enforce_per_user": False,
        },
    )
    assert policy_update.status_code == 200

    global_first = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {created_secret_2}"},
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "global limit"}]},
    )
    assert global_first.status_code == 200

    global_limit_hit = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {created_secret}"},
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "global limit hit"}]},
    )
    assert global_limit_hit.status_code == 429

    no_policy_model_call = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {created_secret_2}"},
        json={"model": "gpt-4.1", "messages": [{"role": "user", "content": "no policy"}]},
    )
    assert no_policy_model_call.status_code == 200

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

    policy_delete = client.delete(
        f"/api/admin/quota-policies/{policy_id}",
        headers={"Authorization": "Bearer admin-secret"},
    )
    assert policy_delete.status_code == 200

    policy_delete_missing = client.delete(
        f"/api/admin/quota-policies/{policy_id}",
        headers={"Authorization": "Bearer admin-secret"},
    )
    assert policy_delete_missing.status_code == 404

    no_auth_call = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert no_auth_call.status_code == 401


