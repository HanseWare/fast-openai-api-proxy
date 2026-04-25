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
            DELETE FROM quota_override_usage;
            DELETE FROM quota_overrides;
            DELETE FROM quota_policy_usage;
            DELETE FROM quota_policies;
            DELETE FROM api_key_usage;
            DELETE FROM api_key_quotas;
            DELETE FROM protected_endpoints;
            DELETE FROM api_keys;
            """
        )


def test_access_key_lifecycle_and_quota_enforcement():
    _reset_access_store()

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
    created_owner_id = created_key["owner_id"]

    create_self_2 = client.post(
        "/api/keys",
        headers={"Authorization": "Bearer user-seed-token-2"},
        json={"name": "my-key-2"},
    )
    assert create_self_2.status_code == 200
    created_key_2 = create_self_2.json()
    created_secret_2 = created_key_2["api_key"]
    created_owner_id_2 = created_key_2["owner_id"]

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

    override_create = client.post(
        "/api/admin/quota-overrides",
        headers={"Authorization": "Bearer admin-secret"},
        json={
            "api_path": "/v1/chat/completions",
            "model": "gpt-4o",
            "owner_id": created_owner_id,
            "window_type": "minute",
            "request_limit": 2,
            "exempt": False,
            "starts_at": None,
            "ends_at": None,
        },
    )
    assert override_create.status_code == 200
    override_id = override_create.json()["id"]

    override_duplicate = client.post(
        "/api/admin/quota-overrides",
        headers={"Authorization": "Bearer admin-secret"},
        json={
            "api_path": "/v1/chat/completions",
            "model": "gpt-4o",
            "owner_id": created_owner_id,
            "window_type": "minute",
            "request_limit": 5,
            "exempt": False,
            "starts_at": None,
            "ends_at": None,
        },
    )
    assert override_duplicate.status_code == 409

    override_list = client.get(
        "/api/admin/quota-overrides",
        headers={"Authorization": "Bearer admin-secret"},
    )
    assert override_list.status_code == 200
    assert any(item["id"] == override_id for item in override_list.json())
    first_override_payload = next(item for item in override_list.json() if item["id"] == override_id)
    assert "created_at" in first_override_payload
    assert "active_now" in first_override_payload
    assert first_override_payload["window_state"] in {"active", "scheduled", "expired"}

    scheduled_override = client.post(
        "/api/admin/quota-overrides",
        headers={"Authorization": "Bearer admin-secret"},
        json={
            "api_path": "/v1/chat/completions",
            "model": "gpt-4.1",
            "owner_id": created_owner_id_2,
            "window_type": "minute",
            "request_limit": 1,
            "exempt": False,
            "starts_at": 9999999999,
            "ends_at": 9999999999 + 60,
        },
    )
    assert scheduled_override.status_code == 200
    scheduled_override_id = scheduled_override.json()["id"]

    filtered_overrides = client.get(
        "/api/admin/quota-overrides",
        headers={"Authorization": "Bearer admin-secret"},
        params={"owner_id": created_owner_id, "model": "gpt-4o"},
    )
    assert filtered_overrides.status_code == 200
    assert all(item["owner_id"] == created_owner_id for item in filtered_overrides.json())
    assert all(item["model"] == "gpt-4o" for item in filtered_overrides.json())

    active_only_overrides = client.get(
        "/api/admin/quota-overrides",
        headers={"Authorization": "Bearer admin-secret"},
        params={"active_only": True},
    )
    assert active_only_overrides.status_code == 200
    assert all(item["active_now"] for item in active_only_overrides.json())
    assert all(item["id"] != scheduled_override_id for item in active_only_overrides.json())

    paginated_overrides = client.get(
        "/api/admin/quota-overrides",
        headers={"Authorization": "Bearer admin-secret"},
        params={"limit": 1, "offset": 1},
    )
    assert paginated_overrides.status_code == 200
    assert len(paginated_overrides.json()) == 1
    assert int(paginated_overrides.headers["X-Limit"]) == 1
    assert int(paginated_overrides.headers["X-Offset"]) == 1
    assert int(paginated_overrides.headers["X-Returned-Count"]) == 1
    assert int(paginated_overrides.headers["X-Total-Count"]) >= 2

    quota_resp = client.put(
        f"/api/admin/keys/{created_id}/quota",
        headers={"Authorization": "Bearer admin-secret"},
        json={"requests_per_minute": 1},
    )
    assert quota_resp.status_code == 200

    own_quota = client.get(
        f"/api/keys/{created_id}/quota",
        headers={"Authorization": "Bearer user-seed-token"},
    )
    assert own_quota.status_code == 200
    assert own_quota.json()["api_key_id"] == created_id
    assert own_quota.json()["requests_per_minute"] == 1

    own_usage = client.get(
        f"/api/keys/{created_id}/usage",
        headers={"Authorization": "Bearer user-seed-token"},
    )
    assert own_usage.status_code == 200
    assert own_usage.json()["api_key_id"] == created_id
    assert own_usage.json()["requests_per_minute"] == 1

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
    assert second_call.status_code == 200

    third_call = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {created_secret}"},
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi once more"}]},
    )
    assert third_call.status_code == 429

    override_invalid_window = client.post(
        "/api/admin/quota-overrides",
        headers={"Authorization": "Bearer admin-secret"},
        json={
            "api_path": "/v1/chat/completions",
            "model": "gpt-4o",
            "owner_id": created_owner_id_2,
            "window_type": "minute",
            "request_limit": 1,
            "exempt": False,
            "starts_at": 200,
            "ends_at": 100,
        },
    )
    assert override_invalid_window.status_code == 400

    override_for_second_user = client.post(
        "/api/admin/quota-overrides",
        headers={"Authorization": "Bearer admin-secret"},
        json={
            "api_path": "/v1/chat/completions",
            "model": "gpt-4o",
            "owner_id": created_owner_id_2,
            "window_type": "minute",
            "request_limit": 1,
            "exempt": True,
            "starts_at": None,
            "ends_at": None,
        },
    )
    assert override_for_second_user.status_code == 200
    override_id_second = override_for_second_user.json()["id"]

    override_update = client.put(
        f"/api/admin/quota-overrides/{override_id}",
        headers={"Authorization": "Bearer admin-secret"},
        json={
            "api_path": "/v1/chat/completions",
            "model": "gpt-4o",
            "owner_id": created_owner_id,
            "window_type": "minute",
            "request_limit": 3,
            "exempt": False,
            "starts_at": None,
            "ends_at": None,
        },
    )
    assert override_update.status_code == 200

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
    assert global_limit_hit.status_code == 200

    global_limit_hit_next = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {created_secret}"},
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "global limit hit next"}]},
    )
    assert global_limit_hit_next.status_code == 429

    # Exempt override bypasses policy/global quota for this specific owner.
    global_exempt_call_1 = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {created_secret_2}"},
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "override exempt 1"}]},
    )
    assert global_exempt_call_1.status_code == 200

    global_exempt_call_2 = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {created_secret_2}"},
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "override exempt 2"}]},
    )
    assert global_exempt_call_2.status_code == 200

    no_policy_model_call = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {created_secret_2}"},
        json={"model": "gpt-4.1", "messages": [{"role": "user", "content": "no policy"}]},
    )
    assert no_policy_model_call.status_code == 200

    default_quota_call = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {created_secret}"},
        json={"model": "gpt-4.1", "messages": [{"role": "user", "content": "default quota"}]},
    )
    assert default_quota_call.status_code == 200

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

    override_delete = client.delete(
        f"/api/admin/quota-overrides/{override_id}",
        headers={"Authorization": "Bearer admin-secret"},
    )
    assert override_delete.status_code == 200

    override_delete_second = client.delete(
        f"/api/admin/quota-overrides/{override_id_second}",
        headers={"Authorization": "Bearer admin-secret"},
    )
    assert override_delete_second.status_code == 200

    scheduled_override_delete = client.delete(
        f"/api/admin/quota-overrides/{scheduled_override_id}",
        headers={"Authorization": "Bearer admin-secret"},
    )
    assert scheduled_override_delete.status_code == 200

    override_delete_missing = client.delete(
        f"/api/admin/quota-overrides/{override_id_second}",
        headers={"Authorization": "Bearer admin-secret"},
    )
    assert override_delete_missing.status_code == 404

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


def test_quota_decision_trace_header_exposes_resolution_source():
    _reset_access_store()

    client = TestClient(_build_app())

    create_key = client.post(
        "/api/keys",
        headers={"Authorization": "Bearer trace-user-token"},
        json={"name": "trace-key"},
    )
    assert create_key.status_code == 200
    created_key = create_key.json()

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

    call = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {created_key['api_key']}",
            "X-FOAP-Debug-Quota": "1",
        },
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "trace"}]},
    )
    assert call.status_code == 200
    assert "source=policy" in call.headers.get("X-FOAP-Quota-Decision", "")


