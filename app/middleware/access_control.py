from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from access_store import store
from auth import extract_bearer_token, get_api_key_context, get_oidc_owner_id


class AccessControlMiddleware(BaseHTTPMiddleware):
    """Optional access-control guard for protected endpoints and per-key quotas."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method.upper()
        model = None
        if method != "GET":
            try:
                payload = await request.json()
                model = payload.get("model") if isinstance(payload, dict) else None
            except Exception:
                model = None

        quota_policy = store.find_quota_policy(api_path=path, model=model) if model else None
        is_protected = store.is_endpoint_protected(path=path, method=method)

        if not is_protected and quota_policy is None:
            return await call_next(request)

        token = extract_bearer_token(request.headers.get("Authorization"))
        key_context = get_api_key_context(token)

        oidc_owner_id = get_oidc_owner_id(token)

        if key_context is None and oidc_owner_id is None:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

        if key_context is not None:
            allowed, retry_after = store.consume_quota(key_context["id"])
            if not allowed:
                headers = {}
                if retry_after is not None:
                    headers["Retry-After"] = str(retry_after)
                return JSONResponse(status_code=429, content={"detail": "Quota exceeded"}, headers=headers)

        if quota_policy is not None:
            if quota_policy["enforce_per_user"]:
                if oidc_owner_id:
                    bucket_key = oidc_owner_id
                elif key_context and key_context.get("owner_id"):
                    bucket_key = key_context["owner_id"]
                elif key_context:
                    bucket_key = f"key:{key_context['id']}"
                else:
                    return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
            else:
                bucket_key = "global"

            allowed, retry_after = store.consume_quota_policy(
                policy_id=quota_policy["id"],
                bucket_key=bucket_key,
                window_type=quota_policy["window_type"],
                request_limit=quota_policy["request_limit"],
            )
            if not allowed:
                headers = {"Retry-After": str(retry_after)}
                return JSONResponse(status_code=429, content={"detail": "Quota policy exceeded"}, headers=headers)

        request.state.api_key = key_context
        request.state.oidc_owner_id = oidc_owner_id
        return await call_next(request)

