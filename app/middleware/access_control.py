from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from access_store import store
from auth import extract_bearer_token, get_api_key_context


class AccessControlMiddleware(BaseHTTPMiddleware):
    """Optional access-control guard for protected endpoints and per-key quotas."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method.upper()

        if not store.is_endpoint_protected(path=path, method=method):
            return await call_next(request)

        token = extract_bearer_token(request.headers.get("Authorization"))
        key_context = get_api_key_context(token)
        if key_context is None:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

        allowed, retry_after = store.consume_quota(key_context["id"])
        if not allowed:
            headers = {}
            if retry_after is not None:
                headers["Retry-After"] = str(retry_after)
            return JSONResponse(status_code=429, content={"detail": "Quota exceeded"}, headers=headers)

        request.state.api_key = key_context
        return await call_next(request)

