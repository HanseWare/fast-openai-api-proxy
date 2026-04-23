from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from access_store import store
from auth import extract_bearer_token, get_api_key_context, get_oidc_owner_id
from config import is_quota_decision_trace_enabled


TRACE_HEADER_NAME = "X-FOAP-Quota-Decision"


def _truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _should_emit_trace(request: Request) -> bool:
    if is_quota_decision_trace_enabled():
        return True
    return _truthy(request.headers.get("X-FOAP-Debug-Quota"))


def _trace_header_value(trace: dict) -> str:
    parts: list[str] = []
    for key in (
        "source",
        "allowed",
        "api_path",
        "model",
        "owner",
        "bucket",
        "policy_id",
        "override_id",
        "retry_after",
    ):
        value = trace.get(key)
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return ";".join(parts)


def _attach_trace(response: JSONResponse, trace: dict, emit_trace: bool) -> JSONResponse:
    if emit_trace:
        response.headers[TRACE_HEADER_NAME] = _trace_header_value(trace)
    return response


class AccessControlMiddleware(BaseHTTPMiddleware):
    """Optional access-control guard for protected endpoints and per-key quotas."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method.upper()
        emit_trace = _should_emit_trace(request)
        model = None
        if method != "GET":
            try:
                payload = await request.json()
                model = payload.get("model") if isinstance(payload, dict) else None
            except Exception:
                model = None

        quota_policy = store.find_quota_policy(api_path=path, model=model) if model else None
        is_protected = store.is_endpoint_protected(path=path, method=method, model=model)

        if not is_protected and quota_policy is None:
            return await call_next(request)

        token = extract_bearer_token(request.headers.get("Authorization"))
        key_context = get_api_key_context(token)

        oidc_owner_id = get_oidc_owner_id(token)

        if key_context is None and oidc_owner_id is None:
            response = JSONResponse(status_code=401, content={"detail": "Unauthorized"})
            return _attach_trace(
                response,
                {
                    "source": "none",
                    "allowed": False,
                    "api_path": path,
                    "model": model,
                },
                emit_trace,
            )

        effective_owner = None
        if oidc_owner_id:
            effective_owner = oidc_owner_id
        elif key_context and key_context.get("owner_id"):
            effective_owner = key_context["owner_id"]
        elif key_context:
            effective_owner = f"key:{key_context['id']}"

        trace: dict = {
            "source": "none",
            "allowed": True,
            "api_path": path,
            "model": model,
            "owner": effective_owner,
        }

        if quota_policy is not None:
            trace["policy_id"] = quota_policy["id"]
            quota_override = (
                store.find_active_quota_override(api_path=path, model=model, owner_id=effective_owner)
                if (model and effective_owner)
                else None
            )
            if quota_override is not None:
                trace["override_id"] = quota_override["id"]
                if quota_override["exempt"]:
                    trace["source"] = "override-exempt"
                    trace["bucket"] = effective_owner
                else:
                    allowed, retry_after = store.consume_quota_override(
                        override_id=quota_override["id"],
                        bucket_key=effective_owner,
                        window_type=quota_override["window_type"],
                        request_limit=quota_override["request_limit"],
                    )
                    trace["source"] = "override"
                    trace["bucket"] = effective_owner
                    trace["allowed"] = allowed
                    if not allowed:
                        trace["retry_after"] = retry_after
                        headers = {"Retry-After": str(retry_after)}
                        response = JSONResponse(status_code=429, content={"detail": "Quota override exceeded"}, headers=headers)
                        return _attach_trace(response, trace, emit_trace)
            else:
                if quota_policy["enforce_per_user"]:
                    if not effective_owner:
                        response = JSONResponse(status_code=401, content={"detail": "Unauthorized"})
                        trace["source"] = "policy"
                        trace["allowed"] = False
                        return _attach_trace(response, trace, emit_trace)
                    bucket_key = effective_owner
                else:
                    bucket_key = "global"

                allowed, retry_after = store.consume_quota_policy(
                    policy_id=quota_policy["id"],
                    bucket_key=bucket_key,
                    window_type=quota_policy["window_type"],
                    request_limit=quota_policy["request_limit"],
                )
                trace["source"] = "policy"
                trace["bucket"] = bucket_key
                trace["allowed"] = allowed
                if not allowed:
                    trace["retry_after"] = retry_after
                    headers = {"Retry-After": str(retry_after)}
                    response = JSONResponse(status_code=429, content={"detail": "Quota policy exceeded"}, headers=headers)
                    return _attach_trace(response, trace, emit_trace)
        elif key_context is not None:
            allowed, retry_after = store.consume_quota(key_context["id"])
            trace["source"] = "default"
            trace["bucket"] = key_context["id"]
            trace["allowed"] = allowed
            if not allowed:
                trace["retry_after"] = retry_after
                headers = {}
                if retry_after is not None:
                    headers["Retry-After"] = str(retry_after)
                response = JSONResponse(status_code=429, content={"detail": "Quota exceeded"}, headers=headers)
                return _attach_trace(response, trace, emit_trace)

        request.state.api_key = key_context
        request.state.oidc_owner_id = oidc_owner_id
        response = await call_next(request)
        if emit_trace:
            response.headers[TRACE_HEADER_NAME] = _trace_header_value(trace)
        return response

