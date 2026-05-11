from typing import Optional
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from access_store import store
from auth import extract_bearer_token, get_api_key_context, get_oidc_owner_id, get_oidc_groups
from config import is_quota_decision_trace_enabled
from models_handler import handler as models_handler
from budget_service import budget_service

TRACE_HEADER_NAME = "X-FOAP-Quota-Decision"

def _truthy(value: Optional[str]) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}

def _should_emit_trace(request: Request) -> bool:
    return is_quota_decision_trace_enabled() or _truthy(request.headers.get("X-FOAP-Debug-Quota"))

def _trace_header_value(trace: dict) -> str:
    return ";".join(f"{k}={v}" for k, v in trace.items() if v is not None)

def _attach_trace(response: JSONResponse, trace: dict, emit_trace: bool) -> JSONResponse:
    if emit_trace:
        response.headers[TRACE_HEADER_NAME] = _trace_header_value(trace)
    return response

class AccessControlMiddleware(BaseHTTPMiddleware):
    """Optional access-control guard for protected endpoints and budgets."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method.upper()
        emit_trace = _should_emit_trace(request)
        model = None

        if method != "GET":
            try:
                body = await request.body()
                if body:
                    import json
                    payload = json.loads(body)
                    model = payload.get("model") if isinstance(payload, dict) else None
                # Restore the body for downstream consumers
                async def receive(): return {"type": "http.request", "body": body}
                request._receive = receive
            except Exception:
                model = None

        is_protected = store.is_endpoint_protected(path=path, method=method, model=model)

        token = extract_bearer_token(request.headers.get("Authorization"))
        key_context = get_api_key_context(token)
        oidc_owner_id = get_oidc_owner_id(token)
        oidc_groups = get_oidc_groups(token)

        entities = []
        if oidc_owner_id:
            entities.append(("user", oidc_owner_id))
            for g in oidc_groups:
                entities.append(("group", g))
        elif key_context and key_context.get("owner_id"):
            entities.append(("user", key_context["owner_id"]))
        elif key_context:
            entities.append(("user", f"key:{key_context['id']}"))

        if not entities and is_protected:
            trace = {"source": "none", "allowed": False, "api_path": path, "model": model}
            response = JSONResponse(status_code=401, content={"detail": "Unauthorized"})
            return _attach_trace(response, trace, emit_trace)

        request.state.api_key = key_context
        request.state.oidc_owner_id = oidc_owner_id
        request.state.entities = entities
        request.state.model_data = None
        request.state.payer_entity = None
        request.state.reserved_credits = 0.0

        if not path.startswith("/v1/"):
            return await call_next(request)

        if model:
            try:
                model_data = models_handler.get_model_data(model, path)
                min_credits = model_data.get("min_credits_per_request", 0.0)
                model_type = model_data.get("type", "llm")
                request.state.model_data = model_data
                
                if min_credits > 0 and entities:
                    allowed, payer_entity = await budget_service.reserve_budget(entities, model_type, min_credits)
                    trace = {"source": "budget", "allowed": allowed, "api_path": path, "model": model, "owner": payer_entity[1] if payer_entity else "none"}
                    if not allowed:
                        response = JSONResponse(status_code=429, content={"detail": "Budget exhausted"})
                        return _attach_trace(response, trace, emit_trace)
                    
                    request.state.payer_entity = payer_entity
                    request.state.reserved_credits = min_credits
            except Exception:
                pass

        response = await call_next(request)

        return response
