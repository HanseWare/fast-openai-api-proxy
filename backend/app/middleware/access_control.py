from typing import Optional
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

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

class AccessControlMiddleware:
    """Optional access-control guard for protected endpoints and budgets."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        path = scope.get("path", request.url.path)
        method = scope.get("method", request.method).upper()
        emit_trace = _should_emit_trace(request)
        model = None
        body = b""
        buffered_disconnect: Optional[Message] = None

        if method != "GET":
            while True:
                message = await receive()
                if message["type"] == "http.request":
                    body += message.get("body", b"")
                    if not message.get("more_body", False):
                        break
                elif message["type"] == "http.disconnect":
                    buffered_disconnect = message
                    break
                else:
                    break

            try:
                if body:
                    import json
                    payload = json.loads(body)
                    model = payload.get("model") if isinstance(payload, dict) else None
            except Exception:
                model = None

        body_sent = method == "GET"
        disconnect_sent = buffered_disconnect is None

        async def replay_receive() -> Message:
            nonlocal body_sent, disconnect_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            if buffered_disconnect is not None and not disconnect_sent:
                disconnect_sent = True
                return buffered_disconnect
            return await receive()

        downstream_receive = receive if method == "GET" else replay_receive

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
            await _attach_trace(response, trace, emit_trace)(scope, downstream_receive, send)
            return

        state = scope.setdefault("state", {})
        state["api_key"] = key_context
        state["oidc_owner_id"] = oidc_owner_id
        state["entities"] = entities
        state["model_data"] = None
        state["payer_entity"] = None
        state["reserved_credits"] = 0.0

        if path.startswith("/v1/") and model:
            try:
                model_data = models_handler.get_model_data(model, path)
                min_credits = model_data.get("min_credits_per_request", 0.0)
                model_type = model_data.get("type", "llm")
                state["model_data"] = model_data

                if min_credits > 0 and entities:
                    allowed, payer_entity = await budget_service.reserve_budget(entities, model_type, min_credits)
                    trace = {"source": "budget", "allowed": allowed, "api_path": path, "model": model, "owner": payer_entity[1] if payer_entity else "none"}
                    if not allowed:
                        response = JSONResponse(status_code=429, content={"detail": "Budget exhausted"})
                        await _attach_trace(response, trace, emit_trace)(scope, downstream_receive, send)
                        return

                    state["payer_entity"] = payer_entity
                    state["reserved_credits"] = min_credits
            except Exception:
                pass

        await self.app(scope, downstream_receive, send)
