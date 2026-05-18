import datetime
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query, Request, Cookie, status, Response
from fastapi.responses import RedirectResponse, JSONResponse

from access_store import store
from auth import extract_bearer_token, get_oidc_groups, get_oidc_owner_id, identity_from_token
from config import get_auth_mode_snapshot, is_self_service_oidc_only_enabled, is_oidc_auth_enabled, get_oidc_client_id, get_oidc_client_secret
from oidc_auth import get_oidc_claims, has_self_service_access, get_owner_id_from_claims, try_refresh_session
from oidc_bff import generate_auth_state, build_authorization_uri, exchange_code_for_token, validate_state, get_base_url_from_request
from session_store import store as session_store
from schemas.access import ApiKeyCreate, ApiKeyCreateResponse, ApiKeyRead, BudgetContextRead, BudgetRead, BudgetUsageRead

router = APIRouter(prefix="/api", tags=["self-service"])

_BUDGET_SCOPE_CATEGORY = {
    "llm": "tokens",
    "embedding": "tokens",
    "audio_transcription": "audio",
    "audio_speech": "audio",
    "stt": "audio",
    "tts": "audio",
    "image": "images",
    "image-gen": "images",
}

_SUMMARY_CATEGORIES = ("tokens", "audio", "images")
_SUMMARY_WINDOWS = ("daily", "monthly")


def _normalize_scope(scope: Optional[str]) -> str:
    return (scope or "").strip()


def _budget_category(scope: Optional[str]) -> Optional[str]:
    normalized = _normalize_scope(scope)
    if not normalized:
        return None
    return _BUDGET_SCOPE_CATEGORY.get(normalized)


def _current_bucket(window: str, timestamp: Optional[int] = None) -> str:
    ts = int(datetime.datetime.now().timestamp()) if timestamp is None else timestamp
    dt = datetime.datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d") if window == "daily" else dt.strftime("%Y-%m")


def _budget_matches_category(budget_scope: Optional[str], category: str) -> bool:
    normalized = _normalize_scope(budget_scope)
    return not normalized or _budget_category(normalized) == category


def _pick_relevant_budget(budgets: list[dict], category: str, window: str) -> tuple[Optional[dict], int]:
    candidates = [
        budget
        for budget in budgets
        if budget.get("window") == window and _budget_matches_category(budget.get("scope"), category)
    ]
    if not candidates:
        return None, 0

    def sort_key(budget: dict) -> tuple[float, bool, int]:
        return (
            float(budget.get("budget_amount") or 0.0),
            budget.get("entity_type") == "user",
            int(budget.get("created_at") or 0),
        )

    selected = max(candidates, key=sort_key)
    return selected, len(candidates)


def _sum_usage_for_budget(usages: list[dict], budget: Optional[dict], window_bucket: str) -> float:
    if not budget:
        return 0.0

    scope = _normalize_scope(budget.get("scope"))
    entity_type = budget.get("entity_type")
    entity_id = budget.get("entity_id")
    window = budget.get("window")
    total = 0.0

    for usage in usages:
        if usage.get("entity_type") != entity_type:
            continue
        if usage.get("entity_id") != entity_id:
            continue
        if usage.get("window") != window:
            continue
        if usage.get("window_bucket") != window_bucket:
            continue
        if _normalize_scope(usage.get("scope")) != scope:
            continue
        total += float(usage.get("cost") or 0.0)

    return total


def _build_budget_summary(budgets: list[dict], usages: list[dict]) -> list[dict]:
    summary: list[dict] = []
    now_bucket_cache = {window: _current_bucket(window) for window in _SUMMARY_WINDOWS}

    for category in _SUMMARY_CATEGORIES:
        for window in _SUMMARY_WINDOWS:
            selected, matched_budget_count = _pick_relevant_budget(budgets, category, window)
            if selected is None:
                summary.append(
                    {
                        "type": category,
                        "window": window,
                        "budget_amount": 0.0,
                        "used": 0.0,
                        "usage_percent": 0,
                        "matched_budget_count": 0,
                        "source_entity_type": None,
                        "source_entity_id": None,
                        "source_scope": None,
                    }
                )
                continue

            used = _sum_usage_for_budget(usages, selected, now_bucket_cache[window])
            budget_amount = float(selected.get("budget_amount") or 0.0)
            usage_percent = int(min(100, round((used / budget_amount) * 100))) if budget_amount > 0 else 0
            summary.append(
                {
                    "type": category,
                    "window": window,
                    "budget_amount": budget_amount,
                    "used": used,
                    "usage_percent": usage_percent,
                    "matched_budget_count": matched_budget_count,
                    "source_entity_type": selected.get("entity_type"),
                    "source_entity_id": selected.get("entity_id"),
                    "source_scope": selected.get("scope"),
                }
            )

    return summary


@router.get("/auth-config", summary="Get self-service auth mode")
async def self_service_auth_config():
    full_snapshot = get_auth_mode_snapshot()
    snapshot = full_snapshot["self_service"]
    mode = snapshot["mode"]
    if mode == "oidc-only":
        login_hint = "Use an OIDC access token from your identity provider."
    elif mode == "oidc-or-token-hash":
        login_hint = "Use an OIDC access token or a static FOAP bearer token."
    else:
        login_hint = "Use your static FOAP bearer token."

    return {**snapshot, "login_hint": login_hint, "oidc_client": full_snapshot.get("oidc_client")}


@router.get("/oidc/login", summary="Initiate OIDC authorization flow (BFF)")
async def oidc_login(response: Response, request: Request):
    """Initiate OIDC login for self-service. Returns authorization URI."""
    if not is_oidc_auth_enabled() or not get_oidc_client_id() or not get_oidc_client_secret():
        raise HTTPException(status_code=400, detail="OIDC BFF not configured")

    state, code_verifier = generate_auth_state()

    # Build full redirect URI respecting X-Forwarded-Proto header
    base_url = get_base_url_from_request(request)
    redirect_uri = f"{base_url}/api/oidc/callback"

    # Store state and verifier in session
    session_id = session_store.create({
        "state": state,
        "code_verifier": code_verifier,
        "scope": "self-service",
    })

    # Set session cookie. Use the session store default TTL so client cookie
    # lifetime matches server-side session expiry (default 24 hours).
    response.set_cookie(
        key="foap_oidc_session",
        value=session_id,
        max_age=session_store.default_ttl,  # align cookie with server session TTL
        httponly=True,
        secure=True,
        samesite="lax",
    )

    auth_uri = build_authorization_uri(redirect_uri, state, code_verifier)
    return {"authorization_uri": auth_uri}


@router.get("/oidc/callback", summary="OIDC callback handler (BFF)")
async def oidc_callback(code: str, state: str, request: Request, oidc_session: Optional[str] = Cookie(default=None, alias="foap_oidc_session")):
    """Handle OIDC callback and set authenticated session."""
    if not is_oidc_auth_enabled() or not get_oidc_client_id() or not get_oidc_client_secret():
        raise HTTPException(status_code=400, detail="OIDC BFF not configured")

    # Retrieve session
    session_data = session_store.get(oidc_session) if oidc_session else None
    if not session_data:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    # Validate state
    if not validate_state(session_data.get("state", ""), state):
        raise HTTPException(status_code=401, detail="State validation failed")

    code_verifier = session_data.get("code_verifier", "")

    # Build full redirect URI (must match the one from /oidc/login) respecting X-Forwarded-Proto header
    base_url = get_base_url_from_request(request)
    redirect_uri = f"{base_url}/api/oidc/callback"

    # Exchange code for token
    token_response = exchange_code_for_token(code, redirect_uri, code_verifier)
    if not token_response or "access_token" not in token_response:
        raise HTTPException(status_code=401, detail="Failed to obtain access token")

    access_token = token_response["access_token"]

    # Validate token and check self-service claim
    claims = get_oidc_claims(access_token)
    if not claims or not has_self_service_access(claims):
        raise HTTPException(status_code=403, detail="No self-service access in token claims")

    # Create authenticated session
    auth_session_id = session_store.create({
        "access_token": access_token,
        "refresh_token": token_response.get("refresh_token"),  # Store refresh token if available
        "owner_id": get_owner_id_from_claims(claims),
        "scope": "self-service",
    }, ttl_seconds=86400)  # 24 hours

    # Clean up old session
    session_store.delete(oidc_session or "")

    # Redirect to account page and set authenticated session cookie on the redirect response
    redirect_response = RedirectResponse(url="/account", status_code=302)
    redirect_response.set_cookie(
        key="foap_session",
        value=auth_session_id,
        max_age=86400,
        httponly=True,
        secure=True,
        samesite="lax",
    )
    redirect_response.set_cookie(
        key="foap_logged_in",
        value="true",
        max_age=86400,
        secure=True,
        samesite="lax",
    )
    return redirect_response


@router.post("/logout", summary="Self-service logout")
async def self_service_logout(
    oidc_session: Optional[str] = Cookie(default=None, alias="foap_oidc_session"),
    foap_session: Optional[str] = Cookie(default=None, alias="foap_session"),
):
    """Invalidate cookie-based self-service sessions and clear client cookies.

    Works without Authorization header; relies on HttpOnly cookies only.
    """
    if oidc_session:
        session_store.delete(oidc_session)
    if foap_session:
        session_store.delete(foap_session)

    response = JSONResponse({"status": "logged_out"})
    # Clear related cookies on client
    for name in ("foap_session", "foap_logged_in", "foap_oidc_session"):
        response.delete_cookie(name, samesite="lax")
    return response


def _require_user_token(authorization: Optional[str], request: Request) -> str:
    """Require an access token; accept either Authorization: Bearer or a cookie-based OIDC session.

    When the frontend uses the BFF cookie session (httponly cookie `foap_session`), the backend
    should accept that session and extract the access_token stored in the session store.
    
    If the stored access token is expired, attempts to refresh using the stored refresh_token.
    """
    token = extract_bearer_token(authorization)

    # If no Authorization header, try cookie-based session (BFF flow)
    if not token:
        session_id = request.cookies.get("foap_session")
        if session_id:
            session_data = session_store.get(session_id)
            if session_data and isinstance(session_data, dict):
                token = session_data.get("access_token")
                
                # If token is expired, try to refresh it
                if token and not get_oidc_claims(token):
                    # Token is invalid/expired; try to refresh
                    if try_refresh_session(session_data):
                        token = session_data.get("access_token")
                    else:
                        token = None

    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    if is_self_service_oidc_only_enabled() and not get_oidc_owner_id(token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Valid OIDC token required")

    return token


def _resolve_owner_id(token: str) -> str:
    oidc_owner = get_oidc_owner_id(token)
    if oidc_owner:
        return oidc_owner
    if is_self_service_oidc_only_enabled():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Valid OIDC subject claim required")
    return identity_from_token(token)


def _require_owned_key(owner_id: str, key_id: str) -> dict:
    key = store.get_api_key(key_id, owner_id=owner_id)
    if key is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")
    return key


@router.get("/health", summary="Self-service API health")
async def self_service_health(request: Request, authorization: Optional[str] = Header(default=None)):
    _require_user_token(authorization, request)
    return {"status": "ok", "scope": "self-service"}


@router.get("/session", summary="Get current self-service session")
async def self_service_session(request: Request, authorization: Optional[str] = Header(default=None)):
    token = _require_user_token(authorization, request)
    owner_id = _resolve_owner_id(token)
    auth_source = "oidc" if get_oidc_owner_id(token) else "token-hash"
    return {
        "owner_id": owner_id,
        "groups": get_oidc_groups(token),
        "auth_source": auth_source,
        "auth_mode": get_auth_mode_snapshot()["self_service"]["mode"],
    }


@router.get("/keys", response_model=list[ApiKeyRead], summary="List own API keys")
async def list_own_keys(request: Request, authorization: Optional[str] = Header(default=None)):
    token = _require_user_token(authorization, request)
    owner_id = _resolve_owner_id(token)
    return store.list_api_keys(owner_id=owner_id)


@router.post("/keys", response_model=ApiKeyCreateResponse, summary="Create own API key")
async def create_own_key(payload: ApiKeyCreate, request: Request, authorization: Optional[str] = Header(default=None)):
    token = _require_user_token(authorization, request)
    owner_id = _resolve_owner_id(token)
    return store.create_api_key(name=payload.name, owner_id=owner_id)


@router.delete("/keys/{key_id}", summary="Delete own API key")
async def delete_own_key(key_id: str, request: Request, authorization: Optional[str] = Header(default=None)):
    token = _require_user_token(authorization, request)
    owner_id = _resolve_owner_id(token)
    deleted = store.revoke_api_key(key_id, owner_id=owner_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"status": "deleted", "key_id": key_id}


@router.get("/budgets", response_model=list[BudgetRead], summary="Get own budgets")
async def get_own_budgets(request: Request, authorization: Optional[str] = Header(default=None)):
    token = _require_user_token(authorization, request)
    owner_id = _resolve_owner_id(token)
    return store.list_budgets(entity_type="user", entity_id=owner_id)


@router.get("/budgets/usage", response_model=list[BudgetUsageRead], summary="Get own budget usage")
async def get_own_budget_usage(request: Request, authorization: Optional[str] = Header(default=None)):
    token = _require_user_token(authorization, request)
    owner_id = _resolve_owner_id(token)
    return store.get_all_budget_usage(entity_type="user", entity_id=owner_id)


@router.get("/budgets/context", response_model=BudgetContextRead, summary="Get own budgets with group-aware summary")
async def get_budget_context(request: Request, authorization: Optional[str] = Header(default=None)):
    token = _require_user_token(authorization, request)
    owner_id = _resolve_owner_id(token)
    groups = get_oidc_groups(token)

    entities: list[tuple[str, str]] = [("user", owner_id)]
    entities.extend(("group", group_id) for group_id in groups if group_id)

    budgets: list[dict] = []
    usage: list[dict] = []
    for entity_type, entity_id in entities:
        budgets.extend(store.list_budgets(entity_type=entity_type, entity_id=entity_id))
        usage.extend(store.get_all_budget_usage(entity_type=entity_type, entity_id=entity_id))

    return {
        "owner_id": owner_id,
        "groups": groups,
        "daily_bucket": _current_bucket("daily"),
        "monthly_bucket": _current_bucket("monthly"),
        "budgets": budgets,
        "usage": usage,
        "summary": _build_budget_summary(budgets, usage),
    }


