import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, Cookie, status
from fastapi.responses import RedirectResponse, JSONResponse

from access_store import store
from auth import extract_bearer_token
from config import (
    get_admin_token,
    get_auth_mode_snapshot,
    is_admin_oidc_only_enabled,
    is_oidc_auth_enabled,
    get_oidc_client_id,
    get_oidc_client_secret,
)
from oidc_auth import get_oidc_claims, has_admin_access, get_owner_id_from_claims, try_refresh_session
from oidc_bff import generate_auth_state, build_authorization_uri, exchange_code_for_token, validate_state, get_base_url_from_request
from session_store import store as session_store
from schemas.access import (
    AuthModeSnapshot,
    AdminApiKeyCreate,
    ApiKeyCreateResponse,
    ApiKeyRead,
    ProtectedEndpointRule,
    ProtectedEndpointRuleRead,
    BudgetRead,
    BudgetCreate,
    BudgetUpdate,
)

router = APIRouter(prefix="/api/admin", tags=["admin"])


def require_admin(authorization: Optional[str] = Header(default=None), foap_session: Optional[str] = Cookie(default=None, alias="foap_session")) -> None:
    """Require admin access. Accepts either:

    - Authorization: Bearer <token> matching FOAP_ADMIN_TOKEN, or
    - an OIDC bearer token with admin claims, or
    - a cookie-based BFF session (`foap_session`) created by the admin OIDC callback.
    """
    expected_token = get_admin_token()
    provided_token = extract_bearer_token(authorization)
    admin_oidc_only = is_admin_oidc_only_enabled()

    # If admin-only via OIDC and Authorization header present
    if admin_oidc_only:
        if provided_token:
            claims = get_oidc_claims(provided_token)
            if claims is None or not has_admin_access(claims):
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="OIDC admin claim missing or invalid")
            return
        # try cookie-based session
        if foap_session:
            session = session_store.get(foap_session)
            if not session:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session")
            access_token = session.get("access_token")
            
            # If access token is expired, try to refresh
            if access_token and not get_oidc_claims(access_token):
                if try_refresh_session(session):
                    access_token = session.get("access_token")
                else:
                    access_token = None
            
            if access_token:
                claims = get_oidc_claims(access_token)
                if claims is None or not has_admin_access(claims):
                    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="OIDC admin claim missing or invalid")
                return
            
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="OIDC bearer token required")

    # If there is an expected static admin token, accept it
    if expected_token and provided_token == expected_token:
        return

    # If Authorization header contains an OIDC token, validate admin claim
    if is_oidc_auth_enabled() and provided_token:
        claims = get_oidc_claims(provided_token)
        if claims is not None and has_admin_access(claims):
            return

    # If Authorization header missing but cookie-based session exists, accept if admin claim present
    if foap_session and is_oidc_auth_enabled():
        session = session_store.get(foap_session)
        if session:
            access_token = session.get("access_token")
            
            # If access token is expired, try to refresh
            if access_token and not get_oidc_claims(access_token):
                if try_refresh_session(session):
                    access_token = session.get("access_token")
                else:
                    access_token = None
            
            if access_token:
                claims = get_oidc_claims(access_token)
                if claims is not None and has_admin_access(claims):
                    return

    if not expected_token and not is_oidc_auth_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API requires FOAP_ADMIN_TOKEN or OIDC configuration.",
        )

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


@router.get("/health", summary="Admin API health")
async def admin_health(_: None = Depends(require_admin)):
    return {"status": "ok", "scope": "admin"}


@router.get("/auth-config", response_model=AuthModeSnapshot, summary="Get active auth mode and claim mappings")
async def get_auth_config():
    return get_auth_mode_snapshot()


@router.get("/oidc/login", summary="Initiate OIDC authorization flow (BFF)")
async def oidc_login(response: Response, request: Request):
    """Initiate OIDC login for admin. Returns authorization URI."""
    if not is_oidc_auth_enabled() or not get_oidc_client_id() or not get_oidc_client_secret():
        raise HTTPException(status_code=400, detail="OIDC BFF not configured")

    state, code_verifier = generate_auth_state()

    # Build full redirect URI respecting X-Forwarded-Proto header
    base_url = get_base_url_from_request(request)
    redirect_uri = f"{base_url}/api/admin/oidc/callback"

    # Store state and verifier in session
    session_id = session_store.create({
        "state": state,
        "code_verifier": code_verifier,
        "scope": "admin",
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
    redirect_uri = f"{base_url}/api/admin/oidc/callback"

    # Exchange code for token
    token_response = exchange_code_for_token(code, redirect_uri, code_verifier)
    if not token_response or "access_token" not in token_response:
        raise HTTPException(status_code=401, detail="Failed to obtain access token")

    access_token = token_response["access_token"]

    # Validate token and check admin claim
    claims = get_oidc_claims(access_token)
    if not claims or not has_admin_access(claims):
        raise HTTPException(status_code=403, detail="No admin access in token claims")

    # Create authenticated session
    auth_session_id = session_store.create({
        "access_token": access_token,
        "refresh_token": token_response.get("refresh_token"),  # Store refresh token if available
        "owner_id": get_owner_id_from_claims(claims),
        "scope": "admin",
    }, ttl_seconds=86400)  # 24 hours

    # Clean up old session
    session_store.delete(oidc_session or "")

    # Redirect to dashboard and set authenticated session cookie on the redirect response
    redirect_response = RedirectResponse(url="/admin", status_code=302)
    redirect_response.set_cookie(
        key="foap_session",
        value=auth_session_id,
        max_age=86400,
        httponly=True,
        secure=True,
        samesite="lax",
    )
    redirect_response.set_cookie(
        key="foap_admin_logged_in",
        value="true",
        max_age=86400,
        secure=True,
        samesite="lax",
    )
    return redirect_response


@router.post("/logout", summary="Admin logout")
async def admin_logout(
    oidc_session: Optional[str] = Cookie(default=None, alias="foap_oidc_session"),
    foap_session: Optional[str] = Cookie(default=None, alias="foap_session"),
):
    """Invalidate admin cookie-based sessions and clear client cookies."""
    if oidc_session:
        session_store.delete(oidc_session)
    if foap_session:
        session_store.delete(foap_session)

    response = JSONResponse({"status": "logged_out"})
    for name in ("foap_session", "foap_admin_logged_in", "foap_oidc_session"):
        response.delete_cookie(name, samesite="lax")
    return response


@router.get("/keys", response_model=list[ApiKeyRead], summary="List managed API keys")
async def list_admin_keys(_: None = Depends(require_admin)):
    return store.list_api_keys(owner_id=None)


@router.post("/keys", response_model=ApiKeyCreateResponse, summary="Create managed API key")
async def create_admin_key(payload: AdminApiKeyCreate, _: None = Depends(require_admin)):
    return store.create_api_key(name=payload.name, owner_id=payload.owner_id)


@router.delete("/keys/{key_id}", summary="Delete managed API key")
async def delete_admin_key(key_id: str, _: None = Depends(require_admin)):
    deleted = store.revoke_api_key(key_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"status": "deleted", "key_id": key_id}


@router.get("/protected-endpoints", response_model=list[ProtectedEndpointRuleRead], summary="List protected endpoints")
async def list_protected_endpoints(_: None = Depends(require_admin)):
    return store.list_protected_endpoints()


@router.post("/protected-endpoints", response_model=ProtectedEndpointRuleRead, summary="Create protected endpoint rule")
async def create_protected_endpoint(rule: ProtectedEndpointRule, _: None = Depends(require_admin)):
    try:
        return store.create_protected_endpoint(path=rule.path, method=rule.method, model_pattern=rule.model_pattern)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Protected endpoint already exists") from exc


@router.delete("/protected-endpoints/{endpoint_id}", summary="Delete protected endpoint rule")
async def delete_protected_endpoint(endpoint_id: str, _: None = Depends(require_admin)):
    deleted = store.delete_protected_endpoint(endpoint_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Protected endpoint not found")
    return {"status": "deleted", "endpoint_id": endpoint_id}


# --- BUDGETS ---

@router.get("/budgets", response_model=list[BudgetRead], summary="List all budgets")
async def list_budgets(
    entity_type: Optional[str] = Query(default=None),
    entity_id: Optional[str] = Query(default=None),
    _: None = Depends(require_admin)
):
    return store.list_budgets(entity_type=entity_type, entity_id=entity_id)


@router.get("/budgets/{budget_id}", response_model=BudgetRead, summary="Get budget by ID")
async def get_budget(budget_id: str, _: None = Depends(require_admin)):
    budget = store.get_budget(budget_id)
    if not budget:
        raise HTTPException(status_code=404, detail="Budget not found")
    return budget


@router.post("/budgets", response_model=BudgetRead, summary="Create budget")
async def create_budget(payload: BudgetCreate, _: None = Depends(require_admin)):
    try:
        return store.create_budget(
            entity_type=payload.entity_type,
            entity_id=payload.entity_id,
            window=payload.window,
            budget_amount=payload.budget_amount,
            scope=payload.scope
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Budget already exists for this entity and window") from exc


@router.put("/budgets/{budget_id}", response_model=BudgetRead, summary="Update budget amount")
async def update_budget(budget_id: str, payload: BudgetUpdate, _: None = Depends(require_admin)):
    updated = store.update_budget(budget_id, budget_amount=payload.budget_amount)
    if updated is None:
        raise HTTPException(status_code=404, detail="Budget not found")
    return updated


@router.delete("/budgets/{budget_id}", summary="Delete budget")
async def delete_budget(budget_id: str, _: None = Depends(require_admin)):
    deleted = store.delete_budget(budget_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Budget not found")
    return {"status": "deleted", "budget_id": budget_id}
