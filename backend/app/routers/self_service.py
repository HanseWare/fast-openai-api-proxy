from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query, Request, Cookie, status, Response
from fastapi.responses import RedirectResponse

from access_store import store
from auth import extract_bearer_token, get_oidc_owner_id, identity_from_token
from config import get_auth_mode_snapshot, is_self_service_oidc_only_enabled, is_oidc_auth_enabled, get_oidc_client_id, get_oidc_client_secret
from oidc_auth import get_oidc_claims, has_self_service_access, get_owner_id_from_claims
from oidc_bff import generate_auth_state, build_authorization_uri, exchange_code_for_token, validate_state, get_base_url_from_request
from session_store import store as session_store
from schemas.access import ApiKeyCreate, ApiKeyCreateResponse, ApiKeyRead, QuotaPolicyRead, QuotaUsageRead

router = APIRouter(prefix="/api", tags=["self-service"])


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

    # Set session cookie
    response.set_cookie(
        key="foap_oidc_session",
        value=session_id,
        max_age=600,  # 10 minutes for auth flow
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
    return redirect_response


def _require_user_token(authorization: Optional[str]) -> str:
    token = extract_bearer_token(authorization)
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
async def self_service_health(authorization: Optional[str] = Header(default=None)):
    _require_user_token(authorization)
    return {"status": "ok", "scope": "self-service"}


@router.get("/session", summary="Get current self-service session")
async def self_service_session(authorization: Optional[str] = Header(default=None)):
    token = _require_user_token(authorization)
    owner_id = _resolve_owner_id(token)
    auth_source = "oidc" if get_oidc_owner_id(token) else "token-hash"
    return {
        "owner_id": owner_id,
        "auth_source": auth_source,
        "auth_mode": get_auth_mode_snapshot()["self_service"]["mode"],
    }


@router.get("/keys", response_model=list[ApiKeyRead], summary="List own API keys")
async def list_own_keys(authorization: Optional[str] = Header(default=None)):
    token = _require_user_token(authorization)
    owner_id = _resolve_owner_id(token)
    return store.list_api_keys(owner_id=owner_id)


@router.post("/keys", response_model=ApiKeyCreateResponse, summary="Create own API key")
async def create_own_key(payload: ApiKeyCreate, authorization: Optional[str] = Header(default=None)):
    token = _require_user_token(authorization)
    owner_id = _resolve_owner_id(token)
    return store.create_api_key(name=payload.name, owner_id=owner_id)


@router.delete("/keys/{key_id}", summary="Delete own API key")
async def delete_own_key(key_id: str, authorization: Optional[str] = Header(default=None)):
    token = _require_user_token(authorization)
    owner_id = _resolve_owner_id(token)
    deleted = store.revoke_api_key(key_id, owner_id=owner_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"status": "deleted", "key_id": key_id}


@router.get("/keys/{key_id}/quota", response_model=QuotaPolicyRead, summary="Get own API key quota")
async def get_own_key_quota(key_id: str, authorization: Optional[str] = Header(default=None)):
    token = _require_user_token(authorization)
    owner_id = _resolve_owner_id(token)
    _require_owned_key(owner_id, key_id)

    quota = store.get_quota(key_id)
    if quota is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Quota not configured")
    return {"api_key_id": key_id, "requests_per_minute": quota}


@router.get("/keys/{key_id}/usage", response_model=QuotaUsageRead, summary="Get own API key usage")
async def get_own_key_usage(key_id: str, authorization: Optional[str] = Header(default=None)):
    token = _require_user_token(authorization)
    owner_id = _resolve_owner_id(token)
    _require_owned_key(owner_id, key_id)

    usage = store.get_quota_usage(key_id)
    if usage is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Quota not configured")
    return usage


@router.get("/usage/summary", summary="Get own aggregated usage summary")
async def get_own_usage_summary(
    authorization: Optional[str] = Header(default=None),
    model: Optional[str] = Query(default=None, min_length=1),
    api_path: Optional[str] = Query(default=None, min_length=1),
    window_size: int = Query(default=6, ge=1, le=96),
):
    token = _require_user_token(authorization)
    owner_id = _resolve_owner_id(token)
    return store.get_usage_summary(owner_id=owner_id, model=model, api_path=api_path, window_size=window_size)

