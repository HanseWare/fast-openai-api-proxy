import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, Cookie, status
from fastapi.responses import RedirectResponse

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
from oidc_auth import get_oidc_claims, has_admin_access, get_owner_id_from_claims
from oidc_bff import generate_auth_state, build_authorization_uri, exchange_code_for_token, validate_state, get_base_url_from_request
from session_store import store as session_store
from schemas.access import (
    AuthModeSnapshot,
    AdminApiKeyCreate,
    ApiKeyCreateResponse,
    ApiKeyRead,
    ModelQuotaPolicyCreate,
    ModelQuotaPolicyRead,
    ModelQuotaPolicyUpdate,
    ProtectedEndpointRule,
    ProtectedEndpointRuleRead,
    QuotaPolicy,
    QuotaOverrideCreate,
    QuotaOverrideRead,
    QuotaOverrideUpdate,
    QuotaPolicyRead,
    QuotaUsageRead,
)

router = APIRouter(prefix="/api/admin", tags=["admin"])


def _validate_override_window(starts_at: int | None, ends_at: int | None) -> None:
    if starts_at is not None and ends_at is not None and ends_at <= starts_at:
        raise HTTPException(status_code=400, detail="ends_at must be greater than starts_at")

def require_admin(authorization: Optional[str] = Header(default=None)) -> None:
    expected_token = get_admin_token()
    provided_token = extract_bearer_token(authorization)
    admin_oidc_only = is_admin_oidc_only_enabled()

    if admin_oidc_only:
        if not provided_token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="OIDC bearer token required")
        claims = get_oidc_claims(provided_token)
        if claims is None or not has_admin_access(claims):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="OIDC admin claim missing or invalid")
        return

    if expected_token and provided_token == expected_token:
        return

    if is_oidc_auth_enabled() and provided_token:
        claims = get_oidc_claims(provided_token)
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
        "owner_id": get_owner_id_from_claims(claims),
        "scope": "admin",
    }, ttl_seconds=86400)  # 24 hours

    # Clean up old session
    session_store.delete(oidc_session or "")

    # Redirect to dashboard and set authenticated session cookie on the redirect response
    redirect_response = RedirectResponse(url="/", status_code=302)
    redirect_response.set_cookie(
        key="foap_session",
        value=auth_session_id,
        max_age=86400,
        httponly=True,
        secure=True,
        samesite="lax",
    )
    return redirect_response


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


@router.put("/keys/{key_id}/quota", response_model=QuotaPolicyRead, summary="Set key quota")
async def set_key_quota(key_id: str, payload: QuotaPolicy, _: None = Depends(require_admin)):
    if not store.key_exists(key_id):
        raise HTTPException(status_code=404, detail="API key not found")

    store.set_quota(api_key_id=key_id, requests_per_minute=payload.requests_per_minute)
    return {"api_key_id": key_id, "requests_per_minute": payload.requests_per_minute}


@router.get("/keys/{key_id}/quota", response_model=QuotaPolicyRead, summary="Get key quota")
async def get_key_quota(key_id: str, _: None = Depends(require_admin)):
    quota = store.get_quota(key_id)
    if quota is None:
        raise HTTPException(status_code=404, detail="Quota not configured")
    return {"api_key_id": key_id, "requests_per_minute": quota}


@router.get("/keys/{key_id}/usage", response_model=QuotaUsageRead, summary="Get current key quota usage")
async def get_key_usage(key_id: str, _: None = Depends(require_admin)):
    usage = store.get_quota_usage(key_id)
    if usage is None:
        raise HTTPException(status_code=404, detail="Quota not configured")
    return usage


@router.get("/quota-policies", response_model=list[ModelQuotaPolicyRead], summary="List quota policies")
async def list_quota_policies(
    response: Response,
    api_path: Optional[str] = Query(default=None, min_length=1),
    model: Optional[str] = Query(default=None, min_length=1),
    limit: Optional[int] = Query(default=None, ge=1),
    offset: Optional[int] = Query(default=None, ge=0),
    _: None = Depends(require_admin),
):
    policies, total = store.list_quota_policies(
        api_path=api_path,
        model=model,
        limit=limit,
        offset=offset,
    )
    response.headers["X-Total-Count"] = str(total)
    response.headers["X-Returned-Count"] = str(len(policies))
    if limit is not None:
        response.headers["X-Limit"] = str(limit)
    if offset is not None:
        response.headers["X-Offset"] = str(offset)
    return policies


@router.get("/quota-policies/{policy_id}", response_model=ModelQuotaPolicyRead, summary="Get quota policy")
async def get_quota_policy(policy_id: str, _: None = Depends(require_admin)):
    policy = store.get_quota_policy(policy_id)
    if not policy:
        raise HTTPException(status_code=404, detail="Quota policy not found")
    return policy


@router.post("/quota-policies", response_model=ModelQuotaPolicyRead, summary="Create quota policy")
async def create_quota_policy(payload: ModelQuotaPolicyCreate, _: None = Depends(require_admin)):
    try:
        return store.create_quota_policy(
            api_path=payload.api_path,
            model=payload.model,
            window_type=payload.window_type,
            request_limit=payload.request_limit,
            enforce_per_user=payload.enforce_per_user,
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Quota policy already exists for api_path + model") from exc


@router.put("/quota-policies/{policy_id}", response_model=ModelQuotaPolicyRead, summary="Update quota policy")
async def update_quota_policy(policy_id: str, payload: ModelQuotaPolicyUpdate, _: None = Depends(require_admin)):
    try:
        updated = store.update_quota_policy(
            policy_id=policy_id,
            api_path=payload.api_path,
            model=payload.model,
            window_type=payload.window_type,
            request_limit=payload.request_limit,
            enforce_per_user=payload.enforce_per_user,
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Quota policy already exists for api_path + model") from exc

    if updated is None:
        raise HTTPException(status_code=404, detail="Quota policy not found")
    return updated


@router.delete("/quota-policies/{policy_id}", summary="Delete quota policy")
async def delete_quota_policy(policy_id: str, _: None = Depends(require_admin)):
    deleted = store.delete_quota_policy(policy_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Quota policy not found")
    return {"status": "deleted", "policy_id": policy_id}


@router.get("/quota-overrides/{override_id}", response_model=QuotaOverrideRead, summary="Get quota override")
async def get_quota_override(override_id: str, _: None = Depends(require_admin)):
    override = store.get_quota_override(override_id)
    if not override:
        raise HTTPException(status_code=404, detail="Quota override not found")
    return override


@router.get("/quota-overrides", response_model=list[QuotaOverrideRead], summary="List quota overrides")
async def list_quota_overrides(
    response: Response,
    owner_id: Optional[str] = Query(default=None, min_length=1),
    api_path: Optional[str] = Query(default=None, min_length=1),
    model: Optional[str] = Query(default=None, min_length=1),
    exempt: Optional[bool] = Query(default=None),
    active_only: bool = Query(default=False),
    limit: Optional[int] = Query(default=None, ge=1),
    offset: Optional[int] = Query(default=None, ge=0),
    _: None = Depends(require_admin),
):
    overrides, total = store.list_quota_overrides(
        owner_id=owner_id,
        api_path=api_path,
        model=model,
        exempt=exempt,
        active_only=active_only,
        limit=limit,
        offset=offset,
    )
    response.headers["X-Total-Count"] = str(total)
    response.headers["X-Returned-Count"] = str(len(overrides))
    if limit is not None:
        response.headers["X-Limit"] = str(limit)
    if offset is not None:
        response.headers["X-Offset"] = str(offset)
    return overrides


@router.post("/quota-overrides", response_model=QuotaOverrideRead, summary="Create quota override")
async def create_quota_override(payload: QuotaOverrideCreate, _: None = Depends(require_admin)):
    _validate_override_window(payload.starts_at, payload.ends_at)
    try:
        return store.create_quota_override(
            api_path=payload.api_path,
            model=payload.model,
            owner_id=payload.owner_id,
            window_type=payload.window_type,
            request_limit=payload.request_limit,
            exempt=payload.exempt,
            starts_at=payload.starts_at,
            ends_at=payload.ends_at,
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Quota override already exists for api_path + model + owner_id") from exc


@router.put("/quota-overrides/{override_id}", response_model=QuotaOverrideRead, summary="Update quota override")
async def update_quota_override(override_id: str, payload: QuotaOverrideUpdate, _: None = Depends(require_admin)):
    _validate_override_window(payload.starts_at, payload.ends_at)
    try:
        updated = store.update_quota_override(
            override_id=override_id,
            api_path=payload.api_path,
            model=payload.model,
            owner_id=payload.owner_id,
            window_type=payload.window_type,
            request_limit=payload.request_limit,
            exempt=payload.exempt,
            starts_at=payload.starts_at,
            ends_at=payload.ends_at,
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Quota override already exists for api_path + model + owner_id") from exc

    if updated is None:
        raise HTTPException(status_code=404, detail="Quota override not found")
    return updated


@router.delete("/quota-overrides/{override_id}", summary="Delete quota override")
async def delete_quota_override(override_id: str, _: None = Depends(require_admin)):
    deleted = store.delete_quota_override(override_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Quota override not found")
    return {"status": "deleted", "override_id": override_id}

