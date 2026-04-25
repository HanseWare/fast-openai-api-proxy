from typing import Optional

from fastapi import APIRouter, Header, HTTPException, status

from access_store import store
from auth import extract_bearer_token, get_oidc_owner_id, identity_from_token
from config import is_self_service_oidc_only_enabled
from schemas.access import ApiKeyCreate, ApiKeyCreateResponse, ApiKeyRead, QuotaPolicyRead, QuotaUsageRead

router = APIRouter(prefix="/api", tags=["self-service"])

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
    owned_ids = {item["id"] for item in store.list_api_keys(owner_id=owner_id)}
    if key_id not in owned_ids:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")
    return {"api_key_id": key_id}


@router.get("/health", summary="Self-service API health")
async def self_service_health(authorization: Optional[str] = Header(default=None)):
    _require_user_token(authorization)
    return {"status": "ok", "scope": "self-service"}


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


