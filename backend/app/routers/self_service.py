from typing import Optional

from fastapi import APIRouter, Header, HTTPException, status

from access_store import store
from auth import extract_bearer_token, get_oidc_owner_id, identity_from_token
from config import is_self_service_oidc_only_enabled
from schemas.access import ApiKeyCreate, ApiKeyCreateResponse, ApiKeyRead

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

