from typing import Optional

from fastapi import APIRouter, Header, HTTPException, status

from access_store import store
from auth import extract_bearer_token, identity_from_token
from schemas.access import ApiKeyCreate, ApiKeyCreateResponse, ApiKeyRead

router = APIRouter(prefix="/api", tags=["self-service"])

def _require_user_token(authorization: Optional[str]) -> str:
    token = extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return token


@router.get("/health", summary="Self-service API health")
async def self_service_health(authorization: Optional[str] = Header(default=None)):
    _require_user_token(authorization)
    return {"status": "ok", "scope": "self-service"}


@router.get("/keys", response_model=list[ApiKeyRead], summary="List own API keys")
async def list_own_keys(authorization: Optional[str] = Header(default=None)):
    token = _require_user_token(authorization)
    owner_id = identity_from_token(token)
    return store.list_api_keys(owner_id=owner_id)


@router.post("/keys", response_model=ApiKeyCreateResponse, summary="Create own API key")
async def create_own_key(payload: ApiKeyCreate, authorization: Optional[str] = Header(default=None)):
    token = _require_user_token(authorization)
    owner_id = identity_from_token(token)
    return store.create_api_key(name=payload.name, owner_id=owner_id)


@router.delete("/keys/{key_id}", summary="Delete own API key")
async def delete_own_key(key_id: str, authorization: Optional[str] = Header(default=None)):
    token = _require_user_token(authorization)
    owner_id = identity_from_token(token)
    deleted = store.revoke_api_key(key_id, owner_id=owner_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"status": "deleted", "key_id": key_id}

