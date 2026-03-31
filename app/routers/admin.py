import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status

from access_store import store
from auth import extract_bearer_token
from config import get_admin_token
from schemas.access import (
    AdminApiKeyCreate,
    ApiKeyCreateResponse,
    ApiKeyRead,
    ProtectedEndpointRule,
    ProtectedEndpointRuleRead,
    QuotaPolicy,
    QuotaPolicyRead,
    QuotaUsageRead,
)

router = APIRouter(prefix="/api/admin", tags=["admin"])

def require_admin(authorization: Optional[str] = Header(default=None)) -> None:
    expected_token = get_admin_token()
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API is enabled but FOAP_ADMIN_TOKEN is not configured.",
        )

    provided_token = extract_bearer_token(authorization)
    if provided_token != expected_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


@router.get("/health", summary="Admin API health")
async def admin_health(_: None = Depends(require_admin)):
    return {"status": "ok", "scope": "admin"}


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
        return store.create_protected_endpoint(path=rule.path, method=rule.method)
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

