from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import sqlite3

from routers.admin import require_admin
from config_store import config_store
from models_handler import handler as models_handler
from schemas.config import (
    ProviderCreate, ProviderRead, ProviderUpdate,
    ProviderModelCreate, ProviderModelUpdate, ProviderModelRead,
    ProviderModelEndpointCreate, ProviderModelEndpointUpdate, ProviderModelEndpointRead,
    ModelAliasCreate, ModelAliasUpdate, ModelAliasRead
)

class ImportPayload(BaseModel):
    config_json: Dict[str, Any]

router = APIRouter(prefix="/api/admin/config", tags=["admin-config"])

# ---------------- Providers ----------------

@router.get("/providers", response_model=List[ProviderRead])
async def list_providers(_: None = Depends(require_admin)):
    return config_store.list_providers()

@router.post("/providers", response_model=ProviderRead)
async def create_provider(payload: ProviderCreate, _: None = Depends(require_admin)):
    try:
        provider = config_store.create_provider(
            name=payload.name,
            api_key_variable=payload.api_key_variable,
            prefix=payload.prefix,
            default_base_url=payload.default_base_url,
            default_request_timeout=payload.default_request_timeout,
            default_health_timeout=payload.default_health_timeout
        )
        models_handler.refresh()
        return provider
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Provider name already exists")

@router.put("/providers/{provider_id}", response_model=ProviderRead)
async def update_provider(provider_id: str, payload: ProviderUpdate, _: None = Depends(require_admin)):
    try:
        updated = config_store.update_provider(
            provider_id=provider_id,
            name=payload.name,
            api_key_variable=payload.api_key_variable,
            prefix=payload.prefix,
            default_base_url=payload.default_base_url,
            default_request_timeout=payload.default_request_timeout,
            default_health_timeout=payload.default_health_timeout
        )
        if not updated:
            raise HTTPException(status_code=404, detail="Provider not found")
        models_handler.refresh()
        return updated
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Provider name conflict")

@router.delete("/providers/{provider_id}")
async def delete_provider(provider_id: str, _: None = Depends(require_admin)):
    if not config_store.delete_provider(provider_id):
        raise HTTPException(status_code=404, detail="Provider not found")
    models_handler.refresh()
    return {"status": "deleted"}

# ---------------- Models ----------------

@router.get("/providers/{provider_id}/models", response_model=List[ProviderModelRead])
async def list_models(provider_id: str, _: None = Depends(require_admin)):
    return config_store.list_models_for_provider(provider_id)

@router.get("/providers/{provider_id}/ratelimits")
async def get_provider_ratelimits(provider_id: str, _: None = Depends(require_admin)):
    provider = config_store.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    from access_store import store
    return store.get_provider_ratelimits(provider["name"]) or {}

@router.post("/models", response_model=ProviderModelRead)
async def create_model(payload: ProviderModelCreate, _: None = Depends(require_admin)):
    try:
        model = config_store.create_model(payload.provider_id, payload.name)
        models_handler.refresh()
        return model
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Model already exists for this provider")

@router.delete("/models/{model_id}")
async def delete_model(model_id: str, _: None = Depends(require_admin)):
    if not config_store.delete_model(model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    models_handler.refresh()
    return {"status": "deleted"}

@router.put("/models/{model_id}", response_model=ProviderModelRead)
async def update_model(model_id: str, payload: ProviderModelUpdate, _: None = Depends(require_admin)):
    try:
        updated = config_store.update_model(model_id, payload.name)
        if not updated:
            raise HTTPException(status_code=404, detail="Model not found")
        models_handler.refresh()
        return updated
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Model name already exists for this provider")

# ---------------- Endpoints ----------------

@router.get("/models/{model_id}/endpoints", response_model=List[ProviderModelEndpointRead])
async def list_endpoints(model_id: str, _: None = Depends(require_admin)):
    return config_store.list_endpoints_for_model(model_id)

@router.post("/endpoints", response_model=ProviderModelEndpointRead)
async def create_endpoint(payload: ProviderModelEndpointCreate, _: None = Depends(require_admin)):
    try:
        ep = config_store.create_endpoint(
            model_id=payload.model_id,
            path=payload.path,
            target_model_name=payload.target_model_name,
            target_base_url=payload.target_base_url,
            request_timeout=payload.request_timeout,
            health_timeout=payload.health_timeout,
            fallback_model_name=payload.fallback_model_name
        )
        models_handler.refresh()
        return ep
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Endpoint path already exists for this model")

@router.delete("/endpoints/{endpoint_id}")
async def delete_endpoint(endpoint_id: str, _: None = Depends(require_admin)):
    if not config_store.delete_endpoint(endpoint_id):
        raise HTTPException(status_code=404, detail="Endpoint not found")
    models_handler.refresh()
    return {"status": "deleted"}

@router.put("/endpoints/{endpoint_id}", response_model=ProviderModelEndpointRead)
async def update_endpoint(endpoint_id: str, payload: ProviderModelEndpointUpdate, _: None = Depends(require_admin)):
    try:
        updated = config_store.update_endpoint(
            endpoint_id,
            path=payload.path,
            target_model_name=payload.target_model_name,
            target_base_url=payload.target_base_url,
            request_timeout=payload.request_timeout,
            health_timeout=payload.health_timeout,
            fallback_model_name=payload.fallback_model_name
        )
        if not updated:
            raise HTTPException(status_code=404, detail="Endpoint not found")
        models_handler.refresh()
        return updated
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Endpoint path already exists for this model")

# ---------------- Aliases (Virtual Models) ----------------

@router.get("/aliases", response_model=List[ModelAliasRead])
async def list_aliases(_: None = Depends(require_admin)):
    return config_store.list_aliases()

@router.post("/aliases", response_model=ModelAliasRead)
async def create_alias(payload: ModelAliasCreate, _: None = Depends(require_admin)):
    try:
        alias = config_store.create_alias(payload.alias_name, payload.target_model_name)
        models_handler.refresh()
        return alias
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Alias name already exists")

@router.put("/aliases/{alias_id}", response_model=ModelAliasRead)
async def update_alias(alias_id: str, payload: ModelAliasUpdate, _: None = Depends(require_admin)):
    try:
        updated = config_store.update_alias(alias_id, payload.alias_name, payload.target_model_name)
        if not updated:
            raise HTTPException(status_code=404, detail="Alias not found")
        models_handler.refresh()
        return updated
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Alias name conflict")

@router.delete("/aliases/{alias_id}")
async def delete_alias(alias_id: str, _: None = Depends(require_admin)):
    if not config_store.delete_alias(alias_id):
        raise HTTPException(status_code=404, detail="Alias not found")
    models_handler.refresh()
    return {"status": "deleted"}

# ---------------- Import Tool ----------------

@router.post("/import")
async def import_config(payload: ImportPayload, _: None = Depends(require_admin)):
    stats = {"providers": 0, "models": 0, "endpoints": 0}
    for provider_name, provider_config in payload.config_json.items():
        p = config_store.get_provider_by_name(provider_name)
        if not p:
            p = config_store.create_provider(
                name=provider_name,
                api_key_variable=provider_config.get("api_key_variable", None),
                prefix=provider_config.get("prefix", ""),
                default_base_url=provider_config.get("target_base_url"),
                default_request_timeout=provider_config.get("request_timeout"),
                default_health_timeout=provider_config.get("health_timeout"),
                max_upstream_retry_seconds=provider_config.get("max_upstream_retry_seconds", 0),
                sync_provider_ratelimits=provider_config.get("sync_provider_ratelimits", False)
            )
            stats["providers"] += 1
            
        models = provider_config.get("models", {})
        for model_name, model_info in models.items():
            m = config_store.get_model_by_name(p["id"], model_name)
            if not m:
                m = config_store.create_model(p["id"], model_name)
                stats["models"] += 1
                
            endpoints = model_info.get("endpoints", [])
            for ep in endpoints:
                e = config_store.get_endpoint_by_path(m["id"], ep["path"])
                if not e:
                    config_store.create_endpoint(
                        model_id=m["id"],
                        path=ep["path"],
                        target_model_name=ep["target_model_name"],
                        target_base_url=ep.get("target_base_url"),
                        request_timeout=ep.get("request_timeout"),
                        health_timeout=ep.get("health_timeout"),
                        fallback_model_name=ep.get("fallback_model_name")
                    )
                    stats["endpoints"] += 1
    
    models_handler.refresh()
    return {"status": "success", "imported": stats}

