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
            base_url=payload.base_url,
            request_timeout=payload.request_timeout,
            health_timeout=payload.health_timeout,
            max_upstream_retry_seconds=payload.max_upstream_retry_seconds,
            sync_provider_ratelimits=payload.sync_provider_ratelimits,
            route_fallbacks=payload.route_fallbacks
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
            base_url=payload.base_url,
            request_timeout=payload.request_timeout,
            health_timeout=payload.health_timeout,
            max_upstream_retry_seconds=payload.max_upstream_retry_seconds,
            sync_provider_ratelimits=payload.sync_provider_ratelimits,
            route_fallbacks=payload.route_fallbacks
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
    return config_store.get_provider_ratelimits(provider["name"]) or {}

@router.post("/models", response_model=ProviderModelRead)
async def create_model(payload: ProviderModelCreate, _: None = Depends(require_admin)):
    try:
        model = config_store.create_model(
            provider_id=payload.provider_id,
            name=payload.name,
            type=payload.type,
            target_model_name=payload.target_model_name,
            target_base_url=payload.target_base_url,
            fallback_model_name=payload.fallback_model_name,
            supported_endpoints=payload.supported_endpoints,
            price_per_unit=payload.price_per_unit,
            min_credits_per_request=payload.min_credits_per_request,
            owned_by=payload.owned_by or 'FOAP',
            hide_on_models_endpoint=payload.hide_on_models_endpoint or False
        )
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
        updated = config_store.update_model(
            model_id=model_id,
            name=payload.name,
            type=payload.type,
            target_model_name=payload.target_model_name,
            target_base_url=payload.target_base_url,
            fallback_model_name=payload.fallback_model_name,
            supported_endpoints=payload.supported_endpoints,
            price_per_unit=payload.price_per_unit,
            min_credits_per_request=payload.min_credits_per_request,
            owned_by=payload.owned_by,
            hide_on_models_endpoint=payload.hide_on_models_endpoint
        )
        if not updated:
            raise HTTPException(status_code=404, detail="Model not found")
        models_handler.refresh()
        return updated
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Model name already exists for this provider")

# ---------------- Aliases (Virtual Models) ----------------

@router.get("/aliases", response_model=List[ModelAliasRead])
async def list_aliases(_: None = Depends(require_admin)):
    return config_store.list_aliases()

@router.post("/aliases", response_model=ModelAliasRead)
async def create_alias(payload: ModelAliasCreate, _: None = Depends(require_admin)):
    try:
        alias = config_store.create_alias(
            alias_name=payload.alias_name,
            target_model_name=payload.target_model_name,
            owned_by=payload.owned_by or 'FOAP',
            hide_on_models_endpoint=payload.hide_on_models_endpoint or False
        )
        models_handler.refresh()
        return alias
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Alias name already exists")

@router.put("/aliases/{alias_id}", response_model=ModelAliasRead)
async def update_alias(alias_id: str, payload: ModelAliasUpdate, _: None = Depends(require_admin)):
    try:
        updated = config_store.update_alias(
            alias_id=alias_id,
            alias_name=payload.alias_name,
            target_model_name=payload.target_model_name,
            owned_by=payload.owned_by,
            hide_on_models_endpoint=payload.hide_on_models_endpoint
        )
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
    stats = {"providers": 0, "models": 0}
    for provider_name, provider_config in payload.config_json.items():
        p = config_store.get_provider_by_name(provider_name)
        if not p:
            p = config_store.create_provider(
                name=provider_name,
                api_key_variable=provider_config.get("api_key_variable", None),
                prefix=provider_config.get("prefix", ""),
                base_url=provider_config.get("base_url") or provider_config.get("default_base_url"),
                request_timeout=provider_config.get("request_timeout") or provider_config.get("default_request_timeout"),
                health_timeout=provider_config.get("health_timeout") or provider_config.get("default_health_timeout"),
                max_upstream_retry_seconds=provider_config.get("max_upstream_retry_seconds", 0),
                sync_provider_ratelimits=provider_config.get("sync_provider_ratelimits", False)
            )
            stats["providers"] += 1
            
        models = provider_config.get("models", {})
        for model_name, model_info in models.items():
            m = config_store.get_model_by_name(p["id"], model_name)
            if not m:
                # Handle old schema import mapping (endpoints list to supported_endpoints array)
                legacy_endpoints = model_info.get("endpoints", [])
                supported_endpoints = model_info.get("supported_endpoints", [])
                target_model_name = model_info.get("target_model_name")
                target_base_url = model_info.get("target_base_url")
                fallback_model_name = model_info.get("fallback_model_name")
                
                if legacy_endpoints and not supported_endpoints:
                    supported_endpoints = [ep["path"] for ep in legacy_endpoints]
                    if not target_model_name:
                        target_model_name = legacy_endpoints[0].get("target_model_name", model_name)
                    if not target_base_url:
                        target_base_url = legacy_endpoints[0].get("target_base_url")
                    if not fallback_model_name:
                        fallback_model_name = legacy_endpoints[0].get("fallback_model_name")

                if not target_model_name:
                    target_model_name = model_name

                m = config_store.create_model(
                    provider_id=p["id"],
                    name=model_name,
                    type=model_info.get("type", "llm"),
                    target_model_name=target_model_name,
                    target_base_url=target_base_url,
                    fallback_model_name=fallback_model_name,
                    supported_endpoints=supported_endpoints,
                    price_per_unit=model_info.get("price_per_unit", 0.0),
                    min_credits_per_request=model_info.get("min_credits_per_request", 0.0),
                    owned_by=model_info.get("owned_by", "FOAP"),
                    hide_on_models_endpoint=model_info.get("hide_on_models_endpoint", False)
                )
                stats["models"] += 1
    
    models_handler.refresh()
    return {"status": "success", "imported": stats}
