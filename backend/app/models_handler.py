from typing import Any, Dict

from fastapi import HTTPException
import os
import json
from config_store import config_store

class ModelsHandler:
    models: Dict[str, Any] = {}
    aliases: Dict[str, str] = {}

    def __init__(self):
        self.refresh()
        
    def refresh(self):
        self.models = {}
        self.aliases = {}
        
        # 1. Load Aliases
        aliases_db = config_store.list_aliases()
        for a in aliases_db:
            self.aliases[a["alias_name"]] = a["target_model_name"]
            
        # 2. Load Providers
        providers = config_store.list_providers()
        
        if not providers:
            # Fallback to JSON configs - seed the DB automatically
            config_dirs = [os.getenv("FOAP_CONFIG_DIR"), "/app/custom_configs", "../configs", "/configs"]
            config_dir = next((d for d in config_dirs if d and os.path.exists(d)), None)
            
            if config_dir:
                for filename in os.listdir(config_dir):
                    if filename.endswith(".json") or filename.endswith(".yaml") or filename.endswith(".yml"):
                        filepath = os.path.join(config_dir, filename)
                        with open(filepath, "r") as f:
                            try:
                                if filename.endswith(".json"):
                                    config = json.load(f)
                                    self._seed_db_from_json(config)
                                else:
                                    import yaml
                                    yml_data = yaml.safe_load(f)
                                    if yml_data and yml_data.get("kind") == "ConfigMap" and "data" in yml_data:
                                        for key, val in yml_data["data"].items():
                                            if key.endswith(".json"):
                                                config = json.loads(val)
                                                self._seed_db_from_json(config)
                                    else:
                                        self._seed_db_from_json(yml_data)
                            except Exception as e:
                                print(f"Failed to load config {filepath}: {e}")
                # Re-fetch providers now that DB is seeded
                providers = config_store.list_providers()
        
        # Load from DB
        for p in providers:
            provider_name = p["name"]
            prefix = p["prefix"]
            api_key_var = p["api_key_variable"]
            api_key = os.getenv(api_key_var, "ignored") if api_key_var else "ignored"
            
            provider_route_fallbacks = p.get("route_fallbacks", {})
            
            models_db = config_store.list_models_for_provider(p["id"])
            for m in models_db:
                full_model_name = f"{prefix}{m['name']}"
                
                endpoints_db = config_store.list_endpoints_for_model(m["id"])
                eps = []
                for e in endpoints_db:
                    endpoint_fallback = e.get("fallback_model_name")
                    if not endpoint_fallback:
                        endpoint_fallback = provider_route_fallbacks.get(e["path"])
                    
                    eps.append({
                        "path": e["path"],
                        "api_key": api_key,
                        "provider": provider_name,
                        "model_requested": m["name"],
                        "target_base_url": e.get("target_base_url") or p.get("default_base_url") or "",
                        "target_model_name": e.get("target_model_name"),
                        "request_timeout": e.get("request_timeout") or p.get("default_request_timeout") or 60,
                        "health_timeout": e.get("health_timeout") or p.get("default_health_timeout") or 60,
                        "fallback_model_name": endpoint_fallback,
                        "max_upstream_retry_seconds": p.get("max_upstream_retry_seconds", 0),
                        "sync_provider_ratelimits": p.get("sync_provider_ratelimits", False)
                    })
                    
                self.models[full_model_name] = {
                    "model_info": {"endpoints": eps},
                    "endpoints": eps
                }

    def _seed_db_from_json(self, config: Dict[str, Any]):
        for provider_name, provider_config in config.items():
            p = config_store.get_provider_by_name(provider_name)
            if not p:
                p = config_store.create_provider(
                    name=provider_name,
                    api_key_variable=provider_config.get("api_key_variable"),
                    prefix=provider_config.get("prefix", ""),
                    default_base_url=provider_config.get("default_base_url"),
                    default_request_timeout=provider_config.get("default_request_timeout"),
                    default_health_timeout=provider_config.get("default_health_timeout"),
                    max_upstream_retry_seconds=provider_config.get("max_upstream_retry_seconds", 0),
                    sync_provider_ratelimits=provider_config.get("sync_provider_ratelimits", False),
                    route_fallbacks=provider_config.get("route_fallbacks", {})
                )
            
            models = provider_config.get("models", {})
            for model_name, model_info in models.items():
                m = config_store.get_model_by_name(p["id"], model_name)
                if not m:
                    m = config_store.create_model(p["id"], model_name)
                
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

    def get_model_data(self, model, api_path=None):
        resolved_model = self.aliases.get(model, model)
        
        model_entry = self.models.get(resolved_model)
        if not model_entry:
            raise HTTPException(status_code=404, detail=f"Model {model} (resolved to {resolved_model}) not found")
        if not api_path:
            return model_entry
        for endpoint in model_entry.get("endpoints", []):
            if endpoint["path"] == api_path:
                return endpoint
        raise HTTPException(status_code=404, detail=f"API path {api_path} not supported for model {model}")

    def get_model(self, model):
        resolved_model = self.aliases.get(model, model)
        if resolved_model not in self.models:
            return None
        return {
            "id": model,
            "object": "model",
            "created": 1714780800,
            "owned_by": "FOAP"
        }

    def get_model_list(self):
        return [
            {
                "id": model,
                "object": "model",
                "created": 1714780800,
                "owned_by": "FOAP"
            } for model in self.models.keys()
        ]


handler = ModelsHandler()
