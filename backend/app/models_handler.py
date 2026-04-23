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
            # Fallback to JSON configs
            config_dir = os.getenv("FOAP_CONFIG_DIR", "/configs")
            if os.path.exists(config_dir):
                for filename in os.listdir(config_dir):
                    if filename.endswith(".json"):
                        filepath = os.path.join(config_dir, filename)
                        with open(filepath, "r") as f:
                            try:
                                config = json.load(f)
                                self.load_config(config)
                            except Exception as e:
                                print(f"Failed to load JSON config {filepath}: {e}")
        else:
            # Load from DB
            for p in providers:
                provider_name = p["name"]
                prefix = p["prefix"]
                api_key_var = p["api_key_variable"]
                api_key = os.getenv(api_key_var, "ignored") if api_key_var else "ignored"
                
                models_db = config_store.list_models_for_provider(p["id"])
                for m in models_db:
                    full_model_name = f"{prefix}{m['name']}"
                    
                    endpoints_db = config_store.list_endpoints_for_model(m["id"])
                    eps = []
                    for e in endpoints_db:
                        eps.append({
                            "path": e["path"],
                            "api_key": api_key,
                            "provider": provider_name,
                            "model_requested": m["name"],
                            "target_base_url": e["target_base_url"] or p["default_base_url"] or "",
                            "target_model_name": e["target_model_name"],
                            "request_timeout": e["request_timeout"] or p["default_request_timeout"] or 60,
                            "health_timeout": e["health_timeout"] or p["default_health_timeout"] or 60,
                            "fallback_model_name": e["fallback_model_name"],
                            "max_upstream_retry_seconds": p["max_upstream_retry_seconds"],
                            "sync_provider_ratelimits": p["sync_provider_ratelimits"]
                        })
                        
                    self.models[full_model_name] = {
                        "model_info": {"endpoints": eps},
                        "endpoints": eps
                    }

    def load_config(self, config: Dict[str, Any]):
        for provider_name, provider_config in config.items():
            prefix = provider_config.get("prefix", "")
            models = provider_config.get("models", {})
            # Use api_key_variable to read the api_key from environment variable, default api_key to "ignored" if no api_key_variable is provided
            api_key = "ignored"
            if provider_config.get("api_key_variable"):
                api_key = os.getenv(provider_config.get("api_key_variable"), "ignored")
            for model_name, model_info in models.items():
                full_model_name = f"{prefix}{model_name}"
                endpoints = model_info.get("endpoints", [])
                for endpoint in endpoints:
                    # Merge provider-level and model-level configurations
                    endpoint["api_key"] = api_key
                    endpoint["provider"] = provider_name
                    endpoint["model_requested"] = model_name
                    endpoint["target_base_url"] = endpoint.get("target_base_url", provider_config.get("target_base_url", ""))
                    endpoint["request_timeout"] = endpoint.get("request_timeout", provider_config.get("request_timeout", 60))
                    endpoint["health_timeout"] = endpoint.get("health_timeout", provider_config.get("health_timeout", 60))
                self.models[full_model_name] = {
                    "model_info": model_info,
                    "endpoints": endpoints
                }

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
