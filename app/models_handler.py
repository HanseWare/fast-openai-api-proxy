from typing import Any, Dict

from fastapi import HTTPException
import os
import json


class ModelsHandler:
    models: Dict[str, Any] = {}

    def __init__(self):
        self.models = {}
        # Read models from config files in folder "configs"
        config_dir = os.getenv("FOAP_CONFIG_DIR", "../configs")
        for filename in os.listdir(config_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(config_dir, filename)
                with open(filepath, "r") as f:
                    config = json.load(f)
                    # Process the config
                    self.load_config(config)

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
        model_entry = self.models.get(model)
        if not model_entry:
            raise HTTPException(status_code=404, detail=f"Model {model} not found")
        if not api_path:
            return model_entry
        for endpoint in model_entry.get("endpoints", []):
            if endpoint["path"] == api_path:
                return endpoint
        raise HTTPException(status_code=404, detail=f"API path {api_path} not supported for model {model}")

    def get_model(self, model):
        if model not in self.models:
            return None
        return {
            "id": model,
            "object": "model",
            "created": 1714780800,
            "owned_by": "myLab"
        }

    def get_model_list(self):
        return [
            {
                "id": model,
                "object": "model",
                "created": 1714780800,
                "owned_by": "myLab"
            } for model in self.models.keys()
        ]


handler = ModelsHandler()
