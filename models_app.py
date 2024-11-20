from typing import Any, Dict

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import os
import json

# TODO: not sure why you need that?!
# from openai import api_key


class ModelsApp(FastAPI):
    models: Dict[str, Any] = {}

    def __init__(self):
        super().__init__()
        self.models = {}
        # Read models from config files in folder "configs"
        config_dir = os.getenv('FOAP_CONFIG_DIR', 'configs')
        for filename in os.listdir(config_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(config_dir, filename)
                with open(filepath, 'r') as f:
                    config = json.load(f)
                    # Process the config
                    self.load_config(config)

    def load_config(self, config: Dict[str, Any]):
        for provider_name, provider_config in config.items():
            prefix = provider_config.get('prefix', '')
            models = provider_config.get('models', {})
            # use api_key_variable to read the api_key from environment variable, default api_key to "ignored" if no api_key_variable is provided
            api_key = 'ignored'
            if provider_config.get('api_key_variable'):
                api_key = os.getenv(provider_config.get('api_key_variable'), 'ignored')
            for model_name, model_info in models.items():
                full_model_name = f"{prefix}{model_name}"
                # Merge provider-level and model-level configurations
                model_info['api_key'] = api_key
                model_info['provider'] = provider_name
                model_info['model_requested'] = model_name
                model_info['target_model'] = model_info.get('target_model', model_name)
                model_info['target_base_url'] = model_info.get('target_base_url',
                                                               provider_config.get('target_base_url', ''))
                model_info['request_timeout'] = model_info.get('request_timeout',
                                                               provider_config.get('request_timeout', 60))
                model_info['health_timeout'] = model_info.get('health_timeout',
                                                                provider_config.get('health_timeout', 60))
                self.models[full_model_name] = model_info

    def get_model_data(self, model, api_path=None):
        model_info = self.models.get(model)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {model} not found")
        if model_info and (not api_path or api_path in model_info.get('apis_supported', [])):
            return model_info
        return None

    def get_model_list(self):
        return list(self.models.keys())


app = ModelsApp()


# Route for model details
@app.get("/")
async def models_details(request: Request):
    model_list = app.get_model_list()
    response_data = {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": 1714780800,
                "owned_by": "myLab"
            } for model in model_list
        ]
    }
    return JSONResponse(content=response_data)
