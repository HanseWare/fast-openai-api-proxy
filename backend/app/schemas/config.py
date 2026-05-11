from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict

# Providers
class ProviderBase(BaseModel):
    name: str = Field(..., min_length=1)
    api_key_variable: Optional[str] = None
    prefix: str = Field(default="")
    base_url: Optional[str] = None
    request_timeout: Optional[int] = None
    health_timeout: Optional[int] = None
    max_upstream_retry_seconds: Optional[int] = Field(default=0)
    sync_provider_ratelimits: Optional[bool] = Field(default=False)
    route_fallbacks: Optional[Dict[str, str]] = Field(default_factory=dict)

class ProviderCreate(ProviderBase):
    pass

class ProviderUpdate(ProviderBase):
    pass

class ProviderRead(ProviderBase):
    id: str
    created_at: int

# Models
class ProviderModelBase(BaseModel):
    name: str = Field(..., min_length=1)
    type: str = Field(default="llm")
    target_model_name: str = Field(..., min_length=1)
    target_base_url: Optional[str] = None
    fallback_model_name: Optional[str] = None
    supported_endpoints: Optional[List[str]] = Field(default_factory=list)
    price_per_unit: float = Field(default=0.0)
    min_credits_per_request: float = Field(default=0.0)
    owned_by: Optional[str] = Field(default='FOAP')
    hide_on_models_endpoint: Optional[bool] = Field(default=False)

class ProviderModelCreate(ProviderModelBase):
    provider_id: str = Field(..., min_length=1)

class ProviderModelUpdate(ProviderModelBase):
    pass

class ProviderModelRead(ProviderModelBase):
    id: str
    provider_id: str

# Aliases
class ModelAliasBase(BaseModel):
    alias_name: str = Field(..., min_length=1)
    target_model_name: str = Field(..., min_length=1)
    owned_by: Optional[str] = Field(default='FOAP')
    hide_on_models_endpoint: Optional[bool] = Field(default=False)

class ModelAliasCreate(ModelAliasBase):
    pass

class ModelAliasUpdate(ModelAliasBase):
    pass

class ModelAliasRead(ModelAliasBase):
    id: str
    created_at: int
