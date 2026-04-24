from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict

# Providers
class ProviderBase(BaseModel):
    name: str = Field(..., min_length=1)
    api_key_variable: Optional[str] = None
    prefix: str = Field(default="")
    default_base_url: Optional[str] = None
    default_request_timeout: Optional[int] = None
    default_health_timeout: Optional[int] = None
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

class ProviderModelCreate(ProviderModelBase):
    provider_id: str = Field(..., min_length=1)

class ProviderModelUpdate(ProviderModelBase):
    pass

class ProviderModelRead(ProviderModelBase):
    id: str
    provider_id: str

# Endpoints
class ProviderModelEndpointBase(BaseModel):
    path: str = Field(..., min_length=1)
    target_model_name: str = Field(..., min_length=1)
    target_base_url: Optional[str] = None
    request_timeout: Optional[int] = None
    health_timeout: Optional[int] = None
    fallback_model_name: Optional[str] = None

class ProviderModelEndpointCreate(ProviderModelEndpointBase):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str = Field(..., min_length=1)

class ProviderModelEndpointUpdate(ProviderModelEndpointBase):
    pass

class ProviderModelEndpointRead(ProviderModelEndpointBase):
    model_config = ConfigDict(protected_namespaces=())
    id: str
    model_id: str

# Aliases
class ModelAliasBase(BaseModel):
    alias_name: str = Field(..., min_length=1)
    target_model_name: str = Field(..., min_length=1)

class ModelAliasCreate(ModelAliasBase):
    pass

class ModelAliasUpdate(ModelAliasBase):
    pass

class ModelAliasRead(ModelAliasBase):
    id: str
    created_at: int
