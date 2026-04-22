from pydantic import BaseModel, Field
from typing import Optional, List

# Providers
class ProviderBase(BaseModel):
    name: str = Field(..., min_length=1)
    api_key_variable: str = Field(..., min_length=1)
    prefix: str = Field(default="")
    default_base_url: Optional[str] = None
    default_request_timeout: Optional[int] = None
    default_health_timeout: Optional[int] = None

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

class ProviderModelEndpointCreate(ProviderModelEndpointBase):
    model_id: str = Field(..., min_length=1)

class ProviderModelEndpointRead(ProviderModelEndpointBase):
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
