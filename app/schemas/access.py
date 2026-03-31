from pydantic import BaseModel, Field


class ApiKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)


class AdminApiKeyCreate(ApiKeyCreate):
    owner_id: str | None = Field(default=None, min_length=1, max_length=128)


class ApiKeyRead(BaseModel):
    id: str
    name: str
    owner_id: str | None = None
    masked_key: str


class ApiKeyCreateResponse(ApiKeyRead):
    api_key: str


class ProtectedEndpointRule(BaseModel):
    path: str = Field(..., min_length=1)
    method: str = Field(..., min_length=3, max_length=10)


class ProtectedEndpointRuleRead(ProtectedEndpointRule):
    id: str


class QuotaPolicy(BaseModel):
    requests_per_minute: int = Field(..., ge=1)


class QuotaPolicyRead(QuotaPolicy):
    api_key_id: str


class QuotaUsageRead(QuotaPolicyRead):
    used: int
    remaining: int
    reset_in_seconds: int


