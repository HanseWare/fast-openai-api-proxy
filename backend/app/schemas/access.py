from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

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
    model_config = ConfigDict(protected_namespaces=())
    path: str = Field(..., min_length=1)
    method: str = Field(..., min_length=1, max_length=10)
    model_pattern: str = Field(default='*')

class ProtectedEndpointRuleRead(ProtectedEndpointRule):
    id: str

# Budgets

class BudgetBase(BaseModel):
    entity_type: str = Field(..., pattern="^(user|group)$")
    entity_id: str = Field(..., min_length=1)
    scope: Optional[str] = None
    window: str = Field(..., pattern="^(daily|monthly)$")
    budget_amount: float = Field(..., ge=0)

class BudgetCreate(BudgetBase):
    pass

class BudgetUpdate(BaseModel):
    budget_amount: float = Field(..., ge=0)

class BudgetRead(BudgetBase):
    id: str
    created_at: int

class BudgetUsageRead(BaseModel):
    entity_type: str
    entity_id: str
    scope: Optional[str] = None
    window: str
    window_bucket: str
    cost: float

class RequestLogRead(BaseModel):
    id: str
    api_key_id: Optional[str] = None
    timestamp: int
    requested_model: Optional[str] = None
    target_model_name: Optional[str] = None
    provider: Optional[str] = None
    scope: Optional[str] = None
    usage: Optional[float] = None
    usage_unit: Optional[str] = None
    price: Optional[float] = None
    price_per_unit: Optional[float] = None
    cost: Optional[float] = None

class AuthModeSection(BaseModel):
    enabled: bool
    mode: str
    oidc_enabled: bool
    oidc_only: bool
    static_token_enabled: bool | None = None

class AuthClaimMappings(BaseModel):
    role_claim: str
    group_claim: str
    subject_claim: str
    admin_values: list[str]
    self_service_values: list[str]

class OidcClientConfig(BaseModel):
    client_id: str
    authority: str
    display_name: str | None = None

class AuthModeSnapshot(BaseModel):
    admin: AuthModeSection
    self_service: AuthModeSection
    claim_mappings: AuthClaimMappings
    oidc_client: OidcClientConfig | None = None

