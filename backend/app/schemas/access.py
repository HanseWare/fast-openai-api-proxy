from pydantic import BaseModel, Field, ConfigDict


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


class QuotaPolicy(BaseModel):
    requests_per_minute: int = Field(..., ge=1)


class QuotaPolicyRead(QuotaPolicy):
    api_key_id: str


class QuotaUsageRead(QuotaPolicyRead):
    used: int
    remaining: int
    reset_in_seconds: int


class ModelQuotaPolicyBase(BaseModel):
    api_path: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    window_type: str = Field(..., pattern="^(minute|hour|day)$")
    request_limit: int = Field(..., ge=1)
    enforce_per_user: bool = Field(default=True)


class ModelQuotaPolicyCreate(ModelQuotaPolicyBase):
    pass


class ModelQuotaPolicyUpdate(ModelQuotaPolicyBase):
    pass


class ModelQuotaPolicyRead(ModelQuotaPolicyBase):
    id: str


class QuotaOverrideBase(BaseModel):
    api_path: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    owner_id: str = Field(..., min_length=1)
    window_type: str = Field(..., pattern="^(minute|hour|day)$")
    request_limit: int = Field(..., ge=1)
    exempt: bool = Field(default=False)
    starts_at: int | None = Field(default=None, ge=0)
    ends_at: int | None = Field(default=None, ge=0)


class QuotaOverrideCreate(QuotaOverrideBase):
    pass


class QuotaOverrideUpdate(QuotaOverrideBase):
    pass


class QuotaOverrideRead(QuotaOverrideBase):
    id: str
    created_at: int
    active_now: bool
    window_state: str


class AuthModeSection(BaseModel):
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


class AuthModeSnapshot(BaseModel):
    admin: AuthModeSection
    self_service: AuthModeSection
    claim_mappings: AuthClaimMappings
    oidc_client: OidcClientConfig | None = None


