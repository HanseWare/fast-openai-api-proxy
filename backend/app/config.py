import os


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def is_admin_api_enabled() -> bool:
    return _env_bool("FOAP_ENABLE_ADMIN_API", False)


def is_self_service_api_enabled() -> bool:
    return _env_bool("FOAP_ENABLE_SELF_SERVICE_API", False)


def is_access_control_enabled() -> bool:
    return _env_bool("FOAP_ENABLE_ACCESS_CONTROL", False)


def is_quota_decision_trace_enabled() -> bool:
    return _env_bool("FOAP_ENABLE_QUOTA_DECISION_TRACE", False)


def get_admin_token() -> str | None:
    token = os.getenv("FOAP_ADMIN_TOKEN")
    if token is None:
        return None
    token = token.strip()
    return token or None


def get_access_db_path() -> str:
    return os.getenv("FOAP_ACCESS_DB_PATH", "data/access.db")


def is_oidc_auth_enabled() -> bool:
    return _env_bool("FOAP_ENABLE_OIDC_AUTH", False)


def is_admin_oidc_only_enabled() -> bool:
    return _env_bool("FOAP_ADMIN_OIDC_ONLY", False)


def is_self_service_oidc_only_enabled() -> bool:
    return _env_bool("FOAP_SELF_SERVICE_OIDC_ONLY", False)


def get_oidc_issuer_url() -> str | None:
    value = os.getenv("FOAP_OIDC_ISSUER_URL")
    if value is None:
        return None
    value = value.strip()
    return value or None


def get_oidc_jwks_url() -> str | None:
    value = os.getenv("FOAP_OIDC_JWKS_URL")
    if value is None:
        return None
    value = value.strip()
    return value or None


def get_oidc_audience() -> str | None:
    value = os.getenv("FOAP_OIDC_AUDIENCE")
    if value is None:
        return None
    value = value.strip()
    return value or None


def get_oidc_role_claim() -> str:
    return os.getenv("FOAP_OIDC_ROLE_CLAIM", "roles")


def get_oidc_group_claim() -> str:
    return os.getenv("FOAP_OIDC_GROUP_CLAIM", "groups")


def get_oidc_subject_claim() -> str:
    return os.getenv("FOAP_OIDC_SUBJECT_CLAIM", "sub")


def get_oidc_admin_values() -> list[str]:
    return _env_csv("FOAP_OIDC_ADMIN_VALUES", "foap-admin")


def get_oidc_self_service_values() -> list[str]:
    return _env_csv("FOAP_OIDC_SELF_SERVICE_VALUES", "")


def get_oidc_client_id() -> str | None:
    value = os.getenv("FOAP_OIDC_CLIENT_ID")
    if value is None:
        return None
    value = value.strip()
    return value or None


def get_oidc_client_secret() -> str | None:
    value = os.getenv("FOAP_OIDC_CLIENT_SECRET")
    if value is None:
        return None
    value = value.strip()
    return value or None


def get_oidc_provider_display_name() -> str | None:
    value = os.getenv("FOAP_OIDC_PROVIDER_DISPLAY_NAME")
    if value is None:
        return None
    value = value.strip()
    return value or None


def get_auth_configuration_errors() -> list[str]:
    errors: list[str] = []

    oidc_enabled = is_oidc_auth_enabled()
    admin_oidc_only = is_admin_oidc_only_enabled()
    self_service_oidc_only = is_self_service_oidc_only_enabled()

    if admin_oidc_only and not oidc_enabled:
        errors.append("FOAP_ADMIN_OIDC_ONLY requires FOAP_ENABLE_OIDC_AUTH=true.")
    if self_service_oidc_only and not oidc_enabled:
        errors.append("FOAP_SELF_SERVICE_OIDC_ONLY requires FOAP_ENABLE_OIDC_AUTH=true.")

    if oidc_enabled:
        if not get_oidc_issuer_url():
            errors.append("FOAP_ENABLE_OIDC_AUTH=true requires FOAP_OIDC_ISSUER_URL.")
        if not get_oidc_role_claim().strip():
            errors.append("FOAP_OIDC_ROLE_CLAIM must not be empty.")
        if not get_oidc_group_claim().strip():
            errors.append("FOAP_OIDC_GROUP_CLAIM must not be empty.")
        if not get_oidc_subject_claim().strip():
            errors.append("FOAP_OIDC_SUBJECT_CLAIM must not be empty.")

    if is_admin_api_enabled() and not get_admin_token() and not oidc_enabled:
        errors.append("Admin API enabled but neither FOAP_ADMIN_TOKEN nor OIDC auth is configured.")

    return errors


def _admin_auth_mode() -> str:
    if is_admin_oidc_only_enabled():
        return "oidc-only"
    if get_admin_token() and is_oidc_auth_enabled():
        return "hybrid"
    if get_admin_token():
        return "static-token-only"
    if is_oidc_auth_enabled():
        return "oidc-only"
    return "disabled"


def _self_service_auth_mode() -> str:
    if is_self_service_oidc_only_enabled():
        return "oidc-only"
    if is_oidc_auth_enabled():
        return "oidc-or-token-hash"
    return "token-hash-only"


def get_auth_mode_snapshot() -> dict:
    oidc_client = None
    if is_oidc_auth_enabled() and get_oidc_client_id() and get_oidc_issuer_url():
        oidc_client = {
            "client_id": get_oidc_client_id(),
            "authority": get_oidc_issuer_url(),
        }
        display_name = get_oidc_provider_display_name()
        if display_name:
            oidc_client["display_name"] = display_name

    return {
        "admin": {
            "enabled": is_admin_api_enabled(),
            "mode": _admin_auth_mode(),
            "oidc_enabled": is_oidc_auth_enabled(),
            "oidc_only": is_admin_oidc_only_enabled(),
            "static_token_enabled": bool(get_admin_token()),
        },
        "self_service": {
            "enabled": is_self_service_api_enabled(),
            "mode": _self_service_auth_mode(),
            "oidc_enabled": is_oidc_auth_enabled(),
            "oidc_only": is_self_service_oidc_only_enabled(),
        },
        "claim_mappings": {
            "role_claim": get_oidc_role_claim(),
            "group_claim": get_oidc_group_claim(),
            "subject_claim": get_oidc_subject_claim(),
            "admin_values": get_oidc_admin_values(),
            "self_service_values": get_oidc_self_service_values(),
        },
        "oidc_client": oidc_client,
    }
