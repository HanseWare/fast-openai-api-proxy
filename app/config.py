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


