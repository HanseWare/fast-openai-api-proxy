import os


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


