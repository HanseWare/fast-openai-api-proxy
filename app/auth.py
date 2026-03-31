import hashlib
from typing import Optional

from access_store import store
from config import is_access_control_enabled


def extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization or " " not in authorization:
        return None
    scheme, token = authorization.split(" ", 1)
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


def identity_from_token(token: str) -> str:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"user_{digest[:24]}"


def get_api_key_context(token: Optional[str]) -> Optional[dict]:
    if not token:
        return None
    return store.verify_api_key(token)


def can_request(model, token):
    # Keep current behavior by default; strict checks activate behind feature flag.
    if not is_access_control_enabled():
        return True

    return get_api_key_context(token) is not None

