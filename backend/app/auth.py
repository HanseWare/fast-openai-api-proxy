import hashlib
from typing import Optional

from access_store import store
from config import is_access_control_enabled, is_oidc_auth_enabled
from oidc_auth import get_oidc_claims, get_owner_id_from_claims, has_self_service_access


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


def get_oidc_owner_id(token: Optional[str]) -> Optional[str]:
    if not token or not is_oidc_auth_enabled():
        return None
    claims = get_oidc_claims(token)
    if claims is None or not has_self_service_access(claims):
        return None
    return get_owner_id_from_claims(claims)


def can_request(model, token):
    # Delegate request admission to middleware when access control is enabled.
    # This keeps unprotected endpoints open while protected/quota-scoped endpoints
    # are enforced centrally in AccessControlMiddleware.
    if not is_access_control_enabled():
        return True

    return True
