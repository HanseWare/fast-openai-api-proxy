import json
import logging
from typing import Any, Optional
from urllib.request import urlopen

import jwt
from jwt import PyJWKClient

logger = logging.getLogger(__name__)

from config import (
    get_oidc_admin_values,
    get_oidc_audience,
    get_oidc_group_claim,
    get_oidc_issuer_url,
    get_oidc_jwks_url,
    get_oidc_role_claim,
    get_oidc_self_service_values,
    get_oidc_subject_claim,
    is_oidc_auth_enabled,
)


def _normalize_claim_values(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {item for item in value if isinstance(item, str)}
    return set()


def _get_nested_claim(claims: dict[str, Any], claim_path: str) -> Any:
    if not claim_path:
        return None
    keys = claim_path.split('.')
    current = claims
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    return current


def _discover_jwks_url(issuer_url: str) -> str:
    well_known = issuer_url.rstrip("/") + "/.well-known/openid-configuration"
    with urlopen(well_known, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    jwks_uri = payload.get("jwks_uri")
    if not jwks_uri:
        raise ValueError("OIDC discovery response does not contain jwks_uri")
    return jwks_uri


class OIDCVerifier:
    def __init__(self):
        self._jwk_client: Optional[PyJWKClient] = None

    def _get_jwk_client(self) -> Optional[PyJWKClient]:
        if self._jwk_client is not None:
            return self._jwk_client

        jwks_url = get_oidc_jwks_url()
        if not jwks_url:
            issuer = get_oidc_issuer_url()
            if not issuer:
                raise ValueError("OIDC is enabled but FOAP_OIDC_ISSUER_URL is not configured")
            jwks_url = _discover_jwks_url(issuer)

        self._jwk_client = PyJWKClient(jwks_url)
        return self._jwk_client

    def verify(self, token: str) -> Optional[dict[str, Any]]:
        if not is_oidc_auth_enabled():
            return None

        issuer = get_oidc_issuer_url()
        if not issuer:
            return None

        try:
            signing_key = self._get_jwk_client().get_signing_key_from_jwt(token)
            options = {"verify_aud": get_oidc_audience() is not None}
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"],
                audience=get_oidc_audience(),
                issuer=issuer,
                options=options,
            )
            return claims
        except jwt.ExpiredSignatureError:
            logger.debug("OIDC access token has expired (exp claim validation failed)")
            return None
        except Exception as e:
            logger.debug(f"OIDC token verification failed: {type(e).__name__}: {e}")
            return None


_verifier = OIDCVerifier()


def try_refresh_session(session_data: dict[str, Any]) -> bool:
    """Attempt to refresh an expired session using the refresh token.
    
    Updates session_data in-place with a new access_token if successful.
    Returns True if refresh succeeded, False otherwise.
    """
    if not session_data:
        return False
    
    access_token = session_data.get("access_token")
    refresh_token = session_data.get("refresh_token")
    
    if not refresh_token:
        logger.debug("No refresh token available for session refresh")
        return False
    
    # Try to refresh
    from oidc_bff import refresh_access_token
    token_response = refresh_access_token(str(refresh_token))
    if not token_response or "access_token" not in token_response:
        logger.warning("Failed to refresh OIDC session: token endpoint returned invalid response")
        return False
    
    # Update session with new access token
    new_access_token = token_response.get("access_token")
    new_refresh_token = token_response.get("refresh_token", refresh_token)  # Some providers rotate, some don't
    
    session_data["access_token"] = new_access_token
    session_data["refresh_token"] = new_refresh_token
    logger.info("Successfully refreshed OIDC session")
    return True


def get_oidc_claims(token: Optional[str]) -> Optional[dict[str, Any]]:
    if not token:
        return None
    return _verifier.verify(token)


def _claim_values_for_mapping(claims: dict[str, Any]) -> set[str]:
    values = set()
    values |= _normalize_claim_values(_get_nested_claim(claims, get_oidc_role_claim()))
    values |= _normalize_claim_values(_get_nested_claim(claims, get_oidc_group_claim()))
    return values


def has_self_service_access(claims: dict[str, Any]) -> bool:
    """Return True if claims allow self-service access.

    Admin access implicitly grants self-service access to avoid role duplication for admins.
    """
    # Admins are allowed implicitly
    if has_admin_access(claims):
        return True

    required_values = set(get_oidc_self_service_values())
    if not required_values:
        return True
    return bool(_claim_values_for_mapping(claims) & required_values)


def has_admin_access(claims: dict[str, Any]) -> bool:
    required_values = set(get_oidc_admin_values())
    if not required_values:
        return False
    return bool(_claim_values_for_mapping(claims) & required_values)


def get_owner_id_from_claims(claims: dict[str, Any]) -> Optional[str]:
    subject_value = _get_nested_claim(claims, get_oidc_subject_claim())
    if not subject_value or not isinstance(subject_value, str):
        return None
    return f"oidc:{subject_value.strip()}"
