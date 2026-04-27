"""OIDC Backend-for-Frontend (BFF) flow handler with client_secret."""

import json
from typing import Optional
from urllib.request import urlopen, Request
from urllib.parse import urlencode
import secrets

from fastapi import Request as FastAPIRequest

from config import (
    get_oidc_issuer_url,
    get_oidc_client_id,
    get_oidc_client_secret,
)


def get_base_url_from_request(request: FastAPIRequest) -> str:
    """Extract base URL from request, respecting X-Forwarded-Proto header for reverse proxies."""
    # Check for X-Forwarded-Proto header (set by reverse proxy like Nginx)
    proto = request.headers.get("X-Forwarded-Proto")
    if not proto:
        proto = request.url.scheme

    # Check for X-Forwarded-Host header
    host = request.headers.get("X-Forwarded-Host")
    if not host:
        host = request.headers.get("Host")
    if not host:
        host = request.url.hostname
        if request.url.port:
            host = f"{host}:{request.url.port}"

    return f"{proto}://{host}"


def _get_token_endpoint(issuer_url: str) -> str:
    """Discover token endpoint from OIDC issuer."""
    well_known = issuer_url.rstrip("/") + "/.well-known/openid-configuration"
    with urlopen(well_known, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    token_endpoint = payload.get("token_endpoint")
    if not token_endpoint:
        raise ValueError("OIDC discovery response does not contain token_endpoint")
    return token_endpoint


def _get_authorization_endpoint(issuer_url: str) -> str:
    """Discover authorization endpoint from OIDC issuer."""
    well_known = issuer_url.rstrip("/") + "/.well-known/openid-configuration"
    with urlopen(well_known, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    auth_endpoint = payload.get("authorization_endpoint")
    if not auth_endpoint:
        raise ValueError("OIDC discovery response does not contain authorization_endpoint")
    return auth_endpoint


def generate_auth_state() -> tuple[str, str]:
    """Generate PKCE state and code_verifier for authorization request.

    Returns (state, code_verifier)
    """
    state = secrets.token_urlsafe(32)
    code_verifier = secrets.token_urlsafe(32)
    return state, code_verifier


def build_authorization_uri(
    redirect_uri: str,
    state: str,
    code_verifier: str,
) -> str:
    """Build the OIDC authorization URI for the user to visit.

    Args:
        redirect_uri: Where to redirect after login (e.g., https://foap.example.com/api/admin/oidc/callback)
        state: PKCE state from generate_auth_state()
        code_verifier: PKCE code verifier from generate_auth_state()

    Returns:
        Full authorization URI.
    """
    issuer_url = get_oidc_issuer_url()
    client_id = get_oidc_client_id()

    if not issuer_url or not client_id:
        raise ValueError("OIDC not properly configured")

    auth_endpoint = _get_authorization_endpoint(issuer_url)

    # Simple PKCE: code_challenge = code_verifier (for confidential clients, we can skip the hash)
    # But we'll use it properly: code_challenge should be base64url(sha256(code_verifier))
    import hashlib
    import base64
    code_sha = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = base64.urlsafe_b64encode(code_sha).decode().rstrip("=")

    params = {
        "client_id": client_id,
        "response_type": "code",
        "scope": "openid profile email",
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }

    return f"{auth_endpoint}?{urlencode(params)}"


def exchange_code_for_token(
    code: str,
    redirect_uri: str,
    code_verifier: str,
) -> Optional[dict]:
    """Exchange authorization code for access token using client_secret.

    Args:
        code: Authorization code from callback
        redirect_uri: Must match the one sent in authorization request
        code_verifier: PKCE code verifier from generate_auth_state()

    Returns:
        Token response dict with 'access_token', 'token_type', etc., or None on error.
    """
    issuer_url = get_oidc_issuer_url()
    client_id = get_oidc_client_id()
    client_secret = get_oidc_client_secret()

    if not issuer_url or not client_id or not client_secret:
        return None

    token_endpoint = _get_token_endpoint(issuer_url)

    body = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }

    try:
        req = Request(
            token_endpoint,
            data=urlencode(body).encode(),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urlopen(req, timeout=5) as response:
            token_response = json.loads(response.read().decode("utf-8"))
        return token_response
    except Exception as e:
        return None


def validate_state(stored_state: str, provided_state: str) -> bool:
    """Validate that the state matches (CSRF protection)."""
    return stored_state == provided_state

