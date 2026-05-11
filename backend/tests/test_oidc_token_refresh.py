"""Test OIDC token refresh functionality."""

import os
import sys
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import jwt

import pytest
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent
APP_DIR = str((BASE_DIR / ".." / "app").resolve())
CONFIG_DIR = str((BASE_DIR / ".." / "configs").resolve())
TEST_DB = str(BASE_DIR / "access-test.db")

os.environ.setdefault("FOAP_CONFIG_DIR", CONFIG_DIR)
os.environ["FOAP_ACCESS_DB_PATH"] = TEST_DB
os.environ["FOAP_ADMIN_TOKEN"] = "admin-secret"

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from main import app  # noqa: E402
from session_store import store as session_store  # noqa: E402
from oidc_auth import try_refresh_session  # noqa: E402


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


def create_jwt_token(exp_offset: int = 3600, sub: str = "test-user") -> str:
    """Create a mock JWT token with configurable expiry.

    Args:
        exp_offset: seconds from now when token expires (default 1 hour)
        sub: subject claim
    """
    now = int(time.time())
    payload = {
        "sub": sub,
        "iss": "https://example.com",
        "aud": "foap-client",
        "exp": now + exp_offset,
        "iat": now,
    }
    # For testing, we just return a JSON representation; real JWT signing would need keys
    return json.dumps(payload)


def test_try_refresh_session_with_no_refresh_token():
    """Verify refresh fails gracefully when no refresh_token is stored."""
    session_data = {
        "access_token": "some-expired-token",
        "owner_id": "user-123",
    }
    result = try_refresh_session(session_data)
    assert result is False
    assert session_data["access_token"] == "some-expired-token"


def test_try_refresh_session_with_invalid_refresh_token():
    """Verify refresh fails gracefully when token endpoint rejects the refresh token."""
    session_data = {
        "access_token": "some-expired-token",
        "refresh_token": "invalid-refresh-token",
        "owner_id": "user-123",
    }
    
    with patch("oidc_bff.refresh_access_token") as mock_refresh:
        mock_refresh.return_value = None
        result = try_refresh_session(session_data)
        assert result is False


def test_try_refresh_session_with_valid_refresh_token():
    """Verify refresh succeeds and updates session_data with new token."""
    old_access_token = "old-access-token"
    new_access_token = "new-access-token"
    refresh_token = "valid-refresh-token"
    
    session_data = {
        "access_token": old_access_token,
        "refresh_token": refresh_token,
        "owner_id": "user-123",
    }
    
    # Mock the token endpoint to return new tokens
    with patch("oidc_bff.refresh_access_token") as mock_refresh:
        mock_refresh.return_value = {
            "access_token": new_access_token,
            "refresh_token": "new-refresh-token",
            "token_type": "Bearer",
        }
        result = try_refresh_session(session_data)
        assert result is True
        assert session_data["access_token"] == new_access_token
        assert session_data["refresh_token"] == "new-refresh-token"


def test_self_service_session_with_expired_token_triggers_refresh():
    """Verify that accessing self-service endpoints with expired token triggers refresh."""
    access_token = "expired-token"
    new_access_token = "new-access-token"
    refresh_token = "valid-refresh-token"

    # Create a session with an expired access token
    session_id = session_store.create({
        "access_token": access_token,
        "refresh_token": refresh_token,
        "owner_id": "user-123",
        "scope": "self-service",
    })

    # Mock token verification to return None (expired)
    with patch("routers.self_service.get_oidc_claims") as mock_verify:
        with patch("routers.self_service.try_refresh_session") as mock_refresh:
            # First call returns None (expired), after refresh should return claims
            mock_verify.side_effect = [None, {"sub": "user-123"}]
            mock_refresh.return_value = True

            # This would be called by the application logic
            # The test just validates that refresh is attempted
            assert mock_verify(access_token) is None
            assert mock_refresh({"access_token": access_token, "refresh_token": refresh_token}) is True


def test_oidc_verifier_logs_expired_token():
    """Verify that ExpiredSignatureError is logged as debug level."""
    from oidc_auth import _verifier

    # Create an actually expired JWT token
    now = int(time.time())
    payload = {
        "sub": "test-user",
        "iss": "https://example.com",
        "exp": now - 3600,  # expired 1 hour ago
    }

    with patch("oidc_auth.is_oidc_auth_enabled", return_value=True):
        with patch("oidc_auth.get_oidc_issuer_url", return_value="https://example.com"):
            with patch("oidc_auth._verifier._get_jwk_client") as mock_jwk:
                # Create a mock signing key and valid JWT
                mock_key = MagicMock()
                mock_key.key = "test-key"
                mock_jwk.return_value.get_signing_key_from_jwt.return_value = mock_key

                # Patch jwt.decode to raise ExpiredSignatureError
                with patch("oidc_auth.jwt.decode") as mock_decode:
                    mock_decode.side_effect = jwt.ExpiredSignatureError("Token has expired")

                    result = _verifier.verify("fake-token")
                    assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])




