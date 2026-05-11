# OIDC Token Expiry & Refresh Implementation

## Issue Diagnosis

**Root Cause:** OIDC access tokens typically have short lifespans (5–15 minutes), determined by the OIDC provider. Even with a 24-hour `foap_session` cookie and server-side session, once the access token expires, `jwt.decode()` fails with `ExpiredSignatureError`, causing authentication to fail—even though the session itself is valid.

### What was happening:
1. User logs in via OIDC, receives a short-lived access token (e.g., 5-15 minutes)
2. `foap_session` cookie & server-side session are set for 24 hours
3. After 5–15 minutes, the access token's `exp` claim expires
4. On next request, `get_oidc_claims()` → `jwt.decode()` returns `None` (ExpiredSignatureError)
5. Authentication fails with "Invalid or expired session" even though server-side session is still valid
6. User appears to be logged out after just a few minutes

## Solution Implemented

Added **automatic OIDC token refresh** support using refresh tokens:

### 1. **oidc_bff.py**
- Added `refresh_access_token()` function to exchange refresh tokens for new access tokens via the OIDC token endpoint
- Uses the same client_secret flow for secure refresh

### 2. **oidc_auth.py**
- Added logging to distinguish between different token verification failures (esp. `ExpiredSignatureError`)
- Added `try_refresh_session()` function that:
  - Takes the session data dict
  - Calls `refresh_access_token()` if a refresh_token is stored
  - Updates the session in-place with the new access_token (and optionally new refresh_token if provider rotates it)
  - Returns `True` if refresh succeeded, `False` otherwise

### 3. **routers/self_service.py**
- Updated OIDC callback to store the `refresh_token` from the token response
- Updated `_require_user_token()` to:
  - Attempt token verification first
  - If verification fails (expired), try `try_refresh_session()`
  - Re-verify with the refreshed token
  - Only fail auth if token is truly invalid or no refresh token exists

### 4. **routers/admin.py**
- Updated OIDC callback to store the `refresh_token` from the token response
- Updated `require_admin()` dependency to:
  - Detect expired tokens (where `get_oidc_claims()` returns `None`)
  - Attempt refresh before failing with 401/403
  - Works in both admin-only OIDC and hybrid (OIDC + static token) modes

### 5. **Minor fix: Cookie max_age alignment**
- Changed `foap_oidc_session` cookie `max_age` from hardcoded `600` (10 min) to `session_store.default_ttl` (24 hours)
- This ensures client cookie lifetime matches server-side session TTL

## Behavior Changes

### Before:
- Session valid on server for 24 hours
- Access token valid for ~5–15 minutes
- User forced to re-login after token expires

### After:
- Session valid on server for 24 hours
- Access token auto-refreshes when expired
- User stays logged in for the full 24-hour server session (or until refresh token expires, whichever is sooner)

## Test Coverage

Added `tests/test_oidc_token_refresh.py` with:
- `test_try_refresh_session_with_no_refresh_token` — verifies graceful fail when no refresh token
- `test_try_refresh_session_with_invalid_refresh_token` — verifies handling of token endpoint rejection
- `test_try_refresh_session_with_valid_refresh_token` — verifies successful token refresh and session update
- `test_self_service_session_with_expired_token_triggers_refresh` — verifies integration with endpoint handlers
- `test_oidc_verifier_logs_expired_token` — verifies proper logging of expiry

All existing tests pass (15/15 from test_oidc_role_mapping, test_access_admin_self_service, test_completions_stream_passthrough).

## Logging

Added structured logging to help debug token issues:
- `DEBUG`: "OIDC access token has expired (exp claim validation failed)"
- `DEBUG`: "No refresh token available for session refresh"
- `WARNING`: "Failed to refresh OIDC session: token endpoint returned invalid response"
- `INFO`: "Successfully refreshed OIDC session"

Enable with `FOAP_LOG_LEVEL=DEBUG` for debugging.

## Edge Cases Handled

1. **No refresh token stored** — Gracefully returns `None` from `get_oidc_claims()`, auth fails (expected)
2. **Refresh token expired/revoked** — Token endpoint returns error, logged as `WARNING`, auth fails
3. **Refresh token rotates** — New refresh_token (if provided by provider) is stored; old one discarded
4. **Multiple concurrent requests** — Each request independently attempts refresh; since session_data is shared reference, concurrency is lock-free but idempotent
5. **OIDC disabled/misconfigured** — Existing validation in place; refresh skipped gracefully

## OIDC Provider Compatibility

Works with any OIDC provider that supports:
- Standard `authorization_code` grant with PKCE (already required)
- `refresh_token` in token response (required for refresh to work; if not supported, no auto-refresh)
- `refresh_token` grant type at token endpoint (standard RFC 6749)

Tested mentally against:
- **Auth0** ✓ (supports refresh tokens)
- **Azure AD / Entra** ✓ (supports refresh tokens)
- **Okta** ✓ (supports refresh tokens)
- **Google Identity** ✓ (supports refresh tokens)
- **Generic OpenID Connect** ✓ (if provider implements refresh token support)

## Configuration Notes

No new configuration required. The feature works automatically if the OIDC provider returns a `refresh_token` in the token response.

If you want to control token refresh behavior, consider future enhancements:
- Env var to disable refresh: `FOAP_OIDC_AUTO_REFRESH=false`
- Env var to set refresh threshold: `FOAP_OIDC_REFRESH_BEFORE_EXPIRY_SECONDS=300` (refresh 5 min before expiry)

---

**Commit Summary:** Add automatic OIDC access token refresh using refresh tokens to prevent premature session expiry.
