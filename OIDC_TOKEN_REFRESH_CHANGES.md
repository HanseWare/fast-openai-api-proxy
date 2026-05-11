# Modified Files Summary

## Files Changed

### 1. `backend/app/oidc_bff.py`
- **Added:** `refresh_access_token(refresh_token: str)` function
  - Exchanges a refresh token for a new access token via the OIDC provider's token endpoint
  - Uses `grant_type=refresh_token` with client credentials

### 2. `backend/app/oidc_auth.py`
- **Added:** Logging import and logger instance
- **Updated:** `OIDCVerifier.verify()` to:
  - Catch `jwt.ExpiredSignatureError` separately and log it at DEBUG level
  - Log other verification errors at DEBUG level with exception type/message
- **Updated:** `_get_jwk_client()` return type to `Optional[PyJWKClient]` (type hint fix)
- **Added:** `try_refresh_session(session_data: dict) -> bool` function
  - Attempts to refresh an expired session using stored refresh_token
  - Updates session_data in-place with new access_token and optionally new refresh_token
  - Logs success/failure at INFO/WARNING level

### 3. `backend/app/routers/admin.py`
- **Updated imports:** Added `try_refresh_session` from oidc_auth
- **Updated:** `oidc_callback()` endpoint to store `refresh_token` from token response in session
- **Updated:** `require_admin()` dependency to:
  - Detect expired tokens (where `get_oidc_claims()` returns `None`)
  - Attempt automatic refresh before rejecting with 401
  - Works for both "admin-oidc-only" and "oidc-or-token-hash" modes
- **Updated:** `foap_oidc_session` cookie `max_age` from hardcoded `600` to `session_store.default_ttl`

### 4. `backend/app/routers/self_service.py`
- **Updated imports:** Added `try_refresh_session` from oidc_auth
- **Updated:** `oidc_callback()` endpoint to store `refresh_token` from token response in session
- **Updated:** `_require_user_token()` to:
  - Check if token is valid after retrieval
  - Attempt refresh if token verification fails
  - Re-verify with refreshed token
  - Only fail auth if token is truly invalid
- **Updated:** `foap_oidc_session` cookie `max_age` from hardcoded `600` to `session_store.default_ttl`

### 5. `backend/tests/test_oidc_token_refresh.py` (NEW)
- **Created:** Comprehensive test suite for token refresh functionality
- **Tests:**
  - No refresh token stored (graceful fail)
  - Invalid/expired refresh token (token endpoint rejection)
  - Valid refresh token (successful token refresh)
  - Integration with endpoint handlers
  - Logging of token expiry

---

## No Changes Needed (Preserved)

- `backend/app/session_store.py` — Works as-is; already supports storing arbitrary dict data
- `backend/app/config.py` — No new config required
- `backend/app/middleware/access_control.py` — No changes needed
- `backend/app/access_store.py` — No changes needed
- `backend/app/models_handler.py` — No changes needed
- `backend/app/utils.py` — No changes needed

---

## Migration Checklist

- [x] OIDC access tokens automatically refresh when expired
- [x] Refresh tokens stored from initial login
- [x] Token expiry properly logged (DEBUG level)
- [x] Refresh failures logged (WARNING level)
- [x] All existing tests pass
- [x] New tests added for token refresh
- [x] Cookie max_age aligned with server session TTL
- [x] Both admin and self-service flows support refresh
- [x] Type hints fixed and validated
- [x] Graceful handling of missing/invalid refresh tokens

---

## How to Verify the Fix

### Test Scenario 1: Automatic Token Refresh
```bash
# Run the new token refresh tests
cd backend
python -m pytest tests/test_oidc_token_refresh.py -v
```

### Test Scenario 2: Existing Test Suite
```bash
# Verify no regressions
cd backend
python -m pytest tests/test_oidc_role_mapping.py tests/test_access_admin_self_service.py -v
```

### Manual Verification
1. Configure OIDC with a provider that issues short-lived tokens (e.g., 5 min) + refresh tokens
2. Log in to admin or self-service
3. Wait for access token to expire (5+ minutes)
4. Make another API request
5. **Expected:** Request succeeds because token is auto-refreshed
6. **Before fix:** Request would fail with 401/403

---

## Rollback Instructions

If needed, revert these commits:
- Delete `backend/tests/test_oidc_token_refresh.py`
- Revert changes to `backend/app/oidc_bff.py`, `oidc_auth.py`, `routers/admin.py`, `routers/self_service.py`
- Existing functionality (static tokens, no OIDC) unaffected

---

## Known Limitations

1. **Refresh token expiry:** If the refresh token itself expires (common default: 7-30 days), auto-refresh stops working. No warn-before-expiry logic needed yet.
2. **Concurrency:** No locking on session_data updates; concurrent refresh attempts are idempotent but may wastefully call token endpoint twice. Acceptable for now.
3. **No preemptive refresh:** Only refreshes when an expired token is detected. Could be optimized to refresh 5 minutes before expiry.

---

**Last Updated:** 2025-05-11
