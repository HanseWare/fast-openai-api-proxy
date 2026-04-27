# OIDC Authentication Fix - Backend-for-Frontend (BFF) Pattern

## Problem Analysis

The OIDC authentication was failing with `error="invalid_client_credentials"` because:

1. **Frontend was attempting client credentials exchange from JavaScript**: The original implementation used `oidc-client-ts` library to handle the complete OIDC flow, including the authorization code-to-token exchange.

2. **Confidential Client Mismatch**: When the Keycloak client is configured as "Confidential" (which requires a `client_secret`), the authorization code exchange MUST include valid credentials.

3. **Security Issue**: JavaScript running in the browser cannot securely store and transmit a `client_secret` - it would be exposed in the source code/network traffic.

4. **Root Cause**: The Keycloak logs showed `error="invalid_client_credentials"` because the frontend's token exchange request couldn't include the secret.

**Log Evidence:**
```
type="CODE_TO_TOKEN_ERROR", clientId="mylab-foap", error="invalid_client_credentials", grant_type="authorization_code"
```

## Solution: Backend-for-Frontend (BFF) Pattern

Migrated to a secure BFF pattern where:
- The **backend** securely holds and uses the `client_secret`
- The **frontend** initiates login but lets the backend handle sensitive exchanges
- The **session cookie** becomes the source of authentication (not bearer tokens from browser)

### New Flow

```
1. User clicks "Sign in with SSO"
   ↓
2. Frontend calls → /api/admin/oidc/login (or /api/oidc/login for self-service)
   ↓
3. Backend generates PKCE state/verifier, stores in session, returns authorization_uri
   ↓
4. Frontend redirects browser → Keycloak authorization endpoint
   ↓
5. User authenticates at Keycloak
   ↓
6. Keycloak redirects → /api/admin/oidc/callback (or /api/oidc/callback)
   ↓
7. Backend securely exchanges authorization code for access token (with client_secret)
   ↓
8. Backend validates token claims (admin/self-service access)
   ↓
9. Backend creates session, sets foap_session cookie, redirects to dashboard
   ↓
10. User is now authenticated via secure session cookie
```

## Changes Made

### 1. **frontend/src/services/oidc.js** - Complete Rewrite
**Before:** Used `oidc-client-ts` UserManager for full OIDC handling  
**After:** Simple BFF orchestration

```javascript
export async function startOidcLogin(apiBasePath = '/api/admin', target = 'admin') {
  // Call backend BFF endpoint
  const response = await fetch(`${apiBasePath}/oidc/login`)
  const { authorization_uri } = await response.json()
  // Browser redirects to Keycloak (not handled by frontend)
  window.location.href = authorization_uri
}
```

**Key Changes:**
- Removed `UserManager` from `oidc-client-ts`
- Removed `getOidcManager()` function
- Removed `completeOidcLogin()` function (no longer needed!)
- Replaced with simple HTTP call to backend BFF endpoint
- Backend handles all PKCE, state, code exchange

### 2. **frontend/src/views/LoginView.vue** - Updated Admin Login
**Changed:** `handleSsoLogin()` function

```javascript
// Old (broken):
await startOidcLogin(oidcClient.value, 'admin')

// New (BFF pattern):
await startOidcLogin('/api/admin', 'admin')
```

**Why:** Pass API base path directly instead of OIDC client config - backend provides auth URI.

### 3. **frontend/src/views/AccountView.vue** - Updated Self-Service Login
**Changed:** `handleSsoLogin()` function

```javascript
// Old (broken):
await startOidcLogin(oidcClient.value, 'account')

// New (BFF pattern):
await startOidcLogin('/api', 'account')
```

**Why:** Use self-service API base path for self-service OIDC endpoint.

### 4. **frontend/src/views/OidcCallbackView.vue** - Simplified Callback Handler
**Changed:** Removed `oidc-client-ts` token exchange logic

**Before:**
```javascript
const user = await mgr.signinCallback()  // ← This was failing!
return user?.access_token || null
```

**After:**
```javascript
// Backend has already handled callback and set foap_session cookie
// Just verify we have a valid session
await fetchApi('/health')  // Validates the session cookie
router.replace({ name: 'dashboard' })
```

**Note:** This view is now a fallback/loading page. In the normal BFF flow:
- Backend handles the callback redirect from Keycloak
- Backend sets `foap_session` cookie
- Backend redirects to "/" or "/account"
- This view might never be reached (unless callback URL is accessed directly)

### 5. **frontend/package.json** - Removed Unused Dependency
**Removed:** `"oidc-client-ts": "^3.5.0"`

Since we're not using the library anymore, removed from dependencies.

## Backend Endpoints (Already Exist)

The backend already has proper BFF implementation in:

### Admin OIDC Flow
- `GET /api/admin/oidc/login` → Returns `{"authorization_uri": "..."}`
- `GET /api/admin/oidc/callback` → Handles callback, sets `foap_session` cookie, redirects to "/"

### Self-Service OIDC Flow
- `GET /api/oidc/login` → Returns `{"authorization_uri": "..."}`
- `GET /api/oidc/callback` → Handles callback, sets `foap_session` cookie, redirects to "/account"

**Location:** `backend/app/routers/admin.py` and `backend/app/routers/self_service.py`

## Environment Variables Needed

Ensure your backend has OIDC BFF configured:

```bash
# Enable OIDC
FOAP_ENABLE_OIDC_AUTH=true

# Keycloak realm configuration
FOAP_OIDC_ISSUER_URL=https://keycloak.example.com/realms/mylab
FOAP_OIDC_CLIENT_ID=mylab-foap
FOAP_OIDC_CLIENT_SECRET=your-client-secret-here  # ← BACKEND HOLDS THE SECRET

# Admin access control
FOAP_OIDC_ADMIN_VALUES=foap-admin
FOAP_ADMIN_OIDC_ONLY=true  # Optional: require OIDC for admin access

# Claim mappings (optional, with defaults)
FOAP_OIDC_ROLE_CLAIM=roles
FOAP_OIDC_GROUP_CLAIM=groups
FOAP_OIDC_SUBJECT_CLAIM=sub
```

## Keycloak Configuration

Ensure your Keycloak `mylab-foap` client is configured correctly:

### Client Settings
- **Client ID:** `mylab-foap` ✓
- **Client Type:** Confidential ✓
- **Authentication:** Client ID and Secret ✓
- **Authorization Code Flow:** Enabled ✓

### Credentials Tab
- **Client Authenticator:** `Client ID and Secret` ✓
- **Client Secret:** `your-client-secret-here` (copy this to `FOAP_OIDC_CLIENT_SECRET`)
- Mark as "Confidential" to require secret

### Advanced Tab
- **PKCE Code Challenge Method:** S256 ✓ (Backend uses this)

### Valid Redirect URIs
Add these exact URIs:
```
https://ai-api.mylab.th-luebeck.de/api/admin/oidc/callback
https://ai-api.mylab.th-luebeck.de/api/oidc/callback
```

### Verify Keycloak Configuration
Test the configuration by visiting:
```
https://keycloak.example.com/realms/mylab/.well-known/openid-configuration
```

Should contain:
- `token_endpoint` - where backend exchanges code for token
- `authorization_endpoint` - where user authenticates
- `jwks_uri` - for token signature validation

## Testing the Fix

### 1. Start Fresh
```bash
# Clear browser cache/storage
# Close all tabs to session

# Backend running with OIDC env vars:
FOAP_ENABLE_OIDC_AUTH=true
FOAP_OIDC_ISSUER_URL=https://keycloak.example.com/realms/mylab
FOAP_OIDC_CLIENT_ID=mylab-foap
FOAP_OIDC_CLIENT_SECRET=<your-secret>
```

### 2. Try Admin Login
- Navigate to `https://ai-api.mylab.th-luebeck.de/#/admin/login`
- Click "Sign in with SSO"
- Should redirect to Keycloak login
- After authentication, should redirect back to dashboard
- Should NOT see any 401/invalid_client_credentials errors

### 3. Try Self-Service Login
- Navigate to `https://ai-api.mylab.th-luebeck.de/#/account`
- Click "Sign in with SSO"
- Should redirect to Keycloak login
- After authentication, should redirect back to account view

### 4. Verify Session
- After successful login, check browser cookies
- Should have `foap_session` cookie (httpOnly, secure, samesite=lax)
- This cookie authenticates subsequent API calls

### 5. Check Logs
```bash
# Backend should NOT show:
# - invalid_client_credentials errors
# - CODE_TO_TOKEN_ERROR in Keycloak logs

# Backend SHOULD show successful token exchange and session creation
```

## Network Trace Expectations

### Before Fix (Broken)
```
POST /realms/mylab/protocol/openid-connect/token
← 401 Unauthorized
← {"error":"unauthorized_client","error_description":"Invalid client or Invalid client credentials"}
```

**Why:** Frontend JavaScript trying to send incomplete/unsigned credentials.

### After Fix (Working)
```
GET /api/admin/oidc/login
← 200 OK {"authorization_uri": "https://keycloak..."}

[Browser redirects to Keycloak authorization endpoint]

[User authenticates at Keycloak]

[Keycloak redirects back to /api/admin/oidc/callback with ?code=... &state=...]

POST /realms/mylab/protocol/openid-connect/token
  Body: grant_type=authorization_code
         code=<auth_code>
         client_id=mylab-foap
         client_secret=<valid_secret>  ← Backend sends this securely!
         redirect_uri=/api/admin/oidc/callback
         code_verifier=<pkce_verifier>
← 200 OK {"access_token": "...", "token_type": "Bearer", ...}

[Backend validates token claims]

[Backend creates session, sets foap_session cookie]

[Backend redirects to /]

[Frontend has authenticated session]
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Completing sign-in…" hangs | Backend callback failed silently | Check backend logs for token exchange errors |
| 401 from health check after callback | Token rejected/no admin claims | Verify Keycloak token contains roles claim |
| "Sign-in failed" error immediately | Backend `/oidc/login` endpoint errors | Ensure `FOAP_OIDC_CLIENT_SECRET` is set |
| Redirect loop | Session cookie not being set | Check backend session store, browser cookie settings |
| Network error in console | CORS issue or network blocked | Ensure Keycloak is accessible, check CORS headers |

## Files Changed

```
frontend/
├── src/
│   ├── services/oidc.js                    ← Complete rewrite (BFF pattern)
│   └── views/
│       ├── LoginView.vue                   ← Updated handleSsoLogin()
│       ├── AccountView.vue                 ← Updated handleSsoLogin()
│       └── OidcCallbackView.vue            ← Simplified, removed oidc-client-ts
└── package.json                             ← Removed oidc-client-ts dependency

backend/
├── app/
│   ├── oidc_bff.py                        ← No changes (already correct)
│   └── routers/
│       ├── admin.py                        ← No changes (already correct)
│       └── self_service.py                 ← No changes (already correct)
```

## Security Notes

✓ **Client secret never exposed to frontend**  
✓ **PKCE enabled for authorization code exchange**  
✓ **State validation prevents CSRF attacks**  
✓ **Session cookie is httpOnly, secure, samesite=lax**  
✓ **Token subject claim extracted for user identification**  
✓ **Admin/self-service claims validated on backend**  

## Next Steps

1. **Deploy updated frontend**
2. **Verify backend has OIDC environment variables set**
3. **Test login flow end-to-end**
4. **Update documentation if needed**
5. (Optional) Monitor Keycloak logs for successful token exchanges

