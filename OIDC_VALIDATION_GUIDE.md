# OIDC Authentication Fix - Quick Validation Guide

## ✅ Changes Summary

### Files Modified:
1. ✅ `frontend/src/services/oidc.js` - Complete rewrite (BFF pattern)
2. ✅ `frontend/src/views/LoginView.vue` - Updated `handleSsoLogin()`
3. ✅ `frontend/src/views/AccountView.vue` - Updated `handleSsoLogin()`
4. ✅ `frontend/src/views/OidcCallbackView.vue` - Simplified callback handler
5. ✅ `frontend/package.json` - Removed `oidc-client-ts` dependency

## 🔍 Key Changes at a Glance

| File | Change | Why |
|------|--------|-----|
| `oidc.js` | Removed `UserManager`, replaced with backend BFF calls | Client-side token exchange was failing; backend must handle with client_secret |
| `LoginView.vue` | `startOidcLogin('/api/admin', 'admin')` instead of passing oidcClient | Pass API path; backend provides auth URI |
| `AccountView.vue` | `startOidcLogin('/api', 'account')` instead of passing oidcClient | Use self-service endpoint for account portal |
| `OidcCallbackView.vue` | Removed token exchange logic; now just validates session | Backend handles callback; frontend just verifies session |
| `package.json` | Removed `oidc-client-ts` | No longer needed; backend handles OIDC |

## 🧪 Testing Checklist

### Prerequisites
- [ ] Backend has OIDC environment variables set correctly:
  ```bash
  FOAP_ENABLE_OIDC_AUTH=true
  FOAP_OIDC_ISSUER_URL=https://keycloak.../realms/mylab
  FOAP_OIDC_CLIENT_ID=mylab-foap
  FOAP_OIDC_CLIENT_SECRET=<your-secret>
  FOAP_OIDC_ADMIN_VALUES=foap-admin
  ```
- [ ] Keycloak client `mylab-foap` is configured as:
  - Client Type: Confidential
  - Valid Redirect URIs include both callback endpoints
  - Client has valid secret in Credentials tab
- [ ] Frontend dependencies updated: `npm install` (or equivalent)

### Test 1: Admin OIDC Login
```
1. Navigate to: http://localhost:5173/#/admin/login
2. Click "Sign in with SSO" button
   ✅ Should redirect to Keycloak login page
   ❌ Should NOT show error immediately
3. Authenticate with Keycloak credentials
4. Should redirect back to dashboard
   ✅ URL should show admin/dashboard or /
   ✅ Should see authenticated content (not login form)
   ❌ Should NOT show JSON error about invalid_client_credentials
```

### Test 2: Self-Service OIDC Login  
```
1. Navigate to: http://localhost:5173/#/account
2. In login section, click SSO button
   ✅ Should redirect to Keycloak login page
   ❌ Should NOT show error immediately
3. Authenticate with Keycloak credentials
4. Should redirect back to account page
   ✅ URL should show /account or /
   ✅ Should see authenticated content (keys, quotas)
   ❌ Should NOT show JSON error
```

### Test 3: Browser Developer Tools Inspection

#### Network Tab
```
Request 1: GET /api/admin/oidc/login
├─ Status: 200 OK
├─ Response: {"authorization_uri": "https://keycloak..."}
└─ ✅ Should see this before redirect to Keycloak

Request 2: GET /api/admin/oidc/callback?code=...&state=...
├─ Status: 302 Found
├─ Headers: Set-Cookie: foap_session=...; HttpOnly; Secure
└─ ✅ Should see this after Keycloak redirects

Request 3: GET / (or dashboard)
├─ Status: 200 OK
├─ Cookie: foap_session=...; (should be present)
└─ ✅ Session cookie automatically sent
```

#### Cookies Tab
```
After successful login:
├─ Name: foap_session
├─ Value: <session_id>
├─ HttpOnly: ✅ (should be checked, not visible to JavaScript)
├─ Secure: ✅ (only sent over HTTPS)
├─ SameSite: Lax ✅
└─ Domain: ai-api.mylab.th-luebeck.de

After logout:
└─ foap_session: Should be deleted/expired
```

#### Console Tab
```
❌ Should NOT see errors like:
   - "OIDC callback error: ..."
   - "Invalid client or Invalid client credentials"
   - "Failed to exchange code for token"

✅ Should see normal Vue/router operations:
   - Navigation to dashboard
   - Component lifecycle logs
```

### Test 4: Token Exchange Flow (Advanced)

Monitor Keycloak logs while testing:
```bash
# Expected flow in Keycloak logs:
✅ type="CODE_TO_TOKEN", clientId="mylab-foap", success=true
✅ Token issued successfully

❌ Should NOT see:
   type="CODE_TO_TOKEN_ERROR", error="invalid_client_credentials"
   type="CODE_TO_TOKEN_ERROR", error="invalid_client_id"
```

### Test 5: Error Scenarios

#### Scenario: Invalid Client Secret
```
Setup: Set FOAP_OIDC_CLIENT_SECRET to wrong value
Action: Try to login with SSO
Expected: Error message appears saying "OIDC BFF not configured" or similar
✅ Should NOT hang or show cryptic error
```

#### Scenario: Missing Admin Claims
```
Setup: Login with user that lacks "foap-admin" role
Action: Try admin OIDC login
Expected: Redirect to login with error message
✅ Should see: "No admin access in token claims"
```

#### Scenario: Network Offline
```
Setup: Backend unreachable during /api/admin/oidc/login
Action: Try admin OIDC login
Expected: Error message: "Failed to start SSO login"
✅ Should NOT hang indefinitely
```

## 📊 Request/Response Validation

### Step 1: /api/admin/oidc/login Response
```javascript
// Should return:
{
  "authorization_uri": "https://keycloak.example.com/realms/mylab/protocol/openid-connect/auth?client_id=mylab-foap&response_type=code&scope=openid+profile+email&redirect_uri=https%3A%2F%2Fai-api.mylab.th-luebeck.de%2Fapi%2Fadmin%2Foidc%2Fcallback&state=<random_state>&code_challenge=<pkce_challenge>&code_challenge_method=S256"
}

// Response headers should include:
Set-Cookie: foap_oidc_session=<session_id>; Max-Age=600; HttpOnly; Secure; SameSite=Lax
```

### Step 2: /api/admin/oidc/callback Response
```
Status: 302 Found
Location: /
Set-Cookie: foap_session=<auth_session_id>; Max-Age=86400; HttpOnly; Secure; SameSite=Lax
```

### Step 3: /api/admin/health Response (after login)
```javascript
{
  "status": "ok",
  "scope": "admin"
}
```

## 🐛 Debugging Tips

### Issue: "Completing sign-in…" page shows error
```
Common causes:
1. Backend /api/admin/oidc/callback endpoint failed
2. Token exchange returned invalid_client_credentials
   → Check: FOAP_OIDC_CLIENT_SECRET is correct
3. User lacks required admin roles/claims
   → Check: User has "foap-admin" role in Keycloak

Debug: Check backend logs for CODE_TO_TOKEN_ERROR
```

### Issue: Infinite loop showing Keycloak login
```
Common causes:
1. State parameter validation failed
2. Session cookie not being set/persisted
3. Redirect URI mismatch

Debug: 
- Check browser cookies: foap_oidc_session set?
- Check browser console: any fetch errors?
- Check backend logs: session store working?
```

### Issue: Logged in but no admin access
```
Common causes:
1. User token doesn't contain admin roles claim
2. FOAP_OIDC_ADMIN_VALUES configured wrong
3. Keycloak role mapper not working

Debug:
- Decode access token at jwt.io
- Check inside: does it have "roles" claim with "foap-admin"?
- Verify Keycloak role mappings for the user
```

## 🔄 Before & After Comparison

### Before (Broken) - Keycloak Logs
```
⚠️  type="CODE_TO_TOKEN_ERROR"
    realmName="mylab"
    clientId="mylab-foap"
    error="invalid_client_credentials"
    grant_type="authorization_code"
    ipAddress="149.249.73.209"
```

### After (Fixed) - Keycloak Logs
```
✅ type="CLIENT_LOGIN"
   realmName="mylab"
   clientId="mylab-foap"
   userId="<user_id>"
   ipAddress="149.249.73.209"

✅ type="CODE_TO_TOKEN"
   clientId="mylab-foap"
   success=true
```

## 📝 Verification Checklist

- [ ] All 5 frontend files have been modified correctly
- [ ] `npm install` completed successfully
- [ ] No console errors on login page
- [ ] Can click "Sign in with SSO" without immediate error
- [ ] Redirects to Keycloak properly
- [ ] Can authenticate at Keycloak
- [ ] Redirected back to dashboard after auth
- [ ] Dashboard shows authenticated content
- [ ] Browser has `foap_session` cookie (check DevTools)
- [ ] Keycloak logs show successful CODE_TO_TOKEN (not error)
- [ ] Backend logs show successful session creation
- [ ] Self-service account login also works with same pattern
- [ ] Logout clears session cookie properly
- [ ] Subsequent login still works

## 🚀 Deployment Readiness

- [ ] Frontend built successfully: `npm run build`
- [ ] Backend environment variables verified
- [ ] Keycloak client configuration matches backend endpoints
- [ ] HTTPS enabled in production (required for secure cookies)
- [ ] CORS configured if frontend and backend on different domains
- [ ] Session store (SQLite) is writable by backend process
- [ ] Browser cookie policy allows httpOnly/Secure/SameSite

## 📞 Support

If tests fail, check:

1. **Backend logs** - Look for OIDC/token exchange errors
2. **Keycloak logs** - Check for CODE_TO_TOKEN_ERROR details
3. **Browser console** - Look for network/fetch errors
4. **Browser DevTools Network** - Inspect request/response headers
5. **Environment variables** - Verify OIDC_CLIENT_SECRET is set
6. **Redirect URIs** - Ensure Keycloak client config is exact match

