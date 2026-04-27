# OIDC Authentication - Before vs After Flow Diagram

## ❌ BEFORE (Broken) - Direct Token Exchange from Browser

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            BROWSER (Frontend)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. User clicks "Sign in with SSO"                                        │
│                          ↓                                                 │
│  2. oidc-client-ts signinRedirect()                                       │
│     ├── Generate code_verifier                                             │
│     ├── Generate state                                                     │
│     └── Redirect to Keycloak authorization endpoint                       │
│                          ↓                                                 │
│  3. User authenticates at Keycloak                                        │
│                          ↓                                                 │
│  4. Browser receives callback: ?code=...&state=...                        │
│     (Routes to /oidc-callback frontend view)                              │
│                          ↓                                                 │
│  5. oidc-client-ts signinCallback()                                       │
│     └── JavaScript tries to POST /token with code                         │
│         ├── client_id: mylab-foap                                          │
│         ├── code: <auth_code>                                              │
│         ├── code_verifier: <pkce_value>                                    │
│         ├── client_secret: ??? (CANNOT SEND SECURELY FROM JS!)             │
│         └── [REQUEST SENT]                                                 │
│                          ↓                                                 │
│  6. ❌ Keycloak Response: 401 Unauthorized                                 │
│     └── {"error":"unauthorized_client",                                    │
│          "error_description":"Invalid client or Invalid client            │
│                                credentials"}                               │
│                          ↓                                                 │
│  7. ❌ Frontend shows error: "Failed to extract access token..."          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

KEYCLOAK LOGS:
⚠️  type="CODE_TO_TOKEN_ERROR", 
    clientId="mylab-foap", 
    error="invalid_client_credentials",
    grant_type="authorization_code"

PROBLEM: JavaScript cannot securely hold/send client_secret!
```

---

## ✅ AFTER (Fixed) - Backend-for-Frontend (BFF) Pattern

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          FRONTEND (Browser)                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│  1. User clicks "Sign in with SSO"                                          │
│                          ↓                                                   │
│  2. startOidcLogin('/api/admin', 'admin')                                   │
│     └── fetch('/api/admin/oidc/login')                                      │
│                          ↓                                                   │
│                    [Network Request]                                         │
│                          ↓                                                   │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                          BACKEND (Node.js/Python)                            │
├──────────────────────────────────────────────────────────────────────────────┤
│  3. GET /api/admin/oidc/login                                               │
│     ├── Generate code_verifier (PKCE)                                       │
│     ├── Generate state (CSRF protection)                                    │
│     ├── Store in session: session_id=<random>                              │
│     ├── Set session cookie: foap_oidc_session=<session_id>                 │
│     └── Return:                                                             │
│         {                                                                    │
│           "authorization_uri":                                              │
│           "https://keycloak/realms/mylab/protocol/openid-connect/auth?      │
│            client_id=mylab-foap&state=<state>&                            │
│            code_challenge=<pkce>&code_challenge_method=S256&..."            │
│         }                                                                    │
│                          ↓                                                   │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                          FRONTEND (Browser)                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│  4. window.location.href = authorizationUri                                 │
│     └── Redirect to Keycloak                                                │
│                          ↓                                                   │
│  5. User authenticates at Keycloak                                          │
│                          ↓                                                   │
│  6. Keycloak redirects: /api/admin/oidc/callback?code=...&state=...        │
│     (NOT to frontend, directly to backend!)                                 │
│                          ↓                                                   │
│                    [Network Request to Backend]                             │
│                          ↓                                                   │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                          BACKEND (Secure Token Exchange)                     │
├──────────────────────────────────────────────────────────────────────────────┤
│  7. GET /api/admin/oidc/callback?code=...&state=...                        │
│     (with Cookie: foap_oidc_session=<session_id>)                          │
│     ├── Retrieve session from database                                      │
│     ├── Validate state parameter                                            │
│     ├── Extract code_verifier from session                                  │
│     └── [Continue to step 8]                                                │
│                          ↓                                                   │
│  8. exchange_code_for_token(code, redirect_uri, code_verifier)            │
│     └── POST /realms/mylab/protocol/openid-connect/token                   │
│         ├── grant_type: authorization_code                                  │
│         ├── code: <auth_code>                                               │
│         ├── client_id: mylab-foap                                            │
│         ├── client_secret: <SECRET FROM ENV VAR>  ← ✅ SECURE!              │
│         ├── redirect_uri: /api/admin/oidc/callback                          │
│         └── code_verifier: <pkce_verifier>                                  │
│                          ↓                                                   │
│  9. ✅ Keycloak Response: 200 OK                                            │
│     {                                                                        │
│       "access_token": "eyJhbGc...",                                        │
│       "token_type": "Bearer",                                              │
│       "expires_in": 300,                                                    │
│       "refresh_token": "..."                                               │
│     }                                                                        │
│                          ↓                                                   │
│  10. Backend validates token:                                              │
│      ├── JWT signature valid?                                               │
│      ├── Claims contain "foap-admin"?                                       │
│      ├── Extract subject (user ID)                                          │
│      └── Create authenticated session                                       │
│                          ↓                                                   │
│  11. Create auth session:                                                   │
│      ├── auth_session_id = generate_session_id()                            │
│      ├── Store: {                                                            │
│      │   access_token: <token>,                                             │
│      │   owner_id: oidc:willnowp,                                           │
│      │   scope: admin                                                       │
│      │ }                                                                      │
│      ├── Set cookie: foap_session=<auth_session_id>                        │
│      │   (httpOnly, secure, samesite=lax, max_age=86400)                   │
│      └── Delete old session: foap_oidc_session                              │
│                          ↓                                                   │
│  12. Return: 302 Redirect to "/"                                           │
│                          ↓                                                   │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                          FRONTEND (Browser)                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│  13. Browser follows redirect: GET /                                        │
│      (Cookie: foap_session=<auth_session_id> automatically sent)           │
│                          ↓                                                   │
│  14. Frontend checks: isAuthenticated?                                      │
│      └── ✅ YES! Session cookie present and valid                          │
│                          ↓                                                   │
│  15. ✅ User sees dashboard                                                 │
│      └── All subsequent API calls include foap_session cookie              │
│          automatically (httpOnly cookies sent by browser)                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

KEYCLOAK LOGS:
✅ type="CODE_TO_TOKEN", clientId="mylab-foap", success=true
✅ Token issued to mylab-foap for user willnowp

KEY IMPROVEMENTS:
✓ client_secret NEVER exposed to browser
✓ Backend performs secure token exchange
✓ PKCE prevents authorization code interception
✓ Session cookie prevents clipboard leakage
✓ Proper state validation prevents CSRF
```

---

## Request/Response Examples

### ✅ AFTER FIX: Backend BFF Flow

#### Step 1: Frontend Initiates Login
```http
GET /api/admin/oidc/login
Host: ai-api.mylab.th-luebeck.de
Accept: application/json
Cookie: <session_cookies>

HTTP/1.1 200 OK
Set-Cookie: foap_oidc_session=abc123def456; Max-Age=600; HttpOnly; Secure; SameSite=Lax
Content-Type: application/json

{
  "authorization_uri": "https://keycloak.example.com/realms/mylab/protocol/openid-connect/auth?client_id=mylab-foap&response_type=code&scope=openid+profile+email&redirect_uri=https%3A%2F%2Fai-api.mylab.th-luebeck.de%2Fapi%2Fadmin%2Foidc%2Fcallback&state=xyz789abc&code_challenge=abcde12345fghij67890klmnop&code_challenge_method=S256"
}
```

#### Step 2: Backend Handles Callback (Keycloak Redirects to Backend)
```http
GET /api/admin/oidc/callback?code=gho_16C7...[long auth code]...&state=xyz789abc
Host: ai-api.mylab.th-luebeck.de
Cookie: foap_oidc_session=abc123def456

[Backend exchanges code for token using client_secret securely]

HTTP/1.1 302 Found
Location: /
Set-Cookie: foap_session=xyz789auth123session; Max-Age=86400; HttpOnly; Secure; SameSite=Lax
```

#### Step 3: Frontend Now Has Authenticated Session
```http
GET /
Host: ai-api.mylab.th-luebeck.de
Cookie: foap_session=xyz789auth123session

[Frontend loads, auth store reads foap_session cookie]
[All subsequent API calls include foap_session automatically]

GET /api/admin/health
Host: ai-api.mylab.th-luebeck.de
Cookie: foap_session=xyz789auth123session

HTTP/1.1 200 OK
{
  "status": "ok",
  "scope": "admin"
}
```

---

## Why This Works

| Aspect | Before (Broken) | After (Fixed) |
|--------|-----------------|---------------|
| **Token Exchange Location** | Browser (JavaScript) | Backend (Secure) |
| **Client Secret Location** | ❌ Not applicable | ✅ Environment variable |
| **Client Secret in Network** | N/A | ✅ Server-to-server only |
| **HTTPS Required** | Not enforced | ✅ Yes (httpOnly cookies) |
| **CSRF Protection** | Missing | ✅ State parameter |
| **Replay Attack Protection** | None | ✅ State + session binding |
| **Clients Trust Level** | Public (must be) | ✅ Confidential (secure) |
| **Token Storage** | Browser localStorage | ✅ Session database (server-side) |
| **Cookie Security** | N/A | ✅ httpOnly, secure, samesite=lax |

---

## Migration Impact

### For End Users
- ✅ **Same login experience** - "Sign in with SSO" button works the same
- ✅ **Better security** - Credentials never exposed
- ✅ **Seamless session** - Automatic session management via cookies
- ✅ **No breaking changes** - Existing tokens still work

### For Administrators  
- ✅ **No migration needed** - Backend endpoints unchanged
- ✅ **Same environment variables** - OIDC config still the same
- ✅ **Better logging** - Server-side token exchange logs in backend
- ✅ **Keycloak unchanged** - Same realm configuration works

### For Developers
- ✅ **Cleaner code** - No oidc-client-ts dependency
- ✅ **Better testability** - BFF endpoints can be mocked
- ✅ **Easier debugging** - Backend handles all complex logic
- ✅ **Standards compliant** - Pure OAuth2 + PKCE (RFC 7636)

