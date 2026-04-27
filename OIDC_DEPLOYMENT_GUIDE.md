# OIDC Authentication Fix - Deployment & Next Steps

## 🎯 What Was Fixed

Your OIDC login was failing with `error="invalid_client_credentials"` because the **frontend JavaScript was trying to exchange an authorization code for an access token directly with Keycloak**. This fails when the Keycloak client is configured as "Confidential" (which requires the `client_secret` to be sent during token exchange).

The frontend cannot securely send the `client_secret` from JavaScript - it would be exposed in the browser.

**Solution:** Migrated to a **Backend-for-Frontend (BFF)** pattern where the backend securely handles the token exchange using the `client_secret` from environment variables.

---

## 📋 Deployment Checklist

### Phase 1: Pre-Deployment Verification

- [ ] **Backend Verification**
  ```bash
  # Check backend environment variables are set:
  echo $FOAP_ENABLE_OIDC_AUTH              # Should be: true
  echo $FOAP_OIDC_ISSUER_URL               # Should be: https://keycloak.../realms/mylab
  echo $FOAP_OIDC_CLIENT_ID                # Should be: mylab-foap
  echo $FOAP_OIDC_CLIENT_SECRET            # Should be set (not empty!)
  echo $FOAP_OIDC_ADMIN_VALUES             # Should be: foap-admin
  ```

- [ ] **Keycloak Client Verification**
  - Login to Keycloak Admin Console
  - Navigate to Realm: `mylab` → Clients → `mylab-foap`
  - Verify:
    - ✅ Client Type: "Confidential" (not Public)
    - ✅ Credentials Tab: Client Secret is generated and matches backend env var
    - ✅ Valid Redirect URIs includes:
      - `https://ai-api.mylab.th-luebeck.de/api/admin/oidc/callback`
      - `https://ai-api.mylab.th-luebeck.de/api/oidc/callback`

- [ ] **Backend Endpoints Accessible**
  ```bash
  # Test from backend server or same network
  curl -I https://ai-api.mylab.th-luebeck.de/api/admin/oidc/login
  # Should return 200 or 400 (not connection refused)
  
  curl -I https://ai-api.mylab.th-luebeck.de/api/oidc/login
  # Should return 200 or 400 (not connection refused)
  ```

### Phase 2: Frontend Update

- [ ] **Update Frontend Code**
  - Ensure these files are modified:
    - `/frontend/src/services/oidc.js` ← Complete rewrite
    - `/frontend/src/views/LoginView.vue` ← Updated handleSsoLogin()
    - `/frontend/src/views/AccountView.vue` ← Updated handleSsoLogin()
    - `/frontend/src/views/OidcCallbackView.vue` ← Simplified
    - `/frontend/package.json` ← Removed oidc-client-ts

- [ ] **Install Dependencies**
  ```bash
  cd frontend
  npm install
  # Should remove oidc-client-ts from node_modules
  ```

- [ ] **Build Frontend**
  ```bash
  npm run build
  # Should complete without errors
  # Check dist/ folder has new files
  ```

- [ ] **Verify Build Artifacts**
  ```bash
  ls -la frontend/dist/
  # Should contain index.html, assets/, etc.
  # Check for any error messages
  ```

### Phase 3: Application Update

- [ ] **Stop Running Services**
  ```bash
  # Stop frontend (if using dev server or reverse proxy)
  # Stop backend (graceful shutdown)
  ```

- [ ] **Deploy Frontend**
  ```bash
  # Option 1: Nginx/reverse proxy
  cp -r frontend/dist/* /var/www/api-portal/
  
  # Option 2: Docker
  docker build -f frontend/Dockerfile -t foap-frontend:latest .
  docker stop foap-frontend
  docker rm foap-frontend
  docker run -d --name foap-frontend -p 3000:80 foap-frontend:latest
  
  # Option 3: Direct serve
  cd frontend && npm run preview
  ```

- [ ] **Start Backend**
  ```bash
  # Backend should have env vars set
  cd backend
  python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
  ```

- [ ] **Verify Services Running**
  ```bash
  # Frontend should be accessible
  curl https://ai-api.mylab.th-luebeck.de/
  
  # Backend should respond to /health
  curl https://ai-api.mylab.th-luebeck.de/api/admin/health
  # May return 401 (expected without auth), not 502
  ```

### Phase 4: Testing

- [ ] **Clear Browser Cache**
  - Quit browser completely
  - Clear cache/cookies/storage
  - Restart browser
  - Navigate to https://ai-api.mylab.th-luebeck.de

- [ ] **Test Admin OIDC Login**
  1. Go to `/#/admin/login`
  2. Click "Sign in with SSO"
  3. Verify redirects to Keycloak
  4. Authenticate with Keycloak
  5. Verify redirects to dashboard
  6. ✅ Should see dashboard (not error)

- [ ] **Test Self-Service OIDC Login**
  1. Go to `/#/account`
  2. Click "Sign in with SSO"
  3. Verify redirects to Keycloak
  4. Authenticate with Keycloak
  5. Verify redirects to account page
  6. ✅ Should see account portal (not error)

- [ ] **Verify Session**
  - Open DevTools → Application → Cookies
  - Check for `foap_session` cookie
  - Should be httpOnly, Secure, SameSite=Lax
  - Should NOT expire for 24 hours

- [ ] **Check Logs**
  ```bash
  # Backend logs should show:
  # ✅ User authenticated successfully
  # ✅ Session created
  # ❌ Should NOT show: invalid_client_credentials
  
  # Keycloak logs should show:
  # ✅ CODE_TO_TOKEN success
  # ❌ Should NOT show: CODE_TO_TOKEN_ERROR
  ```

### Phase 5: Validation

- [ ] **Admin API Access**
  ```bash
  # These should work without passing oidcClient config
  # They just return auth config
  curl https://ai-api.mylab.th-luebeck.de/api/admin/auth-config
  
  # Should be able to access dashoard protected endpoints
  # Browser should have foap_session cookie automatically included
  ```

- [ ] **Self-Service API Access**
  ```bash
  curl https://ai-api.mylab.th-luebeck.de/api/auth-config
  # Should show self-service auth mode
  ```

- [ ] **Fallback Token Auth Still Works**
  - If you have a static token/API key, verify you can still:
    - Use it in password field on login page
    - Or use it as an Authorization header

### Phase 6: Rollback Plan

If something goes wrong:

- [ ] **Keep Old frontend/dist**
  ```bash
  cp -r frontend/dist frontend/dist.backup-$(date +%Y%m%d)
  ```

- [ ] **Keep Old node_modules (optional)**
  - If needed to revert oidc-client-ts:
    ```bash
    git clean -fd node_modules/
    git checkout package-lock.json
    npm install
    ```

- [ ] **Revert Frontend**
  ```bash
  # If deployment fails, restore old dist:
  rm -rf frontend/dist
  cp -r frontend/dist.backup-latest/* frontend/dist/
  
  # Verify old frontend still works
  ```

---

## ⚡ Quick Start (Development)

For local testing:

```bash
# Terminal 1: Backend
cd backend
export FOAP_ENABLE_OIDC_AUTH=true
export FOAP_OIDC_ISSUER_URL=https://keycloak.example.com/realms/mylab
export FOAP_OIDC_CLIENT_ID=mylab-foap
export FOAP_OIDC_CLIENT_SECRET=your-secret-here
export FOAP_OIDC_ADMIN_VALUES=foap-admin
export FOAP_ENABLE_ADMIN_API=true
python -m uvicorn app.main:app --reload

# Terminal 2: Frontend
cd frontend
npm install
npm run dev

# Terminal 3: Access
# Open: http://localhost:5173/#/admin/login
# Keycloak should be accessible from your machine
```

---

## 📊 Success Indicators

### Immediate Signs (Should See Immediately)

✅ Login page loads without errors  
✅ "Sign in with SSO" button is clickable  
✅ Clicking button redirects to Keycloak (not an error page)  

### During Authentication

✅ Can login successfully to Keycloak  
✅ Browser redirects back without showing JSON error  
✅ Page shows "Completing sign-in…" briefly  

### After Authentication

✅ Redirected to dashboard/account page  
✅ Can see authenticated content  
✅ Browser has `foap_session` cookie (DevTools)  
✅ Can access protected API endpoints  

### Logs

✅ No "invalid_client_credentials" errors  
✅ Backend logs show successful token exchange  
✅ Keycloak logs show CODE_TO_TOKEN success  

---

## 🚨 Common Issues & Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| Shows JSON error immediately | Backend BFF not configured | Check FOAP_OIDC_CLIENT_SECRET env var |
| Hangs on "Completing sign-in…" | Backend token exchange failed | Check Keycloak logs for CODE_TO_TOKEN_ERROR |
| Login works but no dashboard | Missing admin claims in token | User must have "foap-admin" role in Keycloak |
| Browser redirects to blank page | Session cookie not set | Check backend session store (SQLite) is working |
| Mixed content warning (HTTPS) | Frontend served over HTTP | Must use HTTPS for secure cookies |
| CORS error in browser console | Frontend and backend different origins | Configure CORS in backend if needed |

---

## 📚 Documentation Files

Supporting documentation has been created:

- **`OIDC_FIX_SUMMARY.md`** - Complete technical summary of changes
- **`OIDC_FLOW_DIAGRAM.md`** - Visual flow diagrams (before/after)
- **`OIDC_VALIDATION_GUIDE.md`** - Detailed testing checklist

---

## 🔗 Related Commands

```bash
# View backend OIDC logs in real-time
tail -f backend.log | grep -i oidc

# Test backend OIDC endpoints
curl -v https://ai-api.mylab.th-luebeck.de/api/admin/oidc/login

# Verify Keycloak configuration
curl https://keycloak.example.com/realms/mylab/.well-known/openid-configuration | jq .

# Check session database
sqlite3 data/access.db "SELECT id, created_at FROM sessions ORDER BY created_at DESC LIMIT 5;"

# Decode JWT token from browser
# Paste token at https://jwt.io or use:
jq -R 'split(".") | .[1] | @base64d | fromjson' <<< "token-from-authorization-header"
```

---

## ✅ Completion Checklist

Mark when complete:

- [ ] Backend environment variables configured
- [ ] Keycloak client settings verified
- [ ] Frontend code updated
- [ ] Dependencies installed (`npm install`)
- [ ] Frontend built (`npm run build`)
- [ ] Services deployed
- [ ] Admin OIDC login tested
- [ ] Self-service OIDC login tested
- [ ] Session cookies verified
- [ ] Error logs checked
- [ ] Keycloak logs show success
- [ ] Documentation reviewed
- [ ] Team informed of changes

---

## 📞 Need Help?

If issues persist:

1. Check `/OIDC_FIX_SUMMARY.md` for detailed technical info
2. Review network traces in browser DevTools
3. Check backend logs for token exchange errors
4. Verify Keycloak logs for CODE_TO_TOKEN errors
5. Ensure `FOAP_OIDC_CLIENT_SECRET` exactly matches Keycloak

Remember: **The backend now handles all OIDC complexity securely!**

