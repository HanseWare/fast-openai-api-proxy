# OIDC SSO Flow using `oidc-client-ts`

You are completely right—rolling our own PKCE crypto logic in vanilla JavaScript is reinventing the wheel and prone to subtle security flaws. 

We will use **`oidc-client-ts`**, the industry-standard TypeScript/JavaScript library for OIDC flows in Single Page Applications.

*(For the backend, we are already perfectly positioned! FOAP acts as a Resource Server and uses `PyJWT` with `PyJWKClient` to fetch public keys via `.well-known/openid-configuration` and cryptographically verify the tokens. This is the industry standard for Python backends).*

## Backend Changes

By using `oidc-client-ts` in the frontend, we don't need to manually expose `authorization_url` or `token_url`. The library will auto-discover them using the issuer URL!

1. **New Configuration Variable (`backend/app/config.py`)**:
   - `FOAP_OIDC_CLIENT_ID`: The OIDC client ID configured in your Identity Provider (e.g. `foap-admin-ui`).
   - `FOAP_OIDC_PROVIDER_DISPLAY_NAME`: Optional provider label for UI button text (e.g. `Keycloak`).
   - *(We already have `FOAP_OIDC_ISSUER_URL` which the library needs as the `authority`).*

2. **API Payload Updates**:
   - Expose `oidc_client_id` and `oidc_issuer_url` inside `AuthModeSnapshot` via `/api/admin/auth-config`.

## Frontend Changes

1. **Dependencies (`frontend/package.json`)**:
   - Run `npm install oidc-client-ts`.

2. **OIDC Service (`frontend/src/services/oidc.js`)**:
   - Create a dedicated wrapper around `UserManager` from `oidc-client-ts`.
   - Configure it dynamically using the `authority` and `client_id` fetched from the backend.

3. **Login Views (`LoginView.vue` & `AccountView.vue`)**:
   - Add a **"Sign in with SSO"** button, replaced by `Sign in with <Provider>` when `FOAP_OIDC_PROVIDER_DISPLAY_NAME` is set.
   - When clicked, call `userManager.signinRedirect()`. The library automatically generates the PKCE challenge, stores it in session storage, and redirects the user to Keycloak.

4. **New Callback Route (`frontend/src/views/OidcCallbackView.vue`)**:
   - Create a `/oidc-callback` route.
   - On mount, it calls `userManager.signinCallback()`. The library automatically extracts the code, hits the token endpoint, verifies the token, and returns the User object containing the `access_token`.
   - Save the `access_token` to the Pinia Auth store and redirect to the dashboard.

## User Review Required

This is a much cleaner architecture. Are you ready for me to install `oidc-client-ts` and wire this up?
