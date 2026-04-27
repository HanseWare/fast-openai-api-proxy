# Fast OpenAI API Proxy

OpenAI-compatible FastAPI proxy that routes one API surface to multiple backend providers, with SQLite-backed routing/config management, virtual model aliases, and passthrough behavior for upstream responses.

## Repository Layout

```text
.
├── backend/
│   ├── app/                 # FastAPI app, routers, middleware, auth, storage
│   ├── configs/             # Provider/model mapping JSON/YAML seed files
│   ├── tests/               # Pytest suites
│   ├── Dockerfile
│   ├── requirements.txt
│   └── requirements-dev.txt
├── AGENTS.md
├── docs/
│   ├── ROADMAP.md
│   └── walkthrough.md
├── frontend/                # Vue admin UI
└── LICENSE
```

## What It Supports

- OpenAI v1-style endpoints under `/v1` (chat/completions/embeddings/audio/images/moderations/models/responses)
- Provider/model/endpoint routing from the SQLite config store, with automatic fallback seeding from `backend/configs/*.json` or `*.yml`/`*.yaml` when the DB is empty
- Virtual models / aliases that resolve to a target upstream model and can be hidden from `/v1/models`
- Streaming passthrough (SSE), file upload passthrough, robust upstream error/status passthrough
- Fallback routing via `fallback_model_name` plus provider-level route fallbacks
- Provider rate-limit sync from upstream `x-ratelimit-*` headers when enabled per provider, with proactive 429 short-circuiting when cached limits are exhausted
- Optional admin + self-service APIs (`/api/admin/*`, `/api/*`)
- Vue self-service account portal at `/account`
- Self-service portal auto-detects OIDC vs static-token mode via `/api/auth-config`
- Admin config APIs for providers, models, endpoints, aliases, and JSON import (`/api/admin/config/*`)
- Optional access control middleware (API keys, protected endpoints, quotas)
- Optional OIDC auth with role/group mapping

## Quick Start (Local)

1. Install dependencies:

```bash
cd backend
python -m pip install -r requirements.txt
```

2. Set minimum environment variables (example):

```bash
export FOAP_CONFIG_DIR="./configs"
export OPENAI_API_TOKEN="<your-provider-key>"
```

3. Run API:

```bash
cd backend/app
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Quick Start (Tests)

```bash
cd backend
python -m pip install -r requirements-dev.txt
python -m pytest tests/test_access_admin_self_service.py tests/test_oidc_role_mapping.py tests/test_completions_stream_passthrough.py -q
```

## Docker

Build with `backend/` as context:

```bash
docker build -t fast-openai-api-proxy ./backend
```

Run:

```bash
docker run -p 8000:8000 \
  -e FOAP_CONFIG_DIR="configs" \
  -e OPENAI_API_TOKEN="<your-provider-key>" \
  fast-openai-api-proxy
```

## Configuration

Key environment variables:

- `FOAP_CONFIG_DIR`: config directory (`backend/configs` in local dev)
- `FOAP_LOGLEVEL`: `DEBUG|INFO|WARNING|ERROR`
- `BASE_URL`: base URL used for proxy-generated URLs (for custom image data route)
- `FOAP_ENABLE_ADMIN_API`: enable `/api/admin/*`
- `FOAP_ENABLE_SELF_SERVICE_API`: enable `/api/*` self-service endpoints
- `FOAP_ENABLE_ACCESS_CONTROL`: enable access middleware enforcement
- `FOAP_ENABLE_QUOTA_DECISION_TRACE`: emits `X-FOAP-Quota-Decision`
- `FOAP_ACCESS_DB_PATH`: SQLite file for keys/quotas/rules

### OIDC & Identity Configuration

FOAP supports both static token authentication and OIDC (OpenID Connect) for Identity Provider integrations (e.g. Keycloak, Auth0). 

- `FOAP_ENABLE_OIDC_AUTH`: set to `true` to enable OIDC token validation
- `FOAP_OIDC_ISSUER_URL`: The issuer URL (e.g. `https://keycloak.example.com/realms/foap`)
- `FOAP_OIDC_JWKS_URL`: (Optional) Explicit JWKS URL, otherwise auto-discovered via `.well-known`
- `FOAP_OIDC_AUDIENCE`: (Optional) Required audience claim

**Configuring Administrators & Users:**
Access to the Admin Dashboard and Self-Service Portal is determined by claim mappings:

- `FOAP_OIDC_ROLE_CLAIM`: The JWT claim to check for Admin access (default: `roles`)
- `FOAP_OIDC_ADMIN_VALUES`: Comma-separated list of values required for Admin access (default: `foap-admin`)
- `FOAP_OIDC_GROUP_CLAIM`: The JWT claim to check for Self-Service portal access (default: `groups`)
- `FOAP_OIDC_SELF_SERVICE_VALUES`: Comma-separated list of values required for Self-Service access (default: `foap-user`)
- `FOAP_OIDC_SUBJECT_CLAIM`: The JWT claim used as the unique user identifier to scope API keys (default: `sub`)

> **Note on Nested Claims**: FOAP supports dot-notation for nested claims. For example, if your Identity Provider places roles inside `realm_access: { roles: [...] }`, you can set `FOAP_OIDC_ROLE_CLAIM="realm_access.roles"`.

**Authentication Modes:**
You can enforce OIDC-only logins to harden the UI against static token usage:
- `FOAP_ADMIN_OIDC_ONLY`: set to `true` to disable static admin tokens.
- `FOAP_SELF_SERVICE_OIDC_ONLY`: set to `true` to disable static user tokens.
- `FOAP_ADMIN_TOKEN`: Static admin token (used as a fallback when `FOAP_ADMIN_OIDC_ONLY` is false).

**SSO Browser Login (Backend-for-Frontend + PKCE with `client_secret`):**

FOAP implements a **Backend-for-Frontend (BFF)** OAuth2/OIDC flow. When you set `FOAP_OIDC_CLIENT_ID` and `FOAP_OIDC_CLIENT_SECRET`, both the Admin and Self-Service UIs will show a **"Sign in with SSO"** button.

Flow:
1. Browser clicks "Sign in with SSO"
2. Frontend calls `POST /api/admin/oidc/login` or `POST /api/oidc/login`
3. Backend generates PKCE state, stores it in an in-memory session, and returns the authorization URI
4. Frontend redirects browser to Identity Provider
5. User logs in and is redirected to `/api/admin/oidc/callback` or `/api/oidc/callback`
6. Backend exchanges the authorization code for an access token using `client_secret`
7. Backend validates the token, checks claims, and sets a **HttpOnly Secure session cookie**
8. Browser is redirected to `/` (admin) or `/account` (self-service)

Configuration:
- `FOAP_OIDC_CLIENT_ID`: The OIDC client ID registered in your Identity Provider (e.g. `foap-admin-ui`).
- `FOAP_OIDC_CLIENT_SECRET`: The OIDC client secret (kept only on the backend, never exposed to the browser).
- `FOAP_OIDC_PROVIDER_DISPLAY_NAME`: Optional label for the login button (e.g., `"Keycloak"` → `"Sign in with Keycloak"`).

**Why BFF?**
- Secrets remain server-side (never in browser code).
- Sessions are cookie-based (stateful on backend).
- More consistent with your existing JupyterHub, GitLab, N8N deployments.

> **Keycloak Setup**: Create a client with "Standard Flow + PKCE" enabled and "Client authentication" set to **On** (Confidential client). Add your FOAP backend URLs (e.g. `https://foap.example.com/api/admin/oidc/callback`) to the "Valid Redirect URIs" list. Copy the client secret into `FOAP_OIDC_CLIENT_SECRET`.


Provider API keys are resolved from env vars configured in model JSON (`api_key_variable`).
Self-service API keys are generated as `foap-<sha256(uuid4)>` and retried until the stored secret hash is unique in the DB.

## Admin and Self-Service APIs

When feature flags are enabled:

- Admin:
  - `GET /api/admin/health`
  - `GET /api/admin/auth-config`
  - `GET|POST|PUT|DELETE /api/admin/config/providers...`
  - `GET|POST|PUT|DELETE /api/admin/config/models...`
  - `GET|POST|PUT|DELETE /api/admin/config/endpoints...`
  - `GET|POST|PUT|DELETE /api/admin/config/aliases...`
  - `POST /api/admin/config/import`
  - `GET|POST|DELETE /api/admin/keys...`
  - `GET|PUT /api/admin/keys/{key_id}/quota`
  - `GET /api/admin/keys/{key_id}/usage`
  - `GET|POST|PUT|DELETE /api/admin/quota-policies...`
  - `GET|POST|PUT|DELETE /api/admin/quota-overrides...`
  - `GET|POST|DELETE /api/admin/protected-endpoints...`

- Self-service:
  - `GET /api/health`
  - `GET /api/session`
  - `GET|POST|DELETE /api/keys...`
  - `GET /api/usage/summary`

### Quota Overrides: List UX

`GET /api/admin/quota-overrides` supports optional filters and pagination:

- Filters: `owner_id`, `api_path`, `model`, `exempt`, `active_only`
- Pagination: `limit`, `offset`
- Response headers: `X-Total-Count`, `X-Returned-Count`, `X-Limit`, `X-Offset`
- Audit-friendly item fields: `created_at`, `active_now`, `window_state`

## Notes

- `backend/app/models_handler.py` loads config from SQLite and falls back to seed files when the DB is empty.
- `backend/app/routers/admin_config.py` exposes CRUD and import APIs for providers, models, endpoints, and aliases.
- Endpoint-model resolution is driven by `api_path + model` mapping, with alias resolution before lookup.
- Request auth to upstream providers uses provider env keys, not client bearer tokens.
- See `AGENTS.md` for contribution patterns and high-signal code paths.

## Project Status

- Phase 1: complete
- Phase 2: complete
- Phase 2.5: complete
- Phase 3: complete — identity, quotas, self-service portal, and provider rate-limit hardening
- Phase 3.5: complete — stateless Responses API payload translation
- Phase 4: planned — stateful intelligence, PostgreSQL, and vector-store features
