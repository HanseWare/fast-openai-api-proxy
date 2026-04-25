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
- `FOAP_ENABLE_OIDC_AUTH`: enable OIDC token validation
- `FOAP_ADMIN_TOKEN`: static admin token (unless OIDC-only mode)
- `FOAP_ADMIN_OIDC_ONLY`, `FOAP_SELF_SERVICE_OIDC_ONLY`: enforce OIDC-only auth modes
- `FOAP_ACCESS_DB_PATH`: SQLite file for keys/quotas/rules

Provider API keys are resolved from env vars configured in model JSON (`api_key_variable`).

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
  - `GET|POST|DELETE /api/keys...`

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
- Phase 3: in progress — identity, quotas, self-service portal, and provider rate-limit hardening
- Phase 4: planned — stateful intelligence, PostgreSQL, and vector-store features

