# ROADMAP

## Achieved So Far (Summary)

- [x] Initial agent guidance added and later tightened (`AGENTS.md`)
- [x] API v1 split into modular routers (`backend/app/routers/*`)
- [x] Stateless Responses API passthrough added (core routes)
- [x] Endpoint hardening pass completed for major surfaces:
  - completions stream/error passthrough improvements
  - transcription parameter handling updates
  - audio/image/moderation/embedding passthrough robustness
  - safer JSON parsing and upstream failure handling
  - improved bearer extraction and auth parsing robustness
- [x] Optional admin and self-service APIs implemented behind feature flags
- [x] Access-control middleware integrated for protected endpoints and quotas
- [x] SQLite-backed storage foundation added (`access_store`)
- [x] OIDC support added with strict config validation and OIDC-only modes
- [x] Quota policy engine baseline implemented:
  - model + endpoint scoped policies (`api_path + model`)
  - minute/hour/day windows
  - per-policy mode: per-user vs global
- [x] Per-user quota overrides/exceptions implemented:
  - override > policy > default resolution order
  - temporary windows (`starts_at` / `ends_at`) and exemption mode
- [x] Quota decision tracing for debugging (`X-FOAP-Quota-Decision`)
- [x] Admin UX-oriented quota override API improvements:
  - filtering and pagination support
  - audit-friendly fields (`created_at`, `active_now`, `window_state`)
- [x] Backend repository structure moved under `backend/` (`app`, `configs`, `tests`, `Dockerfile`, `requirements*`)

## Open Backend Checklist

- [ ] Continue endpoint-by-endpoint OpenAI docs parity pass for remaining routes
- [ ] Harden OIDC operations for rollout (operational playbooks, fallback and diagnostics refinements)
- [ ] Complete admin API flows around quota operations (remaining UX-facing API gaps)
- [ ] Migrate provider/model routing config from JSON to DB:
  - [ ] DB schema for providers/models/endpoints/credentials/timeouts
  - [ ] Admin CRUD APIs for provider/model/endpoint config
  - [ ] JSON import bootstrap path and validation
  - [ ] Diff preview before applying import changes
  - [ ] Staged runtime cutover (DB-first, temporary JSON fallback, then fallback-off)
- [ ] Add stateful interception/storage foundations for Responses and related features:
  - [ ] conversations/messages models
  - [ ] vector storage integration path (pgvector/PostgreSQL)
  - [ ] middleware logic for stateful features
  - [ ] management endpoints for stateful objects
- [ ] General iterative improvements (error handling, logging, structure)

## Open Frontend Checklist

- [ ] Build VueJS admin frontend (`/admin`) for:
  - [ ] API keys management
  - [ ] protected endpoints management
  - [ ] quota policies management
  - [ ] quota overrides management
  - [ ] auth-mode visibility (`/api/admin/auth-config`)
- [ ] Build VueJS self-service frontend (`/account`) for:
  - [ ] own API keys lifecycle
  - [ ] own quota visibility / usage UX
- [ ] Frontend UX for DB-backed provider/model/endpoint config once backend CRUD is ready
- [ ] Frontend UX for JSON import workflow (validation + diff preview)
- [ ] Rollout readiness: auth-aware login/session UX aligned with static-token/OIDC modes
