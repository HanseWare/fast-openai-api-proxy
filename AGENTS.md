# AGENTS.md - Fast OpenAI API Proxy

Purpose: Help coding agents become productive quickly in this repository.

## 1) Scope and Layout

- Active backend code lives in `backend/`.
- High-signal paths:
  - `backend/app/main.py`: root app, health checks, mounting.
  - `backend/app/api_v1.py`: v1 app composition and router wiring.
  - `backend/app/routers/*.py`: endpoint families.
  - `backend/app/routers/admin_config.py`: provider/model/endpoint/alias CRUD and JSON import.
  - `backend/app/utils.py`: request/upload passthrough, streaming + response handling.
  - `backend/app/models_handler.py`: DB-backed model/provider lookup, alias resolution, seed-file fallback.
  - `backend/app/middleware/access_control.py`: protected-endpoint and quota enforcement.
  - `backend/app/access_store.py`: SQLite persistence for keys/rules/quotas/overrides and provider rate-limit state.
  - `backend/app/config_store.py`: SQLite config store for providers, models, endpoints, aliases.

## 2) Core Flow (Request -> Upstream -> Response)

1. Client calls `/v1/*`.
2. Router resolves `model` + `api_path` via `ModelsHandler.get_model_data(...)`, including alias resolution before lookup.
3. `models_handler.py` loads config from SQLite and falls back to `backend/configs/*.json`/YAML seed files when the DB is empty.
4. Proxy layer forwards request to provider endpoint with provider API key from env.
5. Response is passed through with robust status/body behavior (including streaming and non-JSON branches), plus optional fallback model routing and provider rate-limit sync.

If you add new endpoint behavior, keep this flow intact.

## 3) Access Control and Quotas

- Access middleware is optional (`FOAP_ENABLE_ACCESS_CONTROL=true`).
- Decision order for scoped quota enforcement:
  1. user override (`quota_overrides`)
  2. model/endpoint policy (`quota_policies`)
  3. key default quota (`api_key_quotas`)
- Quota decision tracing can be exposed via `FOAP_ENABLE_QUOTA_DECISION_TRACE` or `X-FOAP-Debug-Quota: 1`.

## 4) Admin / Self-Service Surface

- Admin routes: `backend/app/routers/admin.py`
- Admin config routes: `backend/app/routers/admin_config.py`
- Self-service routes: `backend/app/routers/self_service.py`
- OIDC + static token behavior is controlled in `backend/app/config.py` and `backend/app/oidc_auth.py`.

When changing auth behavior, preserve existing mode matrix semantics and error clarity.

## 5) Configuration Patterns

- Provider/model routing is SQLite-driven, with seed fallback from `backend/configs/*.json` and import support in `/api/admin/config/import`.
- Important config keys: `api_key_variable`, `prefix`, `default_base_url`, `default_request_timeout`, `default_health_timeout`, `max_upstream_retry_seconds`, `sync_provider_ratelimits`.
- Endpoint config keys: `path`, `target_model_name`, `target_base_url`, `request_timeout`, `health_timeout`, `fallback_model_name`.
- Alias config keys: `alias_name`, `target_model_name`, `owned_by`, `hide_on_models_endpoint`.
- Timeouts cascade endpoint -> provider defaults.

## 6) Local Workflows

- Install deps: `backend/requirements.txt` / `backend/requirements-dev.txt`.
- Run API from `backend/app` with `uvicorn main:app`.
- Run tests from `backend`:
  - `tests/test_access_admin_self_service.py`
  - `tests/test_oidc_role_mapping.py`
  - `tests/test_completions_stream_passthrough.py`
  - add coverage around config import, alias routing, provider rate-limit sync, and fallback behavior when touching those areas.

## 7) Agent Rules for This Repo

- Prefer minimal, behavior-preserving changes.
- Keep passthrough semantics: do not replace upstream payloads with generic local errors unless required.
- For quota/auth changes, always add or update tests in `backend/tests/`.
- Update `README.md` and `docs/ROADMAP.md` when behavior or structure changes.
