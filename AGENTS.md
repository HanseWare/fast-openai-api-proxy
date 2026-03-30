# AGENTS.md - Fast OpenAI API Proxy

AI agent guide for productive contribution to this OpenAI API proxy project.

## Quick Context

**Fast OpenAI API Proxy** is a FastAPI application that acts as a unified entrypoint to multiple AI model providers. It converts incoming OpenAI API v1-compatible requests into provider-specific calls (OpenAI, Azure OpenAI, local models, etc.) and returns compatible responses.

**Core Value Proposition:** Single unified API → multiple heterogeneous backend model providers

## Architecture Overview

### Entry Points & Data Flow

1. **`app/main.py`** - Root FastAPI application
   - Custom `FOAP` class extending `FastAPI` with `base_url` and `model_handler` attributes
   - Health checks: `GET /health` (general) and `GET /health/{model}` (backend-specific)
   - Mounts `/v1` sub-application via: `app.mount("/v1", app=api_v1_app)`
   - Initializes JSON logging on startup via `setup_logging()`

2. **`app/api_v1.py`** - All OpenAI v1 endpoints
   - Separate `FOAP_API_V1` FastAPI class mounted as sub-app
   - Endpoints: `/chat/completions`, `/completions`, `/embeddings`, `/audio/*`, `/images/*`, `/models`, `/moderations`
   - **Key pattern:** Most endpoints call `handle_request()` then `process_completion_response()` (or `process_response()`)
   - File uploads (audio, images) use `handle_file_upload()` instead

3. **`app/models_handler.py`** - Configuration and model resolution
   - Single `ModelsHandler` instance (`handler`) loaded at startup
   - Reads all `.json` files from `FOAP_CONFIG_DIR` (env var; defaults to `configs/`)
   - **Critical method:** `get_model_data(model, api_path=None)` - resolves model name → endpoint config
   - Config structure: provider → models → endpoints with `target_base_url`, `target_model_name`, timeouts, api_key_variable
   - Model names support prefix: e.g., `prefix: "custom-"` makes `gpt-4o` → `custom-gpt-4o`

4. **`app/utils.py`** - Request proxying and response handling
   - `handle_request()`: JSON request → target backend + streaming support
   - `handle_file_upload()`: multipart/form-data → target backend (audio/images)
   - `process_response()` / `process_completion_response()`: format and stream responses

### Data Flow Example
```
Client Request → api_v1.py endpoint → models_handler.get_model_data(model, api_path)
  → utils.handle_request() → target backend (e.g., OpenAI) → response processing → StreamingResponse
```

## Key Architectural Decisions

- **Model-to-Endpoint Mapping:** One model (e.g., `gpt-4o`) can have multiple endpoints (`chat/completions`, `completions`, etc.), each pointing to potentially different backends. Resolution happens via `api_path` parameter.
- **Bearer Token Handling:** Extracted from `Authorization` header; replaced with provider's API key from environment variables before forwarding.
- **Streaming Support:** Handled via AsyncClient streaming generator; check `stream` param in request body.
- **Health Checks:** Aggregate checks across all backend URLs for a model endpoint; return 503 if ANY backend unhealthy.

## Configuration Format

Config files (JSON) in `FOAP_CONFIG_DIR` follow this structure:

```json
{
  "ProviderName": {
    "api_key_variable": "ENV_VAR_NAME",
    "prefix": "optional-prefix-",
    "target_base_url": "https://api.provider.com",
    "request_timeout": 60,
    "health_timeout": 10,
    "models": {
      "model-id": {
        "endpoints": [
          {
            "path": "v1/chat/completions",
            "target_base_url": "https://...",
            "target_model_name": "actual-model-name",
            "request_timeout": 120
          }
        ]
      }
    }
  }
}
```

**Key Fields:**
- `api_key_variable`: Env var to read API key from
- `target_model_name`: Model name to send to backend (may differ from exposed name)
- `path`: Must match OpenAI API path (e.g., `v1/chat/completions`)
- Timeouts cascade: endpoint → provider level (fallback)

## Common Workflows

### Adding a New Endpoint Type
1. Add endpoint method in `app/api_v1.py` with FastAPI decorator (e.g., `@app.post("/endpoint")`)
2. Use `handle_request()` for JSON bodies or `handle_file_upload()` for multipart
3. Process response with `process_completion_response()` or `process_response()`
4. Add `api_path` matching pattern in config `path` field

### Adding a New Provider
1. Create/update JSON config in `configs/` with provider details
2. Set `api_key_variable` pointing to env var holding the API key
3. Define models with target endpoints and model name mappings
4. Test via `GET /v1/models` and specific endpoint calls

### Debugging Request Flow
- **Log format:** Structured JSON via `pythonjsonlogger`
- **Logged on each request:** provider, model_requested, model_used, api_path, stream status
- Check `FOAP_LOGLEVEL` env var (INFO, DEBUG, WARNING, ERROR)

## Testing & Deployment

- **Local Run:** `python -m uvicorn app.main:app --reload` (requires `cd app/` or adjust PYTHONPATH)
- **Docker:** Builds on Python 3.11-slim; sets `FOAP_CONFIG_DIR=configs` by default
- **Env Vars (important):**
  - `FOAP_CONFIG_DIR`: Config directory (default: `configs` locally, `/configs` in Docker)
  - `FOAP_LOGLEVEL`: Log level (default: `INFO`)
  - `BASE_URL`: Proxy's base URL for response construction (default: `http://localhost:8000`)
  - Provider API keys (e.g., `OPENAI_API_TOKEN`)

## Critical Patterns & Gotchas

1. **Model Resolution:** Always call `models.get_model_data(model, api_path)` before proxying; raises 404 if not found
2. **Authorization:** `can_request(model, token)` in `auth.py` is a placeholder; currently always returns `True`
3. **File Uploads:** Use `await upload_file.read()` to get file content; prepare as `files` dict for httpx
4. **Streaming:** Return `StreamingResponse` from `handle_request()` directly; don't consume the stream early
5. **Response Format:** Audio/image endpoints may return non-JSON (binary); check `Content-Type` header
6. **Model Name Replacement:** Replace request body `model` with `target_model_name` before sending to backend

## Codebase Maintenance Notes

- **ROADMAP.md** lists planned features (e.g. admin endpoints, Responses API, auth improvements)
- No tests present yet; additions welcome
- Single point of initialization: `ModelsHandler` loads all configs once at startup
- **Auth placeholder:** `auth.py::can_request()` needs implementation for actual access control

### MCP Notes
- Always use the OpenAI developer documentation MCP server if you need to work with the OpenAI API, ChatGPT Apps SDK, Codex,… without me having to explicitly ask.