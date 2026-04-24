# FOAP Next Phases: Phase 3 & Phase 4 Implementation Plan

This document outlines the architecture, tasks, and boundaries for the upcoming phases of the Fast OpenAI API Proxy (FOAP) platform. Given the size and complexity of the remaining features, we are splitting the work into two distinct phases to ensure stability and maintain high development velocity.

---

## 🎯 Phase 3: Identity, Quotas & Self-Service (The User Layer)

**Goal:** Finalize the authentication/OIDC flow, enforce actual provider rate limits asynchronously, and build the Vue.js Self-Service portal so users can generate their own API keys and monitor their usage without admin intervention.

### 1. Provider Rate-Limit Synchronization (Backend)
- **Asynchronous Header Processing:** Implement background tasks or middleware to parse `x-ratelimit-*` headers from upstream provider responses (e.g., OpenAI, Anthropic).
- **Global Provider Throttling:** Update the `provider_ratelimits` table in SQLite dynamically.
- **Smart Backoff:** When `sync_provider_ratelimits` is enabled, FOAP will proactively `429 Too Many Requests` clients if the upstream provider limit is known to be exhausted, saving network roundtrips and preventing IP bans.

### 2. OIDC & Auth Hardening (Backend)
- Finalize the `oidc_auth.py` flows.
- Ensure seamless mapping of OIDC roles/groups to FOAP access control logic.
- Complete any remaining admin API flows around quota operations and overrides.

### 3. Self-Service Portal (Frontend - `/account`)
- **Login/Session UX:** Auth-aware login screen that gracefully handles OIDC redirects vs. static token modes.
- **API Key Management:** UI for users to generate, view, and revoke their personal FOAP API keys.
- **Quota Dashboard:** A clean, visual dashboard showing the user's current usage vs. their assigned quotas (minute/hour/day limits) across different models.

---

## 🚀 Phase 4: Stateful Intelligence (The Infrastructure Layer)

**Goal:** Evolve FOAP from a stateless proxy into a stateful intelligence platform by introducing PostgreSQL, `pgvector`, and full support for the OpenAI Assistants, Conversations, and Files APIs.

> [!NOTE]
> **Stateful Features Flag:** Stateful features will be gated behind an environment variable (`FOAP_ENABLE_STATEFUL_FEATURES=true`). If this flag is disabled, FOAP will run in lightweight mode using SQLite, and the Responses API will function purely as a stateless passthrough (with standard streaming support).

### 1. Database Infrastructure: PostgreSQL + pgvector
- Require a PostgreSQL connection string (URL, user, password) if `FOAP_ENABLE_STATEFUL_FEATURES` is true.
- Introduce `pgvector` extension for native high-performance vector similarity search.
- Build Alembic/SQLModel migrations to transition and sync data between SQLite and Postgres seamlessly.

### 2. The Responses & Conversations API
- Implement stateful interception middleware.
- **Conversations & Messages:** Store chat history natively. Allow clients to pass a `thread_id` and have FOAP automatically reconstruct the context window before forwarding to the upstream LLM.
- **Stateless Fallback:** If stateful features are disabled, the `/v1/chat/completions` API acts purely as a stateless router.
- Build management endpoints for administrators to view/audit stateful objects.

### 3. Files & Vector Store API (Assistants API Parity)
- Implement `/v1/files`: Accept file uploads, chunk text, and generate embeddings (using an embedding alias model).
- Implement `/v1/vector_stores`: Group files into vector stores for RAG (Retrieval-Augmented Generation).
- Store resulting embeddings in PostgreSQL via `pgvector`.
- When a user chats with a specific thread/assistant, FOAP will seamlessly perform the RAG retrieval against `pgvector` and inject the context into the upstream prompt.
