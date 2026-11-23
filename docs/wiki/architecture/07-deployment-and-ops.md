# Deployment and Operations

Sources: `Dockerfile`, `Dockerfile.frontend`, `docker-compose.yml`, `docker-entrypoint.sh`, `healthcheck.sh`, `railway.toml`, `vercel.json`, `next.config.ts`, `security-headers.mjs`, `src/proxy.ts`, `api/index.py`, `api/utils/sentry_setup.py`, `api/utils/langsmith_pin.py`, `api/utils/langsmith_tracing.py`, `api/llm_trace_service.py`, `api/trace_service.py`, `api/metrics_service.py`, `api/funnel_service.py`, `api/utils/request_guard.py`, `api/utils/llm_provider.py`, `.github/workflows/*.yml`, `pytest.ini`, `requirements*.txt`, `package.json`, `scripts/README-openproject.md`

Last verified against code: 2026-11-23

This page describes production as it runs today (Vercel + Railway). The planned move of the backend to AWS (cutover DOCBOT-2519, mid-January 2027 per the [Master backlog](master-backlog)) is covered in the [AWS migration exec plan](aws-migration-plan).

## 1. Deployment topology

| Tier | Host | Build | Start | Health |
|------|------|-------|-------|--------|
| Frontend (Next.js 16, React 19) | Vercel | `vercel.json`: framework `nextjs`, `next build`, `npm install`, output `.next` | Vercel-managed | none configured |
| Backend (FastAPI, Python 3.12) | Railway container | `railway.toml:1-3`: `builder = "DOCKERFILE"`, `Dockerfile` | `Dockerfile` CMD: `uvicorn api.index:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-graceful-shutdown 25` | `railway.toml:6`: `healthcheckPath = "/api/health"`, timeout 30 s, restart `ON_FAILURE` max 3 |
| App DB | Railway PostgreSQL | managed | managed | probed by `/api/health` (`api/index.py:889-904`, `SELECT 1`, 503 on failure) |
| Vector store | Chroma `PersistentClient` on the container's disk/volume | n/a | in-process | n/a. Path from `CHROMA_DIR` (legacy `CHROMA_PERSIST_DIR`) (`api/utils/vector_store.py:12-17,72`) |
| PM board | AWS Lightsail (OpenProject), see section 8 | docker compose | docker compose | manual |

Request path: browser → Vercel (Next.js) → rewrite `/api/:path*` and `/v1/:path*` to `${BACKEND_URL}` (`next.config.ts:41-47`) → Railway FastAPI. In development the rewrite target is `http://127.0.0.1:8000` (`next.config.ts:42-43`). A production build fails hard if neither `BACKEND_URL` nor legacy `RAILWAY_BACKEND_URL` is set (`next.config.ts:7-15`, `security-headers.mjs` `backendUrlFromEnv`).

`/landing` permanently redirects to `/` (`next.config.ts:36-38`).

Note: only `/api/*` and `/v1/*` are rewritten. Backend routes under `/admin/*` are not reachable through the Vercel origin (see [Frontend Architecture](architecture-08-frontend-architecture), "Known gaps").

### 1.1 Backend image (`Dockerfile`)

1. Base `python:3.12-slim`, non-root `appuser` created.
2. Installs `unixodbc-dev` and, on amd64 only, Microsoft `msodbcsql18` (Azure SQL / Entra, DOCBOT-208). On arm64 the step is skipped, so Azure SQL is unavailable on that build.
3. `pip install -r requirements.txt`; copies `api/` only (no frontend, no tests).
4. Pre-downloads the local ONNX models as `appuser` (DOCBOT-1511): chromadb `ONNXMiniLM_L6_V2` embeddings and fastembed `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder into `FASTEMBED_CACHE_PATH=/home/appuser/.cache/fastembed`. Best-effort: a build without network still succeeds and downloads on first use.
5. `ENTRYPOINT docker-entrypoint.sh` runs as root only to `chown` the `/data` volume (Railway mounts volumes root-owned; `find ... -exec chown -h`, never follows symlinks), then `setpriv --reuid=appuser` and `exec`s the CMD (DOCBOT-2338).
6. `--timeout-graceful-shutdown 25` lets the lifespan cleanup run before a 30 s host kill timeout, even with SSE streams open.

### 1.2 Frontend image (`Dockerfile.frontend`, on-prem only)

Three-stage `node:20-alpine` build (deps → builder → runner). Sets `DOCKER_BUILD=1`, which switches `next.config.ts:19` to `output: "standalone"`. Runs `node server.js` as uid 1001 on port 3000. Vercel does not use this file.

### 1.3 On-prem stack (`docker-compose.yml`, DOCBOT-605)

Services: `postgres` (15-alpine, healthcheck `pg_isready`), `backend` (Dockerfile, healthcheck `curl /api/health`), `frontend` (Dockerfile.frontend, `RAILWAY_BACKEND_URL=http://backend:8000`), optional `ollama` (profile `ollama`). Bridge network `docbot_net`, volumes `postgres_data`, `ollama_data`. `healthcheck.sh` checks postgres/backend/frontend/ollama and exits non-zero on any failure.

Drift: the compose file sets `DOCBOT_LLM_PROVIDER`, `OLLAMA_BASE_URL` and `OLLAMA_MODEL`, but no file under `api/` reads any of them (grep, 2026-11-23). The real LLM switch is `LLM_PRIMARY` (`groq` default, or `bedrock`) in `api/utils/llm_provider.py:687`. The Ollama profile therefore has no effect on the backend (inferred from the grep).

## 2. Startup and shutdown (FastAPI lifespan)

Module import (`api/index.py`):

| Order | Step | Location |
|-------|------|----------|
| 1 | `load_dotenv()` then `pin_langsmith_env()`, before any other import, so LangChain auto-tracing is forced off | `api/index.py:6-9` |
| 2 | `init_sentry()` (no-op without `SENTRY_DSN`) | `api/index.py:45-46` |
| 3 | CORS middleware from `parse_allowed_origins()`; `abuse_guard` HTTP middleware (write freeze, demo blocking, per-IP rate limit, anon cookie); `SecurityHeadersMiddleware` outermost | `api/index.py:52-136` |
| 4 | Exit with code 1 if `DATABASE_URL` unset; async engine `pool_size=5, max_overflow=10, pool_pre_ping=True` | `api/index.py:147-166` |
| 5 | SQLAlchemy `Table` definitions and `register_*_table(metadata)` calls | `api/index.py:170-440` |

`lifespan()` (`api/index.py:597-784`), in order:

1. `assert_safe_auth_config()` (refuses multi-tenant boot in open mode, DOCBOT-1610) and `assert_safe_cors_config()` (refuses `*` with credentials, DOCBOT-1624).
2. `init_db()` (`api/index.py:512-594`): `metadata.create_all`, then idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` migrations, the audit-log immutability trigger, legacy identity and `email_canonical` backfill plus unique index, hashing of legacy session tokens, default-organization backfill, billing columns, credit-ledger immutability trigger, funnel dedupe indexes. There is no Alembic; schema evolution is these in-code idempotent statements.
3. Rate-limit counters switch to Postgres (`request_guard.SqlCounterStore`).
4. Tool registry, RBAC, demo-session lookup, commerce, connector store, EDGAR cache, envelope encryption, LLM trace writer, usage writer, spend guard, funnel, credit store, lineage, LLM cache, agent traces, ontology, eval, API keys are wired.
5. `credit_service.run_reaper_job()` releases credit reservations orphaned by the previous process.
6. `cleanup_expired_uploads()` (non-fatal).
7. Marketplace connectors restored from `marketplace_connections` into `app.state.connectors` (non-fatal per connector).
8. One `AsyncIOScheduler` (APScheduler) started with these jobs:

| Job | Registered at | Cadence |
|-----|---------------|---------|
| Per-connector background sync (DOCBOT-704) | `api/connectors/sync_jobs.py:154,167,216` | per connector (see that file) |
| Commerce `raw_json` retention cleanup (DOCBOT-1526) | `api/commerce_service.py:512-525` | recurring |
| Stale credit-reservation reaper (DOCBOT-1802) | `api/credit_service.py:786-788` | recurring |
| Vector-store retention cleanup (DOCBOT-1525) | `api/index.py:762-769` | daily, `max_instances=1` |

9. Vector-store retention cleanup runs once at startup (`VECTOR_STORE_RETENTION_DAYS`), then the in-memory `VECTOR_STORES` LRU is warmed from Chroma disk up to its cap (`api/index.py:738`, DOCBOT-2311).

Shutdown (`api/index.py:771-784`): scheduler stop, LLM-call writer drain, usage writer drain, LangSmith flush bounded to 3 s in a thread, `engine.dispose()`.

Single process: one uvicorn worker, no Celery or external queue. Background work is APScheduler plus in-process asyncio tasks (inferred from the CMD and the lack of worker config).

## 3. Environment variables (names only)

Collected with `grep -rnoE "os\.(getenv|environ...)\(['\"]NAME" api` plus the helper readers `_env_int`, `_env_bool`, `_env_num`, `_env_flag`, `_env_true`, and `process.env.*` in `src/`, `next.config.ts` and `security-headers.mjs`. Values are never documented here.

### 3.1 Backend core

| Variable | Purpose | First read at |
|----------|---------|---------------|
| `DATABASE_URL` | App PostgreSQL; process exits if unset | `api/index.py:147` |
| `DB_ENCRYPTION_KEY` | Fernet key for credential blobs | `api/utils/encryption.py:9` |
| `ALLOWED_ORIGINS` | CORS allowlist | `api/utils/cors_config.py`, `api/oauth_service.py:53` |
| `APP_BASE_URL`, `FRONTEND_URL` | Absolute links and OAuth redirects | `api/index.py:2711`, `api/index.py:2632` |
| `ENV`, `APP_ENV`, `ENVIRONMENT`, `RAILWAY_ENVIRONMENT`, `RAILWAY_ENVIRONMENT_NAME`, `ECS_CONTAINER_METADATA_URI_V4` | Deployed-environment detection | `api/rbac_service.py:253-266`, `api/index.py:1669` |
| `CHROMA_DIR` / `CHROMA_PERSIST_DIR` | Chroma persistence path | `api/utils/vector_store.py:72` |
| `VECTOR_STORE_CACHE_MAX`, `VECTOR_STORE_RETENTION_DAYS` | In-memory LRU size and retention window | `api/utils/vector_store.py:101,119` |
| `COMMERCE_RAW_JSON_RETENTION_DAYS` | Commerce raw payload retention | `api/commerce_service.py:86` |

### 3.2 LLM, embeddings, sandbox

| Variable | Purpose | First read at |
|----------|---------|---------------|
| `groq_api_key` | Groq primary LLM | `api/utils/llm_provider.py` and 5 services |
| `GEMINI_API_KEY` | Gemini fallback, LangExtract, EDGAR extraction | `api/utils/llm_provider.py`, `api/index.py:1108` |
| `LLM_PRIMARY` | `groq` (default) or `bedrock` | `api/utils/llm_provider.py:687` |
| `BEDROCK_API_KEY`, `BEDROCK_REGION`, `BEDROCK_MODEL`, `BEDROCK_CODE_MODEL`, `BEDROCK_FALLBACK_MODEL`, `BEDROCK_EXTRACT_MODEL` | AWS Bedrock (OpenAI-compatible endpoint) when `LLM_PRIMARY=bedrock` | `api/utils/llm_provider.py:683-725`, `api/document_extractor.py:362` |
| `SESSION_COST_CEILING_USD` | Per-run soft LLM cost ceiling | `api/utils/llm_provider.py:249` |
| `EMBEDDINGS_PROVIDER`, `huggingface_api_key`, `HUGGINGFACEHUB_API_TOKEN` | Local ONNX (default) or HF API embeddings | `api/utils/embeddings_provider.py:98-99` |
| `RERANKER_PROVIDER` | `local` / `hf` / `off` | `api/utils/reranker.py:108` |
| `SEMANTIC_CHUNK_MAX_CHARS` | Chunker cap | `api/utils/chunker.py:42` |
| `E2B_API_KEY`, `E2B_SANDBOX_TEMPLATE`, `SANDBOX_PIP_INTERNET` | E2B sandbox | `api/sandbox_service.py:350`, `api/utils/sandbox_policy.py:66,117` |
| `ALLOW_ANONYMOUS_SANDBOX` | Allow sandbox for anonymous callers | `api/rbac_service.py:270` |
| `AUTOPILOT_TIMEOUT_SECONDS`, `AUTOPILOT_STEP_TIMEOUT_SECONDS` | Autopilot run and step timeouts | `api/autopilot_service.py:64-65` |

### 3.3 Auth and identity

| Variable | Purpose | First read at |
|----------|---------|---------------|
| `AUTH_REQUIRED`, `APP_AUTH_MODE`, `MULTI_TENANT` | Auth enforcement mode | `api/auth_service.py:119,136`, `api/rbac_service.py:341` |
| `SESSION_TTL_HOURS` | Session cookie TTL | `api/auth_service.py:33` |
| `PLATFORM_ADMIN_USER_IDS` | Cross-org platform admins | `api/rbac_service.py:169` |
| `OWNER_KEY` | Owner bypass of abuse guards | `api/utils/request_guard.py:149` |
| `SAML_SP_ENTITY_ID`, `SAML_SP_ACS_URL`, `SAML_IDP_ENTITY_ID`, `SAML_IDP_SSO_URL`, `SAML_IDP_X509_CERT`, `SAML_SP_X509_CERT`, `SAML_SP_PRIVATE_KEY`, `SAML_WANT_ASSERTIONS_SIGNED`, `SAML_WANT_NAME_ID_ENCRYPTED`, `SAML_DEBUG` | SAML 2.0 SSO | `api/auth_service.py:56-96` |
| `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | OAuth | `api/oauth_service.py:56-59` |
| `RESEND_API_KEY`, `EMAIL_FROM` | Transactional email (verification, waitlist) | `api/email_verification_service.py`, `api/waitlist_service.py:104-113` |
| `TURNSTILE_SECRET_KEY` | Cloudflare Turnstile server check | `api/email_verification_service.py:131` |
| `WAITLIST_NOTIFY_EMAIL` | Waitlist notification recipient | `api/waitlist_service.py:105` |

### 3.4 Abuse guards, limits, billing

| Variable | Purpose | First read at |
|----------|---------|---------------|
| `PUBLIC_DEMO_MODE`, `WRITE_FREEZE`, `GATE_UPLOADS` | Demo path blocking, maintenance write freeze, sign-in gate for uploads | `api/utils/request_guard.py:100,104,140` |
| `RATE_LIMIT_PER_MIN`, `SESSION_BUDGET`, `USER_DAILY_BUDGET`, `TRUSTED_PROXY_HOPS`, `LOGIN_FAILURES_PER_WINDOW` | Per-IP / session / user limits | `api/utils/request_guard.py:105-141` |
| `DEMO_INIT_PER_IP_PER_HOUR`, `DEMO_INIT_GLOBAL_PER_HOUR`, `DEMO_INIT_PER_USER_PER_HOUR`, `DEMO_DAILY_USD_CAP` | Demo limits | `api/utils/request_guard.py:112-114,448` |
| `EDGAR_PER_IP_PER_HOUR`, `EDGAR_PER_USER_PER_HOUR`, `EDGAR_BATCH_MAX` | EDGAR limits | `api/utils/request_guard.py:118-120` |
| `SANDBOX_RUNS_PER_HOUR` | Sandbox limit | `api/utils/request_guard.py:124` |
| `PLATFORM_DAILY_USD_CAP` | Platform-wide daily spend cap (503 when hit) | `api/utils/spend_guard.py:59` |
| `UPLOAD_MAX_FILE_MB`, `UPLOAD_MAX_FILES`, `UPLOAD_PDF_MAX_PAGES`, `UPLOAD_PDF_PARSE_TIMEOUT_SECONDS`, `UPLOAD_SQLITE_MAX_MB`, `UPLOAD_CSV_MAX_MB` | Upload limits | `api/utils/upload_limits.py:28-33` |
| `BILLING_ENFORCE`, `BILLING_MARKUP`, `BILLING_FREE_MONTHLY_CREDITS`, `BILLING_FREE_RUN_CEILING_CREDITS`, `BILLING_RESERVATION_TTL_MINUTES`, `BILLING_EST_CREDITS_<ROUTE>` | Credit metering | `api/credit_service.py:151-187` |
| `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_PACKS_JSON` | Stripe checkout and webhook | `api/billing_service.py:64-190` |

### 3.5 Observability

| Variable | Purpose | First read at |
|----------|---------|---------------|
| `SENTRY_DSN`, `SENTRY_ENVIRONMENT` | Sentry error tracking | `api/utils/sentry_setup.py:89,96` |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY` | DocBot's own metadata-only LangSmith runs | `api/utils/langsmith_pin.py:53`, `api/utils/langsmith_tracing.py:84-85` |
| `LANGCHAIN_TRACING_V2`, `LANGSMITH_TRACING_V2`, `LANGCHAIN_TRACING`, `LANGCHAIN_HANDLER` | Forced off / removed at boot | `api/utils/langsmith_pin.py:15-19` |

### 3.6 Frontend (Vercel)

| Variable | Purpose | Read at |
|----------|---------|---------|
| `BACKEND_URL` (legacy `RAILWAY_BACKEND_URL`) | Rewrite target and CSP `connect-src` | `security-headers.mjs` `backendUrlFromEnv`, `next.config.ts:5` |
| `DOCKER_BUILD` | Standalone output for the Docker image | `next.config.ts:19` |
| `NODE_ENV`, `VERCEL_ENV` | Dev CSP (`unsafe-eval`), preview toolbar origins | `src/proxy.ts:9-10` |
| `NEXT_PUBLIC_SITE_URL` | Canonical site URL / metadata | `src/lib/site.ts:6` |
| `NEXT_PUBLIC_TURNSTILE_SITE_KEY` | Turnstile widget | `src/app/chat/page.tsx:218` |
| `NEXT_PUBLIC_AZURE_CLIENT_ID` | MSAL sign-in for Azure SQL Entra | `src/components/ConnectionPanel.tsx:290` |
| `NEXT_PUBLIC_COMPANY_NAME`, `NEXT_PUBLIC_CONTACT_EMAIL` | Legal pages | `src/components/LegalPage.tsx:10-11` |

## 4. CI/CD workflows (`.github/workflows/`)

| Workflow | Triggers | Jobs | Blocking? |
|----------|----------|------|-----------|
| `ci.yml` | push to any branch; PR to `main` | `test`: Python 3.12, `pip install -r requirements-dev.txt`, `pytest tests/ -v -m "not external and not postgres"` with a CI Fernet secret and a fake `DATABASE_URL`; uploads `.pytest_cache` on failure. `frontend`: Node 20, `npm ci`, `npx tsc --noEmit`, `npm run lint` | yes |
| `security.yml` (DOCBOT-1625) | push any branch; PR to `main`; weekly Mon 06:17 UTC | `bandit -r api -ll` (blocking); `pip-audit -r requirements.txt` with four per-ID chromadb ignores (Chroma HTTP-server CVEs, not reachable since Chroma runs in-process; revisit 2026-12-01) (blocking); `npm audit --audit-level=high` (`continue-on-error`, known baseline); `gitleaks detect` over full history with a checksum-verified CLI (blocking) | mixed |
| `nightly-eval.yml` (DOCBOT-1503) | cron 07:00 UTC daily; manual | `pytest tests/eval -v -m external`. Gate: `test_retrieval_eval.py::test_retrieval_recall` asserts `recall[5] >= 0.7` | n/a (scheduled) |
| `openproject-sync.yml` (DOCBOT-2586) | branch create; PR opened/reopened/ready/closed; push to `main` touching backlog/stories/sync scripts; manual | Moves DOCBOT tickets on the OpenProject board (branch created → In Progress, PR open → Testing, merged to `main` → Done, merged to `dev` → Testing) or runs a full sync. Checks out `main` (never the PR branch) because the step holds the API key; serialized by a concurrency group; skips with a warning when the secret is missing | never a required check |

Deploys are not driven by Actions. Railway builds from the Dockerfile and Vercel builds Next.js through their own Git integrations (inferred: no deploy job exists in any workflow). CI does not run `next build` or any frontend unit tests.

## 5. Tests

`pytest.ini`: `asyncio_mode = auto`, `testpaths = tests`, markers `unit`, `integration`, `external`, `postgres`.

| Directory | Files (`test_*.py`) | Notes |
|-----------|---------------------|-------|
| `tests/unit/` | 110 | Pure logic, no network |
| `tests/integration/` | 15 | SQLite / temp files, no external APIs |
| `tests/eval/` | 3 (+ `run_eval_batch.py`, `eval_latency.py`, gold sets, `README.md`) | Discrepancy eval runs in CI; retrieval/tie-out evals are `external` (nightly) |
| `tests/external/` | 2 | Live API keys, skipped in CI |
| `tests/conftest.py` | n/a | Shared fixtures |

Collected on 2026-11-23 with `DATABASE_URL=<fake> python -m pytest --collect-only -q -m "not external and not postgres"` (Python venv from the main checkout, run against this worktree): **2717 of 2757 tests selected, 40 deselected** (external + postgres).

Dev dependencies: `requirements-dev.txt` (`-r requirements.txt`, pytest 9.2, pytest-asyncio 1.5, aiosqlite 0.22). Frontend scripts: `dev`, `build`, `start`, `lint` (`package.json`); no frontend test runner.

## 6. Observability

| Signal | Mechanism | Storage / sink | Privacy rule |
|--------|-----------|----------------|--------------|
| Errors | Sentry SDK, `traces_sample_rate=0.0`, `send_default_pii=False`, no local vars, no request bodies; `before_send` / `before_breadcrumb` scrub URL credentials, key=value secrets, and Groq/Stripe/OpenAI/Google/E2B key patterns (`api/utils/sentry_setup.py:14-118`) | Sentry (only if `SENTRY_DSN`) | secrets scrubbed |
| LLM call log | `llm_provider._log_llm_call` → `enqueue_call` (thread-safe `queue.Queue`) → single asyncio writer (`api/llm_trace_service.py:132-223`). Drops oldest on overflow, never back-pressures requests | `llm_calls` table | metadata only, no prompt/response text |
| Agent traces + feedback | `log_trace` fire-and-forget task; `record_feedback` from `POST /api/traces/{trace_id}/feedback` (`api/trace_service.py:147,209`) | `agent_traces` table | routing metadata |
| Answer lineage | `api/lineage_service.py` | `answer_lineage` table | served to the Inspector UI |
| LangSmith | Off unless `LANGSMITH_TRACING` and `LANGSMITH_API_KEY` are both set. One root run per DocBot `run_id`, child `llm` runs with empty inputs/outputs and allowlisted metadata (`api/utils/langsmith_tracing.py:1-30,76-103`). LangChain auto-tracing pinned off at import (`api/utils/langsmith_pin.py`) | LangSmith | metadata only |
| Admin metrics | `GET /admin/metrics` → `metrics_service.get_platform_metrics` + `get_llm_cost_metrics` (cost, latency percentiles, tokens from `llm_calls`) (`api/metrics_service.py:90,186`) | PostgreSQL queries | aggregates |
| Activation funnel | `record_funnel_event` fire-and-forget, deduped by unique indexes; `GET /admin/funnel` report (`api/funnel_service.py:97,177`) | `funnel_events` table | user/anon id only |
| Audit log | Append-only `audit_log` with DB immutability trigger (`api/index.py:524-526`) | `audit_log` table | see [Security Architecture](architecture-06-security-architecture) |
| Usage ledger | `api/usage_service.py` background writer | `usage_events` table | per-org metering |
| Health | `GET /api/health` (DB ping) used by Railway and docker-compose | n/a | n/a |
| Frontend | `@vercel/analytics` `<Analytics />` in `src/app/layout.tsx:68` | Vercel | n/a |

There is no APM/tracing backend, no metrics exporter (Prometheus/OTel), and no log shipping beyond the host's stdout capture (inferred from the absence of such dependencies in `requirements.txt` and `package.json`).

## 7. Operational controls

| Control | How | Location |
|---------|-----|----------|
| Maintenance write freeze | `WRITE_FREEZE` → 503 + `Retry-After: 600` for writes, owner included | `api/index.py:79-85` |
| Public demo lock-down | `PUBLIC_DEMO_MODE` blocks upload/DB paths (403) | `api/index.py:90-96` |
| Per-IP rate limit | POST/DELETE on `/api/*` and `/v1/*`, Postgres-backed fixed windows (`rate_limit_counters`) | `api/index.py:98-113`, `api/utils/request_guard.py` |
| Platform spend cap | `_spend_cap` dependency → 503 `platform_capacity` | `api/utils/spend_guard.py` |
| Owner bypass | `docbot_owner` cookie (derived from `OWNER_KEY`, set via `POST /api/auth/owner-unlock`, compared with `hmac.compare_digest`; raw key never accepted as header/query) skips guards, not the write freeze | `api/utils/request_guard.py:149,352-363` |
| Graceful shutdown | 25 s uvicorn grace, bounded LangSmith flush | `Dockerfile` CMD, `api/index.py:771-784` |

## 8. OpenProject PM server

Source: `scripts/README-openproject.md`.

| Item | Value |
|------|-------|
| Purpose | Self-hosted ticket board (project `docbot`), replaced Plane; fed by `scripts/openproject_sync.py` and `.github/workflows/openproject-sync.yml` |
| Host | AWS Lightsail, us-east-1, 2 GB plan, static IP, DNS `pm` A record managed in Vercel |
| Firewall | 22, 443 (80 for certificate renewal) |
| Stack | `/opt/openproject/docker-compose.yml`: `openproject/openproject:17` all-in-one (bundled Postgres) on loopback 8080, `caddy:2` terminating HTTPS; 3 GB swapfile |
| Secrets | Server `.env` (root only); API token in `OPENPROJECT_API_KEY` (GitHub secret / macOS Keychain locally). Not reproduced here |
| Backups | Lightsail automatic snapshots daily, last 7 kept |
| Ops | `sudo docker compose up -d` to restart; snapshot before image upgrades; resize to 4 GB if pages exceed 5 s or the container restarts |
| Sync script | `--self-test`, `--dry-run`, apply (reads `origin/main`), `--set-state DOCBOT-XXXX <state>`; status never moves backwards on full sync |

## 9. Gaps and risks

- No migration tool: schema changes are idempotent DDL inside `init_db()` on every boot (`api/index.py:512-594`). A failing statement blocks startup.
- Chroma data lives on the container volume. Without a mounted persistent volume at `CHROMA_DIR`, uploaded-document vector stores are lost on redeploy (stated in `api/utils/vector_store.py:5-17`).
- Single uvicorn process holds in-memory state (`VECTOR_STORES`, `app.state.connectors`, APScheduler). Horizontal scaling would duplicate scheduler jobs and split caches (inferred).
- `docker-compose.yml` LLM settings (`DOCBOT_LLM_PROVIDER`, Ollama) are not read by the backend.
- `npm audit` job is non-blocking with a known critical/high baseline (`security.yml` header).
