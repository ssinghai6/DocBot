# DocBot — Claude Code Context

## What This Project Is

DocBot is an AI-powered document + database analyst, fully deployed on Railway (backend) + Vercel (frontend). It can:
- Chat with uploaded PDFs using 8 expert personas with smart auto-routing
- Connect to live databases (PostgreSQL, MySQL, SQLite, Azure SQL) and generate SQL
- Upload CSV files — queries run via E2B pandas sandbox (no SQL pipeline)
- Run Python analysis via E2B sandboxes with matplotlib chart capture
- Answer hybrid questions spanning both docs and live data with dual citations and discrepancy detection
- Run multi-step deep research using a LangGraph 5-node state machine
- Authenticate via SAML SSO (Okta, Azure AD), GitHub OAuth, Google OAuth, or email/password
- Enforce RBAC (viewer / analyst / admin), audit logging, and PII auto-masking

**The core differentiator:** Hybrid Docs+DB synthesis with discrepancy detection. No other tool on the market does this.

## Current Branch

`main` — all work goes here. Railway auto-deploys from `main` on push.

## Key Files

| File | Purpose |
|------|---------|
| `api/index.py` | FastAPI backend (~1700 lines). App setup, EXPERT_PERSONAS, init_db(), all route handlers |
| `api/db_service.py` | All DB connectivity, schema introspection, 7-step SQL pipeline |
| `api/sandbox_service.py` | E2B sandbox execution, Python/pandas code generation (Qwen via Groq), CSV→E2B pipeline |
| `api/hybrid_service.py` | Intent classification, parallel RAG+SQL retrieval, discrepancy detection |
| `api/autopilot_service.py` | Analytical Autopilot — LangGraph multi-step investigation state machine |
| `api/deep_research_service.py` | LangGraph Deep Research — 5-node state machine (plan→retrieve→evaluate→gap→synthesize) |
| `api/auth_service.py` | SAML 2.0 SSO, session management, JIT user provisioning |
| `api/oauth_service.py` | GitHub OAuth, Google OAuth, email/password auth with bcrypt |
| `api/rbac_service.py` | RBAC — viewer/analyst/admin roles, `require_role()` FastAPI dependency |
| `api/audit_service.py` | Append-only audit log, PostgreSQL immutability trigger, CSV export |
| `api/artifact_service.py` | Session artifact store — persists charts, code, SQL results |
| `api/document_extractor.py` | LangExtract financial extraction (Gemini 2.5 Flash, full-doc coverage) |
| `api/file_upload_service.py` | CSV/SQLite file uploads — CSV goes to E2B pandas, SQLite to SQL pipeline |
| `api/utils/` | Shared helpers: encryption, SSRF validator, SQL validator, embeddings, PII masking, table selector, context compressor |
| `src/app/page.tsx` | Monolithic React frontend (~1800 lines). All UI state via useState |
| `requirements.txt` | Python dependencies |
| `project-tasks/docbot-v2-project-tracking.md` | 38 user stories, 9 epics, sprint plan, Definition of Done — **primary ticket tracker** |
| `project-tasks/docbot-db-master-plan.md` | Full architecture, security model, phased build plan |

## Current Architecture (v2 — live on Railway)

```
Next.js 16 (Vercel) → FastAPI (Railway container) → Groq / Gemini
                                ↓
         LangChain RAG  +  SQLAlchemy DB  +  E2B Python Sandbox
                 ↓                ↓                   ↓
         HuggingFace        PostgreSQL           LangGraph
         Embeddings         (Railway)         (Autopilot + Deep Research)
```

## Tech Stack

- **Frontend**: Next.js 16, React 19, TailwindCSS 4, TypeScript, lucide-react, react-markdown
- **Backend**: FastAPI, Python 3.12, Groq (Llama 3.3-70b), LangChain, PyMuPDF
- **AI/ML**: Groq Qwen/qwen3-32b (code gen), Gemini 2.5 Flash via LangExtract (financial docs), LangGraph (agentic flows)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 via HuggingFace API
- **Storage**: PostgreSQL on Railway (sessions, connections, audit log, schema cache)
- **Sandbox**: E2B code-interpreter (Python/pandas execution, matplotlib charts)
- **Auth**: python3-saml (SAML 2.0), httpx (OAuth), bcrypt (passwords), azure-identity (Entra)
- **DB Connectors**: asyncpg (PostgreSQL), pymysql (MySQL), pyodbc (Azure SQL), aiosqlite (SQLite)
- **Deployment**: Vercel (Next.js frontend), Railway (FastAPI backend + PostgreSQL)

## Local Dev Commands

```bash
# Backend
source .venv/bin/activate
python3 -m uvicorn api.index:app --host 0.0.0.0 --port 8000 --reload

# Frontend (separate terminal)
npm run dev -- --port 3000
```

Next.js proxies `/api/*` to `localhost:8000` in development.

## Environment Variables

| Variable | Required | Notes |
|----------|----------|-------|
| `groq_api_key` | Yes | Groq API key |
| `huggingface_api_key` | Yes | For embeddings model |
| `DATABASE_URL` | Yes | Railway PostgreSQL connection string |
| `DB_ENCRYPTION_KEY` | Yes | Fernet key for credential encryption — never hardcode |
| `E2B_API_KEY` | Yes | E2B sandbox API key (CSV queries + Python analysis) |
| `GEMINI_API_KEY` | Yes | Gemini 2.5 Flash for LangExtract financial extraction |
| `ALLOWED_ORIGINS` | No | Comma-separated CORS origins, defaults to localhost:3000 |
| `SESSION_TTL_HOURS` | No | Session cookie TTL in hours, default 8 |
| `SAML_SP_ENTITY_ID` | SSO | SP entity ID for SAML 2.0 |
| `SAML_SP_ACS_URL` | SSO | SP Assertion Consumer Service URL |
| `SAML_IDP_ENTITY_ID` | SSO | IdP entity ID (from IdP metadata) |
| `SAML_IDP_SSO_URL` | SSO | IdP SSO redirect URL |
| `SAML_IDP_X509_CERT` | SSO | IdP public certificate (base64, no headers) |
| `GITHUB_CLIENT_ID` | OAuth | GitHub OAuth app client ID |
| `GITHUB_CLIENT_SECRET` | OAuth | GitHub OAuth app client secret |
| `GOOGLE_CLIENT_ID` | OAuth | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | OAuth | Google OAuth client secret |
| `OAUTH_REDIRECT_BASE_URL` | OAuth | Base URL for OAuth callback redirect |

Never commit `.env`. Never log credentials. Never pass connection strings to LLM context.

## Branch Naming Convention

```
feature/DOCBOT-XXX-short-description
```

Example: `feature/DOCBOT-801-smart-agent-routing`

One ticket per branch. One branch per session when possible.

## Ticket Reference

All work is tracked in `project-tasks/docbot-v2-project-tracking.md`.

**Current state (2026-03-25):**
- EPIC-01 through EPIC-06 — **Done** (archived)
- Consumer Auth (DOCBOT-701) — **Done**
- EPIC-08 Smart Agent Auto-Routing (DOCBOT-801–805) — **Done**
- EPIC-09 LangGraph Deep Research (DOCBOT-901–904) — **Done**
- **EPIC-10 RAG Quality Enhancement (DOCBOT-1001–1004) — Active work** (16 pts)
  - DOCBOT-1001: Chroma persistent store (replaces InMemoryVectorStore)
  - DOCBOT-1002: Cross-encoder reranker post-retrieval
  - DOCBOT-1003: SemanticChunker for financial/legal docs
  - DOCBOT-1004: FinanceBench accuracy baseline test suite
- **EPIC-07 Commerce Connectors (DOCBOT-702–705) — Gated**
  - Gate condition: begin after 3 of 5 discovery interviews confirm the commerce/seller segment

> **PageIndex evaluated 2026-03-25 — not integrating.** Hard blockers: OpenAI-only (Groq incompatible), not on PyPI (Railway brittleness), no streaming (SSE conflict). Revisit if PyPI package + multi-backend support ships.

## Critical Architecture Rules — DO NOT VIOLATE

**Never use `create_sql_agent()`** — it has unbounded LLM loops (3–8 calls typical). Use the 7-step bounded pipeline defined in the master plan instead.

**Never use the `vanna` package** — 150MB of deps (Chromadb, fastembed). Use sqlglot + existing embedding model.

**Never use regex for SQL validation** — trivially bypassed. Always use sqlglot AST parsing.

**3-layer read-only enforcement** (all three required):
1. LLM prompt: "Read-only queries ONLY"
2. sqlglot AST: reject any non-SELECT root statement
3. DB transaction: `SET TRANSACTION READ ONLY`

**SSRF prevention**: Validate all DB connection strings against RFC 1918 private addresses, loopback, and link-local before connecting.

**Credential protection**: Connection strings with passwords are NEVER logged, NEVER passed to LLM context, NEVER stored in plain text. Always Fernet-encrypt before persisting.

**CSV queries never use SQL pipeline**: CSV uploads store raw bytes in the encrypted creds blob (`dialect="csv"`). Queries bypass `run_sql_pipeline` entirely and go to `run_csv_query_on_e2b()` in `sandbox_service.py`.

## SQL Query Pipeline (7 bounded steps)

```
NL Question
  → [1] Schema Retrieval     (cache → miss → introspect)
  → [2] Table Selector       (semantic similarity → LLM fallback)
  → [3] Few-Shot Retrieval   (cosine similarity on stored queries)
  → [4] SQL Generator        (LLM call #1)
  → [5] SQL Validator        (sqlglot AST — DETERMINISTIC, no LLM)
  → [6] Executor             (SQLAlchemy, 15s timeout, 500 row cap)
  → [7] Answer Generator     (LLM call #2, Groq streaming)
```

Max 2–3 LLM calls. No loops. CSV dialect short-circuits before Step 1.

## LLM Routing

- **Groq Llama 3.3-70b**: SQL generation, hybrid synthesis, intent classification, answer generation
- **Groq Qwen/qwen3-32b**: Python/pandas code generation for E2B sandboxes (CSV queries + chart analysis)
- **Gemini 2.5 Flash via LangExtract**: Financial document extraction (full-document chunked extraction with char_interval source grounding)

## Code Style

- Python: No `type: ignore`, no bare `except`. Use specific exception types.
- TypeScript: No `any`. Use Zod for runtime validation on API responses.
- All new API routes need Pydantic models for request/response.
- All new frontend API calls need Zod schemas.
- No hardcoded secrets anywhere. Check with `grep -r "sk-\|api_key\s*=" --include="*.py" --include="*.ts"` before committing.

## Git Branch
When asked to work on new Epic or Feature always create a new git branch based on that feature.

## Testing

**All tests live in `tests/`** — never in root, never inline in service files.

```
tests/
  conftest.py              # shared pytest fixtures (Fernet key, temp DBs, mocked engine)
  unit/                    # pure logic, no network, no DB, no API keys — always run in CI
    test_ssrf_validator.py
    test_encryption.py
    test_sql_validator.py
    test_embeddings.py
    test_db_service_helpers.py
  integration/             # hit real SQLite / temp files — still run in CI (no external APIs)
    test_db_pipeline.py
    test_file_upload_service.py
```

**Rules:**
- Every new service module gets a matching test file in `tests/unit/` or `tests/integration/`.
- Tests that require external API keys (Groq, HuggingFace, E2B) are marked `@pytest.mark.external` and **skipped in CI**.
- Tests that require a live PostgreSQL are marked `@pytest.mark.postgres` and **skipped in CI**.
- Unit tests must never make network calls. If a function does I/O, mock it.
- Run tests locally: `pytest tests/ -v`
- Run only unit tests: `pytest tests/unit/ -v`
- CI command: `pytest tests/ -v -m "not external and not postgres"`

**Dev dependencies** are in `requirements-dev.txt` (not `requirements.txt`).

## Module Structure
**Never dump new logic into `api/index.py`.** That file is the FastAPI entrypoint only — it should contain:
- App setup (middleware, lifespan, engine, table definitions)
- Route handlers (thin — call service functions, return responses)

All business logic lives in dedicated service/util modules:
- `api/db_service.py` — DB connectivity, schema, 7-step SQL pipeline
- `api/sandbox_service.py` — E2B sandbox, Python/pandas code gen, CSV→E2B pipeline
- `api/hybrid_service.py` — intent classification, RAG retrieval, hybrid chat pipeline
- `api/autopilot_service.py` — Analytical Autopilot LangGraph state machine
- `api/deep_research_service.py` — Deep Research LangGraph 5-node state machine
- `api/auth_service.py` — SAML SSO, session management, JIT provisioning
- `api/oauth_service.py` — GitHub/Google OAuth, email/password auth
- `api/rbac_service.py` — role enforcement via FastAPI dependency
- `api/audit_service.py` — append-only audit log
- `api/artifact_service.py` — session artifact persistence
- `api/document_extractor.py` — LangExtract financial extraction
- `api/file_upload_service.py` — CSV/SQLite upload handling
- `api/utils/` — shared helpers (embeddings, encryption, validators, etc.)

**Rule:** If a function is more than ~15 lines or has no direct dependency on the HTTP request/response, it belongs in a service or utils module, not in index.py.

Always prefer to create utils and helper functions whenever required.

## Commit
Always have Author: Sanshrit Singhai
Never use Co-Authored-By Claude

## Jira Sync Rule

**Whenever a DOCBOT ticket is implemented and its branch is merged into `main`, immediately update the corresponding Jira ticket to Done.**

- Jira project: `https://dbdocbot.atlassian.net` — project key `SCRUM`
- Credentials: stored in `scripts/jira_update_status.py` (EMAIL + API_TOKEN)
- Done transition ID: `51`
- Also transition all subtasks of the story to Done
- Run `python scripts/jira_update_status.py --dry-run` first to verify, then without `--dry-run`
- Use `--epic EPIC-XX` to target a single epic

**How to update:**
1. Add the new ticket to the `TICKETS` list in `scripts/jira_update_status.py`
2. Run `python scripts/jira_update_status.py` (script auto-resolves SCRUM keys by summary search)

## README Update Rule

**Whenever a Phase is fully completed and merged to `main`, update `README.md` to reflect the current state of the product.**

The README must always accurately describe what is actually shipped — not what is planned. On phase completion:
1. Update the Features section to include all newly shipped capabilities
2. Update the API Reference with any new routes
3. Update the Tech Stack table if new dependencies were added
4. Update the Environment Variables table if new vars are required
5. Move completed items out of Roadmap and into the main feature sections
6. Update the test count badge and test table

Do not update the README for individual ticket completions within a phase — only on full phase completion or for significant standalone features.
