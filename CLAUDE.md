# DocBot — Claude Code Context

## What This Project Is

DocBot is an AI-powered document + database analyst. Currently deployed on Vercel as a PDF chat tool with 7 expert personas (v1). Being extended into a full "Friendly Data/Business Analyst" product (v2) that can:
- Chat with uploaded PDFs (existing)
- Connect to live databases and generate SQL
- Run Python analysis via E2B sandboxes
- Answer hybrid questions spanning both docs and live data in one answer with dual citations

**The core differentiator:** Hybrid Docs+DB synthesis with discrepancy detection. No other tool on the market does this.

## Current Branch

`v2` — all new work goes here. `main` is the stable Vercel-deployed v1.

## Key Files

| File | Purpose |
|------|---------|
| `api/index.py` | FastAPI backend (~940 lines). Contains EXPERT_PERSONAS, VECTOR_STORES, init_db(), all routes |
| `src/app/page.tsx` | Monolithic React frontend (~1800 lines). All UI state via useState |
| `requirements.txt` | Python dependencies |
| `vercel.json` | Routes all /api/* to api/index.py |
| `project-tasks/docbot-db-master-plan.md` | Full architecture, security model, phased build plan |
| `project-tasks/docbot-v2-project-tracking.md` | 32 user stories, 6 epics, sprint plan, Definition of Done |

## Architecture: Current (v1)

```
Next.js 16 (Vercel) → FastAPI (Vercel Serverless) → Groq (Llama 3.3-70b)
                                ↓
              LangChain + InMemoryVectorStore + PyMuPDF
                                ↓
              SQLite at /tmp/docbot_sessions.db (lost on cold start)
```

## Architecture: Target (v2)

```
Next.js 16 (Vercel) → FastAPI (Railway container) → Groq / Claude claude-sonnet-4-6
                                ↓
         LangChain RAG  +  SQLAlchemy DB  +  E2B Python Sandbox
                                ↓
              PostgreSQL (Railway) — persistent session + schema store
```

## Tech Stack

- **Frontend**: Next.js 16, React 19, TailwindCSS 4, TypeScript, lucide-react, react-markdown
- **Backend**: FastAPI, Python 3.12, Groq (Llama 3.3-70b), LangChain, PyMuPDF
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 via HuggingFace API
- **Storage**: SQLite /tmp (v1), PostgreSQL on Railway (v2)
- **Deployment**: Vercel (frontend + current backend), Railway (v2 backend)

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
| `ALLOWED_ORIGINS` | No | Comma-separated CORS origins, defaults to localhost:3000 |
| `DB_ENCRYPTION_KEY` | v2 | Fernet key for credential encryption — never hardcode |
| `DATABASE_URL` | v2 | Railway PostgreSQL connection string |
| `E2B_API_KEY` | v2 | E2B sandbox API key |
| `GEMINI_API_KEY` | v2 | Gemini API key for LangExtract financial extraction (free tier works; set max_workers=2) |

Never commit `.env`. Never log credentials. Never pass connection strings to LLM context.

## Branch Naming Convention

```
feature/DOCBOT-XXX-short-description
```

Example: `feature/DOCBOT-101-railway-migration`

One ticket per branch. One branch per session when possible.

## Ticket Reference

All work is tracked against tickets in `project-tasks/docbot-v2-project-tracking.md`.

Sprint 0 priorities (do these first, in order):
1. **DOCBOT-101** — Railway migration (Dockerfile, railway.toml, health check)
2. **DOCBOT-102** — PostgreSQL session store (migrate from SQLite /tmp)
3. **DOCBOT-103** — E2B sandbox integration

Phase 1 starts after Sprint 0 and 5 discovery interviews are complete.

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

## Vercel Constraints (current backend)

| Constraint | Limit |
|-----------|-------|
| Execution time | 30s max |
| Writable filesystem | `/tmp` only |
| Bundle size | 250MB |
| Process memory | Per-invocation (no persistent state) |

`VECTOR_STORES = {}` is in-memory and lost on every cold start. This is a known limitation addressed by the Railway migration.

## SQL Query Pipeline (7 bounded steps)

```
NL Question
  → [1] Schema Retrieval     (cache → miss → introspect)
  → [2] Table Selector       (LLM call #1)
  → [3] Few-Shot Retrieval   (cosine similarity on stored queries)
  → [4] SQL Generator        (LLM call #2)
  → [5] SQL Validator        (sqlglot AST — DETERMINISTIC, no LLM)
  → [6] Executor             (SQLAlchemy, 15s timeout, 500 row cap)
  → [7] Answer Generator     (LLM call #3, Groq streaming)
```

Max 3 LLM calls. No loops.

## LLM Routing (v2)

- **Groq Llama 3.3-70b**: SQL generation, hybrid synthesis, intent classification (speed priority, ~82% accuracy on business queries)
- **Groq Qwen/qwen3-32b**: Python code generation for E2B sandboxes (free, coding-optimised)
- **Gemini 2.5 Flash via LangExtract**: Financial document extraction (full-document chunked extraction with char_interval source grounding)

## Code Style

- Python: No type: ignore, no bare except. Use specific exception types.
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
- `api/db_service.py` — all database connectivity, schema, SQL pipeline logic
- `api/sandbox_service.py` — E2B sandbox logic + Python code generation (Qwen via Groq)
- `api/hybrid_service.py` — intent classification, RAG retrieval, hybrid chat pipeline
- `api/document_extractor.py` — LangExtract financial extraction (Gemini 2.5 Flash, full-doc coverage)
- `api/utils/` — shared helpers (embeddings, encryption, validators, etc.)

**Rule:** If a function is more than ~15 lines or has no direct dependency on the HTTP request/response, it belongs in a service or utils module, not in index.py.

Always prefer to create utils and helper functions whenever required.


### Commit
Always have Author : Sanshrit Singhai
Never use Co-Authored By Claude

## Jira Sync Rule

**Whenever a DOCBOT ticket is implemented and its branch is merged into `main`, immediately update the corresponding Jira ticket to Done.**

- Jira project: `https://dbdocbot.atlassian.net` — project key `SCRUM`
- Credentials: stored in `scripts/jira_update_status.py` (EMAIL + API_TOKEN)
- Done transition ID: `51`
- Also transition all subtasks of the story to Done

**How to update:**
1. Find the SCRUM key by searching Jira for the story summary (e.g. `DOCBOT-201`)
2. POST to `/rest/api/3/issue/{key}/transitions` with `{"transition": {"id": "51"}}`
3. Repeat for all child subtasks via `/rest/api/3/search/jql` with `parent={key}`
4. If the ticket has a phase/priority correction needed, add a comment explaining the change

Use `scripts/jira_update_status.py` as a reference implementation. Add new ticket mappings to it as work is completed.

## README Update Rule

**Whenever a Phase is fully completed and merged to `main`, update `README.md` to reflect the current state of the product.**

The README must always accurately describe what is actually shipped — not what is planned. On phase completion:
1. Update the Features section to include all newly shipped capabilities
2. Update the API Reference with any new routes
3. Update the Tech Stack table if new dependencies were added
4. Update the Environment Variables table if new vars are required
5. Move completed items out of Roadmap and into the main feature sections
6. Update the test count badge and test table

Do not update the README for individual ticket completions within a phase — only on full phase completion or for significant standalone features (like DOCBOT-208 which added a new dialect).