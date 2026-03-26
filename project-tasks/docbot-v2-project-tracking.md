# DocBot v2 — Complete Project Tracking Document
> Generated: 2026-03-17
> **Last Updated: 2026-03-25** (Audit pass + Fix #1: real discrepancy detection — numeric extraction engine, delta computation, 29 unit tests)
> Team size: 1–2 engineers
> Tracking tool recommendation: Linear (see Section 6)

---

## Table of Contents

1. [Universal Definition of Done](#1-universal-definition-of-done)
2. [Epic Structure](#2-epic-structure)
3. [User Stories with Tasks](#3-user-stories-with-tasks)
4. [Sprint Plan — Phase 0 + Phase 1](#4-sprint-plan--phase-0--phase-1)
5. [Tracking Tool Recommendation](#5-tracking-tool-recommendation)
6. [Risk Register Reference](#6-risk-register-reference)

---

## 1. Universal Definition of Done

Every story is only "done" when ALL of the following are true. No exceptions.

**Code Quality**
- [ ] Code is committed to the correct feature branch (naming: `feature/DOCBOT-XXX-short-description`)
- [ ] No linting errors (Python: `ruff` / TypeScript: `eslint`)
- [ ] No hardcoded secrets, connection strings, or API keys
- [ ] All new environment variables documented in `.env.example`

**Testing**
- [ ] Happy path manually tested end-to-end in local dev
- [ ] At least one explicit failure/error case tested (bad input, network timeout, etc.)
- [ ] For API routes: tested via `curl` or Postman with documented example request/response
- [ ] For UI components: tested in both mobile (375px) and desktop (1280px) viewports

**Security**
- [ ] No PII or credentials appear in logs
- [ ] All new API routes have input validation (Pydantic models on backend, Zod on frontend)
- [ ] SQL-touching code reviewed for injection risk (sqlglot AST validation in place)

**Documentation**
- [ ] Inline comments for any non-obvious logic
- [ ] New API routes documented with request/response shape in the story's Linear ticket
- [ ] `requirements.txt` updated if new Python packages added
- [ ] `package.json` updated if new npm packages added

**Deployment**
- [ ] Feature tested on Vercel preview deployment (frontend) or Railway staging (backend)
- [ ] No new Vercel function exceeds 250MB bundle size
- [ ] Response time for new API routes under 5 seconds for P95 (excluding E2B sandbox cold start)

---

## 2. Epic Structure

| Epic ID | Epic Name | Phase(s) | Status | Description |
|---------|-----------|----------|--------|-------------|
| EPIC-01 | Infrastructure Migration | 0 | ✅ Done | Move backend to Railway, PostgreSQL session store, E2B integration |
| EPIC-02 | Database Connectivity | 1, 3 | ✅ Done | DB connections, SQL generation pipeline, query execution |
| EPIC-03 | Analytical Loop (Python) | 1, 2 | ✅ Done | Python code execution via E2B, chart rendering, analysis |
| EPIC-04 | Hybrid Intelligence | 1, 2 | ✅ Done | Cross-source synthesis, discrepancy detection, planner/router |
| EPIC-05 | Memory and Context | 2 | ✅ Done | Session artifacts, context compression, multi-hop queries |
| EPIC-06 | Enterprise Readiness | 4 | ✅ Done | SSO, RBAC, audit logging, PII masking, Docker Compose, admin UI — all shipped |
| EPIC-07 | Commerce Connectors | 4+ | 🔄 Active (Phase 1) | Marketplace API integrations. **Phase 1 (unblocked):** Connector interface, unified commerce schema, Amazon SP-API (DOCBOT-701–703, 29 pts). **Phase 2 (post-funding):** Background sync worker, Shopify connector (DOCBOT-704–705, 16 pts). |
| EPIC-08 | Smart Agent Auto-Routing | 3 | ✅ Done | Replace static persona picker with intelligent per-question agent routing, per-agent badges and rendering |
| EPIC-09 | LangGraph Deep Research | 3+ | ✅ Done | Multi-step reasoning graph replacing single-shot Deep Research prompt — query planner, parallel retrieval, gap detection, streaming synthesis |
| EPIC-10 | RAG Quality Enhancement | 4+ | 🔄 Active | Chroma persistent store, cross-encoder reranker, SemanticChunker for financial/legal docs, FinanceBench accuracy baseline. PageIndex evaluated and rejected (2026-03-25). |

---

## 3. User Stories with Tasks

Story numbering format: `DOCBOT-[epic][sequence]`, e.g. DOCBOT-101 is Epic 1, story 1.

---

### EPIC-01: Infrastructure Migration

---

#### DOCBOT-101: Backend Migration to Railway

**Story**
As a developer, I want the FastAPI backend running on Railway as a persistent container, so that DB connections stay alive between requests and we are no longer limited by Vercel's 30-second function timeout.

**Phase**: 0
**Priority**: Must Have
**Story Points**: 5
**Dependencies**: None
**Status**: ✅ Done

**Acceptance Criteria**
- [ ] FastAPI server runs on Railway and returns 200 on GET /api/health
- [ ] Next.js frontend on Vercel successfully calls Railway backend API
- [ ] CORS configured correctly for Vercel domain
- [ ] All existing v1 routes (/api/chat, /api/upload) continue to work after migration
- [ ] Railway service auto-restarts on crash (health check configured)

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Create `Dockerfile` for FastAPI backend with all requirements | DevOps | 2h |
| 2 | Create `railway.toml` with start command, health check route, and env var references | DevOps | 1h |
| 3 | Add `GET /api/health` route to `api/index.py` | Backend | 0.5h |
| 4 | Update all `NEXT_PUBLIC_API_URL` references in frontend to point to Railway URL | Frontend | 1h |
| 5 | Update CORS `allow_origins` in FastAPI to include Vercel production domain | Backend | 0.5h |
| 6 | Smoke test all v1 routes end-to-end after migration; document any regressions | Full-stack | 2h |

---

#### DOCBOT-102: PostgreSQL Session Store

**Story**
As a developer, I want session data stored in PostgreSQL instead of SQLite at `/tmp`, so that sessions persist across server restarts and we can support multiple concurrent users.

**Phase**: 0
**Priority**: Must Have
**Story Points**: 5
**Dependencies**: DOCBOT-101
**Status**: ✅ Done

**Acceptance Criteria**
- [ ] A Railway-provisioned PostgreSQL instance is running
- [ ] All 5 session-related SQLite tables migrated to PostgreSQL with equivalent schema
- [ ] Existing v1 session history (message storage, retrieval) works correctly against PostgreSQL
- [ ] `init_db()` function creates tables idempotently on first run
- [ ] SQLite is no longer used anywhere in production code paths

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Provision PostgreSQL add-on on Railway; store `DATABASE_URL` in Railway env vars | DevOps | 0.5h |
| 2 | Add `asyncpg` and `sqlalchemy[asyncio]` to `requirements.txt` | Backend | 0.5h |
| 3 | Rewrite `init_db()` to use PostgreSQL DDL; keep SQLAlchemy ORM models portable | Backend | 3h |
| 4 | Replace all `sqlite3` calls with SQLAlchemy session management | Backend | 3h |
| 5 | Write migration script to copy existing local SQLite data to PostgreSQL for dev continuity | Backend | 1h |
| 6 | Test: create session, add messages, retrieve history — all against PostgreSQL | Backend | 1h |

---

#### DOCBOT-103: E2B Sandbox Integration

**Story**
As a developer, I want Python code execution routed through E2B sandboxes, so that user-generated code runs in an isolated environment with no risk to the host server.

**Phase**: 0
**Priority**: Must Have
**Story Points**: 8
**Dependencies**: DOCBOT-101
**Status**: ✅ Done

**Acceptance Criteria**
- [ ] E2B SDK installed and `E2B_API_KEY` loaded from environment
- [ ] A sandbox can be created, Python code executed, and output/error returned within 30 seconds
- [ ] `matplotlib` chart generation returns a PNG as base64 string
- [ ] Sandbox is always closed in a `finally` block (no sandbox leaks)
- [ ] Timeout of 25 seconds enforced; returns error to user if exceeded

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Add `e2b-code-interpreter` to `requirements.txt` | Backend | 0.5h |
| 2 | Create `api/sandbox_service.py` with `run_python(code: str) -> SandboxResult` wrapper | Backend | 3h |
| 3 | Implement `SandboxResult` Pydantic model: `{stdout, stderr, charts: list[str], error}` | Backend | 1h |
| 4 | Add 25-second timeout + `finally` sandbox cleanup | Backend | 1h |
| 5 | Write test: execute `import pandas as pd; print(pd.__version__)` and verify output | Backend | 1h |
| 6 | Write test: generate a matplotlib bar chart and verify base64 PNG returned | Backend | 1h |

---

### EPIC-02: Database Connectivity

---

#### DOCBOT-201: Database Connection API

**Story**
As Maya (Finance Manager), I want to connect my PostgreSQL database by entering a connection string, so that DocBot can query my live business data alongside my board deck.

**Phase**: 1
**Priority**: Must Have
**Story Points**: 8
**Dependencies**: DOCBOT-102
**Status**: ✅ Done

**Acceptance Criteria**
- [ ] POST /api/db/connect accepts dialect, host, port, dbname, user, password
- [ ] Connection is validated (test query runs successfully before storing)
- [ ] Credentials are Fernet-encrypted before storage; plain-text password never appears in logs
- [ ] SSRF prevention blocks RFC 1918 addresses (10.x, 172.16.x, 192.168.x), loopback, and link-local
- [ ] Returns `connection_id` and basic schema summary on success
- [ ] Unsupported dialects return a clear 400 error

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Create `DBConnectionRequest` Pydantic model with SSRF validator using `ipaddress` stdlib | Backend | 2h |
| 2 | Implement Fernet encryption: `encrypt_credentials()` and `decrypt_credentials()` in `api/db_service.py` | Backend | 2h |
| 3 | Implement `connect_database()`: validate → encrypt → store in `db_connections` table → introspect schema | Backend | 3h |
| 4 | Add `POST /api/db/connect` route to `api/index.py` | Backend | 0.5h |
| 5 | Add `DELETE /api/db/disconnect/{id}` route for cleanup | Backend | 0.5h |
| 6 | Test: attempt connection to `localhost` → expect SSRF block; test valid connection → expect `connection_id` | Backend | 1h |

---

#### DOCBOT-202: Schema Introspection and Cache

**Story**
As a developer, I want the schema of a connected database cached with TTL expiry, so that every query does not re-introspect a potentially large database schema.

**Phase**: 1
**Priority**: Must Have
**Story Points**: 5
**Dependencies**: DOCBOT-201
**Status**: ✅ Done

**Acceptance Criteria**
- [ ] On first connect, schema introspected and stored in `schema_cache` table with a 24-hour TTL
- [ ] Subsequent queries use cached schema; DB is not re-queried until TTL expires
- [ ] GET /api/db/schema/{id} returns table names, column names, and types
- [ ] Schema cache invalidated when user explicitly disconnects
- [ ] Handles databases with 100+ tables without context overflow (returns top 50 tables by row count if over limit)

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Implement `introspect_schema()` using SQLAlchemy `inspect()` — returns `{table: [{col, type}]}` | Backend | 3h |
| 2 | Add `schema_cache` table to PostgreSQL DDL with `expires_at` column | Backend | 1h |
| 3 | Implement cache read/write with TTL check in `get_schema()` function | Backend | 2h |
| 4 | Add `GET /api/db/schema/{id}` route | Backend | 0.5h |
| 5 | Handle large schemas: sort tables by information_schema row estimate, cap at 50 | Backend | 1h |

---

#### DOCBOT-203: SQL Generation Pipeline

**Story**
As Maya, I want to ask questions in plain English and receive a validated SQL query with a plain-English explanation, so that I can verify what the system is querying before trusting the answer.

**Phase**: 1
**Priority**: Must Have
**Story Points**: 13
**Dependencies**: DOCBOT-202
**Status**: ✅ Done

**Acceptance Criteria**
- [ ] 7-step pipeline completes in under 15 seconds for simple queries on a cached schema
- [ ] Generated SQL is validated via `sqlglot` AST: rejects any non-SELECT root statement
- [ ] Executor enforces a 500-row result cap and 15-second query timeout
- [ ] Response includes: generated SQL, plain-English explanation, result data, and row count
- [ ] If SQL validation fails, system returns a safe error with the reason (not the invalid SQL)
- [ ] Few-shot retrieval: if ≥3 similar past queries exist, they are included in the generation prompt

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Implement Step 1–2: schema retrieval + Groq LLM call for table selection | Backend | 3h |
| 2 | Implement Step 3: cosine similarity retrieval over `query_embeddings` using `all-MiniLM-L6-v2` | Backend | 2h |
| 3 | Implement Step 4: SQL generation prompt with schema + few-shot examples | Backend | 2h |
| 4 | Implement Step 5: `sqlglot` AST validator — reject non-SELECT, inject LIMIT if missing | Backend | 2h |
| 5 | Implement Step 6: SQLAlchemy executor with 15s statement timeout and 500-row cap | Backend | 2h |
| 6 | Implement Step 7: Groq LLM call for plain-English answer in persona voice | Backend | 1.5h |
| 7 | Store successful query + embedding in `query_history` and `query_embeddings` tables | Backend | 1h |

---

#### DOCBOT-204: DB Chat API Route

**Story**
As Raj (Operations Manager), I want to send a natural language question to a connected database and receive a complete answer, so that I can get custom reports without writing SQL.

**Phase**: 1
**Priority**: Must Have
**Story Points**: 5
**Dependencies**: DOCBOT-203
**Status**: ✅ Done

**Acceptance Criteria**
- [ ] POST /api/db/chat accepts `{connection_id, question, persona, session_id}`
- [ ] Returns `{answer, sql_query, explanation, result_preview, row_count, sources}`
- [ ] Errors (bad connection_id, query timeout, validation failure) return structured JSON with `error_type`
- [ ] Selected persona influences the answer tone (Finance Expert vs. Doctor vs. Data Analyst)
- [ ] Streaming response used for answer generation step (FastAPI `StreamingResponse`)

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Define `DBChatRequest` and `DBChatResponse` Pydantic models | Backend | 1h |
| 2 | Wire the 7-step pipeline into `POST /api/db/chat` route | Backend | 2h |
| 3 | Implement streaming response for Step 7 (answer generation) | Backend | 2h |
| 4 | Add structured error handling: `QueryValidationError`, `ExecutionTimeoutError`, `ConnectionNotFoundError` | Backend | 1h |

---

#### DOCBOT-205: MySQL Connector

**Story**
As Raj, I want to connect my clinic's MySQL database, so that I can query patient records alongside clinical protocol PDFs.

**Phase**: 1
**Priority**: Must Have
**Story Points**: 3
**Dependencies**: DOCBOT-204
**Status**: ✅ Done

**Acceptance Criteria**
- [ ] POST /api/db/connect accepts `dialect: "mysql"`
- [ ] MySQL connection validated with test query on connect
- [ ] Read-only enforcement via `SET SESSION TRANSACTION READ ONLY` before every query
- [ ] `pymysql` driver used (not `mysqlclient`, which has build issues on Railway)
- [ ] Dialect-specific LIMIT syntax handled correctly by `sqlglot`

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Add `pymysql>=1.1.0` to `requirements.txt`; test Railway build | DevOps | 1h |
| 2 | Add MySQL dialect branch in `connect_database()` and read-only enforcement | Backend | 2h |
| 3 | Test: connect MySQL, run SELECT query, verify read-only enforcement blocks INSERT | Backend | 1h |

---

#### DOCBOT-206: SQLite File Upload as Data Source

**Story**
As Sarah (Strategy Analyst), I want to upload a SQLite file as a data source without entering any credentials, so that I can query project data without needing a live database connection.

**Phase**: 1
**Priority**: Must Have
**Story Points**: 5
**Dependencies**: DOCBOT-204
**Status**: ✅ Done

**Acceptance Criteria**
- [ ] SQLite file accepted via multipart upload (max 100MB)
- [ ] File stored in `/tmp` with a unique UUID filename
- [ ] Schema introspected identically to live DB connections
- [ ] Same 7-step SQL pipeline used (no separate code path)
- [ ] File cleaned up when session ends or after 2-hour TTL

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Add `POST /api/db/upload` route accepting multipart SQLite file | Backend | 2h |
| 2 | Generate UUID-based temp path; save file; register as `db_connection` with `dialect: sqlite` | Backend | 1h |
| 3 | Add TTL-based cleanup job (Railway cron or background task on startup) | Backend | 2h |
| 4 | Test: upload a 50MB SQLite file, run a JOIN query, verify result | Backend | 1h |

---

#### DOCBOT-207: CSV File Upload as Data Source

**Story**
As Sarah, I want to upload a CSV file as a queryable data source, so that I can query spreadsheet data without needing a database.

**Phase**: 1
**Priority**: Must Have
**Story Points**: 5
**Dependencies**: DOCBOT-206
**Status**: ✅ Done

**Acceptance Criteria**
- [ ] CSV accepted via multipart upload (max 50MB)
- [ ] CSV converted to an in-memory SQLite table using pandas on upload
- [ ] Column types auto-inferred (numeric strings cast to INT/FLOAT, date strings to DATE)
- [ ] Same SQL pipeline used for querying
- [ ] Schema returns inferred column names and types

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Parse CSV with pandas; infer dtypes; write to an in-memory SQLite DB | Backend | 2h |
| 2 | Register the in-memory SQLite as a `db_connection` so the same pipeline applies | Backend | 1h |
| 3 | Handle common CSV issues: BOM encoding, quoted commas, mixed-type columns | Backend | 2h |
| 4 | Test with a real-world CSV (>10 columns, >1000 rows, messy types) | Backend | 1h |

---

#### DOCBOT-208: Microsoft Entra (Azure AD) Auth for Azure SQL

**Story**
As an enterprise user, I want to connect to Azure SQL Database using Microsoft Entra authentication, so that I can query my organization's Azure SQL databases without exposing username/password credentials.

**Phase**: 1 (Enterprise Add-on)
**Priority**: Must Have (for enterprise tier)
**Story Points**: 8
**Dependencies**: DOCBOT-201 (Fernet encryption baseline)
**Status**: ✅ Done

**Acceptance Criteria**
- [x] `azure_sql` added as a supported dialect with `mssql+pyodbc` driver
- [x] ODBC Driver 18 for SQL Server installed in Docker image
- [x] `DBConnectionRequest` accepts `auth_type`, `tenant_id`, `client_id`, `client_secret` fields with model validation
- [x] `_get_entra_token()` acquires token via `azure.identity.ClientSecretCredential`; safe error messages only
- [x] `_build_entra_connect_args()` encodes token as UTF-16-LE struct for `SQL_COPT_SS_ACCESS_TOKEN` (attr 1256)
- [x] Connection URL contains no credentials (`mssql+pyodbc:///?odbc_connect=...`)
- [x] Encryption at rest: SP credentials stored Fernet-encrypted alongside other connection credentials
- [x] Frontend: Azure SQL option in dialect dropdown with conditional Tenant ID / Client ID / Client Secret fields (SP flow)
- [x] 131 unit tests passing (including 15 new tests for Azure SQL/Entra)
- [x] `auth_type: entra_interactive` — accepts pre-acquired MSAL access token from frontend
- [x] `_parse_token_expiry()` decodes JWT exp claim (stdlib only) for TTL storage
- [x] `TokenExpiredError` raised when stored token within 5 min of expiry; caught in db/chat stream as `requires_reauth: true`
- [x] Frontend: "Sign in with Microsoft" button via `@azure/msal-browser` replaces SP credential fields
- [ ] **[BACKLOG]** End-to-end test with real Azure SQL database and Azure AD app registration (requires Azure account setup — `NEXT_PUBLIC_AZURE_CLIENT_ID`)

**Implementation Notes**
- `_resolve_connection(creds)` dispatches to `entra_interactive`, `entra_sp`, or password path
- Interactive tokens stored encrypted with `token_expires_at` ISO timestamp; re-auth triggered by frontend on 401
- `SET TRANSACTION ISOLATION LEVEL SNAPSHOT` used for read-only enforcement (Azure SQL doesn't support PostgreSQL's `SET TRANSACTION READ ONLY`)
- `pyodbc>=5.0.0`, `azure-identity>=1.17.0`, `@azure/msal-browser^5.6.1` added to deps

---

### EPIC-03: Analytical Loop (Python Execution)

---

#### DOCBOT-301: Python Code Generation

**Story**
As a developer, I want the system to generate Python (pandas/numpy/matplotlib) code to analyze SQL query results, so that answers can include statistical insights and visualizations beyond what SQL alone can express.

**Phase**: 1
**Priority**: Must Have
**Story Points**: 8
**Dependencies**: DOCBOT-103, DOCBOT-203
**Status**: ✅ Done

**Acceptance Criteria**
- [ ] After SQL execution, a second LLM call (Claude claude-sonnet-4-6) generates Python code to analyze the result DataFrame
- [ ] Python code uses only `pandas`, `numpy`, `matplotlib`, `datetime` — no external HTTP calls
- [ ] Generated code always assigns final output to `result` variable and saves charts to `chart.png`
- [ ] If SQL result has fewer than 5 rows, Python step is skipped (not enough data to analyze)
- [ ] Generated code is displayed in the UI as a collapsible `CodeDisplay` block

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Create `generate_analysis_code()` in `api/sandbox_service.py` using Claude claude-sonnet-4-6 API | Backend | 3h |
| 2 | Define system prompt: "Generate pandas code to analyze this DataFrame. Use only stdlib + pandas/numpy/matplotlib." | Backend | 1h |
| 3 | Convert SQL result (list of dicts) to a pandas DataFrame constructor string injected into the generated code | Backend | 1h |
| 4 | Add logic to skip Python generation when result row count < 5 | Backend | 0.5h |
| 5 | Test: SQL result of monthly revenue figures → code generates bar chart + YoY growth calculation | Backend | 1h |

---

#### DOCBOT-302: Python Code Execution via E2B

**Story**
As a developer, I want generated Python code executed in an E2B sandbox, so that computation runs safely without access to the host filesystem or network.

**Phase**: 1
**Priority**: Must Have
**Story Points**: 5
**Dependencies**: DOCBOT-103, DOCBOT-301
**Status**: ✅ Done

**Acceptance Criteria**
- [ ] Code runs in E2B `CodeInterpreter` sandbox, not locally
- [ ] stdout, stderr, and any matplotlib figures captured and returned
- [ ] Execution timeout: 25 seconds; returns error if exceeded
- [ ] Sandbox closed in `finally` block regardless of success or failure
- [ ] Network access disabled in sandbox (E2B option: `allow_internet=False`)

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Wire `generate_analysis_code()` output into `run_python()` in `sandbox_service.py` | Backend | 1h |
| 2 | Extract figures from `execution.results` (E2B `Result` objects with `.png` attribute) | Backend | 2h |
| 3 | Convert PNG bytes to base64 string for JSON transport | Backend | 0.5h |
| 4 | Add 25-second asyncio timeout wrapper around E2B call | Backend | 1h |
| 5 | Test: timeout case (infinite loop) → verify error returned within 26 seconds | Backend | 1h |

---

#### DOCBOT-303: Chart Rendering in Chat

**Story**
As Maya, I want charts generated from my data to appear inline in the chat response, so that I can visually understand trends without switching to another tool.

**Phase**: 1
**Priority**: Must Have
**Story Points**: 5
**Dependencies**: DOCBOT-302
**Status**: ✅ Done

**Acceptance Criteria**
- [ ] Base64 PNG strings from E2B rendered as `<img>` tags in the chat response
- [ ] Charts displayed below the text answer, above citation sources
- [ ] User can click a chart to open it full-screen
- [ ] Charts have a download button (saves as PNG)
- [ ] If no chart generated, UI renders gracefully (no empty image placeholder)

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Create `ChartDisplay` React component that accepts `base64Png: string[]` prop | Frontend | 2h |
| 2 | Add lightbox/full-screen view on chart click using a simple modal | Frontend | 2h |
| 3 | Add download button that triggers `<a download>` with the base64 PNG | Frontend | 1h |
| 4 | Integrate `ChartDisplay` into existing chat message component | Frontend | 1h |

---

#### DOCBOT-304: Data Analyst Persona

**Story**
As a developer, I want an 8th "Data Analyst" persona in the persona system, so that data-focused queries receive responses framed with analytical precision and SQL transparency.

**Phase**: 1
**Priority**: Must Have
**Story Points**: 2
**Dependencies**: None (can be done independently)
**Status**: ✅ Done

**Acceptance Criteria**
- [ ] "Data Analyst" persona appears in the persona selector UI alongside the 7 existing personas
- [ ] Persona definition includes: always shows SQL, explains query logic, flags data quality issues (NULLs, outliers)
- [ ] Persona is automatically selected when user connects a database (can be overridden)
- [ ] Persona tone: direct, quantitative, neutral — no clinical or legal caveats

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Add `data_analyst` entry to `EXPERT_PERSONAS` dict in `api/index.py` with full `persona_def` | Backend | 1.5h |
| 2 | Add `data_analyst` card to persona selector in `src/app/page.tsx` with a chart/data icon | Frontend | 1h |
| 3 | Add logic: when DB connection is established, auto-select `data_analyst` persona | Frontend | 0.5h |

---

#### DOCBOT-305: Advanced Charts & Visualization ✅ Done

**Story**
As a user, I want to choose different chart types for my data analysis, see chart metadata captions, and zoom/download charts, so that I get the most informative visualization for each dataset.

**Phase**: 2
**Priority**: Should Have
**Story Points**: 5
**Dependencies**: DOCBOT-103 (E2B sandbox)
**Status**: ✅ Done (SCRUM-398, merged 2026-03-23)

**Acceptance Criteria**
- [x] Chart type selector (auto/bar/line/scatter/heatmap/box/multi) visible above DB input
- [x] `chart_type` param flows from frontend → API → `generate_analysis_code()`
- [x] LLM receives chart-type-specific instructions for each type
- [x] Generated code emits `CHART_META:{json}` stdout line after plt.show()
- [x] Metadata (type, title, x_label, y_label, series_count) captured and returned in SSE stream
- [x] ChartDisplay shows metadata caption row below chart
- [x] Fullscreen zoom modal with Download PNG + Close buttons

**Engineering Tasks**

| # | Task | Role | Status |
|---|------|------|--------|
| 1 | Add `VALID_CHART_TYPES`, `ChartMetadata`, `chart_metadata` to `sandbox_service.py` | Backend | ✅ Done |
| 2 | Extend `_extract_charts()` to parse `CHART_META:` stdout lines (3-tuple return) | Backend | ✅ Done |
| 3 | Add `_chart_type_instructions()` with per-type LLM guidance | Backend | ✅ Done |
| 4 | Update `generate_analysis_code()` to accept `chart_type` and require `CHART_META` output | Backend | ✅ Done |
| 5 | Add `chart_type` to `DBChatRequest`; thread through `run_sql_pipeline()` | Backend | ✅ Done |
| 6 | Include chart metadata in SSE `chart` chunks from `db_service.py` | Backend | ✅ Done |
| 7 | Chart type selector pills in `page.tsx`; capture `chartMetas` from SSE | Frontend | ✅ Done |
| 8 | `ChartDisplay` metadata caption + zoom modal + Download PNG | Frontend | ✅ Done |
| 9 | 26 unit tests in `tests/unit/test_sandbox_service.py` | Testing | ✅ Done |

---

### EPIC-04: Hybrid Intelligence

---

#### DOCBOT-401: Query Intent Classifier

**Story**
As a developer, I want incoming questions classified as "db-only", "docs-only", or "hybrid" before processing, so that each query is routed to the correct pipeline with a single cheap LLM call.

**Phase**: 1
**Priority**: Must Have
**Story Points**: 3
**Dependencies**: DOCBOT-203
**Status**: ✅ Done

**Acceptance Criteria**
- [ ] Single Groq LLM call with a focused classification prompt returns one of: `sql`, `doc`, `hybrid`
- [ ] Classification latency under 1 second (use `llama-3.3-70b` with max_tokens=10)
- [ ] If user has no DB connected, classifier always returns `doc` regardless of question
- [ ] If user has no PDFs uploaded, classifier always returns `sql` regardless of question
- [ ] Classification decision logged for future accuracy analysis

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Create `classify_intent()` in `api/hybrid_service.py` with minimal Groq call | Backend | 2h |
| 2 | Add fallback rules (no DB → force `doc`; no docs → force `sql`) | Backend | 1h |
| 3 | Log `{session_id, question_hash, classification, timestamp}` to `query_history` | Backend | 0.5h |

---

#### DOCBOT-402: Hybrid Chat Pipeline

**Story**
As Maya, I want to ask one question that spans my board deck PDF and my live database, so that I get a single synthesized answer with citations to both sources.

**Phase**: 1
**Priority**: Must Have
**Story Points**: 13
**Dependencies**: DOCBOT-401, DOCBOT-204
**Status**: ✅ Done

**Acceptance Criteria**
- [ ] POST /api/hybrid/chat executes RAG retrieval and SQL execution in parallel (asyncio.gather)
- [ ] Final synthesis uses a single LLM call with both doc context and DB results in the prompt
- [ ] Response includes dual citations: `[Source: file.pdf, Page X]` for docs and `[DB: table_name]` for DB results
- [ ] Total response time under 20 seconds for typical hybrid query on cached schema
- [ ] If one source returns an error, the other source's result is still returned with a note about the failure

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Create `hybrid_chat()` in `api/hybrid_service.py` | Backend | 2h |
| 2 | Implement parallel async execution: `asyncio.gather(rag_retrieve(), sql_execute())` | Backend | 2h |
| 3 | Write hybrid synthesis prompt (persona + doc context + DB result + fusion instruction) | Backend | 2h |
| 4 | Implement dual citation formatter for hybrid responses | Backend | 2h |
| 5 | Add `POST /api/hybrid/chat` route to `api/index.py` | Backend | 0.5h |
| 6 | Handle partial failure: if SQL fails, return doc-only answer; if RAG fails, return SQL-only answer | Backend | 1.5h |

---

#### DOCBOT-403: Discrepancy Detection

**Story**
As Maya, I want the system to explicitly flag when my PDF documents and database data disagree on a number, so that I immediately see the gap without having to compare the two sources myself.

**Phase**: 1
**Priority**: Must Have
**Story Points**: 8
**Dependencies**: DOCBOT-402
**Status**: ✅ Done

**Acceptance Criteria**
- [ ] Synthesis prompt instructs the LLM to explicitly flag disagreements between sources
- [ ] When a discrepancy is detected, the answer includes a highlighted "Discrepancy" section
- [ ] Discrepancy section format: `"Doc says X [Source: file, Page N]. DB shows Y [DB: table]. Delta: Z (±P%)"`
- [ ] Works for numeric values; non-numeric discrepancies ("Q3" vs. "Q4") also flagged as inconsistencies
- [ ] No false positives when doc and DB agree (system should not say "no discrepancy" unless explicitly asked)

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Update hybrid synthesis prompt with discrepancy detection instruction and output format | Backend | 2h |
| 2 | Add post-processing step to detect `[DISCREPANCY]` marker in LLM output and style it differently | Backend | 1.5h |
| 3 | Update frontend chat renderer to display discrepancy block with warning styling | Frontend | 2h |
| 4 | Test: board deck with $2.4M ARR target + DB showing $2.1M → verify discrepancy section appears | Full-stack | 1.5h |

---

#### DOCBOT-404: HybridModeToggle UI Component

**Story**
As a user, I want a 3-way toggle to switch between Docs mode, DB mode, and Hybrid mode, so that I have explicit control over which sources the system queries.

**Phase**: 1
**Priority**: Must Have
**Story Points**: 3
**Dependencies**: DOCBOT-402
**Status**: ✅ Done

**Acceptance Criteria**
- [ ] Toggle displays three options: "Documents", "Database", "Hybrid"
- [ ] Toggle disabled when relevant source is not connected (DB option grayed out if no DB connected)
- [ ] Selected mode persists for the duration of the session
- [ ] Changing mode does not clear the conversation history
- [ ] Hybrid option only shows when both a PDF and a DB are connected

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Create `HybridModeToggle` React component with three-state logic | Frontend | 2h |
| 2 | Add disabled state logic based on `hasDocuments` and `hasDatabase` booleans from app state | Frontend | 1h |
| 3 | Connect toggle state to `POST /api/hybrid/chat` vs. `POST /api/chat` vs. `POST /api/db/chat` routing | Frontend | 1h |

---

#### DOCBOT-406: LangExtract — Structured Financial Document Extraction ✅ Done

**Story**
As Maya (Finance Manager), I want financial documents I upload to have their numeric projections extracted as verified, typed values — not just retrieved as text — so that discrepancy detection produces arithmetic comparisons with exact figures rather than vague prose.

**Phase**: 1 (shipped — merged to main 2026-03-22)
**Priority**: Should Have
**Story Points**: 5
**Dependencies**: DOCBOT-403

**Implementation Notes**
Uses [google/langextract](https://github.com/google/langextract) (v1.2.0) backed by Gemini 2.5 Flash.
- Full document coverage via LangExtract chunked parallel extraction (no 8,000 char limit)
- Precise `char_interval` source grounding replaces manual substring verification
- `max_workers=2` to stay within Gemini free tier (10 RPM cap)
- Reads `GEMINI_API_KEY` from environment; returns `[]` gracefully if key is missing
- Runs as `asyncio.create_task()` background job — zero upload latency impact

**Acceptance Criteria**
- [x] Financial documents (P&L, board decks, budgets, forecast slides) are detected by file type and content heuristics at upload time
- [x] `api/document_extractor.py` uses LangExtract + Gemini 2.5 Flash for full-document chunked extraction with `char_interval` source grounding
- [x] Extraction output includes: field name, value (typed float), unit (USD, %, units), source span text, page number, verified flag
- [x] Extracted fields stored in `EXTRACTED_FIELDS` in-memory dict keyed by session_id
- [x] Hybrid synthesis uses extracted typed values for arithmetic comparisons
- [x] Standard PyMuPDF + LangChain RAG unchanged for non-financial documents
- [x] Extraction runs in background — does not block upload response

**Engineering Tasks**

| # | Task | Role | Status |
|---|------|------|--------|
| 1 | Create `api/document_extractor.py` with LangExtract + Gemini 2.5 Flash extraction | Backend | ✅ Done |
| 2 | Define `ExtractedField(name, value, unit, span_text, page, verified)` Pydantic schema | Backend | ✅ Done |
| 3 | Implement `is_financial_document()` heuristic (keyword density + numeric density) | Backend | ✅ Done |
| 4 | Update `DOCBOT-403` hybrid synthesis prompt with `format_extracted_fields_for_prompt()` | Backend | ✅ Done |
| 5 | Wire background extraction task in upload route (`asyncio.create_task`) | Backend | ✅ Done |
| 6 | Cap `max_workers=2` for Gemini free tier rate limit compliance | Backend | ✅ Done |

---

#### SCRUM-391: LangExtract — Broaden to Non-Financial Documents ✅ Done

**Story**
As a developer, I want the document extraction pipeline to support non-financial document types (legal, medical, research, general), so that structured field extraction is not limited to financial PDFs and all uploaded documents benefit from typed value extraction.

**Phase**: 1 (shipped — merged to main 2026-03-23)
**Priority**: Should Have
**Story Points**: 5
**Dependencies**: DOCBOT-406
**Status**: ✅ Done
**Implemented**: 2026-03-23 — detect_document_type() classifies into financial/legal/medical/research/general. Each type has its own LangExtract prompt and few-shot examples. extract_document_fields() is the new entry point; extract_financial_fields() kept as backward-compat alias.

**Acceptance Criteria**
- [x] `detect_document_type()` classifies documents into 5 types
- [x] Each type has its own prompt and few-shot examples
- [x] `extract_document_fields()` is the unified entry point
- [x] `extract_financial_fields()` kept as backward-compat alias
- [x] Existing financial extraction behaviour unchanged

**Engineering Tasks**

| # | Task | Role | Status |
|---|------|------|--------|
| 1 | Implement `detect_document_type()` with keyword heuristics for 5 types | Backend | ✅ Done |
| 2 | Write per-type LangExtract prompts and few-shot examples | Backend | ✅ Done |
| 3 | Implement `extract_document_fields()` as unified entry point | Backend | ✅ Done |
| 4 | Keep `extract_financial_fields()` as backward-compat alias | Backend | ✅ Done |
| 5 | Update call site in index.py to use `is_extractable_document()` | Backend | ✅ Done |

---

#### DOCBOT-405: Analytical Autopilot — Agentic Multi-Step Investigation (Phase 2) ✅ Done

**Story**
As Maya (Finance Manager), I want to delegate a diagnostic objective like "Why are we at 87.5% of Q3 revenue target?" and have DocBot autonomously run the investigation — querying the database and cross-referencing uploaded documents across multiple steps — so that I get a diagnosis with ranked hypotheses, not just a single data point.

**Phase**: 2
**Priority**: Should Have
**Story Points**: 13
**Dependencies**: DOCBOT-402, DOCBOT-501
**Status**: ✅ Done (SCRUM-399, merged 2026-03-23)

**Design Rationale**
This is DocBot's first agentic feature. The agent loops over the existing 7-step hybrid pipeline using a planner node. The SQL execution layer is unchanged — the agent controls retrieval strategy and step sequencing, never SQL execution directly. LangGraph is used over Claude Agent SDK because it supports hard step limits, explicit node/edge control, and integrates with the existing LangChain setup.

**Architecture**
```
User objective
  → [Planner node]    LLM decomposes into 3–5 investigation steps (one LLM call)
  → [Executor loop]   Each step runs the existing 7-step hybrid pipeline
  → [Synthesizer]     Aggregates step results into a ranked diagnosis

Hard limits: max 5 steps · 90-second wall-clock timeout · partial answer fallback
Lives in: api/agent_service.py (new module, not api/index.py)
```

**Acceptance Criteria**
- [ ] User can submit a diagnostic objective (e.g. "Diagnose the revenue miss") in addition to regular questions
- [ ] Planner decomposes objective into ≤5 investigation steps (sql_query, doc_retrieve, python_analyze, synthesize)
- [ ] Each step result is passed as context to the next step
- [ ] Agent never executes SQL directly — all SQL goes through the existing 7-step pipeline with AST validation
- [ ] Max 5 steps enforced via LangGraph `recursion_limit`; hard timeout at 90 seconds
- [ ] Partial answer returned if timeout is hit (with "investigation incomplete" indicator)
- [ ] UI shows a step-by-step "Thinking" progress panel — every step visible and auditable by the user
- [ ] Final output: ranked hypotheses with supporting evidence + source citations per finding
- [ ] All existing single-question hybrid queries are unaffected (agent path is gated behind intent detection)

**Engineering Tasks**

| # | Task | Role | Status |
|---|------|------|--------|
| 1 | `api/autopilot_service.py`: LangGraph StateGraph (PlannerNode → ExecutorNode loop → SynthesizerNode) | Backend | ✅ Done |
| 2 | `_select_tool()` heuristic routes steps to sql_query / doc_search / python_analysis | Backend | ✅ Done |
| 3 | `make_executor_node()` captures all DB tables/session factories via closure | Backend | ✅ Done |
| 4 | MAX_ITERATIONS=5 enforced via `_should_continue()`; TOTAL_TIMEOUT_S=90 wall-clock guard | Backend | ✅ Done |
| 5 | Executor saves sql_result and chart artifacts via `artifact_service.save_artifact()` | Backend | ✅ Done |
| 6 | `_planner_node()` and `_synthesizer_node()` graceful fallbacks when groq_api_key absent | Backend | ✅ Done |
| 7 | `POST /api/autopilot/run` SSE route; `AutopilotRequest` Pydantic model in `api/index.py` | Backend | ✅ Done |
| 8 | `langgraph>=0.2.0` added to `requirements.txt` | Backend | ✅ Done |
| 9 | Frontend: `AutopilotStep` type; autopilotMode/Running/Steps/Plan state; toggle button | Frontend | ✅ Done |
| 10 | Frontend: `/api/autopilot/run` SSE path in `handleSendMessage` (takes priority when ON) | Frontend | ✅ Done |
| 11 | Frontend: live step-by-step progress panel (plan + completed steps + tool badges + charts) | Frontend | ✅ Done |
| 12 | 29 unit tests in `tests/unit/test_autopilot_service.py` | Testing | ✅ Done |

---

### EPIC-05: Memory and Context

---

#### DOCBOT-501: Session Artifact Store

**Story**
As a developer, I want DataFrames and charts generated in a session stored as artifacts, so that follow-up questions can reference earlier results without re-executing queries.

**Phase**: 2
**Priority**: Should Have
**Story Points**: 8
**Dependencies**: DOCBOT-302, DOCBOT-102
**Status**: ✅ Done (SCRUM-394, merged 2026-03-23)

**Acceptance Criteria**
- [ ] Every query result DataFrame serialized to Parquet and stored in `session_artifacts` PostgreSQL table
- [ ] Every chart PNG stored as bytea or referenced file path in the same table
- [ ] Artifacts referenced by artifact_id in subsequent messages
- [ ] LLM context for follow-up questions includes artifact summaries (row count, column names, date range)
- [ ] Artifacts older than 24 hours purged by background task

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Add `session_artifacts` table to PostgreSQL schema: `{id, session_id, type, data, summary, created_at}` | Backend | 1h |
| 2 | Implement `store_artifact()` and `retrieve_artifact()` functions | Backend | 2h |
| 3 | Serialize DataFrame to Parquet bytes using `pandas.to_parquet()` with pyarrow | Backend | 1h |
| 4 | Auto-generate artifact summary (row count, columns, min/max date) for LLM context injection | Backend | 2h |
| 5 | Implement 24-hour TTL cleanup task (PostgreSQL `DELETE WHERE created_at < NOW() - INTERVAL '24 hours'`) | Backend | 1h |

---

#### DOCBOT-502: Context Compression

**Status**: ✅ Done (SCRUM-396, merged 2026-03-23)

**Story**
As a developer, I want long conversation histories compressed before being sent to the LLM, so that sessions with many turns do not exceed the context window or incur excessive token costs.

**Phase**: 2
**Priority**: Should Have
**Story Points**: 5
**Dependencies**: DOCBOT-102

**Acceptance Criteria**
- [ ] Conversations with more than 20 messages have older turns summarized into a running summary
- [ ] Summary includes key facts, numbers mentioned, and conclusions reached
- [ ] Only the last 5 messages + the running summary are sent to the LLM per turn
- [ ] Running summary stored in `sessions` table and updated incrementally
- [ ] User's actual conversation history in the UI is never truncated (only LLM context is compressed)

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Implement `compress_context()`: takes messages[0..N-5], calls Groq to summarize into ~200 tokens | Backend | 3h |
| 2 | Add `context_summary` column to `sessions` table | Backend | 0.5h |
| 3 | Update chat handler to inject `context_summary` at top of LLM message list when present | Backend | 1h |
| 4 | Trigger compression when message count in session exceeds 20 | Backend | 0.5h |

---

#### DOCBOT-503: Schema-Aware RAG

**Story**
As a developer, I want LLM-generated table descriptions embedded and searchable, so that queries about tables with cryptic names can still find the right tables semantically.

**Phase**: 2
**Priority**: Should Have
**Story Points**: 8
**Dependencies**: DOCBOT-202
**Status**: ✅ Done (SCRUM-395, merged 2026-03-23)

**Acceptance Criteria**
- [ ] On DB connect, Groq generates a one-sentence description of each table from column names + 3 sample rows
- [ ] Descriptions embedded using `all-MiniLM-L6-v2` and stored in `table_embeddings` PostgreSQL table
- [ ] Table selection (Step 2 of SQL pipeline) uses cosine similarity over embeddings, not keyword matching
- [ ] System handles tables with names like `cust_ord_hdr_rec` correctly
- [ ] Embedding generation does not block the connection response (runs as background task)

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Add `table_embeddings` table to PostgreSQL schema | Backend | 0.5h |
| 2 | Implement `generate_table_descriptions()`: for each table, call Groq with schema + sample rows | Backend | 3h |
| 3 | Embed descriptions and store in `table_embeddings` using existing `get_embeddings()` | Backend | 1.5h |
| 4 | Replace keyword-based table selection in Step 2 with cosine similarity retrieval | Backend | 2h |
| 5 | Run description generation as a FastAPI `BackgroundTask` on DB connect | Backend | 1h |

---

#### DOCBOT-504: Query History Panel UI

**Status**: ✅ Done (SCRUM-397, merged 2026-03-23)

**Story**
As Sarah, I want to see a history of past queries with their natural language question and SQL snippet, so that I can re-run or refine previous analyses quickly.

**Phase**: 2
**Priority**: Could Have
**Story Points**: 5
**Dependencies**: DOCBOT-204

**Acceptance Criteria**
- [ ] Right sidebar panel shows past queries in chronological order
- [ ] Each entry shows: NL question (truncated to 60 chars), SQL snippet (first 80 chars), timestamp
- [ ] Clicking an entry re-populates the chat input with the original question
- [ ] Panel loads query history from GET /api/db/sessions
- [ ] Panel is collapsible; collapsed state persists in localStorage

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Create `QueryHistoryPanel` React component | Frontend | 3h |
| 2 | Add GET /api/db/sessions route returning recent queries for a session | Backend | 1h |
| 3 | Implement collapse/expand with localStorage persistence | Frontend | 1h |
| 4 | Wire click handler to populate chat input field | Frontend | 0.5h |

---

### EPIC-06: Enterprise Readiness

---

#### DOCBOT-601: SSO / SAML Integration

**Story**
As an enterprise IT administrator, I want to integrate DocBot with our Okta or Azure AD identity provider, so that employees can log in with their corporate credentials and we maintain centralized access control.

**Phase**: 4
**Priority**: Must Have (for enterprise tier)
**Story Points**: 13
**Dependencies**: None (but requires auth layer not yet built)
**Status**: ✅ Done

**Implementation Notes**
SP-initiated SAML 2.0 implemented via `python3-saml`. JIT user provisioning on first ACS success. Session tokens stored in PostgreSQL with configurable TTL (default 8 hours). HttpOnly, SameSite=Lax session cookie. SP metadata exposed at `GET /api/auth/saml/metadata`. App operates in open mode when `SAML_*` env vars are unset.

**Acceptance Criteria**
- [x] SAML 2.0 SP-initiated SSO flow works with Okta test tenant
- [x] SAML 2.0 SP-initiated SSO flow works with Azure AD test tenant
- [x] User attributes (email, name, groups) mapped from SAML assertion to local user record
- [x] JIT (just-in-time) user provisioning creates account on first SSO login
- [x] Session cookie is secure, HttpOnly, SameSite=Lax

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Evaluate `python-saml` vs. `onelogin/python3-saml` and select | Backend | 2h |
| 2 | Implement SP metadata endpoint and ACS (Assertion Consumer Service) handler | Backend | 5h |
| 3 | Create `users` and `user_sessions` PostgreSQL tables | Backend | 1h |
| 4 | Implement JIT provisioning logic in ACS handler | Backend | 2h |
| 5 | Test full SSO flow with Okta Developer account | Backend | 3h |
| 6 | Test full SSO flow with Azure AD free tenant | Backend | 3h |

---

#### DOCBOT-602: Audit Logging

**Story**
As an enterprise compliance officer, I want an immutable log of all queries and data access events, so that we can demonstrate compliance and investigate any data misuse.

**Phase**: 4
**Priority**: Must Have (for enterprise tier)
**Story Points**: 8
**Dependencies**: DOCBOT-601
**Status**: ✅ Done

**Implementation Notes**
Append-only `audit_log` PostgreSQL table with a `BEFORE UPDATE OR DELETE` trigger enforcing immutability at the DB level. `AuditEvent` Pydantic model and `log_event()` utility in `api/audit_service.py`. Writes are fire-and-forget — failures never block request handlers. CSV export via `GET /admin/audit-log?format=csv`.

**Acceptance Criteria**
- [x] Every query event logged: user_id, timestamp, session_id, question_hash, SQL executed, row count returned
- [x] Audit log is append-only (no UPDATE or DELETE on audit table, enforced at DB level via trigger)
- [x] Audit records exportable as CSV via admin endpoint
- [x] Log includes connection events: DB connected, DB disconnected, by whom, when
- [x] PII (raw question text) stored hashed; full text only if admin enables explicit logging

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Create `audit_log` PostgreSQL table with immutability trigger | Backend | 2h |
| 2 | Create `AuditEvent` Pydantic model and `log_event()` utility | Backend | 1h |
| 3 | Inject `log_event()` call into all query, upload, and connection handlers | Backend | 2h |
| 4 | Add `GET /admin/audit-log` route with CSV export | Backend | 2h |
| 5 | Test: attempt to DELETE from audit_log → verify PostgreSQL trigger blocks it | Backend | 1h |

---

#### DOCBOT-603: Role-Based Access Control

**Story**
As an enterprise admin, I want to assign roles (Viewer, Analyst, Admin) to users, so that I can control who can query which databases and who can manage connections.

**Phase**: 4
**Priority**: Must Have (for enterprise tier)
**Story Points**: 8
**Dependencies**: DOCBOT-601
**Status**: ✅ Done

**Implementation Notes**
Three-tier IntEnum role hierarchy (viewer / analyst / admin) enforced via `require_role()` FastAPI `Depends` dependency. Admin self-demotion guard prevents accidental lockout. Role management UI in admin panel. No-op in open mode (no SAML configured). `GET /admin/users` and `PATCH /admin/users/{user_id}/role` routes added.

**Acceptance Criteria**
- [x] Three roles defined: Viewer (read results only), Analyst (run queries), Admin (manage connections)
- [x] Role assignments stored in PostgreSQL
- [x] API routes return 403 if user's role does not have required permission
- [x] Admin UI panel for assigning roles (basic table view)
- [x] Role checked on every request via FastAPI Depends dependency (not scattered inline checks)

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Add `roles` and `user_roles` tables to PostgreSQL | Backend | 1h |
| 2 | Implement `require_role()` FastAPI dependency for route decoration | Backend | 2h |
| 3 | Apply `require_role()` to all DB and admin routes | Backend | 2h |
| 4 | Create simple admin table UI in Next.js for role assignment | Frontend | 3h |

---

#### DOCBOT-604: PII Auto-Detection and Masking

**Story**
As a compliance officer, I want personally identifiable information in query results to be automatically detected and masked, so that sensitive data is not inadvertently exposed in chat responses.

**Phase**: 4
**Priority**: Must Have (for enterprise tier)
**Story Points**: 8
**Dependencies**: DOCBOT-204
**Status**: ✅ Done

**Implementation Notes**
spaCy NER + regex patterns detect full names, email addresses, phone numbers (US + international), SSNs, and credit card numbers. Masks with typed placeholders: `[NAME]`, `[EMAIL]`, `[PHONE]`, `[SSN]`, `[CC_NUMBER]`. Runs on DB query results before LLM synthesis. Configurable per-request via `mask_pii: true/false` in request body.

**Acceptance Criteria**
- [x] PII detection runs on every query result before it is sent to the LLM or returned to the frontend
- [x] Detects: full names, email addresses, phone numbers (US + international), US SSNs, credit card patterns
- [x] Masking uses typed placeholders: `[NAME]`, `[EMAIL]`, `[PHONE]`, `[SSN]`, `[CC_NUMBER]`
- [x] Configurable per-request via `mask_pii` flag
- [x] PII detection adds no more than 200ms to response time

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Implement regex-based PII detector for email, phone, SSN, credit card | Backend | 3h |
| 2 | Apply PII masking to DataFrame before LLM synthesis and before frontend return | Backend | 2h |
| 3 | Add `pii_masking_enabled` flag to `db_connections` table and connection UI | Backend + Frontend | 2h |
| 4 | Test: DataFrame with real-looking SSNs and emails → verify masking output | Backend | 1h |

---

#### DOCBOT-605: On-Premise Deployment (Docker)

**Story**
As an enterprise customer with strict data residency requirements, I want to run DocBot entirely on my own infrastructure, so that no data leaves my network.

**Phase**: 4
**Priority**: Could Have
**Story Points**: 13
**Dependencies**: DOCBOT-101
**Status**: ✅ Done

**Implementation Notes**
`docker-compose.yml` ships `frontend`, `backend`, and `postgres` services. ODBC Driver 18 for SQL Server included in Dockerfile image for Azure SQL connectivity. All env vars documented in `.env.example`. Health check at `GET /api/health`.

**Acceptance Criteria**
- [x] Single `docker-compose.yml` starts the full stack: Next.js, FastAPI, PostgreSQL
- [x] All environment variables documented in `.env.example` with descriptions
- [x] Dockerfile includes ODBC Driver 18 for SQL Server
- [x] README includes deployment instructions
- [x] Health check at GET /api/health validates service is running

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Create `docker-compose.yml` with `frontend`, `backend`, `postgres` services | DevOps | 3h |
| 2 | Add Ollama service option to `docker-compose.yml` with environment variable toggle | DevOps | 2h |
| 3 | Create `healthcheck.sh` script that pings all services and reports status | DevOps | 1h |
| 4 | Write on-premise setup README section | Full-stack | 2h |

---

### Consumer Authentication (Standalone — Shipped)

---

#### DOCBOT-701: Consumer Auth — GitHub OAuth, Google OAuth, Email/Password

**Story**
As an individual user or small team, I want to sign up and log in using GitHub, Google, or email/password without needing any enterprise IdP, so that I can persist my workspace and return to previous sessions from any device.

**Phase**: 4 (shipped alongside EPIC-06)
**Priority**: Must Have (for consumer tier)
**Story Points**: 8
**Dependencies**: DOCBOT-601 (session cookie infrastructure)
**Status**: ✅ Done

**Implementation Notes**
GitHub OAuth and Google OAuth implemented via standard authorization code flow — backend exchanges code for token, fetches user profile, upserts user record, and issues a DocBot session cookie. Email/password registration uses bcrypt for password hashing. Guest mode preserves session state in browser only. `GET /api/auth/config` exposes which providers are enabled so the frontend renders only available login options. `GET /api/auth/workspace` returns the authenticated user's saved sessions and DB connections.

**Acceptance Criteria**
- [x] GitHub OAuth: one-click login; user record created on first login; session cookie issued
- [x] Google OAuth: one-click login; user record created on first login; session cookie issued
- [x] Email/password: `POST /api/auth/register` creates account; `POST /api/auth/login` authenticates; bcrypt used for hashing
- [x] Guest mode: session is browser-local only; no account required
- [x] `GET /api/auth/config` exposes enabled providers
- [x] `GET /api/auth/workspace` returns authenticated user's saved sessions and DB connections
- [x] All consumer auth flows share the same session cookie and RBAC infrastructure as SAML SSO

**Engineering Tasks**

| # | Task | Role | Status |
|---|------|------|--------|
| 1 | Implement GitHub OAuth authorization code flow in `api/auth_service.py` | Backend | ✅ Done |
| 2 | Implement Google OAuth authorization code flow in `api/auth_service.py` | Backend | ✅ Done |
| 3 | Implement `POST /api/auth/register` and `POST /api/auth/login` with bcrypt | Backend | ✅ Done |
| 4 | Add `GET /api/auth/config` route exposing enabled providers | Backend | ✅ Done |
| 5 | Add `GET /api/auth/workspace` route for authenticated workspace restoration | Backend | ✅ Done |
| 6 | Frontend: login page with conditional provider buttons based on `/api/auth/config` response | Frontend | ✅ Done |

---

### EPIC-07: Commerce Connectors

> **Gate lifted 2026-03-25.** Phase 1 (DOCBOT-701–703: connector interface, commerce schema, Amazon SP-API) unblocked for investor demo sprint. Phase 2 (DOCBOT-704–705: background sync, Shopify) deferred to post-funding.

---

#### DOCBOT-701: Marketplace Connector Interface + Credential Vault Extension

**Story**
As a developer, I want a pluggable connector architecture with an extended credential vault, so that adding a new marketplace (Amazon, Shopify, Etsy) means implementing a defined interface rather than rewriting the integration layer.

**Phase**: 4
**Priority**: Must Have (for Commerce tier)
**Story Points**: 8
**Dependencies**: DOCBOT-201 (Fernet encryption baseline)

**Acceptance Criteria**
- [ ] `api/connectors/base.py` defines `MarketplaceConnector` ABC with abstract methods: `validate_credentials`, `refresh_token`, `fetch_orders_incremental`, `fetch_products_incremental`, `fetch_inventory`
- [ ] `api/connectors/registry.py` provides `register(connector)` and `get_connector(slug)` — adding a new marketplace requires only calling `register()`
- [ ] `marketplace_credentials` PostgreSQL table stores encrypted OAuth tokens with `token_expires_at` plaintext for background refresh scheduling
- [ ] `CredentialService` handles encrypt/decrypt boundary and token refresh — credentials never touched outside this service
- [ ] `TokenBucket` rate limiter implemented per-credential (Amazon Orders: 0.0167 req/s sustained burst)
- [ ] `NormalizedOrder`, `NormalizedProduct`, `NormalizedLineItem` dataclasses defined; all connectors map to these

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Create `api/connectors/base.py`: `MarketplaceConnector` ABC, `NormalizedOrder`, `NormalizedProduct`, `NormalizedLineItem`, `SyncCursor` dataclasses, `ConnectorAuthError`, `ConnectorRateLimitError`, `ConnectorAPIError` exceptions | Backend | 3h |
| 2 | Create `api/connectors/registry.py`: `register()`, `get_connector()`, `list_connectors()` | Backend | 0.5h |
| 3 | Create `api/connectors/credential_service.py`: extend existing Fernet logic with `is_token_near_expiry()` and OAuth refresh scheduling | Backend | 2h |
| 4 | Create `api/connectors/rate_limiter.py`: async `TokenBucket` with per-credential keying; include `AMAZON_LIMITS` constants for each API section | Backend | 2h |
| 5 | Create `api/connectors/status_maps.py`: `normalize_order_status()` mapping Amazon/Shopify/Etsy raw statuses to unified set (`pending`, `confirmed`, `shipped`, `delivered`, `cancelled`, `refunded`) | Backend | 1h |
| 6 | Write migration `004_marketplace_credentials.sql`: `marketplace_credentials` table with encrypted blob, `token_expires_at`, `seller_id`, `marketplace_ids[]`, `scopes_granted[]` | Backend | 1h |

---

#### DOCBOT-702: Unified Commerce Schema + Multi-Tenant RLS

**Story**
As a developer, I want a normalized commerce schema in PostgreSQL with row-level security, so that the existing 7-step SQL pipeline can query marketplace data uniformly regardless of source, and each seller's data is isolated from other tenants.

**Phase**: 4
**Priority**: Must Have (for Commerce tier)
**Story Points**: 8
**Dependencies**: DOCBOT-701, DOCBOT-102

**Acceptance Criteria**
- [ ] Tables created: `tenants`, `marketplace_connections`, `products`, `inventory_snapshots`, `orders`, `order_line_items`
- [ ] All money values stored as `BIGINT` cents (not DECIMAL) for aggregation accuracy
- [ ] Each table has `JSONB raw_attributes` column preserving 100% of marketplace-specific fields
- [ ] Row-level security enabled on all commerce tables; app role uses `SET app.current_tenant_id` session variable to scope access
- [ ] `set_tenant_context(session, tenant_id)` utility called at the start of every request touching commerce tables
- [ ] Background job calls `set_tenant_context()` before any sync SQL
- [ ] Order status normalized via `status_maps.py`; unknown statuses logged as `unknown`, never raise

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Write migration `005_unified_commerce_schema.sql`: all 6 tables with indexes, BIGINT cents, JSONB columns, `PRIMARY KEY`, `UNIQUE` constraints | Backend | 3h |
| 2 | Write migration `006_rls_policies.sql`: enable RLS on all commerce tables; create `docbot_app` role; create `tenant_isolation` policy using `current_setting('app.current_tenant_id')::UUID` | Backend | 2h |
| 3 | Write migration `007_materialized_views.sql`: `mv_daily_sales` (refreshed nightly) and `mv_product_revenue` (refreshed hourly) with `UNIQUE INDEX` to support `CONCURRENTLY` refresh | Backend | 1.5h |
| 4 | Create `api/connectors/db_utils.py`: `set_tenant_context()`, upsert helpers for orders/products/inventory | Backend | 2h |
| 5 | Register commerce tables with existing `schema_cache` so the 7-step SQL pipeline discovers them (cache key: `marketplace:{connector_slug}:{connection_id}`) | Backend | 1h |
| 6 | Test: create two tenant records, insert orders for each, verify RLS prevents cross-tenant row access | Backend | 1.5h |

---

#### DOCBOT-703: Amazon SP-API Connector

**Story**
As Alex (Amazon seller), I want to connect my Amazon Seller Central account to DocBot, so that I can ask natural language questions about my orders, inventory, fees, and profitability.

**Phase**: 4
**Priority**: Must Have (for Commerce tier)
**Story Points**: 13
**Dependencies**: DOCBOT-701, DOCBOT-702

**Acceptance Criteria**
- [ ] `AmazonSPConnector` implements `MarketplaceConnector` ABC fully
- [ ] LWA OAuth flow: exchange refresh token for access token (60-min expiry); `CredentialService` auto-rotates before expiry
- [ ] Seller connects by providing: LWA refresh token, client ID, client secret, seller ID, marketplace ID (US: `ATVPDKIKX0DER`)
- [ ] `validate_credentials()` calls `GET /sellers/v1/marketplaceParticipations` as a probe; returns `True` or raises `ConnectorAuthError`
- [ ] `fetch_orders_incremental()` uses Orders API with `NextToken` pagination; respects `orders` token bucket (0.0167 req/s)
- [ ] `fetch_orders_incremental()` defaults to last 7 days on first sync; uses stored `sync_cursor` on subsequent syncs
- [ ] Orders normalized via `NormalizedOrder`; raw response preserved in `raw_attributes`
- [ ] Finances API sync: `fetch_financial_events()` pulls settlement line items into `order_line_items` with `transaction_type` (FBAFee, Referral, ItemPrice)
- [ ] `python-amazon-sp-api` used for Reports API (multi-step async report generation); direct `httpx` used for Orders + Finances (hot path, needs async)
- [ ] 429 responses raise `ConnectorRateLimitError(retry_after_seconds=N)` — never silently swallowed

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Add `python-amazon-sp-api>=1.0` to `requirements.txt`; verify Railway build | DevOps | 1h |
| 2 | Create `api/connectors/amazon_sp.py`: `AmazonSPConnector` class with `slug = "amazon_sp"`, `display_name = "Amazon Seller Central"` | Backend | 1h |
| 3 | Implement `refresh_token()`: POST to `https://api.amazon.com/auth/o2/token` with LWA grant; return updated credentials dict | Backend | 2h |
| 4 | Implement `validate_credentials()`: call `GET /sellers/v1/marketplaceParticipations` via `_get()` helper; handle 401/403 as `ConnectorAuthError` | Backend | 1h |
| 5 | Implement `_get()` helper: authenticated `httpx` GET with `x-amz-access-token` header; handles 429 → `ConnectorRateLimitError`, 401/403 → `ConnectorAuthError` | Backend | 2h |
| 6 | Implement `fetch_orders_incremental()`: paginate Orders API with `NextToken`; normalize via `_normalize_order()`; respect `_orders_bucket.acquire()` | Backend | 3h |
| 7 | Implement Finances API sync: `fetch_financial_events()` returning settlement line items as `NormalizedLineItem` list | Backend | 2h |
| 8 | Register `AmazonSPConnector()` in `api/connectors/__init__.py` | Backend | 0.5h |
| 9 | Add `POST /api/marketplace/connect` and `DELETE /api/marketplace/disconnect/{id}` routes | Backend | 1h |
| 10 | Test: connect with real SP-API sandbox credentials; verify orders sync into `orders` table; verify RLS isolates data | Backend | 2h |

---

#### DOCBOT-704: Background Sync Worker

**Story**
As a developer, I want marketplace data synced continuously in the background, so that sellers always query fresh data without waiting for a live API call during their chat session.

**Phase**: 4
**Priority**: Must Have (for Commerce tier)
**Story Points**: 8
**Dependencies**: DOCBOT-703

**Acceptance Criteria**
- [ ] `APScheduler` runs as part of the FastAPI app lifecycle (started in `lifespan`)
- [ ] `sync_marketplace_incremental(connection_id, tenant_id)` fetches records updated since `sync_cursor`, upserts into unified schema, updates `last_synced_at` and `sync_cursor` in `marketplace_connections`
- [ ] Amazon Orders synced every 15 min; Inventory synced every 60 min; Finances synced every 4 hours
- [ ] `REFRESH MATERIALIZED VIEW CONCURRENTLY` called after each successful Orders sync
- [ ] `register_sync_schedules(connection_id, tenant_id, connector_slug)` called when a new marketplace is connected; removes schedules on disconnect
- [ ] `ConnectorRateLimitError` triggers exponential backoff (not immediate retry)
- [ ] `ConnectorAuthError` during sync disables the schedule and marks connection `sync_status = 'auth_failed'` — user notified on next chat interaction
- [ ] Sync errors logged with `connection_id` and `error_type`; never crash the main app

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Add `apscheduler>=3.10` to `requirements.txt` | DevOps | 0.5h |
| 2 | Create `api/connectors/sync_jobs.py`: `sync_marketplace_incremental()`, `refresh_materialized_views()`, `register_sync_schedules()`, `deregister_sync_schedules()` | Backend | 4h |
| 3 | Wire `AsyncIOScheduler` into FastAPI `lifespan` context manager (start on startup, shutdown on teardown) | Backend | 1h |
| 4 | Implement exponential backoff on `ConnectorRateLimitError`: schedule retry at `retry_after_seconds * 1.5` | Backend | 1h |
| 5 | Implement auth failure handler: set `sync_status = 'auth_failed'` in `marketplace_connections`; inject warning into next chat context | Backend | 1.5h |
| 6 | Test: mock a 429 response → verify retry scheduled correctly; mock auth failure → verify sync disabled and status set | Backend | 1.5h |

---

#### DOCBOT-705: Shopify Connector

**Story**
As a DTC brand owner on Shopify, I want to connect my Shopify store to DocBot, so that I can ask questions about my orders, products, and inventory alongside my business documents.

**Phase**: 4
**Priority**: Should Have
**Story Points**: 8
**Dependencies**: DOCBOT-701, DOCBOT-702, DOCBOT-704

**Acceptance Criteria**
- [ ] `ShopifyConnector` implements `MarketplaceConnector` ABC fully
- [ ] OAuth 2.0 offline token flow: user authorizes DocBot app in Shopify admin; offline access token stored encrypted
- [ ] `validate_credentials()` calls `GET /admin/api/2024-01/shop.json` as a probe
- [ ] `fetch_orders_incremental()` uses Shopify Orders API with cursor-based pagination (`page_info`)
- [ ] `fetch_products_incremental()` uses Shopify Products API
- [ ] `fetch_inventory()` uses Inventory Levels API; returns current stock per location
- [ ] Webhook registration: `orders/create`, `orders/updated`, `orders/cancelled`, `products/create`, `products/update` registered on connect
- [ ] `POST /api/marketplace/webhook/shopify` handles incoming Shopify webhooks; triggers incremental sync for affected resources
- [ ] Orders normalized to `NormalizedOrder` — Shopify status mapped via `status_maps.py`
- [ ] Same SQL pipeline queries Shopify data identically to Amazon data (no separate query path)

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Create `api/connectors/shopify.py`: `ShopifyConnector` with `slug = "shopify"`, `display_name = "Shopify"` | Backend | 1h |
| 2 | Implement OAuth flow helpers: `get_authorization_url()`, `exchange_code_for_token()`, `validate_hmac_signature()` | Backend | 3h |
| 3 | Implement `fetch_orders_incremental()` with Shopify cursor pagination (`page_info` in `Link` header) | Backend | 2h |
| 4 | Implement `fetch_products_incremental()` and `fetch_inventory()` | Backend | 2h |
| 5 | Add `POST /api/marketplace/webhook/shopify` route with HMAC signature verification; trigger background incremental sync | Backend | 2h |
| 6 | Add Shopify status normalization to `status_maps.py` | Backend | 0.5h |
| 7 | Register `ShopifyConnector()` in `api/connectors/__init__.py` | Backend | 0.5h |
| 8 | Test: connect Shopify development store; verify orders sync; verify webhook triggers incremental update | Backend | 2h |

---

---

### EPIC-08: Smart Agent Auto-Routing

---

#### DOCBOT-801: Extend EXPERT_PERSONAS with Structured Output Contracts

**Story**
As a developer, I want each expert persona to define a structured output contract (required sections, detection keywords, output conventions) so that LLM responses are consistently structured per agent and can be rendered differently in the frontend.

**Phase**: 3
**Priority**: Must Have
**Story Points**: 3
**Dependencies**: None

**Acceptance Criteria**
- [ ] All 8 personas in `EXPERT_PERSONAS` have 5 new fields: `response_format`, `required_sections`, `detection_keywords`, `tool_preference`, `output_conventions`
- [ ] Each `persona_def` string ends with an OUTPUT FORMAT CONTRACT block that mandates section order and formatting rules
- [ ] Finance Expert responses include `## Key Metrics` table; Lawyer responses include `## Risk Flags` with `**RISK:**` prefixes; Doctor responses include `## Medical Disclaimer`
- [ ] `/api/personas` route exposes `response_format`, `detection_keywords`, and `output_conventions` in its JSON response
- [ ] No changes to any service file logic — only `EXPERT_PERSONAS` data and route response payload

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Add `response_format`, `required_sections`, `detection_keywords`, `tool_preference`, `output_conventions` fields to all 8 personas in `api/index.py` | Backend | 1.5h |
| 2 | Append OUTPUT FORMAT CONTRACT block to each `persona_def` string with agent-specific section rules | Backend | 1h |
| 3 | Update `/api/personas` route response to include new fields | Backend | 0.5h |
| 4 | Manual test: ask Finance Expert a financial question — verify `## Key Metrics` table appears in response | Full-stack | 0.5h |

---

#### DOCBOT-802: Client-Side Question Routing Function + Auto/Manual Mode State

**Story**
As a user, I want DocBot to automatically detect the best expert for my question so that I get a domain-appropriate response without having to pre-select a persona.

**Phase**: 3
**Priority**: Must Have
**Story Points**: 5
**Dependencies**: DOCBOT-801

**Acceptance Criteria**
- [ ] `routeQuestion(question, chatMode, isDbConnected, hasDocSession)` function exists in frontend and scores keywords against each persona's `detection_keywords`
- [ ] Confidence thresholds: score ≥ 6 = "high", score ≥ 3 = "medium", score < 3 = "low" (no routing change)
- [ ] `isAutoMode` state (default: `true`) controls whether routing overrides manual selection
- [ ] `Message` type extended with `agentPersona?: string` and `agentPersonas?: string[]`
- [ ] Each assistant message tagged with the persona that answered it
- [ ] "high" or "medium" confidence routes auto-selected persona to the API; "low" keeps current selection
- [ ] `tool_preference: "sql_first"` biases chatMode to "database" when DB is connected; `"rag_first"` biases to "docs"

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Extend frontend `EXPERT_PERSONAS` constant with new fields mirrored from backend | Frontend | 0.5h |
| 2 | Write `routeQuestion()` weighted keyword scorer function | Frontend | 1h |
| 3 | Add `isAutoMode` state and extend `Message` type with `agentPersona` | Frontend | 0.5h |
| 4 | Insert routing call in `handleSend` before API request; tag assistant message with persona used | Frontend | 1h |
| 5 | Console.log test: verify Finance/Lawyer/AI routing via browser console before wiring UI | Frontend | 0.5h |

---

#### DOCBOT-803: Sidebar Auto/Override UX Transformation

**Story**
As a user, I want the Expert Mode sidebar to default to "Auto" routing so that I don't need to manually pick a persona, but I can still override to a specific agent when I want.

**Phase**: 3
**Priority**: Must Have
**Story Points**: 3
**Dependencies**: DOCBOT-802

**Acceptance Criteria**
- [ ] Sidebar Expert Mode section shows AUTO / Manual toggle pills instead of the bare 2×4 grid
- [ ] AUTO mode is active by default on page load
- [ ] Existing 2×4 persona grid is collapsed inside a "Manual Override" expandable section
- [ ] Clicking a persona card in Manual Override: sets `isAutoMode = false` and pins that persona
- [ ] "Reset to Auto" link appears when a persona is manually pinned; click resets to Auto mode
- [ ] No functional regression — `selectedPersona` state and API `persona:` field still work correctly

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Replace sidebar persona section heading with AUTO / Manual pill toggle (Wand2 / UserCog icons) | Frontend | 1h |
| 2 | Wrap existing 2×4 grid in collapsible "Manual Override" div, collapsed when `isAutoMode = true` | Frontend | 0.5h |
| 3 | Add "Reset to Auto" small link below grid; show only when `!isAutoMode` | Frontend | 0.5h |
| 4 | Test: Auto mode → ask question → correct agent routes. Manual pin → ask question → pinned agent used. Reset → auto resumes. | Frontend | 0.5h |

---

#### DOCBOT-804: Per-Message Agent Badge Display

**Story**
As a user, I want to see which expert agent answered each message so that I understand why the response style and structure differs from other messages.

**Phase**: 3
**Priority**: Must Have
**Story Points**: 2
**Dependencies**: DOCBOT-802

**Acceptance Criteria**
- [ ] Each assistant message header shows a colored pill badge with the agent's name (e.g., "Finance Expert" in amber, "Lawyer" in red, "Doctor" in green)
- [ ] Badge color comes from `output_conventions.accent_color` for the agent that answered
- [ ] Hybrid queries (both doc + DB agents) show two side-by-side badges
- [ ] Badge does not appear on user messages
- [ ] If `agentPersona` is undefined (legacy messages), falls back to "DocBot" label

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Replace static "DocBot" span in message header with dynamic agent badge component | Frontend | 1h |
| 2 | Handle `agentPersonas` array case (hybrid): render two badges side by side | Frontend | 0.5h |
| 3 | Visual test: send messages in Auto mode for 3 different domains — verify correct badge color per response | Frontend | 0.5h |

---

#### DOCBOT-805: Per-Agent Response Rendering

**Story**
As a user, I want Finance Expert responses to show styled metric tables, Lawyer responses to highlight risk items, and Doctor responses to visually separate the medical disclaimer, so that each expert's output is immediately recognizable and easier to scan.

**Phase**: 3
**Priority**: Should Have
**Story Points**: 5
**Dependencies**: DOCBOT-801, DOCBOT-804

**Acceptance Criteria**
- [ ] Finance Expert: `## Key Metrics` table header row has amber background tint
- [ ] Lawyer: text matching the risk highlight pattern (`RISK`, `WARNING`, `BREACH`, `PENALTY`, etc.) rendered with red-tinted background highlight
- [ ] Doctor: `## Medical Disclaimer` section rendered in a green left-border callout box rather than plain markdown
- [ ] Data Analyst: SQL fenced code block under `## SQL Query` uses existing collapsible code component
- [ ] All other agents: standard ReactMarkdown rendering (no visual change)
- [ ] Per-agent rendering is applied only when `msg.agentPersona` is set and `response_format !== "general"`

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Extend `renderMessageContent` to detect `msg.agentPersona` and apply per-format pre-processing | Frontend | 1h |
| 2 | Finance: add wrapper div with `agent-finance` class; Tailwind `prose` override for table header amber | Frontend | 1h |
| 3 | Lawyer: apply `highlight_pattern` regex over content, wrap matches in `<mark>` with red-tinted style | Frontend | 1h |
| 4 | Doctor: detect `## Medical Disclaimer` heading + content, wrap in styled callout div | Frontend | 0.5h |
| 5 | Visual test each format: Finance table, Lawyer highlights, Doctor disclaimer box, Data SQL block | Frontend | 1h |

---

### EPIC-09: LangGraph Deep Research

---

#### DOCBOT-901: LangGraph Deep Research State Machine Core

**Story**
As a user enabling Deep Research, I want DocBot to decompose my question, search across multiple focused sub-questions, detect coverage gaps, and synthesize a comprehensive answer — instead of a single-shot LLM response.

**Phase**: 3+
**Priority**: Must Have
**Story Points**: 8
**Dependencies**: None
**Status**: ✅ Done (SCRUM-405/409, merged 2026-03-24)

---

#### DOCBOT-902: Streaming Queue Bridge + /api/chat Wiring

**Story**
As a developer, I want the Deep Research graph to stream progress events and answer tokens over the existing SSE transport so the user sees live step updates and token streaming instead of a blank wait.

**Phase**: 3+
**Priority**: Must Have
**Story Points**: 5
**Dependencies**: DOCBOT-901
**Status**: ✅ Done (SCRUM-410, merged 2026-03-24)

---

#### DOCBOT-903: Frontend Deep Research Progress Strip UI

**Story**
As a user in Deep Research mode, I want to see a live progress strip showing which step the graph is on (planning, searching, evaluating, composing) so I understand what's happening instead of watching a spinner.

**Phase**: 3+
**Priority**: Must Have
**Story Points**: 3
**Dependencies**: DOCBOT-902
**Status**: ✅ Done (SCRUM-411, merged 2026-03-24)

---

#### DOCBOT-904: Unit + Integration Tests for Deep Research

**Story**
As a developer, I want comprehensive unit tests for `_parse_json_list` and `gap_router` so that CI catches regressions in the core graph logic without requiring API keys.

**Phase**: 3+
**Priority**: Must Have
**Story Points**: 3
**Dependencies**: DOCBOT-901
**Status**: ✅ Done (SCRUM-412, merged 2026-03-24)

---

### EPIC-10: RAG Quality Enhancement

> **PageIndex Evaluation Decision (2026-03-25):** VectifyAI's PageIndex was evaluated as a full RAG replacement. Decision: **Do not integrate.** Hard blockers:
> 1. **OpenAI-only** — PageIndex requires `gpt-4o`. DocBot uses Groq for cost efficiency. Adding OpenAI adds ~$0.02–$0.10/doc at index time and ~$0.01–$0.04/query. Cannot be swapped — the library is not designed for backend abstraction.
> 2. **Not on PyPI** — Must install via GitHub clone. Railway builds would depend on an unversioned external repo with no deprecation guarantees.
> 3. **No streaming** — PageIndex returns complete JSON responses. DocBot's entire pipeline is SSE-streaming-first; every doc query would block token output until retrieval completes.
>
> Problems PageIndex solves are already partially addressed: LangExtract + Gemini 2.5 Flash handles structured financial extraction; multi-query expansion handles semantic mismatch; AcroForm prepending handles form fields. EPIC-10 fixes the remaining RAG limitations within the existing stack — no new vendors, no new API keys.
>
> **Revisit PageIndex when:** PyPI package ships + pluggable LLM backend supporting Groq or local models is available.

---

#### DOCBOT-1001: Replace InMemoryVectorStore with Chroma Persistent Store

**Story**
As a developer, I want the document vector store to persist across Railway restarts so that users do not lose their uploaded document sessions when the backend redeploys.

**Phase**: 4+
**Priority**: Must Have
**Story Points**: 5
**Dependencies**: None
**Status**: 🔜 To Do

**Acceptance Criteria**
- [ ] `InMemoryVectorStore` replaced with `Chroma` in `api/index.py`; collections keyed by `session_id`
- [ ] In-memory `VECTOR_STORES` dict removed; Chroma handles collection lifecycle
- [ ] `rag_retrieve()` in `api/hybrid_service.py` uses Chroma retriever (interface unchanged)
- [ ] `parallel_retriever` in `api/deep_research_service.py` updated to use Chroma
- [ ] Unit test: write docs → delete Python collection object → re-fetch → assert docs still present

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Add `chromadb` to `requirements.txt` | Backend | 0.25h |
| 2 | Replace `InMemoryVectorStore` with `Chroma.get_or_create_collection(session_id)` in `api/index.py` | Backend | 2h |
| 3 | Update `rag_retrieve()` in `api/hybrid_service.py` to use Chroma retriever | Backend | 1h |
| 4 | Update `parallel_retriever` node in `api/deep_research_service.py` | Backend | 1h |
| 5 | Write unit test for persistence across collection object deletion | Backend | 1h |

**Branch**: `feature/DOCBOT-1001-chroma-persistent-store`

---

#### DOCBOT-1002: Cross-Encoder Reranker Post-Retrieval

**Story**
As a user, I want document retrieval to surface the most relevant chunks first so that the LLM's synthesis context is precise and answers are more accurate.

**Phase**: 4+
**Priority**: Should Have
**Story Points**: 3
**Dependencies**: DOCBOT-1001
**Status**: 🔜 To Do

**Acceptance Criteria**
- [ ] `api/utils/reranker.py` exports `rerank(question: str, docs: list[Document], top_k: int) -> list[Document]`
- [ ] Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, zero API cost)
- [ ] After k=8 cosine retrieval in `rag_retrieve()`, reranker reduces to top-5
- [ ] Reranker skipped if fewer than 3 docs returned (no-op fallback)
- [ ] Unit test: mock cross-encoder, verify output ordering and count truncation

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Add `sentence-transformers` cross-encoder to `requirements.txt` | Backend | 0.25h |
| 2 | Create `api/utils/reranker.py` with `rerank()` function | Backend | 1.5h |
| 3 | Wire `rerank()` into `rag_retrieve()` in `api/hybrid_service.py` | Backend | 1h |
| 4 | Write unit test in `tests/unit/test_reranker.py` | Backend | 1h |

**Branch**: `feature/DOCBOT-1002-cross-encoder-reranker`

---

#### DOCBOT-1003: SemanticChunker for Financial and Legal Documents

**Story**
As Maya (Finance Manager), I want financial documents chunked along natural semantic boundaries so that tables and financial data are not split across chunks, improving answer accuracy.

**Phase**: 4+
**Priority**: Should Have
**Story Points**: 3
**Dependencies**: DOCBOT-1001
**Status**: 🔜 To Do

**Acceptance Criteria**
- [ ] `SemanticChunker` from `langchain_experimental` used for `financial` and `legal` doc types in `upload_documents()` in `api/index.py`
- [ ] `RecursiveCharacterTextSplitter` (1500/200) kept for general, medical, research docs
- [ ] Branching uses `detect_document_type()` output (already called at upload time in `api/document_extractor.py`)
- [ ] `SemanticChunker` reuses the existing cached `get_embeddings()` model — no new deps or API keys
- [ ] Integration test: upload two-page financial PDF, assert no chunks split mid-sentence at table boundaries

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Add `langchain-experimental` to `requirements.txt` if not present | Backend | 0.25h |
| 2 | Import `SemanticChunker` and add doc-type branching logic in `upload_documents()` | Backend | 1.5h |
| 3 | Write integration test for financial PDF chunking | Backend | 1h |

**Branch**: `feature/DOCBOT-1003-semantic-chunker`

---

#### DOCBOT-1004: FinanceBench Accuracy Baseline Test Suite

**Story**
As a developer, I want an automated accuracy test harness against FinanceBench questions so I can measure before/after retrieval improvements and catch regressions when the RAG pipeline changes.

**Phase**: 4+
**Priority**: Should Have
**Story Points**: 5
**Dependencies**: DOCBOT-1001, DOCBOT-1002, DOCBOT-1003
**Status**: 🔜 To Do

**Acceptance Criteria**
- [ ] `tests/external/test_financebench_accuracy.py` with 20 curated FinanceBench questions + ground truth answers
- [ ] Marked `@pytest.mark.external` — skipped in CI (requires live Groq + HuggingFace keys)
- [ ] Reports per-question pass/fail + aggregate accuracy (exact match + ±2% numeric fuzzy match)
- [ ] Baseline LangChain RAG score recorded as comment at top of file BEFORE DOCBOT-1001–1003 merge
- [ ] Post-improvement score recorded after — target: ≥15% improvement
- [ ] Results pasted into ticket comments before closing

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Create `tests/external/` with `__init__.py` | Backend | 0.25h |
| 2 | Source 20 FinanceBench questions with verified ground truth answers | Backend | 2h |
| 3 | Write `test_financebench_accuracy.py` with fuzzy numeric matcher | Backend | 2h |
| 4 | Run baseline (pre-improvement) and record score | Backend | 1h |
| 5 | Run post-improvement and record delta | Backend | 0.5h |

**Branch**: `feature/DOCBOT-1004-financebench-tests`

---

## 4. Sprint Plan — Phase 0 + Phase 1

### Current Status (as of 2026-03-24)

| Sprint | Tickets | Points | Status |
|--------|---------|--------|--------|
| Sprint 0 | DOCBOT-101, 102, 103 | 18 | ✅ Complete |
| Sprint 1 | DOCBOT-201, 202, 203, 204, 205, 206, 207 | 49 | ✅ Complete |
| Sprint 2 | DOCBOT-301, 302, 303, 304 | 20 | ✅ Complete |
| Sprint 3 | DOCBOT-401, 402, 403, 404, 406 | 35 | ✅ Complete |
| Phase 1 | All Phase 1 tickets | 122 | ✅ Complete |
| Enterprise Add-on | DOCBOT-208 (Azure SQL / Entra auth) | 8 | ✅ Complete |
| Phase 2 Sprint 1 | DOCBOT-501, 503, 502, 504, 305, 405 | 44 | ✅ Complete |
| Phase 2 Bug Fixes | Chart rendering (Qwen3 `<think>` strip), Autopilot auto-detect nudge | — | ✅ Complete |
| Phase 3 Sprint 1 | DOCBOT-801, 802, 803, 804, 805 | 18 | ✅ Complete |
| Phase 3 Fixes | Persona format contract removal, routing fallback fix, AcroForm RAG fix, SSE streaming, parallel retrieval | — | ✅ Complete |
| EPIC-09 Sprint 1 | DOCBOT-901, 902, 903, 904 | 19 | ✅ Complete |
| Enterprise Data Pipeline Hardening | CSV section splitter, DB pipeline upgrades, hybrid routing fix | — | ✅ Complete |
| EPIC-10 Sprint 1 | DOCBOT-1001, 1002, 1003, 1004 | 16 | 🔄 To Do |
| EPIC-07 Phase 1 | DOCBOT-701, 702, 703 (connector interface + commerce schema + Amazon SP-API) | 29 | 🔄 To Do |
| Investor Readiness | CI pipeline, landing page, metrics endpoint, LLM fallback, frontend split | — | 🔄 To Do |
| Human Testing | 85-test manual regression across all features | — | 🔄 To Do |

**Total delivered**: 198 story points across 31 tickets + full test suite (385 tests) | **Remaining**: EPIC-10 (16 pts) + EPIC-07 Phase 1 (29 pts) + investor readiness = **~20 day sprint to investor-demo-ready**

---

**Phase 2 Complete** — All 6 Phase 2 tickets shipped. Two post-ship bugs resolved:
- Fixed DB chat charts not rendering (Qwen3 `<think>` blocks in generated Python broke E2B sandbox execution)
- Added Autopilot auto-detect nudge (keyword-triggered suggestion banner above input)

**Phase 3 Complete** — EPIC-08: Smart Agent Auto-Routing (DOCBOT-801–805, 18 points). Replaced static persona picker with intelligent per-question routing (routeQuestion() keyword scorer), AUTO/Manual sidebar toggle, colored agent badges on messages, per-agent response rendering.

**Phase 3 Post-Ship Fixes (2026-03-24)**:
- Removed rigid OUTPUT FORMAT CONTRACT from all personas — was forcing identical rigid sections on every response. Personas now tone-only; Deep Research toggle enables structured output.
- Fixed routing fallback: low-confidence routing now falls back to Generalist (not upload-recommended persona)
- Fixed AcroForm field extraction: fillable PDFs (LCA, I-9, tax forms) now have widget values extracted before chunking — was invisible to `fitz.get_text()`
- Added `api/utils/query_expansion.py`: static synonym expansion for short queries (no LLM calls) — "His position?" now searches 6 parallel variants
- Converted `/api/chat` from buffered JSON to SSE streaming — eliminates Railway 30s timeout on Deep Research
- Parallelized multi-query retrieval via `ThreadPoolExecutor + asyncio.gather` — reduced retrieval latency ~75%

**EPIC-09 Complete — LangGraph Deep Research (DOCBOT-901–904, 19 points)**:
Replaced single-shot `DEEP_RESEARCH_ADDON` prompt with a proper 5-node LangGraph state machine:
- `query_planner` (LLM #1): decomposes question into 3–5 focused sub-questions
- `parallel_retriever`: concurrent vector search per sub-question using synonym expansion
- `evidence_evaluator`: deterministic coverage scoring — identifies gaps
- `gap_router`: loops back to retriever if gaps exist and iterations < 2 (0 LLM calls)
- `synthesizer` (LLM #2): streams comprehensive structured answer with per-section citations
- asyncio.Queue bridge streams progress events + tokens to frontend in real time
- Frontend progress strip: 🧠 → 🔍 → ✅ → 📝 with live step messages
- Max 2 LLM calls per request (hard ceiling)

**Enterprise Data Pipeline Hardening (2026-03-25)** — Cross-cutting improvements to CSV, DB, and hybrid pipelines:

*Hybrid Routing Fix (4 commits):*
- Fixed routing misfire when both PDF and CSV are connected — `effectiveChatMode` local variable prevents async setState race condition
- Tool preference override (`sql_first`, `rag_first`) now only fires for single-source scenarios, not when both sources are present
- Backend intent classifier decides per-question when both sources are active

*Enterprise CSV Pipeline (new module: `api/utils/csv_preprocessor.py`):*
- Multi-section CSV detection: detects EXHIBIT/SHEET/TABLE/SECTION/APPENDIX boundaries in Excel exports
- Per-section metadata extraction: columns, row counts, header detection with >=40% non-null string heuristic
- `load_section(idx)` pattern: E2B sandbox preamble generates `_SECTIONS` dict + helper function for LLM to query any section
- Mandatory data cleaning preamble: column normalization, `nan` column removal, currency/percentage parsing, date preservation
- Section manifest injected into LLM prompt for multi-exhibit awareness
- Tested against real 7-exhibit financial CSV (497 rows, Elon Musk W30170-XLS-ENG.csv)

*Enterprise DB Pipeline Upgrades:*
1. **Connection engine pooling** — LRU cache (max 20) with `pool_pre_ping`, 30-min recycle. Replaces create/dispose per query.
2. **Views support** — `inspector.get_view_names()` included alongside tables, tagged `is_view: True` for LLM awareness
3. **Higher caps** — Table cap 50→200, column preview 10→20 per table, table selector limit 5→8
4. **Structured error taxonomy** — `classify_db_error()` maps raw driver exceptions to actionable messages (auth, network, SSL, missing objects, syntax, timeouts, deadlocks)
5. **Schema drift detection** — On table/column-not-found execution errors, automatically invalidates cache, re-introspects, regenerates SQL, retries once
6. **Manual schema refresh** — New `POST /api/db/refresh-schema/{connection_id}` endpoint

*Hybrid Synthesis Fix:*
- `_collect_sql_result()` now captures both `metadata` AND `token` chunks — CSV pandas output was invisible to synthesis because only metadata chunks were collected

*asyncio Fix:*
- All `asyncio.get_event_loop()` → `asyncio.get_running_loop()` across db_service.py, table_selector.py, file_upload_service.py (Python 3.12+ deprecation)

*Test suite: 385 tests passing (was 263).*

**EPIC-10 Active — RAG Quality Enhancement (DOCBOT-1001–1004, 16 points)**:
PageIndex (VectifyAI) evaluated 2026-03-25 and rejected: OpenAI-only backend, no PyPI package, no streaming support. Replacing `InMemoryVectorStore` with Chroma for persistence, adding cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) for retrieval precision, adding `SemanticChunker` for financial/legal documents, and establishing a FinanceBench accuracy baseline to measure improvement delta.

**EPIC-07 Phase 1 Unblocked — Amazon Commerce Connector (DOCBOT-701–703, 29 points)**:
Gate lifted 2026-03-25 for investor demo sprint. Building connector interface + credential vault (DOCBOT-701), unified commerce schema with multi-tenant RLS (DOCBOT-702), and Amazon SP-API connector (DOCBOT-703). Background sync worker (DOCBOT-704) and Shopify connector (DOCBOT-705) deferred to post-funding.

---

### Audit Reality — Feature Status Table (2026-03-25)

> Brutal audit: what actually ships vs. what the marketing says. Fix before investor demo.

| Feature | Claimed | Actual Status | Fix Required |
|---------|---------|---------------|--------------|
| Hybrid discrepancy detection | "Flags conflicts between docs and DB with delta %" | **PARTIAL → FIXED (2026-03-25)** — `api/utils/discrepancy_detector.py` now extracts numeric values from both sources, matches by label similarity, computes deltas in code. 29 tests passing. | ✅ Done |
| Smart agent per-question routing | "Per-question routing to the right expert" | **STUB** — upload-time keyword count only. `detection_keywords` in persona definitions are never read at query time. `/api/chat` uses persona passed from frontend, not re-classified per question. | Fix #2 |
| Chroma persistent store | "Documents survive server restarts" | **MISSING** — `InMemoryVectorStore`. All uploaded docs lost on every Railway restart / redeploy. | Fix #3 (DOCBOT-1001) |
| RBAC route enforcement | "Viewer/analyst/admin role enforcement" | **PARTIAL** — `require_role()` exists but `/api/chat`, `/api/db/chat`, `/api/hybrid/chat`, `/api/autopilot/run` have no `Depends(require_role(...))`. Non-SSO users bypass all role checks. | Fix #4 |
| Audit log — SQL execution | "All queries are audited" | **PARTIAL** — upload, login/logout, db_connect/disconnect are logged. SQL query execution is never logged. IP address missing from all events. | Fix #5 |
| PII masking | "PII auto-masked in responses" | **PARTIAL** — email/phone/SSN patterns in structured DB results. Not applied to CSV E2B output or LLM answer text. No international formats. | Fix #6 |
| Deep Research LangGraph | "5-node state machine" | **REAL** — plan→retrieve→evaluate→gap→synthesize fully implemented and tested. | ✅ Shipped |
| E2B Python sandbox | "Run Python in isolated sandbox" | **REAL** — E2B code-interpreter, 25s timeout, finally cleanup. | ✅ Shipped |
| 7-step SQL pipeline | "Bounded 2–3 LLM calls" | **REAL** — sqlglot AST validation + executor + drift retry. | ✅ Shipped |
| SAML SSO | "Okta/Azure AD enterprise SSO" | **REAL** — python3-saml, JIT provisioning, session cookies. | ✅ Shipped |

---

### Investor Demo Sprint — 23-Day Revised Plan (2026-03-25)

**Goal**: Fix the three critical gaps first (real behaviour ≠ marketing claims), then ship EPIC-10 + EPIC-07 Phase 1 + polish. Product must be demo-ready and honest.

| Day | Phase | Work Item | Deliverable |
|-----|-------|-----------|-------------|
| 1 | **Fix #1** | ~~Discrepancy detection~~ | ✅ **Done** — `api/utils/discrepancy_detector.py`, 29 tests |
| 2-3 | **Fix #2** | Per-question persona routing | Read `detection_keywords` at query time; re-classify persona on `/api/chat` before synthesis |
| 4 | **Fix #3** | DOCBOT-1001: Chroma persistent store | `api/utils/vector_store.py`, replace `InMemoryVectorStore` throughout |
| 5 | **Fix #4** | RBAC route guards | Add `Depends(require_role("viewer"))` to `/api/chat`, `/api/db/chat`, `/api/hybrid/chat`, `/api/autopilot/run` |
| 5 | **Fix #5** | Audit SQL execution + IP | Log `query_executed` event in `run_sql_pipeline`; extract IP from request headers |
| 6 | EPIC-10 | DOCBOT-1002: Cross-encoder reranker | `api/utils/reranker.py`, wired into `rag_retrieve()` |
| 7 | EPIC-10 | DOCBOT-1003: SemanticChunker | `api/utils/chunker.py`, doc-type branching in upload route |
| 7-8 | EPIC-10 | DOCBOT-1004: FinanceBench baseline | `tests/benchmarks/`, 20 questions, before/after accuracy delta |
| 9-10 | Polish | Frontend route splitting | `page.tsx` < 600 lines, components in `src/components/` |
| 10 | Polish | Landing page | Auth gate, hero + features + CTA at `/` |
| 11 | Polish | Metrics endpoint + CI pipeline | `api/metrics_service.py`, `GET /admin/metrics`, GitHub Actions enhanced |
| 12 | Polish | LLM fallback | `api/utils/llm_provider.py` — Groq primary, Gemini 2.5 Flash fallback |
| 13-15 | EPIC-07 | DOCBOT-701: Connector interface + credential vault | `api/connectors/base.py`, `registry.py`, `credential_service.py` |
| 16-18 | EPIC-07 | DOCBOT-702: Unified commerce schema + RLS | 6 tables, RLS policies, materialized views |
| 19-22 | EPIC-07 | DOCBOT-703: Amazon SP-API connector | `api/connectors/amazon_sp.py`, LWA OAuth, Orders + Finances sync |
| 23 | Testing | Human testing: 85 tests across all feature areas + final deploy | Investor-demo-ready production |

**Exit Criteria:**
1. All 85 manual tests pass on production (Railway + Vercel)
2. All 6 audit fixes confirmed resolved — no gap between marketing claims and code
3. EPIC-10 complete: Chroma + reranker + SemanticChunker active
4. FinanceBench accuracy documented (target: >60%)
5. Amazon SP-API connector live: connect → sync orders → NL questions answered
6. CI green with 420+ tests
7. Landing page live, metrics endpoint working, LLM fallback tested
8. Zero P0/P1 bugs from human testing

---

### Sprint 0 — Infrastructure Foundation (Days 1–14)

**Goal**: Production backend on Railway, PostgreSQL running, E2B validated. All v1 features still work.

**Stories in Sprint**:

| Story | Title | Points | Owner |
|-------|-------|--------|-------|
| DOCBOT-101 | Backend Migration to Railway | 5 | Full-stack |
| DOCBOT-102 | PostgreSQL Session Store | 5 | Backend |
| DOCBOT-103 | E2B Sandbox Integration | 8 | Backend |

**Total Points**: 18

**Sprint 0 Exit Criteria**:
- GET /api/health returns 200 from Railway URL
- All v1 routes (chat, upload, sessions) work against Railway + PostgreSQL
- E2B sandbox executes pandas code and returns a matplotlib chart as base64 PNG
- Zero regressions on existing v1 functionality

**Daily Checkpoints**:
- Day 3: Railway service deployed and accessible
- Day 7: PostgreSQL session store live; v1 routes tested
- Day 11: E2B sandbox smoke test passing
- Day 14: Full v1 regression test; sprint review

---

### Sprint 1 — DB Connection + SQL Pipeline (Days 15–28)

**Goal**: User can connect a PostgreSQL or MySQL database and ask natural language questions against it.

**Stories in Sprint**:

| Story | Title | Points | Owner |
|-------|-------|--------|-------|
| DOCBOT-201 | Database Connection API | 8 | Backend |
| DOCBOT-202 | Schema Introspection and Cache | 5 | Backend |
| DOCBOT-203 | SQL Generation Pipeline | 13 | Backend |
| DOCBOT-304 | Data Analyst Persona | 2 | Full-stack |

**Total Points**: 28

**Sprint 1 Exit Criteria**:
- User can POST /api/db/connect with PostgreSQL credentials and receive a `connection_id`
- Credentials stored encrypted; plain-text password not in any log
- SQL generated for "show total revenue by month" against a test DB
- sqlglot validator rejects `DROP TABLE` and `INSERT INTO`
- Data Analyst persona visible in persona selector

**Daily Checkpoints**:
- Day 18: Connection API + encryption working
- Day 21: Schema introspection + cache working
- Day 25: SQL pipeline end-to-end (NL → SQL → execute → answer) working
- Day 28: Sprint review; test with real PostgreSQL dump

---

### Sprint 2 — DB Chat + Python Execution + Charts (Days 29–42)

**Goal**: Full DB chat route live, Python analysis executes, charts render in browser.

**Stories in Sprint**:

| Story | Title | Points | Owner |
|-------|-------|--------|-------|
| DOCBOT-204 | DB Chat API Route | 5 | Backend |
| DOCBOT-205 | MySQL Connector | 3 | Backend |
| DOCBOT-301 | Python Code Generation | 8 | Backend |
| DOCBOT-302 | Python Code Execution via E2B | 5 | Backend |
| DOCBOT-303 | Chart Rendering in Chat | 5 | Frontend |

**Total Points**: 26

**Sprint 2 Exit Criteria**:
- Maya scenario works end-to-end: connect PostgreSQL → ask revenue question → get answer + chart
- MySQL connection tested with a real MySQL dump
- Chart renders inline in chat response
- Code display block visible and collapsible

---

### Sprint 3 — Hybrid Mode + Frontend Polish (Days 43–56)

**Goal**: Hybrid mode ships. User can connect a PDF + DB and ask a question that spans both.

**Stories in Sprint**:

| Story | Title | Points | Owner |
|-------|-------|--------|-------|
| DOCBOT-401 | Query Intent Classifier | 3 | Backend |
| DOCBOT-402 | Hybrid Chat Pipeline | 13 | Backend |
| DOCBOT-403 | Discrepancy Detection | 8 | Backend |
| DOCBOT-404 | HybridModeToggle UI | 3 | Frontend |

**Total Points**: 27

**Sprint 3 Exit Criteria**:
- The 60-second demo script works: board deck PDF + PostgreSQL → revenue attainment answer with discrepancy flagged
- Dual citations appear in hybrid answer
- HybridModeToggle shows/hides based on available sources
- Discrepancy section highlighted visually in chat

---

### Sprint 4 — Phase 1 Hardening + DB Frontend (Days 57–70)

**Goal**: Frontend components for DB complete; edge cases handled; ready for beta users.

**Stories in Sprint**:

| Story | Title | Points | Owner |
|-------|-------|--------|-------|
| DBConnectForm UI | Dialect selector, connection fields, test button | 3 (inline) | Frontend |
| SchemaBrowser UI | Collapsible table/column explorer | 3 (inline) | Frontend |
| SQLDisplay UI | Syntax-highlighted SQL + copy button | 2 (inline) | Frontend |
| ResultsTable UI | Virtualized table for query results | 3 (inline) | Frontend |
| Error handling review | All error types return structured JSON | 3 (inline) | Backend |

**Sprint 4 Exit Criteria**:
- All 8 frontend components from the spec are implemented and functional
- 5 beta users from discovery interviews onboarded
- Query Trust Rate baseline established (are users clicking "run" on shown SQL?)

---

## 5. Tracking Tool Recommendation

### Recommendation: Linear

**For a 1–2 person team building an ambitious technical product, use Linear.**

**Why not the alternatives:**

| Tool | Why It Fails for This Team |
|------|---------------------------|
| Jira | Configuration overhead is 4–6 hours before you write a single story. The UI actively slows down fast-moving small teams. Built for 10+ person enterprise teams. |
| Notion | Flexible but no built-in issue tracking primitives. You'll spend time building the system instead of building the product. |
| GitHub Projects | Adequate, but no cycle time analytics, no priority algorithms, weaker sprint tooling. Fine as a last resort. |
| Trello | Card-only model does not support sub-tasks or story points natively. You will outgrow it by Sprint 2. |
| Asana | Better for project management than engineering. Missing sprint velocity, cycle time, and PR linking. |

**Why Linear works for DocBot v2:**

1. **GitHub integration**: Linear auto-closes issues when PRs are merged. For a solo or pair developer, this removes the entire administrative layer of manually updating ticket status.

2. **Sprint cycles built-in**: Linear's "Cycles" map directly to the 2-week sprints above. Velocity tracking starts automatically from Sprint 1 with zero configuration.

3. **Fast keyboard-driven UI**: Create a story in 10 seconds with `C`. No dropdowns, no forms. For a team that is also writing code, speed of capture matters.

4. **Priority algorithm**: Linear's automatic priority sorting means you always see the most important unblocked work at the top without manually sorting.

5. **Sub-issues**: The story + task hierarchy in this document maps directly to Linear's issue + sub-issue model.

**How to structure the Linear board:**

```
Workspace: DocBot

Teams:
  - Engineering (all issues live here)

Projects (map to Epics):
  - EPIC-01: Infrastructure Migration
  - EPIC-02: Database Connectivity
  - EPIC-03: Analytical Loop (Python)
  - EPIC-04: Hybrid Intelligence
  - EPIC-05: Memory and Context
  - EPIC-06: Enterprise Readiness

Cycles (map to Sprints):
  - Sprint 0: Infrastructure Foundation
  - Sprint 1: DB Connection + SQL Pipeline
  - Sprint 2: DB Chat + Python + Charts
  - Sprint 3: Hybrid Mode
  - Sprint 4: Hardening + Frontend

Labels:
  - backend
  - frontend
  - devops
  - full-stack
  - security
  - blocked

Priorities (use Linear defaults):
  - Urgent = Must Have (current sprint)
  - High = Must Have (next sprint)
  - Medium = Should Have
  - Low = Could Have

Views to create:
  1. "Active Sprint" — filter by current cycle, grouped by status
  2. "My Work" — filter by assignee = me
  3. "Blocked" — filter by label = blocked
  4. "Backlog by Epic" — all unstarted issues grouped by project
```

**Identifier setup:**
Set the team identifier to `DOCBOT` so issue IDs become `DOCBOT-101`, `DOCBOT-201`, etc. exactly matching this document.

**One workflow rule to add:**
When a PR is opened with `DOCBOT-XXX` in the title → move issue to "In Review" automatically. When PR is merged → move to "Done". This takes 2 minutes to configure and eliminates all manual status updates.

---

## 6. Risk Register Reference

These risks should be tracked as Linear "issues" with label `risk` and priority `Urgent` on the backlog. Review at the start of each sprint.

| Risk ID | Risk | Sprint to Mitigate | Mitigation |
|---------|------|--------------------|-----------|
| RISK-01 | Users won't share production DB credentials | Before Sprint 1 | Ship SQLite file upload first (DOCBOT-206). Prove value before asking for trust. |
| RISK-02 | SQL hallucination — valid syntax, wrong business logic | Sprint 2 onward | Always show SQL + plain-English explanation. User verifies before trusting (Query Trust Rate metric). |
| RISK-03 | Vercel/Railway timeout on hybrid queries | Sprint 3 | Streaming responses in DOCBOT-402. Test worst-case latency in Sprint 2. |
| RISK-04 | Target users have data in Google Sheets, not SQL | Pre-Sprint 0 | 5 discovery interviews. If true, build Google Sheets connector before PostgreSQL. |
| RISK-05 | E2B sandbox cold start > 10 seconds | Sprint 2 | Pre-warm sandbox on DB connect. Cache sandbox handle per session if E2B supports it. |
| RISK-06 | PII in query results exposed to LLM | Sprint 2 | Add basic PII masking in Sprint 2 even before DOCBOT-604 (Phase 4 full version). Simple regex pass before LLM synthesis. |
| RISK-07 | Competitor (Vanna.ai) ships document integration | Continuous | The persona depth is the moat. Generic RAG + SQL is not Finance Expert reasoning. Move to Sprint 3 fast. |

---

## Story Points Summary by Phase

| Phase | Stories | Total Points | Status |
|-------|---------|-------------|--------|
| Phase 0 | DOCBOT-101, 102, 103 | 18 pts | ✅ Done |
| Phase 1 | DOCBOT-201–208, 301–304, 401–406 | 122 pts | ✅ Done |
| Phase 2 | DOCBOT-305, 405, 501–504 | 44 pts | ✅ Done |
| Phase 3 | DOCBOT-801–805, 901–904 | 37 pts | ✅ Done |
| Phase 4 (Enterprise) | DOCBOT-601–605, DOCBOT-701 (consumer auth) | 58 pts | ✅ Done |
| Phase 4 (Commerce — Remaining) | DOCBOT-701 (connector), 702–705 | ~45 pts | 🔜 Planned |
| **Delivered total** | **38 stories + post-ship fixes** | **~279 pts** | ✅ |

Note: EPIC-07 (Commerce Connectors) is the only outstanding work. Gate condition: ≥3 of 5 discovery interviews must confirm the commerce/seller segment before starting.

---

## Immediate Next Actions (This Week)

All phases through Phase 4 (Enterprise) are complete. The only remaining work is EPIC-07 (Commerce Connectors), which is gated on discovery interviews.

1. Conduct ≥3 of 5 discovery interviews to validate the commerce/seller segment (RISK-04)
2. If interviews confirm the segment: begin DOCBOT-701 (Marketplace Connector Interface) as the EPIC-07 foundation
3. If interviews do not confirm the segment: defer EPIC-07 indefinitely; prioritize direct user feedback on shipped features
