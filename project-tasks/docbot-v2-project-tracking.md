# DocBot v2 — Complete Project Tracking Document
> Generated: 2026-03-17
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

| Epic ID | Epic Name | Phase(s) | Description |
|---------|-----------|----------|-------------|
| EPIC-01 | Infrastructure Migration | 0 | Move backend to Railway, PostgreSQL session store, E2B integration |
| EPIC-02 | Database Connectivity | 1, 3 | DB connections, SQL generation pipeline, query execution |
| EPIC-03 | Analytical Loop (Python) | 1, 2 | Python code execution via E2B, chart rendering, analysis |
| EPIC-04 | Hybrid Intelligence | 1, 2 | Cross-source synthesis, discrepancy detection, planner/router |
| EPIC-05 | Memory and Context | 2 | Session artifacts, context compression, multi-hop queries |
| EPIC-06 | Enterprise Readiness | 4 | SSO, RBAC, audit logging, PII masking, on-premise |
| EPIC-07 | Commerce Connectors | 4+ | Marketplace API integrations (Amazon SP-API, Shopify), unified commerce schema, multi-tenant RLS, background sync |

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

#### DOCBOT-208: Microsoft Entra (Azure AD) Service Principal Auth for Azure SQL

**Story**
As an enterprise user, I want to connect to Azure SQL Database using Microsoft Entra Service Principal credentials (tenant_id, client_id, client_secret), so that I can query my organization's Azure SQL databases without exposing username/password credentials.

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
- [x] Frontend: Azure SQL option in dialect dropdown with conditional Tenant ID / Client ID / Client Secret fields
- [x] 131 unit tests passing (including 15 new tests for Azure SQL/Entra)

**Implementation Notes**
- `_resolve_connection(creds)` dispatches to Entra or standard credential path for all dialects
- Tokens re-acquired on every `get_schema()` / `run_sql_pipeline()` call (no server-side token cache; tokens valid 60–75 min)
- `SET TRANSACTION ISOLATION LEVEL SNAPSHOT` used for read-only enforcement (Azure SQL doesn't support PostgreSQL's `SET TRANSACTION READ ONLY`)
- `pyodbc>=5.0.0` and `azure-identity>=1.17.0` added to `requirements.txt`

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

#### DOCBOT-305: Advanced Analysis (Phase 2)

**Story**
As Maya, I want the system to perform statistical forecasting on my data, so that I can see projected trends, not just historical summaries.

**Phase**: 2
**Priority**: Should Have
**Story Points**: 8
**Dependencies**: DOCBOT-302

**Acceptance Criteria**
- [ ] `statsmodels` and `scipy` available in E2B sandbox
- [ ] For time-series data, system can generate a simple linear trend forecast
- [ ] Forecast output includes confidence interval
- [ ] System does not attempt forecasting on non-time-series data
- [ ] Forecast chart labeled clearly with "Actual" vs. "Projected" series

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Update E2B sandbox template (or requirements) to include `statsmodels`, `scipy` | DevOps | 1h |
| 2 | Add "time-series detection" logic to Python code generator (check for date columns) | Backend | 2h |
| 3 | Add forecasting prompt branch in `generate_analysis_code()` for time-series data | Backend | 2h |
| 4 | Test: 12-month revenue DataFrame → verify forecast chart with confidence bands generated | Backend | 2h |

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

#### DOCBOT-405: Multi-Step Planner (Phase 2)

**Story**
As Sarah, I want to ask complex multi-part questions that require several sequential steps to answer, so that I can get compound insights without asking multiple separate questions.

**Phase**: 2
**Priority**: Should Have
**Story Points**: 13
**Dependencies**: DOCBOT-402

**Acceptance Criteria**
- [ ] Planner decomposes complex questions into a sequence of sub-steps (max 5)
- [ ] Sub-steps can include: sql_query, doc_retrieve, python_analyze, synthesize
- [ ] Each sub-step result is passed as context to the next step
- [ ] User sees a "Thinking" progress indicator showing each step as it completes
- [ ] Total execution time capped at 45 seconds; returns partial answer if exceeded

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Design `PlannerStep` data model: `{step_type, description, dependencies, result}` | Backend | 1h |
| 2 | Implement `plan_query()` LLM call that returns a JSON array of steps | Backend | 3h |
| 3 | Implement sequential step executor with context threading | Backend | 4h |
| 4 | Add streaming progress events via Server-Sent Events for step completion | Backend | 2h |
| 5 | Create `ThinkingIndicator` frontend component showing active step | Frontend | 2h |

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

**Acceptance Criteria**
- [ ] SAML 2.0 SP-initiated SSO flow works with Okta test tenant
- [ ] SAML 2.0 SP-initiated SSO flow works with Azure AD test tenant
- [ ] User attributes (email, name, groups) mapped from SAML assertion to local user record
- [ ] JIT (just-in-time) user provisioning creates account on first SSO login
- [ ] Session cookie is secure, HttpOnly, SameSite=Strict

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

**Acceptance Criteria**
- [ ] Every query event logged: user_id, timestamp, session_id, question_hash, SQL executed, row count returned
- [ ] Audit log is append-only (no UPDATE or DELETE on audit table, enforced at DB level via trigger)
- [ ] Audit records exportable as CSV via admin endpoint
- [ ] Log includes connection events: DB connected, DB disconnected, by whom, when
- [ ] PII (raw question text) stored hashed; full text only if admin enables explicit logging

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

**Acceptance Criteria**
- [ ] Three roles defined: Viewer (read results only), Analyst (run queries), Admin (manage connections)
- [ ] Role assignments stored in PostgreSQL
- [ ] API routes return 403 if user's role does not have required permission
- [ ] Admin UI panel for assigning roles (basic table view, no complex UI needed)
- [ ] Role checked on every request via middleware (not scattered inline checks)

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

**Acceptance Criteria**
- [ ] PII detection runs on every query result before it is sent to the LLM or returned to the frontend
- [ ] Detects: email addresses, phone numbers, US SSNs, credit card patterns
- [ ] Masking strategy: email → `j***@example.com`, SSN → `***-**-1234`, phone → `***-***-4567`
- [ ] Admin can configure per-connection whether masking is enabled
- [ ] PII detection adds no more than 200ms to response time

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

**Acceptance Criteria**
- [ ] Single `docker-compose.yml` starts the full stack: Next.js, FastAPI, PostgreSQL
- [ ] All environment variables documented in `.env.example` with descriptions
- [ ] Ollama supported as a local LLM alternative to Groq (for full air-gap deployment)
- [ ] README includes step-by-step setup instructions for non-technical admins
- [ ] Health check script validates all services are running correctly

**Engineering Tasks**

| # | Task | Role | Est. Hours |
|---|------|------|-----------|
| 1 | Create `docker-compose.yml` with `frontend`, `backend`, `postgres` services | DevOps | 3h |
| 2 | Add Ollama service option to `docker-compose.yml` with environment variable toggle | DevOps | 2h |
| 3 | Create `healthcheck.sh` script that pings all services and reports status | DevOps | 1h |
| 4 | Write on-premise setup README section | Full-stack | 2h |

---

### EPIC-07: Commerce Connectors

> **Gate:** Begin only after Phase 3 ships AND ≥3 of 5 discovery interviews confirm the commerce/seller segment. The horizontal platform (Phases 0–3) is the prerequisite — Phase 4 is a vertical specialization on top of it.

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

## 4. Sprint Plan — Phase 0 + Phase 1

### Current Status (as of 2026-03-23)

| Sprint | Tickets | Points | Status |
|--------|---------|--------|--------|
| Sprint 0 | DOCBOT-101, 102, 103 | 18 | ✅ Complete |
| Sprint 1 | DOCBOT-201, 202, 203, 204, 205, 206, 207 | 49 | ✅ Complete |
| Sprint 2 | DOCBOT-301, 302, 303, 304 | 20 | ✅ Complete |
| Sprint 3 | DOCBOT-401, 402, 403, 404, 406 | 35 | ✅ Complete |
| Phase 1 | All Phase 1 tickets | 122 | ✅ Complete |
| Enterprise Add-on | DOCBOT-208 (Azure SQL / Entra auth) | 8 | ✅ Complete |

**Total delivered**: 135 story points across 21 tickets + full test suite (131 tests) + GitHub Actions CI

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

| Phase | Stories | Total Points | Estimated Calendar Time |
|-------|---------|-------------|------------------------|
| Phase 0 | DOCBOT-101, 102, 103 | 18 pts | 2 weeks |
| Phase 1 | DOCBOT-201–205, 301–304, 401–404 | 87 pts | 8 weeks (4 sprints) |
| Phase 2 | DOCBOT-305, 405, 501–504 | 47 pts | 4 weeks (2 sprints) |
| Phase 3 | DOCBOT-206, 207 + DW connectors | ~35 pts | 4 weeks (2 sprints) |
| Phase 4 | DOCBOT-601–605 | 50 pts | 6 weeks (3 sprints) |
| **Total** | **32 stories** | **~237 pts** | **~24 weeks** |

Note: Timeline assumes 1 full-time engineer. A 2-person team cuts this roughly in half. Phase 4 can begin in parallel with Phase 3 if a second engineer joins after Phase 2.

---

## Immediate Next Actions (This Week)

1. Create the Linear workspace and set identifier to `DOCBOT`
2. Create the 6 epics as Linear Projects
3. Paste DOCBOT-101, 102, 103 into Sprint 0 Cycle and assign to yourself
4. Run 5 discovery interviews (see Risk RISK-04) before writing a single line of Phase 1 code
5. Set up Railway account and validate that a simple FastAPI Docker container deploys successfully (unblocks Sprint 0 Day 3 checkpoint)
