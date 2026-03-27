# DocBot

<p align="center">
  <img src="https://img.shields.io/badge/License-BSL_1.1-orange.svg" alt="License">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Next.js-16+-black.svg" alt="Next.js">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-teal.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Tests-567 passing-brightgreen.svg" alt="Tests">
</p>

**Ask Anything About Your Data.**

DocBot is an AI-powered document + database analyst. Upload PDFs, connect a live database, upload CSV files, or combine sources — get instant answers with source citations, SQL explanations, Python-generated charts, discrepancy detection, session memory, and a multi-step Autopilot agent that investigates complex questions autonomously.

---

## What DocBot Does

| Capability | Description |
|---|---|
| **PDF Chat** | Upload PDFs, ask questions, get cited answers from 8 expert personas with smart auto-routing |
| **Live Database Chat** | Connect PostgreSQL, MySQL, SQLite, or Azure SQL — ask in plain English, get SQL + results |
| **CSV Intelligence** | Upload CSV files — queries run via E2B pandas sandbox with automatic data profiling, error retry, and adaptive limits |
| **Hybrid Mode** | One question answered from both your documents and your database in a single response |
| **Discrepancy Detection** | Automatically flags when a number in your doc differs from what the DB shows |
| **Python Analysis + Charts** | E2B sandbox executes pandas/matplotlib code and returns charts — bar, line, scatter, heatmap, box, multi-panel |
| **Structured Extraction** | Pulls typed values (financial metrics, legal dates, medical measurements) from any document using Gemini 2.5 Flash |
| **Deep Research** | Sub-question decomposition, parallel retrieval, and gap-fill loop for thorough document analysis |
| **Analytical Autopilot** | LangGraph multi-step investigation agent — works with PDF + CSV + SQL, uses Deep Research for doc search |
| **Commerce Connectors** | Marketplace connector framework with Amazon SP-API integration — OAuth, Orders, Finances |
| **LLM Fallback** | Groq primary, Gemini 2.5 Flash automatic fallback — wired to all 8 production callsites |
| **Conversational Memory** | Follow-up questions rephrased into standalone queries across all pipelines (CSV, SQL, hybrid, autopilot) |
| **Azure SQL / Entra Auth** | Enterprise Microsoft Entra (Azure AD) Service Principal authentication for Azure SQL |
| **SAML 2.0 SSO** | SP-initiated SSO with Okta, Azure AD, or any SAML 2.0 IdP; JIT user provisioning on first login |
| **Consumer Auth** | GitHub OAuth, Google OAuth, email+password, or guest mode — no enterprise SSO required |
| **Persistent Workspace** | Login restores your previous chat sessions and saved database connections across devices |
| **Role-Based Access Control** | Three-tier roles (viewer / analyst / admin) enforced as FastAPI dependencies; admin UI for role management |
| **Append-Only Audit Log** | Immutable PostgreSQL audit trail for all queries, logins, uploads, and connection events with CSV export |
| **PII Masking** | Auto-detects and redacts emails, phone numbers, SSNs, credit cards before LLM synthesis — applied to all SSE, sandbox, and audit paths |
| **Session Artifact Store** | DataFrames and charts from every query persisted in PostgreSQL; referenceable across turns |
| **Context Compression** | Sessions with 20+ messages automatically summarized so long conversations stay fast and cheap |
| **Schema-Aware Table Selection** | Question embeddings matched against table embeddings (cosine similarity) so the right tables are always selected — even with cryptic names like `cust_ord_hdr_rec` |
| **Query History Panel** | Sidebar panel showing all past queries — click to re-run or inspect SQL |

**Core differentiator**: Hybrid Docs+DB synthesis with discrepancy detection + Analytical Autopilot + Commerce Connectors. No other tool does this.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/ssinghai6/DocBot.git
cd DocBot
npm install

# 2. Set up Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Fill in your API keys (see Environment Variables below)

# 4. Run both services
# Terminal 1 — Backend
python3 -m uvicorn api.index:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend
npm run dev -- --port 3000
```

Open [http://localhost:3000](http://localhost:3000)

---

## Features

### Expert Persona System

8 specialized AI personas, each with domain-optimized prompts:

| Persona | Best For |
|---|---|
| **Generalist** | Broad questions, general summarization |
| **Doctor** | Medical documents, clinical notes, lab reports |
| **Finance Expert** | Financial reports, P&L, investor materials |
| **Engineer** | Technical specs, architecture documents |
| **AI/ML Expert** | Research papers, model cards, ML documentation |
| **Lawyer** | Contracts, legal agreements, case summaries |
| **Consultant** | Strategy documents, business proposals |
| **Data Analyst** | Data-heavy documents, statistical analysis, BI reports |

### Live Database Connectivity

Connect to your database and ask questions in plain English:

- **Supported dialects**: PostgreSQL, MySQL, SQLite, Azure SQL (Entra auth)
- **SSRF protection**: All connection hosts validated against RFC 1918 private ranges before connecting
- **Credential security**: Fernet-encrypted at rest; never logged, never passed to LLM context
- **Schema-aware table selection**: Question and table schemas embedded with `all-MiniLM-L6-v2`; cosine similarity selects the top-5 most relevant tables per query instead of dumping all tables to the LLM
- **Read-only enforcement**: 3-layer protection — LLM prompt + sqlglot AST validation + read-only transaction
- **Result cap**: 500 rows, 15-second query timeout
- **Views support**: Schema introspection discovers both tables and views (up to 200 objects)
- **LRU engine pool**: Database connection engines pooled and reused with least-recently-used eviction
- **Schema drift detection**: If a query fails due to table/column-not-found, the pipeline invalidates cache, re-introspects, regenerates SQL, and retries once automatically
- **Manual schema refresh**: Force a schema cache refresh via `POST /api/db/refresh-schema/{connection_id}`

### SQL Generation Pipeline (7 Bounded Steps)

```
NL Question
  → [1] Schema Retrieval     (cache → miss → introspect tables+views, 200 cap)
  → [2] Table Selector       (cosine similarity → LLM fallback, 20 cols/table, 8 tables max)
  → [3] Few-Shot Retrieval   (cosine similarity on stored queries)
  → [4] SQL Generator        (LLM call #1)
  → [5] SQL Validator        (sqlglot AST — deterministic, no LLM)
  → [6] Executor             (pooled engine, 15s timeout, 500 row cap, drift retry)
  → [7] Answer Generator     (LLM call #2, streaming)
```

Max 2-3 LLM calls per query. No loops. Predictable latency and cost.

### CSV Intelligence

Upload CSV files and query them using natural language — no SQL required:

- **E2B pandas sandbox**: Queries execute as Python/pandas code in an isolated cloud sandbox, not through the SQL pipeline
- **Automatic data profiling**: On upload, a `DataProfile` is computed (dtypes, sample rows, `describe()`, datetime column detection, null percentages, frequency inference) with zero LLM calls
- **Profile-aware code generation**: The LLM sees actual data context (types, samples, statistics) when generating pandas code — not just column names
- **Adaptive limits**: Complex queries (predict, forecast, model) get extended resources — 4000 max tokens, 150-line code limit, 60-second timeout
- **Error retry with feedback**: If sandbox execution fails, the error message is fed back to the LLM for one corrective retry attempt
- **Multi-section detection**: CSVs with multiple header rows or embedded sections are automatically detected and split

### Hybrid Mode

Ask one question — DocBot queries both your uploaded documents and your live database and synthesizes a single answer with dual citations. When a number in your document differs from the database, `[DISCREPANCY]` markers appear automatically with the delta and percentage difference.

### Python Analysis + Advanced Charts

After SQL execution, DocBot generates pandas/matplotlib code and runs it in an isolated E2B cloud sandbox:

- **Chart types**: bar, line, scatter, heatmap (correlation matrices), box (distribution), multi-panel (2x2 subplots)
- **Chart type selector**: choose before submitting or leave on "auto" to let the LLM decide
- **Zoom & export**: click any chart to expand full-screen; download as PNG
- **Chart metadata**: title, axis labels, series count shown as caption below each chart
- Code shown in collapsible "Analysis code" block for full transparency
- Sandbox isolation: no network access, no filesystem persistence

### Deep Research

Sub-question decomposition with parallel retrieval and iterative gap-fill for thorough document analysis:

- Decomposes complex questions into targeted sub-questions
- Retrieves answers for each sub-question in parallel
- Runs a gap-detection loop — if retrieved context is insufficient, generates additional sub-questions and retrieves again
- Integrated into Autopilot's `doc_search` tool for multi-step investigations
- Also available as a standalone route for direct deep retrieval

### Analytical Autopilot (LangGraph Agent)

Multi-step investigation agent that auto-triggers on analytical queries (why, diagnose, investigate, compare, trend, forecast):

```
PlannerNode  →  ExecutorNode (loop, ≤5 steps)  →  SynthesizerNode
```

- **PlannerNode**: Groq Llama decomposes the question into ≤5 investigation steps with dynamic tool selection
- **ExecutorNode**: For each step, routes to the right tool — `sql_query`, `python_analysis`, or `doc_search` — and executes it
- **Deep doc search**: The `doc_search` tool uses `deep_retrieve()` (sub-question decomposition + parallel retrieval + gap-fill) instead of single-pass RAG
- **Universal**: Works with PDF + CSV + SQL sources — the planner dynamically selects available tools based on connected data sources
- **SynthesizerNode**: Groq Llama synthesizes all step results into a final cited answer, streamed token-by-token
- **Hard limits**: max 5 iterations, 90-second wall-clock timeout — no infinite loops
- **Streaming UI**: step-by-step progress shown live; charts and results visible per step; full investigation persists in the message bubble after completion

### Conversational Memory

All pipelines support multi-turn conversations with context-aware follow-up handling:

- Frontend sends the last 6 messages as conversation context
- All pipelines (CSV, SQL, hybrid, autopilot) rephrase follow-up questions into standalone queries before processing
- Eliminates ambiguous references ("show me that again", "break it down by region") by resolving them against conversation history

### LLM Fallback

Automatic failover from Groq to Gemini 2.5 Flash across all production callsites:

- Primary: Groq (Llama 3.3-70b for reasoning, Qwen/qwen3-32b for code generation)
- Fallback: Gemini 2.5 Flash — activates automatically on Groq errors or rate limits
- Wired to all 8 production callsites — SQL generation, hybrid synthesis, intent classification, answer generation, code generation, autopilot planning, autopilot synthesis, and deep research

### Commerce Connectors (Phase 1)

Pluggable marketplace connector framework for e-commerce data integration:

- **Connector interface**: Abstract `MarketplaceConnector` base class with credential vault, normalized data models, and async token-bucket rate limiter
- **Unified commerce schema**: Two-table schema (orders + financials) with PostgreSQL row-level security per `connection_id` — multi-tenant by design
- **Amazon SP-API connector**: Full LWA OAuth flow, Orders API, and Finances API with automatic retry on throttling
- **Frontend UI**: `MarketplacePanel.tsx` for registering, syncing, and disconnecting marketplace connections
- Phase 2 deferred to post-funding: background sync worker (DOCBOT-704), Shopify connector (DOCBOT-705)

### RAG Quality Enhancement

Production-grade retrieval pipeline:

- **Chroma persistent vector store**: Replaces in-memory vector store for durable document embeddings
- **Cross-encoder reranker**: Re-ranks retrieval results for higher relevance precision
- **SemanticChunker**: Splits documents at semantic boundaries (via `langchain-experimental`) instead of fixed character counts — better context preservation for financial and legal documents

### Session Artifact Store

Every query result is persisted as a session artifact in PostgreSQL:

- DataFrames stored as JSON (records orient, up to 500 rows)
- Charts stored as base64 PNG alongside the DataFrame
- Artifacts referenceable by `artifact_id` in Autopilot multi-step workflows
- List artifacts via `GET /api/artifacts/{session_id}`

### Context Compression

Long conversations stay fast and cost-efficient:

- After every 20 messages in a session, older turns are summarized into a rolling `context_summary`
- Compression runs as a background task — no latency impact
- LLM receives: `context_summary` + last 10 messages (instead of the full unbounded history)
- The full conversation remains visible in the UI — only the LLM context is trimmed

### Query History Panel

A collapsible sidebar panel in DB chat mode shows all past queries for the active connection:

- Natural language question + timestamp + row count badge
- Click any entry to re-populate the input for a follow-up
- Expand inline to see the SQL and results

### Structured Document Extraction

LangExtract + Gemini 2.5 Flash extracts typed, span-verified values from any document type:

| Document Type | Extracts |
|---|---|
| **Financial** | Revenue, ARR, margins, forecasts, EBITDA |
| **Legal** | Effective dates, penalty amounts, contract duration, jurisdiction |
| **Medical** | Blood pressure, glucose, medication doses, diagnoses |
| **Research** | Sample size, p-values, effect sizes, confidence intervals |
| **General** | Any key numeric or categorical fact grounded in the text |

### Microsoft Entra (Azure AD) Authentication

Connect to Azure SQL Database using Service Principal credentials — no username/password required:

- Tenant ID + Client ID + Client Secret -> token via `azure.identity`
- Token injected via `SQL_COPT_SS_ACCESS_TOKEN` — credentials never appear in the connection string
- ODBC Driver 18 for SQL Server included in Docker image

### Enterprise SSO (SAML 2.0)

SP-initiated SAML 2.0 SSO with any standards-compliant IdP (Okta, Azure AD, Google Workspace):

- JIT user provisioning — users created on first login; no pre-registration needed
- Session tokens stored in PostgreSQL with configurable TTL (default 8 hours)
- Session cookie is `HttpOnly; SameSite=Lax` — no token exposure to JavaScript
- SP metadata available at `GET /api/auth/saml/metadata` for IdP registration
- App stays fully functional without an IdP configured (open mode)

### Consumer Authentication (GitHub / Google / Email)

For individual users and small teams, DocBot supports consumer OAuth without any IdP setup:

- **GitHub OAuth** — one-click login with your GitHub account
- **Google OAuth** — one-click login with your Google account
- **Email + password** — register and log in with any email address
- **Guest mode** — use DocBot without an account; session is browser-local only

Environment variables required:
| Variable | Required | Notes |
|----------|----------|-------|
| `GITHUB_CLIENT_ID` | GitHub auth | From GitHub OAuth App settings |
| `GITHUB_CLIENT_SECRET` | GitHub auth | From GitHub OAuth App settings |
| `GOOGLE_CLIENT_ID` | Google auth | From Google Cloud Console |
| `GOOGLE_CLIENT_SECRET` | Google auth | From Google Cloud Console |
| `APP_BASE_URL` | OAuth | Your backend public URL — must match OAuth app callback settings |
| `FRONTEND_URL` | OAuth | Your frontend URL — where users land after login |

### Role-Based Access Control (RBAC)

Three-tier role hierarchy enforced as FastAPI `Depends` dependencies:

| Role | Permissions |
|---|---|
| **viewer** | Read query results and documents |
| **analyst** | Run queries, manage own DB connections, upload documents (default for JIT-provisioned users) |
| **admin** | Full access — audit log, user management, all connections |

- Role enforcement is a no-op when `AUTH_REQUIRED` is not set (open/demo mode)
- Admins can promote/demote users via the Admin Panel or `PATCH /admin/users/{user_id}/role`
- Admin self-demotion guard prevents accidental lockout

### Append-Only Audit Log

Every security-relevant event written to an immutable PostgreSQL table:

| Event | Logged When |
|---|---|
| `query` | SQL/NL query executed against a live DB connection |
| `db_connect` | New database connection created |
| `db_disconnect` | Database connection removed |
| `upload` | File uploaded (PDF, CSV, SQLite) |
| `login` | SSO login (SAML ACS success) |
| `logout` | Session logout |

- PostgreSQL `BEFORE UPDATE OR DELETE` trigger enforces immutability at the DB level
- Admin export: `GET /admin/audit-log?format=csv` for compliance downloads
- Writes are fire-and-forget — audit failures never block request handlers

### PII Masking

Sensitive data detected and redacted before it reaches the LLM:

- Detects: email addresses, phone numbers (US, UK, India, EU), SSNs, credit card numbers
- Masks with typed placeholders: `[EMAIL]`, `[PHONE]`, `[SSN]`, `[CC_NUMBER]`
- Applied to all SSE responses, sandbox outputs, and audit response paths
- Configurable: can be toggled per-request via `mask_pii: true/false` in the request body
- Pure regex implementation — no external NLP dependencies, adds less than 5ms per 500 rows

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Next.js 16 Frontend                       │
│         React 19, TailwindCSS 4, TypeScript                  │
└──────────────────────────┬──────────────────────────────────┘
                           │ /api/*
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                FastAPI Backend (Railway)                      │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ db_service  │  │hybrid_service│  │ sandbox_service    │  │
│  │ SQL pipeline│  │ RAG + intent │  │ E2B Python exec    │  │
│  │ LRU pool   │  │              │  │ CSV code gen       │  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │autopilot_   │  │deep_research_│  │document_extractor  │  │
│  │service      │  │service       │  │LangExtract+Gemini  │  │
│  │(LangGraph)  │  │deep_retrieve │  │                    │  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │commerce_    │  │connectors/   │  │file_upload_        │  │
│  │service      │  │amazon, rate  │  │service             │  │
│  │unified schma│  │limiter, base │  │CSV + SQLite upload │  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │auth_service │  │rbac_service  │  │audit_service       │  │
│  │SAML 2.0 SSO │  │viewer/analyst│  │append-only log     │  │
│  │JIT provision│  │/admin roles  │  │immutability trigger│  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │oauth_service│  │artifact_     │  │metrics_service     │  │
│  │GitHub/Google│  │service       │  │admin dashboard     │  │
│  │email/pass   │  │              │  │                    │  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  utils/  encryption · ssrf_validator · sql_validator    │ │
│  │          embeddings · few_shot_store · table_selector   │ │
│  │          context_compressor · pii_masking · reranker    │ │
│  │          chunker · vector_store · llm_provider          │ │
│  │          csv_preprocessor · gemini_wrapper              │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         PostgreSQL      Groq         E2B
         (Railway)   Llama/Qwen    Sandbox
              │        Gemini
              │       (fallback)
              ▼
           Chroma
        (vector store)
```

### Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 16, React 19, TailwindCSS 4, TypeScript, lucide-react, react-markdown |
| Backend | FastAPI, Python 3.12, SQLAlchemy 2.x |
| Agentic Orchestration | LangGraph (StateGraph — Planner -> Executor -> Synthesizer) |
| LLM — SQL + Hybrid + Autopilot | Groq Llama 3.3-70b (primary), Gemini 2.5 Flash (fallback) |
| LLM — Python codegen | Groq Qwen/qwen3-32b |
| LLM — Doc extraction | Gemini 2.5 Flash (via LangExtract) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (HuggingFace API) |
| Vector Store | Chroma (persistent) + PostgreSQL table_embeddings |
| Retrieval Enhancement | Cross-encoder reranker, SemanticChunker (langchain-experimental) |
| SQL Validation | sqlglot AST parsing |
| Credential Encryption | Fernet (cryptography) |
| SSO / Auth | python3-saml (SAML 2.0), HttpOnly session cookies |
| Consumer Auth | httpx (OAuth flows), bcrypt (passwords) |
| RBAC | FastAPI Depends pattern, IntEnum role hierarchy |
| PII Detection | Regex patterns (email, phone, SSN, credit card) |
| DB Drivers | psycopg2, pymysql, pyodbc, azure-identity |
| Python Sandbox | E2B Cloud (code-interpreter) |
| Commerce | Pluggable connector framework, Amazon SP-API, async rate limiter |
| Storage | PostgreSQL on Railway |
| Deployment | Railway (backend), Vercel (frontend) |

---

## API Reference

### Document Upload

```bash
POST /api/upload
Content-Type: multipart/form-data
# Accepts: PDF files (multiple allowed)
# Returns: { session_id, documents: [{filename, pages, ...}] }
```

### Document Chat

```bash
POST /api/chat
{
  "session_id": "sess-abc123",
  "message": "What is the ARR target for Q3?",
  "persona": "Finance Expert"
}
# Returns: streaming SSE with { response, sources: [{filename, page, text}] }
```

### Database Connection

```bash
POST /api/db/connect
{
  "session_id": "sess-abc123",
  "dialect": "postgresql",          # postgresql | mysql | sqlite | azure_sql
  "host": "db.example.com",
  "port": 5432,
  "dbname": "mydb",
  "user": "readonly_user",
  "password": "..."
}

# For Azure SQL (Entra auth):
{
  "dialect": "azure_sql",
  "host": "myserver.database.windows.net",
  "port": 1433,
  "dbname": "mydb",
  "auth_type": "entra_sp",
  "tenant_id": "...",
  "client_id": "...",
  "client_secret": "..."
}
```

### Database Chat

```bash
POST /api/db/chat
{
  "connection_id": "conn-xyz",
  "question": "What are the top 10 customers by revenue?",
  "session_id": "sess-abc123",
  "persona": "Data Analyst",
  "chart_type": "auto"              # auto | bar | line | scatter | heatmap | box | multi
}
# Returns: SSE stream — token, metadata, analysis_code, chart, done chunks
```

### Schema Refresh

```bash
POST /api/db/refresh-schema/{connection_id}
# Invalidates cached schema and re-introspects tables + views
# Returns: { success: true, tables_count, message }
```

### File Upload (CSV / SQLite)

```bash
# Upload SQLite database file
POST /api/db/upload
Content-Type: multipart/form-data
# Accepts: .db, .sqlite, .sqlite3 files
# Returns: { connection_id, dialect: "sqlite", tables: [...] }

# Upload CSV file
POST /api/db/upload/csv
Content-Type: multipart/form-data
# Accepts: .csv files
# Returns: { connection_id, dialect: "csv", filename, rows, columns, data_profile: {...} }
```

### Hybrid Chat

```bash
POST /api/hybrid/chat
{
  "session_id": "sess-abc123",
  "connection_id": "conn-xyz",
  "message": "Is the revenue in our board deck consistent with the database?",
  "persona": "Finance Expert"
}
# Returns: streaming SSE; response includes [DISCREPANCY] markers when values differ
```

### Analytical Autopilot

```bash
POST /api/autopilot/run
{
  "session_id": "sess-abc123",
  "connection_id": "conn-xyz",
  "question": "Diagnose why we missed Q3 revenue targets",
  "persona": "Finance Expert"
}
# Returns: SSE stream
#   {type: "plan",   steps: [...]}
#   {type: "step",   step_num, tool, step_label, content, chart_b64?, artifact_id?, error?}
#   {type: "answer", content: "<markdown synthesis>"}
#   {type: "done",   citations: [...]}
```

### Artifacts

```bash
GET /api/artifacts/{session_id}
# Returns: { artifacts: [{ id, artifact_type, name, row_count, columns, created_at }] }

GET /api/artifacts/item/{artifact_id}
# Returns: full artifact including data_json and chart_b64
```

### Query History

```bash
GET /api/db/history/{connection_id}?limit=20
# Returns: [{ id, question, sql, executed_at, row_count }]
```

### Schema

```bash
GET /api/db/schema/{connection_id}
# Returns: { tables: [{ name, columns: [{ name, type }] }] }
```

### Commerce Connectors

```bash
# Register a marketplace connector
POST /api/connectors/register
{
  "connector_type": "amazon",
  "credentials": { "refresh_token": "...", "lwa_client_id": "...", "lwa_client_secret": "..." }
}

# List registered connectors
GET /api/connectors

# List available connector types
GET /api/connectors/types

# Fetch orders from a connector
POST /api/connectors/{connector_id}/orders

# Fetch financials from a connector
POST /api/connectors/{connector_id}/financials

# Trigger sync (fetch + persist to unified schema)
POST /api/connectors/{connector_id}/sync

# Query persisted commerce data
GET /api/commerce/{connector_id}/orders
GET /api/commerce/{connector_id}/financials
```

### Sandbox Execution

```bash
POST /api/sandbox/execute
{
  "code": "import pandas as pd\nprint(pd.DataFrame({'a': [1,2,3]}).describe())",
  "session_id": "sess-abc123"
}
# Returns: { stdout, stderr, charts: [{b64, metadata}], error? }
```

### Admin Metrics

```bash
GET /admin/metrics
# Returns: aggregated usage metrics (requires admin role)
```

### Authentication (SSO)

```bash
# Initiate SSO login — redirects to IdP
GET /api/auth/saml/login

# SAML Assertion Consumer Service (IdP POSTs here)
POST /api/auth/saml/acs
# On success: sets HttpOnly docbot_session cookie, redirects to frontend

# SP metadata for IdP registration
GET /api/auth/saml/metadata

# Current user info
GET /api/auth/me
# Returns: { authenticated: true, user: { id, email, name, role } }
# Or:      { authenticated: false, saml_configured: false }

# Logout (clears session cookie)
POST /api/auth/logout

# Consumer OAuth
GET /api/auth/github        # Returns {"url": "..."} — redirect browser to this URL
GET /api/auth/github/callback   # OAuth callback (handled server-side)
GET /api/auth/google        # Returns {"url": "..."} — redirect browser to this URL
GET /api/auth/google/callback   # OAuth callback (handled server-side)
POST /api/auth/register     # Body: {email, password, name?} — email+password registration
POST /api/auth/login        # Body: {email, password} — email+password login
GET /api/auth/config        # Returns {email: true, github: bool, google: bool, saml: bool}
GET /api/auth/workspace     # Returns user's saved sessions and DB connections (requires auth)
```

### Admin — User Management (admin role required)

```bash
# List all users
GET /admin/users
# Returns: { users: [{ id, email, name, role, provider, last_login_at }] }

# Update a user's role
PATCH /admin/users/{user_id}/role
{ "role": "analyst" }    # viewer | analyst | admin
# Returns: { success: true, user_id, new_role }
```

### Admin — Audit Log (admin role required)

```bash
# Paginated audit log viewer
GET /admin/audit-log?limit=50&offset=0&event_type=login
# Returns: { events: [{ id, event_type, session_id, user_id, detail, metadata_json, occurred_at }], total }

# Download as CSV
GET /admin/audit-log?format=csv
# Returns: text/csv attachment
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `groq_api_key` | Yes | Groq API key for Llama + Qwen inference |
| `huggingface_api_key` | Yes | HuggingFace API for embeddings model |
| `DATABASE_URL` | Yes | Railway PostgreSQL connection string |
| `DB_ENCRYPTION_KEY` | Yes | Fernet key for credential encryption — generate with `cryptography.fernet.Fernet.generate_key()` |
| `E2B_API_KEY` | Yes | E2B sandbox API key (CSV queries + Python analysis) |
| `GEMINI_API_KEY` | Yes | Gemini API key for LangExtract extraction and LLM fallback |
| `ALLOWED_ORIGINS` | No | Comma-separated CORS origins (defaults to localhost:3000) |
| `FRONTEND_URL` | Auth | Frontend public URL for post-login redirects (SAML and OAuth) |
| `SESSION_TTL_HOURS` | No | SSO session lifetime in hours (default: 8) |
| `AUTH_REQUIRED` | No | Set to `true` to enforce RBAC login on all protected routes. Default off (open/demo mode) |
| `SAML_SP_ENTITY_ID` | SSO | SP Entity ID — e.g. `https://docbot.example.com` |
| `SAML_SP_ACS_URL` | SSO | ACS callback URL — e.g. `https://docbot.example.com/api/auth/saml/acs` |
| `SAML_IDP_ENTITY_ID` | SSO | IdP Entity ID from your IdP metadata |
| `SAML_IDP_SSO_URL` | SSO | IdP SSO redirect URL |
| `SAML_IDP_X509_CERT` | SSO | IdP public certificate (base64, no headers) |
| `SAML_SP_X509_CERT` | SSO opt | SP certificate for signed requests |
| `SAML_SP_PRIVATE_KEY` | SSO opt | SP private key for signed requests |
| `GITHUB_CLIENT_ID` | Consumer auth | GitHub OAuth App client ID |
| `GITHUB_CLIENT_SECRET` | Consumer auth | GitHub OAuth App client secret |
| `GOOGLE_CLIENT_ID` | Consumer auth | Google OAuth 2.0 client ID |
| `GOOGLE_CLIENT_SECRET` | Consumer auth | Google OAuth 2.0 client secret |
| `APP_BASE_URL` | Auth | Backend public URL used for OAuth callback URIs |
| `OAUTH_REDIRECT_BASE_URL` | OAuth | Base URL for OAuth callback redirect |

Never commit `.env`. Never hardcode secrets.

> **SSO setup**: Set all `SAML_*` variables to enable SSO. When unset, the app runs in open mode — all role checks pass and the login UI is hidden.

> **Auth enforcement**: Set `AUTH_REQUIRED=true` to require login on all protected routes. When unset (default), the app runs in open/demo mode — RBAC is decoupled from SAML configuration.

---

## Local Development

```bash
# Backend (Terminal 1)
source .venv/bin/activate
python3 -m uvicorn api.index:app --host 0.0.0.0 --port 8000 --reload

# Frontend (Terminal 2)
npm run dev -- --port 3000
```

Next.js proxies `/api/*` to `localhost:8000` in development.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Unit tests only (no network, no DB, no API keys)
pytest tests/unit/ -v

# CI command (skips external + postgres tests)
pytest tests/ -v -m "not external and not postgres"
```

**567 tests passing** across:

| File | Coverage |
|---|---|
| `tests/unit/test_ssrf_validator.py` | SSRF protection — private IPs, loopback, link-local |
| `tests/unit/test_encryption.py` | Fernet credential encryption/decryption |
| `tests/unit/test_sql_validator.py` | sqlglot AST validation — write query rejection |
| `tests/unit/test_db_service_helpers.py` | Connection URL building, Entra token struct, Pydantic models |
| `tests/unit/test_embeddings.py` | Embedding cosine similarity, few-shot retrieval |
| `tests/unit/test_artifact_service.py` | Artifact save/list/get, 500-row truncation, type validation |
| `tests/unit/test_table_selector.py` | Cosine similarity table selection, schema summary, fallback |
| `tests/unit/test_context_compressor.py` | Compression thresholds, summary injection, bypass |
| `tests/unit/test_sandbox_service.py` | Chart type routing, metadata extraction, `<think>` stripping |
| `tests/unit/test_sandbox_formatting.py` | Sandbox stdout formatting, markdown table conversion |
| `tests/unit/test_autopilot_service.py` | Planner decomposition, executor routing, hard limits, SSE serialization |
| `tests/unit/test_hybrid_service.py` | Intent classification, hybrid synthesis, discrepancy detection |
| `tests/unit/test_hybrid_chat.py` | Hybrid chat routing, dual-citation format |
| `tests/unit/test_personas.py` | Persona def completeness, response format contracts |
| `tests/unit/test_persona_router.py` | Smart auto-routing — persona selection from query content |
| `tests/unit/test_code_generation.py` | Python codegen prompt construction, chart type injection |
| `tests/unit/test_query_expansion.py` | NL query expansion, synonym injection |
| `tests/unit/test_deep_research_service.py` | Multi-source research orchestration, citation dedup |
| `tests/unit/test_pii_masking.py` | Name/email/phone/SSN/CC detection and redaction |
| `tests/unit/test_auth_service.py` | SAML settings builder, session CRUD, JIT provisioning |
| `tests/unit/test_oauth_service.py` | GitHub/Google OAuth flows, email/password auth, token handling |
| `tests/unit/test_audit_service.py` | Event types, DB write, fire-and-forget dispatcher, immutability DDL |
| `tests/unit/test_rbac_service.py` | Role hierarchy, `require_role` dependency, wire-up |
| `tests/unit/test_amazon_connector.py` | Amazon SP-API connector — LWA OAuth, Orders, Finances (22 tests) |
| `tests/unit/test_commerce_service.py` | Unified commerce schema, RLS, persist/query helpers (31 tests) |
| `tests/unit/test_discrepancy_detector.py` | Discrepancy detection — delta calculation, threshold logic |
| `tests/unit/test_reranker.py` | Cross-encoder reranker scoring, fallback behavior |
| `tests/unit/test_chunker.py` | SemanticChunker splitting, boundary detection |
| `tests/unit/test_vector_store.py` | Chroma persistent vector store operations |
| `tests/unit/test_llm_provider.py` | LLM fallback provider — Groq primary, Gemini fallback |
| `tests/unit/test_metrics_service.py` | Admin metrics aggregation |
| `tests/integration/test_db_pipeline.py` | Full SQL pipeline against SQLite |
| `tests/integration/test_file_upload_service.py` | PDF upload and chunk extraction |
| `tests/integration/test_artifact_service.py` | SQLite-backed artifact round-trip |
| `tests/external/test_financebench_accuracy.py` | FinanceBench 20-question accuracy suite (requires live API keys) |

---

## Deployment

### Railway (Backend)

The backend is containerized and deployed to Railway:

```bash
# Build and test locally
docker build -t docbot .
docker run -p 8000:8000 --env-file .env docbot

# Deploy via Railway CLI
railway up
```

The Dockerfile installs ODBC Driver 18 for SQL Server (required for Azure SQL connectivity).

### Vercel (Frontend)

```bash
vercel deploy
```

Set all environment variables in the Railway and Vercel dashboards.

---

## Security

- **SSRF prevention**: All DB connection hosts validated against RFC 1918, loopback (127.x, ::1), and link-local ranges before any network call
- **SQL injection prevention**: sqlglot AST rejects any non-SELECT root statement — regex-free, bypass-resistant
- **Read-only enforcement**: Three independent layers — LLM prompt, AST check, database transaction
- **Credential encryption**: All DB credentials Fernet-encrypted before PostgreSQL storage; never logged or passed to LLMs
- **Entra auth**: Azure SQL token injection via `SQL_COPT_SS_ACCESS_TOKEN`; no credentials in connection string
- **Autopilot hard limits**: Max 5 iterations and 90-second wall-clock timeout prevent runaway LLM loops
- **SAML assertion validation**: `python3-saml` validates signatures, conditions, and recipient before trusting any assertion
- **HttpOnly session cookies**: Session tokens inaccessible to JavaScript; `SameSite=Lax` prevents CSRF
- **Immutable audit log**: PostgreSQL `BEFORE UPDATE OR DELETE` trigger blocks any modification of audit records at the DB level
- **PII masking**: Personally identifiable information detected and redacted before reaching the LLM — applied to all SSE, sandbox, and audit response paths
- **Multi-tenant RLS**: Commerce data isolated per `connection_id` with PostgreSQL row-level security

---

## Roadmap

### Phase 2: Commerce Connectors (Post-Funding)

- DOCBOT-704: Background sync worker — APScheduler, incremental fetch, exponential backoff on rate limits
- DOCBOT-705: Shopify connector — OAuth offline token, webhook-driven incremental sync

### Future

- FinanceBench accuracy benchmarking (test suite written, run pending)
- Additional marketplace connectors (eBay, Walmart)
- Real-time webhook-driven data ingestion

---

## Third-Party Licenses

DocBot depends on the following key open-source libraries. All dependencies use permissive licenses compatible with commercial use.

| Dependency | License | Notes |
|---|---|---|
| [LangChain](https://github.com/langchain-ai/langchain) | MIT | Core RAG framework — langchain, langchain-core, langchain-community, langchain-groq, langchain-huggingface |
| [LangGraph](https://github.com/langchain-ai/langgraph) | MIT | Agentic state machine for Autopilot |
| [LangExtract](https://github.com/google/langextract) | Apache 2.0 | Structured document extraction (Google) |
| [langchain-experimental](https://github.com/langchain-ai/langchain-experimental) | MIT | SemanticChunker for document splitting |
| [ChromaDB](https://github.com/chroma-core/chroma) | Apache 2.0 | Persistent vector store |
| [E2B Code Interpreter](https://github.com/e2b-dev/code-interpreter) | MIT | Cloud sandbox for Python/pandas execution |
| [sqlglot](https://github.com/tobymao/sqlglot) | MIT | SQL parsing, validation, and transpilation |
| [FastAPI](https://github.com/tiangolo/fastapi) | MIT | Backend web framework |
| [python3-saml](https://github.com/SAML-Toolkits/python3-saml) | MIT | SAML 2.0 SSO |
| [google-generativeai](https://github.com/google/generative-ai-python) | Apache 2.0 | Gemini SDK for LLM fallback |
| [cryptography](https://github.com/pyca/cryptography) | Apache 2.0 / BSD | Fernet credential encryption |
| [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy) | MIT | Database ORM and connection management |

No copyleft (GPL/LGPL/AGPL) dependencies are included. All dependency licenses are permissive and compatible with the BSL 1.1 license used by DocBot.

---

## License

**Business Source License 1.1** (c) 2026 [Sanshrit Singhai](https://github.com/ssinghai6)

You can use, modify, and self-host DocBot freely — but you cannot offer it as a competing commercial AI document/database assistant service. On **2030-03-27** (or 4 years after each version's release), the code automatically converts to **Apache 2.0**.

For commercial licensing inquiries, contact: singhai.sanshrit@gmail.com

See [LICENSE](./LICENSE) for full terms.
