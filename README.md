# DocBot

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Next.js-16+-black.svg" alt="Next.js">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-teal.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Tests-232 passing-brightgreen.svg" alt="Tests">
</p>

**Ask Anything About Your Data.**

DocBot is an AI-powered document + database analyst. Upload PDFs, connect a live database, or do both — get instant answers with source citations, SQL explanations, Python-generated charts, discrepancy detection, session memory, and a multi-step Autopilot agent that investigates complex questions autonomously.

---

## What DocBot Does

| Capability | Description |
|---|---|
| **PDF Chat** | Upload PDFs, ask questions, get cited answers from 8 expert personas |
| **Live Database Chat** | Connect PostgreSQL, MySQL, SQLite, or Azure SQL — ask in plain English, get SQL + results |
| **Hybrid Mode** | One question answered from both your documents and your database in a single response |
| **Discrepancy Detection** | Automatically flags when a number in your doc differs from what the DB shows |
| **Python Analysis + Charts** | E2B sandbox executes pandas/matplotlib code and returns charts — bar, line, scatter, heatmap, box, multi-panel |
| **Structured Extraction** | Pulls typed values (financial metrics, legal dates, medical measurements) from any document using Gemini 2.5 Flash |
| **Azure SQL / Entra Auth** | Enterprise Microsoft Entra (Azure AD) Service Principal authentication for Azure SQL |
| **Session Artifact Store** | DataFrames and charts from every query persisted in PostgreSQL; referenceable across turns |
| **Context Compression** | Sessions with 20+ messages automatically summarized so long conversations stay fast and cheap |
| **Schema-Aware Table Selection** | Question embeddings matched against table embeddings (cosine similarity) so the right tables are always selected — even with cryptic names like `cust_ord_hdr_rec` |
| **Query History Panel** | Sidebar panel showing all past queries — click to re-run or inspect SQL |
| **Analytical Autopilot** | LangGraph multi-step investigation agent — decomposes complex questions into ≤5 steps, executes SQL + Python + doc search autonomously, streams step-by-step progress, synthesizes a final cited answer |

**Core differentiator**: Hybrid Docs+DB synthesis with discrepancy detection + Analytical Autopilot. No other tool does this.

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

### SQL Generation Pipeline (7 Bounded Steps)

```
NL Question
  → [1] Schema Retrieval     (cache → miss → introspect)
  → [2] Table Selector       (cosine similarity on table embeddings, top-5)
  → [3] Few-Shot Retrieval   (cosine similarity on stored queries)
  → [4] SQL Generator        (LLM call #1)
  → [5] SQL Validator        (sqlglot AST — deterministic, no LLM)
  → [6] Executor             (SQLAlchemy, 15s timeout, 500 row cap)
  → [7] Answer Generator     (LLM call #2, streaming)
```

Max 2 LLM calls per query. No loops. Predictable latency and cost.

### Hybrid Mode

Ask one question — DocBot queries both your uploaded documents and your live database and synthesizes a single answer with dual citations. When a number in your document differs from the database, `[DISCREPANCY]` markers appear automatically with the delta and percentage difference.

### Python Analysis + Advanced Charts

After SQL execution, DocBot generates pandas/matplotlib code and runs it in an isolated E2B cloud sandbox:

- **Chart types**: bar, line, scatter, heatmap (correlation matrices), box (distribution), multi-panel (2×2 subplots)
- **Chart type selector**: choose before submitting or leave on "auto" to let the LLM decide
- **Zoom & export**: click any chart to expand full-screen; download as PNG
- **Chart metadata**: title, axis labels, series count shown as caption below each chart
- Code shown in collapsible "Analysis code" block for full transparency
- Sandbox isolation: no network access, no filesystem persistence

### Analytical Autopilot (LangGraph Agent)

Enable the Autopilot toggle in the DB chat header to activate multi-step investigation mode:

```
PlannerNode  →  ExecutorNode (loop, ≤5 steps)  →  SynthesizerNode
```

- **PlannerNode**: Groq Llama decomposes the question into ≤5 investigation steps
- **ExecutorNode**: For each step, routes to the right tool — `sql_query`, `python_analysis`, or `doc_search` — and executes it
- **SynthesizerNode**: Groq Llama synthesizes all step results into a final cited answer, streamed token-by-token
- **Hard limits**: max 5 iterations, 90-second wall-clock timeout — no infinite loops
- **Streaming UI**: step-by-step progress shown live; charts and results visible per step; full investigation persists in the message bubble after completion
- **Auto-detect nudge**: when your question contains diagnostic keywords (why, diagnose, investigate, compare, trend, forecast…) and Autopilot is off, a suggestion banner appears above the input

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

- Tenant ID + Client ID + Client Secret → token via `azure.identity`
- Token injected via `SQL_COPT_SS_ACCESS_TOKEN` — credentials never appear in the connection string
- ODBC Driver 18 for SQL Server included in Docker image

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
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │autopilot_   │  │artifact_     │  │document_extractor  │  │
│  │service      │  │service       │  │LangExtract+Gemini  │  │
│  │(LangGraph)  │  │(DOCBOT-501)  │  │                    │  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  utils/  encryption · ssrf_validator · sql_validator    │ │
│  │          embeddings · few_shot_store · table_selector   │ │
│  │          context_compressor                             │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         PostgreSQL      Groq         E2B
         (Railway)   Llama/Qwen    Sandbox
```

### Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 16, React 19, TailwindCSS 4, TypeScript |
| Backend | FastAPI, Python 3.12, SQLAlchemy 2.x |
| Agentic Orchestration | LangGraph (StateGraph — Planner → Executor → Synthesizer) |
| LLM — SQL + Hybrid + Autopilot | Groq Llama 3.3-70b |
| LLM — Python codegen | Groq Qwen/qwen3-32b |
| LLM — Doc extraction | Gemini 2.5 Flash (via LangExtract) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (HuggingFace API) |
| Vector Search | LangChain InMemoryVectorStore + PostgreSQL table_embeddings |
| SQL Validation | sqlglot AST parsing |
| Credential Encryption | Fernet (cryptography) |
| DB Drivers | psycopg2, pymysql, pyodbc, azure-identity |
| Python Sandbox | E2B Cloud |
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

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `groq_api_key` | Yes | Groq API key for Llama + Qwen inference |
| `huggingface_api_key` | Yes | HuggingFace API for embeddings model |
| `DATABASE_URL` | Yes | Railway PostgreSQL connection string |
| `DB_ENCRYPTION_KEY` | Yes | Fernet key for credential encryption — generate with `cryptography.fernet.Fernet.generate_key()` |
| `E2B_API_KEY` | Yes | E2B sandbox API key |
| `GEMINI_API_KEY` | Yes | Gemini API key for LangExtract document extraction |
| `ALLOWED_ORIGINS` | No | Comma-separated CORS origins (defaults to localhost:3000) |

Never commit `.env`. Never hardcode secrets.

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

**232 tests passing** across:

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
| `tests/unit/test_autopilot_service.py` | Planner decomposition, executor routing, hard limits, SSE serialization |
| `tests/integration/test_db_pipeline.py` | Full SQL pipeline against SQLite |
| `tests/integration/test_file_upload_service.py` | PDF upload and chunk extraction |
| `tests/integration/test_artifact_service.py` | SQLite-backed artifact round-trip |

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

---

## Roadmap

### Phase 3 (Next)
- BigQuery, Snowflake, Redshift connectors
- CSV / Google Sheets file upload as queryable data source

### Phase 4
- Standing Monitors — proactive alerts when doc-defined conditions breach in live data
- SSO / SAML integration
- Audit logging
- PII masking before LLM synthesis

---

## License

MIT © [Sanshrit Singhai](https://github.com/ssinghai6)
