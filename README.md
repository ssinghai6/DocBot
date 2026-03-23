# DocBot

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Next.js-16+-black.svg" alt="Next.js">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-teal.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Tests-131 passing-brightgreen.svg" alt="Tests">
</p>

**Ask Anything About Your Data.**

DocBot is an AI-powered document + database analyst. Upload PDFs, connect a live database, or do both — get instant answers with source citations, SQL explanations, Python-generated charts, and discrepancy detection when your documents and data disagree.

---

## What DocBot Does

| Capability | Description |
|---|---|
| **PDF Chat** | Upload PDFs, ask questions, get cited answers from 8 expert personas |
| **Live Database Chat** | Connect PostgreSQL, MySQL, SQLite, or Azure SQL — ask in plain English, get SQL + results |
| **Hybrid Mode** | One question answered from both your documents and your database in a single response |
| **Discrepancy Detection** | Automatically flags when a number in your doc differs from what the DB shows |
| **Python Analysis** | E2B sandbox executes pandas/matplotlib code and returns charts and statistical insights |
| **Structured Extraction** | Pulls typed values (financial metrics, legal dates, medical measurements, research stats) from any document using Gemini 2.5 Flash |
| **Azure SQL / Entra Auth** | Enterprise-grade Microsoft Entra (Azure AD) Service Principal authentication for Azure SQL |

**Core differentiator**: Hybrid Docs+DB synthesis with discrepancy detection. No other tool does this.

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
- **Schema introspection**: Auto-detects tables and columns; cached for performance
- **Read-only enforcement**: 3-layer protection — LLM prompt + sqlglot AST validation + read-only transaction
- **Result cap**: 500 rows, 15-second query timeout

### SQL Generation Pipeline (7 Bounded Steps)

```
NL Question
  → [1] Schema Retrieval     (cache → miss → introspect)
  → [2] Table Selector       (LLM call #1)
  → [3] Few-Shot Retrieval   (cosine similarity on stored queries)
  → [4] SQL Generator        (LLM call #2)
  → [5] SQL Validator        (sqlglot AST — deterministic, no LLM)
  → [6] Executor             (SQLAlchemy, 15s timeout, 500 row cap)
  → [7] Answer Generator     (LLM call #3, streaming)
```

Max 3 LLM calls per query. No loops. Predictable latency and cost.

### Hybrid Mode

Ask one question — DocBot queries both your uploaded documents and your live database and synthesizes a single answer with dual citations. When a number in your document differs from the database, `[DISCREPANCY]` markers appear automatically with the delta and percentage difference.

### Python Analysis via E2B

After SQL execution, DocBot generates pandas/matplotlib code and runs it in an isolated E2B cloud sandbox:
- Statistical summaries, trend analysis, correlations
- Charts returned as base64-encoded PNG
- Code displayed in a collapsible block for full transparency
- Sandbox isolation: no network access, no filesystem persistence

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
│  ┌─────────────────────────────────────────────────────────┐ │
│  │               document_extractor                         │ │
│  │         LangExtract + Gemini 2.5 Flash                  │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────── ┐ │
│  │  utils/  encryption · ssrf_validator · sql_validator     │ │
│  │          embeddings · few_shot_store                     │ │
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
| LLM — SQL + Hybrid | Groq Llama 3.3-70b |
| LLM — Python codegen | Groq Qwen/qwen3-32b |
| LLM — Doc extraction | Gemini 2.5 Flash (via LangExtract) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (HuggingFace API) |
| Vector Search | LangChain InMemoryVectorStore |
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
  "persona": "Data Analyst"
}
# Returns: { answer, sql, results, explanation, chart_base64? }
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

**131 tests passing** across:

| File | Coverage |
|---|---|
| `tests/unit/test_ssrf_validator.py` | SSRF protection — private IPs, loopback, link-local |
| `tests/unit/test_encryption.py` | Fernet credential encryption/decryption |
| `tests/unit/test_sql_validator.py` | sqlglot AST validation — write query rejection |
| `tests/unit/test_db_service_helpers.py` | Connection URL building, Entra token struct, Pydantic models |
| `tests/unit/test_embeddings.py` | Embedding cosine similarity, few-shot retrieval |
| `tests/integration/test_db_pipeline.py` | Full SQL pipeline against SQLite |
| `tests/integration/test_file_upload_service.py` | PDF upload and chunk extraction |

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

---

## Roadmap

### Phase 2 (Next)
- **DOCBOT-405**: Analytical Autopilot — LangGraph-based multi-step investigation agent ("Diagnose why we missed Q3")
- **DOCBOT-305**: Advanced chart types and visualization options
- **DOCBOT-501**: Session artifact store (persist DataFrames and charts across turns)
- **DOCBOT-502**: Context compression for long sessions
- **DOCBOT-503**: Schema-aware semantic table selection

### Phase 3
- BigQuery, Snowflake, Redshift connectors

### Phase 4
- Standing Monitors — proactive alerts when doc-defined conditions breach in live data
- SSO / SAML integration
- Audit logging

---

## License

MIT © [Sanshrit Singhai](https://github.com/ssinghai6)
