# Archify Input: DocBot Architecture Spec

Sources: `01-system-overview.md`, `02-request-flows.md`, `04-endpoint-reference.md`, `api/index.py`, `api/utils/llm_provider.py`, `next.config.ts`, `Dockerfile`, `railway.toml`

Last verified against code: 2026-11-23

Machine-friendly spec for diagram generation. IDs are stable and match the participant names in `02-request-flows.md` (Browser, Client, NextJS, API, PG, Chroma, E2B, Groq, Gemini, Bedrock, TargetDB, SEC, Shopify, Amazon, Stripe, IdP). `kind` is one of `client | service | datastore | external | queue`. There is no queue in this system: background work runs on an in-process APScheduler, modelled as the `Scheduler` service inside the API zone.

## Zones (trust boundaries)

```yaml
zones:
  - id: Z0
    label: Public internet
  - id: Z1
    label: Vercel (frontend edge)
  - id: Z2
    label: Railway container (FastAPI backend)
  - id: Z3
    label: Railway data plane
  - id: Z4
    label: LLM providers
  - id: Z5
    label: Untrusted code execution
  - id: Z6
    label: Third-party and customer systems
```

## Nodes

```yaml
nodes:
  - {id: Browser,     label: "Browser (Next.js UI)",                 kind: client,    zone: Z0}
  - {id: Client,   label: "Developer API client (Bearer key)",    kind: client,    zone: Z0}
  - {id: NextJS,      label: "Next.js 16 on Vercel (CSP proxy + /api,/v1 rewrite)", kind: service, zone: Z1}
  - {id: VercelAnalytics, label: "Vercel Analytics",                 kind: external,  zone: Z1}
  - {id: API,         label: "FastAPI api.index:app (uvicorn)",       kind: service,   zone: Z2}
  - {id: Guard,       label: "Middleware: security headers, CORS, abuse_guard", kind: service, zone: Z2}
  - {id: RAG,         label: "Doc RAG: pdf_io, chunker, retriever, reranker, personas", kind: service, zone: Z2}
  - {id: SQLPipe,     label: "SQL pipeline: db_service (7 steps, sqlglot)", kind: service, zone: Z2}
  - {id: Hybrid,      label: "Hybrid service + discrepancy detector", kind: service,   zone: Z2}
  - {id: Autopilot,   label: "Autopilot LangGraph + deep_retrieve",   kind: service,   zone: Z2}
  - {id: Sandbox,     label: "Sandbox client (sandbox_service)",      kind: service,   zone: Z2}
  - {id: Connectors,  label: "Connectors: Shopify, Amazon, EDGAR",    kind: service,   zone: Z2}
  - {id: Scheduler,   label: "APScheduler (sync, retention, credit reaper)", kind: service, zone: Z2}
  - {id: Identity,    label: "Auth: SAML, OAuth, password, RBAC, API keys", kind: service, zone: Z2}
  - {id: Billing,     label: "Credits, metering, spend guard, Stripe client", kind: service, zone: Z2}
  - {id: LLMProvider, label: "llm_provider (primary + fallback, cost log)", kind: service, zone: Z2}
  - {id: ONNX,        label: "Local ONNX embeddings + reranker",      kind: service,   zone: Z2}
  - {id: Chroma,      label: "Chroma vector store (files on /data volume)", kind: datastore, zone: Z2}
  - {id: PG,          label: "PostgreSQL (Railway)",                  kind: datastore, zone: Z3}
  - {id: Groq,        label: "Groq: gpt-oss-20b / gpt-oss-120b",       kind: external,  zone: Z4}
  - {id: Bedrock,     label: "Amazon Bedrock (optional primary/fallback)", kind: external, zone: Z4}
  - {id: Gemini,      label: "Gemini 2.5 Flash (fallback + LangExtract)", kind: external, zone: Z4}
  - {id: E2B,         label: "E2B code-interpreter sandbox",          kind: external,  zone: Z5}
  - {id: TargetDB,  label: "Customer DB (Postgres/MySQL/Azure SQL)", kind: datastore, zone: Z6}
  - {id: SEC,         label: "SEC EDGAR",                             kind: external,  zone: Z6}
  - {id: Shopify,     label: "Shopify Admin API",                     kind: external,  zone: Z6}
  - {id: Amazon,      label: "Amazon SP-API + LWA",                   kind: external,  zone: Z6}
  - {id: Stripe,      label: "Stripe",                                kind: external,  zone: Z6}
  - {id: IdP,         label: "GitHub / Google OAuth, SAML IdP",       kind: external,  zone: Z6}
  - {id: Resend,      label: "Resend (verification email)",           kind: external,  zone: Z6}
  - {id: Turnstile,   label: "Cloudflare Turnstile",                  kind: external,  zone: Z6}
  - {id: Sentry,      label: "Sentry (errors only)",                  kind: external,  zone: Z6}
  - {id: LangSmith,   label: "LangSmith (opt-in tracing)",            kind: external,  zone: Z6}
```

## Edges

```yaml
edges:
  - {id: e1,  from: Browser,     to: NextJS,      label: "pages, /api/* fetch, SSE",       protocol: HTTPS}
  - {id: e2,  from: Browser,     to: VercelAnalytics, label: "page analytics",             protocol: HTTPS}
  - {id: e3,  from: Client,   to: NextJS,      label: "/v1/* with Bearer API key",      protocol: HTTPS}
  - {id: e4,  from: NextJS,      to: Guard,       label: "rewrite /api/*, /v1/* to BACKEND_URL", protocol: HTTPS}
  - {id: e5,  from: Stripe,      to: NextJS,      label: "POST /api/billing/webhook",      protocol: HTTPS}
  - {id: e6,  from: Shopify,     to: NextJS,      label: "POST /api/marketplace/webhook/shopify", protocol: HTTPS}
  - {id: e7,  from: Guard,       to: API,         label: "rate limit, write freeze, demo block, anon cookie", protocol: in-process}
  - {id: e8,  from: Guard,       to: PG,          label: "rate_limit_counters upsert",     protocol: SQL/asyncpg}
  - {id: e9,  from: API,         to: Identity,    label: "require_principal / require_role / require_api_key", protocol: in-process}
  - {id: e10, from: Identity,    to: PG,          label: "users, user_sessions, api_keys, organizations", protocol: SQL/asyncpg}
  - {id: e11, from: Identity,    to: IdP,         label: "OAuth code exchange / SAML assertion", protocol: HTTPS}
  - {id: e12, from: Identity,    to: Resend,      label: "verification email",             protocol: HTTPS}
  - {id: e13, from: Identity,    to: Turnstile,   label: "sign-up token verify",           protocol: HTTPS}
  - {id: e14, from: API,         to: Billing,     label: "begin_billing reserve/settle",   protocol: in-process}
  - {id: e15, from: Billing,     to: PG,          label: "credit balances, ledger, usage_events", protocol: SQL/asyncpg}
  - {id: e16, from: Billing,     to: Stripe,      label: "create Checkout session",        protocol: HTTPS}
  - {id: e17, from: API,         to: RAG,         label: "upload, /api/chat",              protocol: in-process}
  - {id: e18, from: RAG,         to: ONNX,        label: "embed chunks/query; rerank on hybrid path only",     protocol: in-process}
  - {id: e19, from: RAG,         to: Chroma,      label: "per-session collection read/write", protocol: in-process/file}
  - {id: e20, from: RAG,         to: Gemini,      label: "LangExtract financial extraction", protocol: HTTPS}
  - {id: e21, from: API,         to: SQLPipe,     label: "/api/db/chat",                   protocol: in-process}
  - {id: e22, from: SQLPipe,     to: PG,          label: "schema_cache, query_history, db_connections", protocol: SQL/asyncpg}
  - {id: e23, from: SQLPipe,     to: TargetDB,  label: "read-only SELECT (SSRF-checked, pinned DNS)", protocol: SQL (driver)}
  - {id: e24, from: SQLPipe,     to: Sandbox,     label: "CSV dialect short-circuit",      protocol: in-process}
  - {id: e25, from: API,         to: Hybrid,      label: "/api/hybrid/chat, /v1/hybrid/chat", protocol: in-process}
  - {id: e26, from: Hybrid,      to: RAG,         label: "doc retrieval",                  protocol: in-process}
  - {id: e27, from: Hybrid,      to: SQLPipe,     label: "SQL retrieval",                  protocol: in-process}
  - {id: e28, from: Hybrid,      to: Sandbox,     label: "CSV retrieval",                  protocol: in-process}
  - {id: e29, from: API,         to: Autopilot,   label: "/api/autopilot/run, /v1/autopilot/run", protocol: in-process}
  - {id: e30, from: Autopilot,   to: RAG,         label: "doc_search via deep_retrieve",   protocol: in-process}
  - {id: e31, from: Autopilot,   to: SQLPipe,     label: "sql_query tool",                 protocol: in-process}
  - {id: e32, from: Autopilot,   to: Sandbox,     label: "python_analysis tool",              protocol: in-process}
  - {id: e33, from: Sandbox,     to: E2B,         label: "create sandbox, run code, fetch charts", protocol: HTTPS}
  - {id: e34, from: RAG,         to: LLMProvider, label: "answer stream",                  protocol: in-process}
  - {id: e35, from: SQLPipe,     to: LLMProvider, label: "SQL gen + answer",               protocol: in-process}
  - {id: e36, from: Hybrid,      to: LLMProvider, label: "intent + synthesis",             protocol: in-process}
  - {id: e37, from: Autopilot,   to: LLMProvider, label: "plan, step, synthesize",         protocol: in-process}
  - {id: e38, from: Sandbox,     to: LLMProvider, label: "pandas code gen (gpt-oss-120b)", protocol: in-process}
  - {id: e39, from: LLMProvider, to: Groq,        label: "primary (default)",              protocol: HTTPS (OpenAI-compatible)}
  - {id: e40, from: LLMProvider, to: Bedrock,     label: "primary when LLM_PRIMARY=bedrock", protocol: HTTPS (OpenAI-compatible)}
  - {id: e41, from: LLMProvider, to: Gemini,      label: "fallback on retriable error",    protocol: HTTPS}
  - {id: e42, from: LLMProvider, to: PG,          label: "llm_calls log, llm_response_cache", protocol: SQL/asyncpg}
  - {id: e43, from: LLMProvider, to: LangSmith,   label: "opt-in trace export",            protocol: HTTPS}
  - {id: e44, from: API,         to: Connectors,  label: "register / sync / EDGAR ingest", protocol: in-process}
  - {id: e45, from: Scheduler,   to: Connectors,  label: "periodic sync",                  protocol: in-process}
  - {id: e46, from: Connectors,  to: Shopify,     label: "orders (cursor pagination)",     protocol: HTTPS}
  - {id: e47, from: Connectors,  to: Amazon,      label: "LWA token + Orders/Finances",    protocol: HTTPS}
  - {id: e48, from: Connectors,  to: SEC,         label: "company search, filing download (10 req/s)", protocol: HTTPS}
  - {id: e49, from: Connectors,  to: PG,          label: "marketplace_connections, commerce_*, edgar_filings_cache", protocol: SQL/asyncpg}
  - {id: e50, from: Connectors,  to: RAG,         label: "EDGAR filing chunk + embed",     protocol: in-process}
  - {id: e51, from: API,         to: PG,          label: "sessions, messages, audit_log, lineage, traces", protocol: SQL/asyncpg}
  - {id: e52, from: API,         to: Sentry,      label: "exceptions (scrubbed)",          protocol: HTTPS}
  - {id: e53, from: Scheduler,   to: Chroma,      label: "vector-store retention cleanup (daily)", protocol: in-process/file}
  - {id: e54, from: Scheduler,   to: PG,          label: "commerce retention, credit reaper", protocol: SQL/asyncpg}
```

## Key flows (ordered edge lists)

```yaml
flows:
  - id: F1
    name: "PDF upload and RAG chat (SSE)"
    steps: [e1, e4, e7, e9, e17, e18, e19, e20, e51, e1, e4, e7, e9, e14, e17, e18, e19, e34, e39, e42, e15]
    notes: "First pass = POST /api/upload (chunk, embed, persist, optional LangExtract). Second pass = POST /api/chat: persona route, retrieve, rerank, stream answer, settle credits."
  - id: F2
    name: "Live DB question (7-step SQL pipeline)"
    steps: [e1, e4, e7, e9, e14, e21, e22, e35, e39, e23, e35, e39, e22, e15]
    notes: "Schema from cache, table select, few-shot, SQL gen (LLM 1), sqlglot validate, execute read-only on customer DB (drift retry once), answer (LLM 2)."
  - id: F3
    name: "CSV question via E2B"
    steps: [e1, e4, e7, e9, e14, e21, e24, e38, e39, e33, e15]
    notes: "CSV dialect bypasses SQL; DataProfile in prompt; one corrective retry on sandbox error."
  - id: F4
    name: "Hybrid docs + data with discrepancy detection"
    steps: [e1, e4, e7, e9, e14, e25, e36, e39, e26, e18, e19, e27, e23, e28, e33, e36, e39, e15]
    notes: "Intent classification, parallel doc and data retrieval, synthesis with dual citations, numeric discrepancy flags."
  - id: F5
    name: "Autopilot investigation"
    steps: [e1, e4, e7, e9, e14, e29, e37, e39, e30, e18, e19, e31, e23, e32, e33, e37, e39, e51, e15]
    notes: "LangGraph plan -> execute tool steps -> synthesize; doc_search uses deep_retrieve (sub-questions, gap fill); per-run cost ceiling."
  - id: F6
    name: "Sign-in (OAuth) and credit purchase"
    steps: [e1, e4, e9, e11, e10, e1, e4, e14, e16, e5, e15]
    notes: "OAuth callback creates user + docbot_session cookie; admin starts Stripe Checkout; Stripe webhook grants credits."
  - id: F7
    name: "Commerce connector sync"
    steps: [e44, e49, e45, e46, e47, e49]
    notes: "Register stores encrypted credentials and schedules sync; scheduler pulls orders/financials and upserts commerce_* rows."
```
