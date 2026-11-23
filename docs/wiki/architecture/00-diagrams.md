# Architecture diagrams

Sources: `docs/wiki/architecture/docbot-architecture.json` (Archify spec), `archify-input.md`, `01-system-overview.md`, `02-request-flows.md`
Last verified against code: 2026-11-23

## System map

![DocBot system architecture (light)](ATTACH:docbot-architecture.light.png)

**Interactive version:** download the attachment `docbot-architecture.html` at the bottom of this page and open it in a browser. It supports pan/zoom, search, dark mode, three guided views (question to answer, where data lives, identity and connectors) and PNG/SVG export.

| Node | What it is | Detail page |
|---|---|---|
| Browser / API client | Next.js UI, or a developer client sending `Authorization: Bearer dk_live_...` to `/v1/*` | [Public API: current state](api-01-current-state) |
| Next.js 16 (Vercel) | Pages + CSP proxy; rewrites `/api/*` and `/v1/*` to the Railway backend | [Frontend architecture](architecture-08-frontend-architecture) |
| FastAPI gateway | 82 routes; security headers, CORS, abuse_guard rate limit, auth, RBAC, credit reserve/settle | [Endpoint reference](architecture-04-endpoint-reference) |
| Identity providers | GitHub / Google OAuth, SAML IdP | [Security architecture](architecture-06-security-architecture) |
| Analysis engines | Doc RAG, 7-step SQL pipeline, hybrid docs+DB with discrepancy detection, Autopilot (LangGraph) | [Request flows](architecture-02-request-flows) |
| Chroma + ONNX | Per-session vector collections; local MiniLM embeddings and reranker | [AI and LLM architecture](architecture-05-ai-llm-architecture) |
| llm_provider | Primary (Groq gpt-oss-20b, or Bedrock when `LLM_PRIMARY=bedrock`) with Gemini fallback, cost log, response cache | [AI and LLM architecture](architecture-05-ai-llm-architecture) |
| E2B sandbox | Untrusted generated pandas/matplotlib code for CSV and Python analysis | [Security architecture](architecture-06-security-architecture) |
| Customer databases | Read-only SELECT only; SSRF check with pinned DNS; sqlglot AST validation | [Request flows](architecture-02-request-flows) |
| PostgreSQL | 33 application tables: users, orgs, sessions, credits, audit, caches | [Data model](architecture-03-data-model) |
| Connectors | Shopify, Amazon SP-API, SEC EDGAR; APScheduler syncs | [Request flows](architecture-02-request-flows) |

## Flow diagrams

Each flow (upload + RAG chat, SQL pipeline, CSV to E2B, hybrid, Autopilot, EDGAR, connectors, auth, billing, API keys) has a Mermaid sequence diagram on [Request flows](architecture-02-request-flows). The data model ER diagram is on [Data model](architecture-03-data-model).

## Regenerating

```bash
cd ~/.claude/skills/archify
node bin/archify.mjs deliver architecture docs/wiki/architecture/docbot-architecture.json docs/wiki/architecture/docbot-architecture.html --quality showcase --json
node bin/archify.mjs visual-check docs/wiki/architecture/docbot-architecture.html --json
```
Edit the JSON spec when components or connections change, then re-import the wiki.
