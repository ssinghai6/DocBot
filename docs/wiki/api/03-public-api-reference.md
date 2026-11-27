# DocBot API v1 Reference (Proposed)

Sources: `api/index.py` (`/v1/*` handlers, `ChatRequest`, `HybridChatRequest`), `api/db_service.py` (`DBChatRequest`), `api/demo_service.py`, `api/session_models.py`, `api/utils/lineage.py`, `api/routes/tieout_routes.py`, `api/credit_service.py`, `api/billing_models.py`, `api/utils/upload_limits.py`, `api/api_key_service.py`

Last verified against code: 2026-11-23

> **This is the PROPOSED v1 contract.** Only `POST /v1/hybrid/chat`, `POST /v1/autopilot/run` and `GET /v1/lineage/{run_id}` exist today, and they still return the old `{"detail": ...}` errors and the old field set. Everything else here is design, to be built per [Public API Implementation Backlog](api-04-implementation-backlog). Each endpoint names the internal route or service it wraps.

## Conventions

| Item | Value |
|---|---|
| Base URL | `https://docbot.space/v1` |
| Auth | `Authorization: Bearer dk_test_...` or `Bearer dk_live_...`. Never in the URL (400 `key_in_url`). |
| Content type | `application/json` unless noted (`/v1/documents` is `multipart/form-data`). |
| Request id | Every response has `X-Request-Id`. Quote it in support requests. |
| Idempotency | Optional `Idempotency-Key` header on POSTs marked "idempotent". 24h window. |
| Rate-limit headers | `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` on every response; `Retry-After` on 429. |
| Cost header | `X-DocBot-Credits-Charged` on cost-bearing calls. 1 credit = $0.001 retail. |
| Limits | Test key: 10 req/min, 1 concurrent long run, 300 credits/day. Live key: 60 req/min, 3 concurrent long runs, org balance. |
| Pagination | `?limit=1..100&starting_after=<cursor>` returns `{data, has_more, next_cursor}`. |
| Streaming | Add `Accept: text/event-stream` on endpoints marked "streamable". Default is one JSON body. |
| Personas | `Generalist`, `Finance Expert`, `Data Analyst`, `Strategy Analyst` (+ backend-only ones listed by `/v1/personas`). |

Credit cost below is the **reservation** held up front (`api/credit_service.py:126-132`); the actual debit is metered model and sandbox cost times markup, usually lower. Read-only calls cost 0.

### Error envelope

```json
{
  "error": {
    "type": "permission_error",
    "code": "missing_scope",
    "message": "API key missing required scope: autopilot:run.",
    "param": null,
    "request_id": "req_7f3c2a9e",
    "doc_url": "https://docbot.space/apis#errors-missing_scope"
  }
}
```

Common errors on every endpoint (not repeated below): 401 `missing_key` / `invalid_key` / `revoked_key` / `expired_key`, 403 `missing_scope`, 429 `rate_limited`, 503 `maintenance` / `platform_capacity`. Full table in [Public API and Developer Portal Plan](api-02-public-api-plan#6-error-envelope).

Setup for all examples (response values are illustrative; field names and types are the contract; demo company, filing, tables and questions are the real `quickbite` dataset in `api/demo_service.py:37-39`, `:150-180`, `:205-208`):

```bash
export DOCBOT_API_KEY="dk_test_..."   # from docbot.space/apis
export HOST="https://docbot.space"
```

---

## Key and catalog

### GET /v1/me

Describe the calling key. Use it to check the key works.

| | |
|---|---|
| Scope | none (any valid key) |
| Wraps | `api_key_service.validate_api_key` record (new handler) |
| Rate limit / cost | standard / 0 credits |

Response `200`:

| Field | Type | Notes |
|---|---|---|
| `key_id` | string | |
| `key_prefix` | string | e.g. `dk_test_a1b2c3d4` |
| `mode` | `"test"` \| `"live"` | |
| `org_id` | string | |
| `scopes` | string[] | |
| `expires_at` | string (ISO 8601) \| null | |
| `rate_limit` | `{requests_per_minute: int, concurrent_runs: int}` | |
| `test_allowance` | `{daily_credits: int, remaining_credits: int, resets_at: string}` \| null | test keys only |

```bash
curl -sS "$HOST/v1/me" -H "Authorization: Bearer $DOCBOT_API_KEY"
```

```json
{
  "key_id": "6c1d0f3e-...",
  "key_prefix": "dk_test_a1b2c3d4",
  "mode": "test",
  "org_id": "org_...",
  "scopes": ["sessions:read", "sessions:write", "query:run", "autopilot:run", "lineage:read", "usage:read"],
  "expires_at": "2026-12-23T00:00:00Z",
  "rate_limit": {"requests_per_minute": 10, "concurrent_runs": 1},
  "test_allowance": {"daily_credits": 300, "remaining_credits": 263, "resets_at": "2026-11-24T00:00:00Z"}
}
```

### GET /v1/personas

List valid `persona` values.

| | |
|---|---|
| Scope | `sessions:read` |
| Wraps | `GET /api/personas` (`api/index.py:907`) |
| Rate limit / cost | standard / 0 |

Response `200`: `{"data": [{"name": "Finance Expert", "description": "..."}]}`.

```bash
curl -sS "$HOST/v1/personas" -H "Authorization: Bearer $DOCBOT_API_KEY"
```

### GET /v1/openapi.json

The machine-readable spec for everything on this page. No auth required (it only describes public endpoints). Wraps FastAPI `get_openapi()` filtered to `/v1` routes.

---

## Sessions

A **session** holds uploaded documents (vector store) and chat history. A **connection** is a database (or demo SQLite, or CSV) attached to it.

### POST /v1/sandbox/sessions

Create a ready-to-query sandbox: a demo 10-K-style PDF set plus a matching SQLite ledger with deliberate discrepancies. This is the starting point for test keys.

| | |
|---|---|
| Scope | `sessions:write` |
| Modes | test and live (live keys may use it for demos; still marked `is_sandbox`) |
| Wraps | `POST /api/demo/init` -> `demo_service.init_demo_session` (`api/index.py:969-989`, `api/demo_service.py:604-711`) |
| Idempotent | yes |
| Rate limit | test: 5 sandbox sessions/hour per key owner |
| Cost | 0 credits (embedding is local); counts toward demo spend breaker |

Request:

| Field | Type | Required | Notes |
|---|---|---|---|
| `dataset` | `"quickbite"` \| `"fuel"` \| `"tieout"` | no, default `quickbite` | `quickbite`: docs vs DB discrepancy. `fuel`: crude-oil series + policy PDF (forecasting). `tieout`: draft 10-K vs trial balance |

Response `201`:

| Field | Type |
|---|---|
| `session_id` | string |
| `connection_id` | string |
| `dataset` | string |
| `company` | string |
| `filing` | string |
| `schema_summary` | `{table_count: int, tables: string[]}` |
| `suggested_questions` | string[] |
| `expires_at` | string (24h after creation) |

```bash
curl -sS -X POST "$HOST/v1/sandbox/sessions" \
  -H "Authorization: Bearer $DOCBOT_API_KEY" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: 4f0e6b8a-1d2c-4a57-9c7e-2b1f0e9d3a11" \
  -d '{"dataset": "quickbite"}'
```

```json
{
  "session_id": "sess_...",
  "connection_id": "conn_...",
  "dataset": "quickbite",
  "company": "QuickBite Inc.",
  "filing": "Annual Report (10-K) — Fiscal Year 2025",
  "schema_summary": {"table_count": 4, "tables": ["financials", "quarterly", "segments", "balance_sheet"]},
  "suggested_questions": [
    "What was QuickBite's total revenue and net income in 2025?",
    "Are there any discrepancies between the filing and the database?"
  ],
  "expires_at": "2026-11-24T10:00:00Z"
}
```

Errors: 400 `unsupported_dataset`, 429 `rate_limited` (sandbox window).

### GET /v1/sessions

List sessions this key can reach (test key: its sandbox sessions; live key: user-owned sessions in the org). Newest first.

| | |
|---|---|
| Scope | `sessions:read` |
| Wraps | `GET /api/sessions` -> `session_service.list_sessions` with an org-scoped `CallerScope` (`api/index.py:1585-1589`) |
| Cost | 0 |

Query: `limit`, `starting_after`.

Response `200`: `{"data": [SessionSummary], "has_more": bool, "next_cursor": string|null}` where `SessionSummary` is (`api/session_models.py:10-16`) plus `is_sandbox`:

| Field | Type |
|---|---|
| `session_id` | string |
| `created_at` / `updated_at` | string \| null |
| `persona` | string \| null |
| `file_count` | int \| null |
| `files_info` | object[] (file name, pages, size) |
| `is_sandbox` | bool |

```bash
curl -sS "$HOST/v1/sessions?limit=20" -H "Authorization: Bearer $DOCBOT_API_KEY"
```

### GET /v1/sessions/{session_id}

| | |
|---|---|
| Scope | `sessions:read` |
| Wraps | `GET /api/session/{session_id}` (`api/index.py:1592`) |
| Cost | 0 |

Response `200`: `{"session": SessionSummary, "messages": [{"role": "user"|"assistant", "content": string}]}`. 404 `resource_not_found` if not reachable by this key.

### DELETE /v1/sessions/{session_id}

Delete a session, its vector store and its messages.

| | |
|---|---|
| Scope | `sessions:write` |
| Wraps | `DELETE /api/session/{session_id}` (`api/index.py:1603`) |
| Cost | 0 |

Response `200`: `{"deleted": true, "session_id": "..."}`.

```bash
curl -sS -X DELETE "$HOST/v1/sessions/$SESSION_ID" -H "Authorization: Bearer $DOCBOT_API_KEY"
```

### POST /v1/documents

Upload PDFs into a new or existing session. **Live keys only** (phase 3).

| | |
|---|---|
| Scope | `sessions:write` |
| Wraps | `POST /api/upload` (`api/index.py:992`) |
| Idempotent | yes |
| Limits | `UPLOAD_MAX_FILE_MB` 25 per file, `UPLOAD_MAX_FILES` 10, `UPLOAD_PDF_MAX_PAGES` 500 (`api/utils/upload_limits.py:28-31`) |
| Cost | reservation 20 credits per call (embedding + optional extraction; inferred) |

Request (`multipart/form-data`):

| Field | Type | Required |
|---|---|---|
| `files` | file[] (PDF) | yes |
| `session_id` | string | no (omit to create a new session) |

Response `201`: `{"session_id": string, "files": [{"name": string, "pages": int, "chunks": int}]}`.

```bash
curl -sS -X POST "$HOST/v1/documents" \
  -H "Authorization: Bearer $DOCBOT_API_KEY" \
  -F "files=@./10-K.pdf"
```

Errors: 403 `test_key_not_allowed`, 413 `payload_too_large`, 400 `invalid_body` (password-protected or unparsable PDF).

### GET /v1/db/connections

List database connections reachable by this key (no credentials ever returned).

| | |
|---|---|
| Scope | `sessions:read` |
| Wraps | `GET /api/db/sessions` (`api/index.py:2042`) |
| Cost | 0 |

Response `200`: `{"data": [{"connection_id": string, "dialect": "postgresql"|"mysql"|"sqlite"|"mssql"|"csv", "database_name": string, "created_at": string, "is_sandbox": bool}], "has_more": bool, "next_cursor": string|null}`.

### GET /v1/db/connections/{connection_id}/schema

| | |
|---|---|
| Scope | `sessions:read` |
| Wraps | `GET /api/db/schema/{connection_id}` (`api/index.py:1892`) |
| Cost | 0 |

Response `200`: `{"connection_id": string, "tables": [{"name": string, "kind": "table"|"view", "columns": [{"name": string, "type": string}]}]}`.

---

## Questions

All question endpoints accept an optional `history` (last 6 messages, `[{role, content}]`) so follow-ups are rephrased into standalone questions, exactly as in the UI.

### POST /v1/chat

Ask a question over the session's documents only.

| | |
|---|---|
| Scope | `query:run` |
| Wraps | `POST /api/chat` (`api/index.py:1202`), buffered |
| Idempotent / streamable | yes / yes |
| Cost | reservation 20 credits |

Request (mirrors `ChatRequest`, `api/index.py:800-804`, renamed `message` -> `question` for consistency):

| Field | Type | Required | Default |
|---|---|---|---|
| `session_id` | string | yes | |
| `question` | string (1-4000 chars) | yes | |
| `persona` | string | no | `Generalist` |
| `history` | `{role, content}[]` (max 6) | no | `[]` |

Response `200`:

| Field | Type |
|---|---|
| `answer` | string (markdown, PII-masked) |
| `citations` | `{source: string, page: int, text: string}[]` |
| `run_id` | string |

```bash
curl -sS -X POST "$HOST/v1/chat" \
  -H "Authorization: Bearer $DOCBOT_API_KEY" -H "Content-Type: application/json" \
  -d '{"session_id": "'"$SESSION_ID"'", "question": "What were the main risk factors?", "persona": "Finance Expert"}'
```

```json
{
  "answer": "The filing lists three primary risks: ...",
  "citations": [{"source": "annual_report.pdf", "page": 14, "text": "Competition in food delivery ..."}],
  "run_id": "run_..."
}
```

Errors: 404 `resource_not_found`, 402 `insufficient_credits`, 429 `test_allowance_exhausted`, 502 `pipeline_failed`.

### POST /v1/db/query

Natural language to SQL over a connected database (bounded 7-step pipeline, read-only enforced 3 ways). CSV connections are routed to the pandas sandbox automatically.

| | |
|---|---|
| Scope | `query:run` |
| Wraps | `POST /api/db/chat` (`api/index.py:1964`, `DBChatRequest` at `api/db_service.py:653-659`) |
| Idempotent / streamable | yes / yes |
| Cost | reservation 30 credits |

Request:

| Field | Type | Required | Default |
|---|---|---|---|
| `connection_id` | string | yes | |
| `session_id` | string | yes | |
| `question` | string | yes | |
| `persona` | string | no | `Generalist` |
| `chart_type` | `auto`\|`bar`\|`line`\|`scatter`\|`heatmap`\|`box`\|`multi`\|`none` | no | `auto` |
| `history` | `{role, content}[]` | no | `[]` |

Response `200`:

| Field | Type | Notes |
|---|---|---|
| `answer` | string | |
| `sql_query` | string \| null | generated SQL (null for CSV) |
| `explanation` | string \| null | |
| `row_count` | int | capped at 500 |
| `result_preview` | object[] | first rows, PII-masked when the connection has masking on |
| `chart` | `{type: string, png_base64: string}` \| null | |
| `run_id` | string | |

```bash
curl -sS -X POST "$HOST/v1/db/query" \
  -H "Authorization: Bearer $DOCBOT_API_KEY" -H "Content-Type: application/json" \
  -d '{"session_id": "'"$SESSION_ID"'", "connection_id": "'"$CONNECTION_ID"'", "question": "Revenue by quarter", "chart_type": "none"}'
```

```json
{
  "answer": "2025 revenue by quarter: Q1 $1.12B, Q2 $1.18B, Q3 $1.21B, Q4 $1.30B.",
  "sql_query": "SELECT quarter, revenue_m FROM quarterly ORDER BY quarter",
  "explanation": "Reads revenue (in $M) for each quarter.",
  "row_count": 4,
  "result_preview": [{"quarter": "Q1", "revenue_m": 1120.0}],
  "chart": null,
  "run_id": "run_..."
}
```

### POST /v1/hybrid/chat (exists, contract changes)

Ask across documents **and** the database; returns dual citations and any discrepancies between the two.

| | |
|---|---|
| Scope | `query:run` (or legacy `hybrid:read`) |
| Wraps | existing `/v1/hybrid/chat` (`api/index.py:3659-3753`) over `hybrid_service.hybrid_chat` |
| Idempotent / streamable | yes / yes |
| Cost | reservation 40 credits |

Request (unchanged from `V1HybridChatRequest`, `api/index.py:3649-3656`):

| Field | Type | Required | Default |
|---|---|---|---|
| `question` | string | yes | |
| `session_id` | string | yes | |
| `connection_id` | string \| null | no | null (docs only) |
| `persona` | string | no | `Data Analyst` |
| `has_docs` | bool | no | `true` (hint only) |
| `history` | `{role, content}[]` | no | `[]` |
| `chart_type` | string | no | `auto` |

Response `200` (new fields marked *):

| Field | Type | Notes |
|---|---|---|
| `answer` | string | final answer **after** discrepancy correction (fixes today's dropped `replace` event) |
| `citations` | object[] | document citations |
| `intent` | `docs`\|`db`\|`hybrid` | fixed to never be overwritten by the SQL metadata event |
| `has_sql` / `has_docs` | bool | |
| `sql_query`* | string \| null | |
| `discrepancies`* | `LineageDiscrepancy[]` | `{id, metric, period, report_value, books_value, delta, delta_pct, severity, unit, doc_citation{file,page}, db_citation{table,query}}` (`api/utils/lineage.py:93-112`) |
| `run_id` / `trace_id` | string | |

```bash
curl -sS -X POST "$HOST/v1/hybrid/chat" \
  -H "Authorization: Bearer $DOCBOT_API_KEY" -H "Content-Type: application/json" \
  -d '{"session_id": "'"$SESSION_ID"'", "connection_id": "'"$CONNECTION_ID"'", "question": "Does Q4 net income in the report match the ledger?", "persona": "Finance Expert"}'
```

```json
{
  "answer": "No. The report states Q4 net income of $325M; the ledger totals $330M, a $5M (1.5%) gap ...",
  "citations": [{"source": "annual_report.pdf", "page": 31, "text": "Net income for the fourth quarter was $325 million"}],
  "intent": "hybrid",
  "has_sql": true,
  "has_docs": true,
  "sql_query": "SELECT net_income_m FROM quarterly WHERE quarter = 'Q4'",
  "discrepancies": [{
    "id": "d1", "metric": "net_income", "period": "Q4 2025",
    "report_value": 325000000, "books_value": 330000000, "delta": 5000000, "delta_pct": 1.54,
    "severity": "medium", "unit": "currency",
    "doc_citation": {"file": "annual_report.pdf", "page": 31},
    "db_citation": {"table": "quarterly", "query": "SELECT ..."}
  }],
  "run_id": "run_...",
  "trace_id": "trc_..."
}
```

SSE (with `Accept: text/event-stream`): events `metadata`, `token`, `replace`, `done`, `error` as defined in [Public API and Developer Portal Plan](api-02-public-api-plan#9-streaming-sse-contract). `replace` carries the full corrected answer; clients must swap, not append.

### POST /v1/autopilot/run (exists)

Multi-step investigation: plan (up to 5 steps) -> execute tools (doc search with deep retrieval, SQL, pandas) -> synthesize. Long-running (often 20-90s, inferred); prefer streaming or, later, the async job form (DOCBOT-3902).

| | |
|---|---|
| Scope | `autopilot:run` |
| Wraps | existing `/v1/autopilot/run` (`api/index.py:3768-3879`) over `autopilot_service.run_autopilot` |
| Idempotent / streamable | yes / yes |
| Concurrency | counts toward the per-key concurrent long-run limit |
| Cost | reservation 150 credits; per-run soft ceiling `SESSION_COST_CEILING_USD` |

Request (`V1AutopilotRequest`, `api/index.py:3756-3765`):

| Field | Type | Required | Default |
|---|---|---|---|
| `session_id` | string | yes | |
| `question` | string | yes | |
| `connection_id` | string \| null | no | null |
| `persona` | string | no | `Generalist` |
| `history` | `{role, content}[]` | no | `[]` |
| `has_docs` / `has_db` / `has_csv` | bool | no | hints only; the server resolves real sources |

Response `200`:

| Field | Type |
|---|---|
| `plan` | `{index: int, tool: string, description: string}[]` |
| `steps` | `{index: int, tool: string, status: "done"\|"failed"\|"skipped", summary: string}[]` |
| `answer` | string |
| `citations` | object[] |
| `warnings` | string[] (e.g. cost ceiling reached) |
| `run_id`* | string |

```bash
curl -sS -X POST "$HOST/v1/autopilot/run" \
  -H "Authorization: Bearer $DOCBOT_API_KEY" -H "Content-Type: application/json" \
  -d '{"session_id": "'"$SESSION_ID"'", "connection_id": "'"$CONNECTION_ID"'", "question": "Explain the margin decline and quantify each driver"}'
```

Errors: 429 `concurrency_limited`, 402 `insufficient_credits`, 502 `pipeline_failed`.

---

## Provenance

### GET /v1/lineage/{run_id} (exists)

Full provenance for one answer: steps, sources with retrieval scores, SQL, discrepancies, model calls, PII masking counts.

| | |
|---|---|
| Scope | `lineage:read` |
| Wraps | existing handler (`api/index.py:3882-3897`) over `lineage_service.get_lineage` |
| Cost | 0 |

Response `200`: the `Lineage` model (`api/utils/lineage.py:153-169`): `run_id`, `mode` (`docs`\|`db`\|`csv`\|`hybrid`\|`autopilot`), `intent`, `question`, `standalone_query`, `expanded_queries[]`, `sub_questions[]`, `steps[] {name, tool, status, latency_ms, detail, lane}`, `sources[] {id, kind, label, page, snippet, score, rerank_score, cited}`, `sql {sql, tables_selected, row_count, execution_time_ms, drift_retry, result_preview}`, `discrepancies[]`, `model_calls[] {provider, model, latency_ms, input_tokens, output_tokens, estimated_cost_usd, fallback_triggered}`, `pii {masked, counts}`, `cache_hit`, `persona`, `total_latency_ms`.

```bash
curl -sS "$HOST/v1/lineage/$RUN_ID" -H "Authorization: Bearer $DOCBOT_API_KEY"
```

### GET /v1/lineage/{run_id}/tieout.{xlsx|pdf}

Download the tie-out workpaper for a run (the auditor-ready deliverable).

| | |
|---|---|
| Scope | `lineage:read` |
| Wraps | `GET /api/lineage/{run_id}/tieout.{fmt}` (`api/routes/tieout_routes.py:37-44`) |
| Cost | 0 |

Response `200`: binary, `Content-Type` `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` or `application/pdf`, `Content-Disposition: attachment`.

```bash
curl -sS -o tieout.xlsx "$HOST/v1/lineage/$RUN_ID/tieout.xlsx" -H "Authorization: Bearer $DOCBOT_API_KEY"
```

---

## Usage

### GET /v1/usage

Credits consumed by **this key**.

| | |
|---|---|
| Scope | `usage:read` |
| Wraps | usage events / credit ledger filtered by `api_key_id` (new; models `UsageSummary`, `UsagePage` in `api/billing_models.py:345-377`) |
| Cost | 0 |

Query: `from`, `to` (ISO dates, default last 30 days), `limit`, `starting_after`.

Response `200`:

```json
{
  "key_id": "6c1d0f3e-...",
  "mode": "test",
  "period": {"from": "2026-10-24", "to": "2026-11-23"},
  "total_credits": 412,
  "by_route": [
    {"route": "hybrid", "calls": 7, "credits": 189},
    {"route": "autopilot", "calls": 2, "credits": 201},
    {"route": "chat", "calls": 3, "credits": 22}
  ],
  "data": [{"at": "2026-11-23T09:12:03Z", "route": "hybrid", "run_id": "run_...", "credits": 31}],
  "has_more": false,
  "next_cursor": null
}
```

---

## Not in v1

DB connect/disconnect, CSV and SQLite upload, code execution, connectors, EDGAR, artifacts, audit log export, admin and billing. Reasons per route are in [Public API and Developer Portal Plan](api-02-public-api-plan#17-endpoint-decisions-every-existing-backend-route).
