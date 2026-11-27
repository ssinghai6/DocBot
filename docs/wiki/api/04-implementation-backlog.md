# Public API Implementation Backlog

Sources: `project-tasks/docbot-master-backlog.md` (EPIC-39 section 7.13, DOCBOT-2907, DOCBOT-3305), `project-tasks/docbot-enterprise-roadmap.md` (EPIC-39), `CLAUDE.md` (F-4 route rule, testing rules), [Developer API: Current State](api-01-current-state), [Public API and Developer Portal Plan](api-02-public-api-plan), [DocBot API v1 Reference (Proposed)](api-03-public-api-reference)

Last verified against code: 2026-11-23

## How this maps to existing tickets

EPIC-39 "API, SDK and MCP Server" already exists (7 stories, 30 pts, parked, track E; master backlog section 7.13), building on DOCBOT-1530 (developer API, shipped) and DOCBOT-2907 (scoped keys with expiry, parked). This plan reuses those ids where they fit. The two live `/v1` defects are the bugs already filed as DOCBOT-2591 and DOCBOT-2592 (being fixed now, not re-filed). Every other story was filed on 2026-11-23 (DOCBOT-2595) as DOCBOT-2596..2599 and DOCBOT-2611..2618 (2600..2610 are EPIC-26). Story files: `project-tasks/stories/DOCBOT-XXXX.md`; backlog rows: `project-tasks/docbot-master-backlog.md` sections 7.3, 7.13 and 8.

Repo rules every story inherits (from `CLAUDE.md`):

- New routes in `api/routes/<area>_routes.py` as `APIRouter`; `api/index.py` only gets `app.include_router(...)`.
- Request/response models in `api/<area>_models.py` (here `api/v1_models.py`, `api/developer_models.py`) with Zod mirrors in `src/lib/schemas/<area>.ts` (identical field names).
- Every route has an auth guard (`require_api_key(...)` for `/v1`, `require_role`/`require_principal` for `/api`) or is listed in `PUBLIC_ROUTES` in `tests/unit/test_route_guards.py` with a justification.
- Tests in `tests/unit/` (no network) or `tests/integration/`; no `type: ignore`, no bare `except`, no `any` in TS.
- `copy_ctx_run()` for any executor call so principal and meter propagate (`api/utils/billing_context.py`).

## Stories

| Id | Title | Acceptance criteria | Pts | Deps | Sprint |
|---|---|---|---|---|---|
| **DOCBOT-2591** | Org-scope `/v1/connectors*` (P0 security) | `GET /v1/connectors` passes `org_id=key.org_id`; `POST` saves with `org_id`; `sync` uses `connector_in_org` like `api/routes/connector_routes.py:69-86`; cross-org test per route returns 404; routes hidden from spec until done | 2 | 1530, 1602 | S1 |
| **DOCBOT-2596** | Close default FastAPI docs in prod | `docs_url`/`redoc_url`/`openapi_url` are `None` unless `ENV=dev`; `/docs` 404 on prod backend; CSP exemption in `api/utils/security_headers.py:39` removed or dev-only; test asserts both modes | 1 | none | S1 |
| **DOCBOT-2592** | Fix buffered `/v1/hybrid/chat` answer | Honor `replace` event (answer equals corrected text); merge `metadata` events so `intent`/`has_sql`/`has_docs` are never overwritten by the SQL metadata event; unit test feeds a fake stream with both cases | 2 | none | S1 |
| **DOCBOT-3901** | Public API v1 contract (existing, expanded) | `/v1` handlers moved from `api/index.py:3580-3981` to `api/routes/v1_routes.py`; models in `api/v1_models.py` + `src/lib/schemas/v1.ts`; `response_model` on every route; `/v1` error envelope handler + `X-Request-Id`; per-key limiter on the shared counter store returns 429 + `Retry-After` + `X-RateLimit-*` (replaces sleeping in-memory bucket `api/api_key_service.py:129-132`); `GET /v1/openapi.json` filtered spec + snapshot test; `GET /v1/me`; versioning and deprecation policy doc; reject keys in query string (400 `key_in_url`) | 5 | DOCBOT-2591, DOCBOT-2596 | S1-S2 |
| **DOCBOT-2907** | Key modes, expiry, owner (existing, expanded) | `api_keys` gains `mode` (`test`/`live`), `expires_at`, `created_by`; `create_api_key(mode=...)` emits `dk_test_` or `dk_live_`; expired key -> 401 `expired_key`; existing rows backfill `mode=live`; audit events for mint/revoke/expire; scope set extended (`sessions:read`, `sessions:write`, `query:run`, `usage:read`; `hybrid:read` kept as alias); service-account part of 2907 stays as written | 3 | 3901 | S2 |
| **DOCBOT-2597** | Sandbox sessions for test keys | `POST /v1/sandbox/sessions` wraps `init_demo_session` with owner = key user + org and `sessions.is_sandbox=true`; test keys get 404 on any non-sandbox session/connection; per-owner sandbox window (5/hour); daily test credit allowance (default 300) charged via `cost_to_credits`, never debits org balance, 429 `test_allowance_exhausted`; sandbox sessions purged after 24h | 5 | 2907, 3901 | S2 |
| **DOCBOT-2598** | Self-serve personal test key | `POST /api/developer/test-key` and `DELETE /api/developer/test-key/{id}` in `api/routes/developer_routes.py`, guarded by signed-in + email-verified user (viewer+); returns raw key once; max 2 active per user; preset scopes; 30-day expiry; models in `api/developer_models.py` + `src/lib/schemas/developer.ts` | 2 | 2907 | S2 |
| **DOCBOT-3905** | Developer portal `/apis` with Scalar (existing, expanded) | `src/app/apis/page.tsx` renders Scalar from `/v1/openapi.json`; signed-out visitors see read-only docs + sign-in CTA; signed-in users click "Get test key", key held in memory only (no URL, no persisted storage), injected as bearer auth; "Try it" works same-origin through the Vercel `/v1` rewrite; quickstart (sandbox session -> hybrid question -> lineage) gives a first answer in under 10 minutes; works with nonce CSP from `src/proxy.ts`; page passes WCAG 2.1 AA basics; light/dark | 5 | 3901, DOCBOT-2597, DOCBOT-2598 | S2 |
| **DOCBOT-2599** | Session read/delete + catalog endpoints | `GET /v1/sessions`, `GET/DELETE /v1/sessions/{id}`, `GET /v1/db/connections`, `GET /v1/db/connections/{id}/schema`, `GET /v1/personas`; cursor pagination; all scoped by key org (and sandbox for test keys); cross-org and cross-mode tests | 3 | 3901, 2907 | S3 |
| **DOCBOT-2611** | `/v1/chat` and `/v1/db/query` | Buffered wrappers over `/api/chat` and `/api/db/chat` pipelines with credit reservation (`chat` 20, `db_chat` 30), audit with `api_key_id`, PII masking contract test (seeded email/SSN comes back masked) | 3 | 3901, DOCBOT-2599 | S3 |
| **DOCBOT-2612** | `/v1/documents` upload (live keys) | Multipart PDF upload reusing `/api/upload` validation and `upload_limits`; test keys 403 `test_key_not_allowed`; 413 on size; new session stamped with org and `created_by` so `session_in_org` accepts it | 3 | DOCBOT-2599 | S3 |
| **DOCBOT-2613** | Per-key usage metering | `api_key_id` recorded on usage events / ledger metadata for every `/v1` call; `GET /v1/usage` (by route + paged events); `X-DocBot-Credits-Charged` header; per-key spend visible in `/apis` and admin key list | 3 | 3901 | S3 |
| **DOCBOT-2614** | Idempotency keys | `Idempotency-Key` on cost-bearing POSTs; `(api_key_id, key)` store with request hash, 24h TTL; replay returns stored body + `Idempotent-Replayed: true` and no charge; 400 on body mismatch; 409 while in flight | 2 | 3901 | S3 |
| **DOCBOT-2615** | Opt-in SSE streaming on `/v1` | `Accept: text/event-stream` on chat, db query, hybrid, autopilot; named events `metadata`/`token`/`replace`/`step`/`done`/`error`; exactly one terminal event; 15s pings; credits settle on client disconnect; concurrency limit enforced | 3 | 3901, DOCBOT-2592 | S3 |
| **DOCBOT-2616** | `/v1` tie-out workpaper download | `GET /v1/lineage/{run_id}/tieout.{xlsx,pdf}` reusing `tieout_routes` export with the key-org `run_row_accessible` check | 1 | 3901 | S3 |
| **DOCBOT-3902** | Autopilot / tie-out as async job (existing) | As written in backlog: job result equals UI result, scoped to key; `POST /v1/jobs` + `GET /v1/jobs/{id}` | 5 | 3901, 3305 | later |
| **DOCBOT-3903** | Signed webhooks (existing) | As written: HMAC signed, retries with backoff | 3 | 3902 | later |
| **DOCBOT-3904** | MCP server (existing) | As written; reuse `/v1` scopes and test-key sandbox for MCP onboarding | 8 | 3901, 3902 | later |
| **DOCBOT-3906** | Python SDK (existing) | Generated from `/v1/openapi.json`; examples for sandbox quickstart | 3 | 3901 | later |
| **DOCBOT-3907** | TypeScript SDK (existing) | Generated from spec; Zod types from `src/lib/schemas/v1.ts` | 3 | 3901 | later |
| **DOCBOT-2617** | Re-expose connectors in v1 | After DOCBOT-2591: connectors back in the spec with `/api/connectors*` parity (types, orders, financials, delete) | 3 | DOCBOT-2591, 3901 | later |
| **DOCBOT-2618** | `/v1/audit-log` export | `audit:read` scope, live admin-minted keys only, cursor pagination, CSV option | 2 | 3901, 2907 | later |

## Totals

| Sprint | Stories | Points |
|---|---|---|
| S1 (fix and contract start) | DOCBOT-2591, DOCBOT-2596, DOCBOT-2592, DOCBOT-3901 (start) | 10 |
| S2 (play: test keys + portal) | DOCBOT-2907, DOCBOT-2597, DOCBOT-2598, DOCBOT-3905 (+ 3901 finish) | 15 |
| S3 (build for real) | DOCBOT-2599, DOCBOT-2611, DOCBOT-2612, DOCBOT-2613, DOCBOT-2614, DOCBOT-2615, DOCBOT-2616 | 18 |
| Later (EPIC-39 remainder + extras) | 3902, 3903, 3904, 3906, 3907, DOCBOT-2617, DOCBOT-2618 | 27 |
| **Total** | 22 stories | **70** |

Minimum to answer the founder's ask ("docbot.space/apis, token-gated, play with the API"): S1 + S2 = 25 points (DOCBOT-2592 is filed at 2 points, the plan first sized it at 1). S1 items DOCBOT-2591 and DOCBOT-2592 should ship regardless, because they fix live defects in the current `/v1` surface.

Board sprints (master backlog, 2026-11-23): plan S1 + S2 (the 25-point "play with the API" slice) are in board sprint **S1**, except DOCBOT-2591 and DOCBOT-2592, which are Phase 0 bugs fixed now. Plan S3 is in board sprint **S2**. "Later" is not scheduled: 3902-3904, 3906, 3907 stay parked in section 7.13; DOCBOT-2617 and DOCBOT-2618 are filed with no sprint.

## Dependency graph

```mermaid
flowchart LR
  N01[DOCBOT-2591 connectors org scope] --> D3901[DOCBOT-3901 v1 contract]
  N02[DOCBOT-2596 close /docs] --> D3901
  D3901 --> D2907[DOCBOT-2907 key modes + expiry]
  D2907 --> N03[DOCBOT-2597 sandbox sessions]
  D2907 --> N04[DOCBOT-2598 self-serve test key]
  N03 --> D3905[DOCBOT-3905 /apis portal]
  N04 --> D3905
  D3901 --> N05[DOCBOT-2599 sessions + catalog]
  N05 --> N06[DOCBOT-2611 chat + db query]
  N05 --> N07[DOCBOT-2612 documents]
  D3901 --> N08[DOCBOT-2613 usage]
  D3901 --> N09[DOCBOT-2614 idempotency]
  N12[DOCBOT-2592 hybrid buffer fix] --> N10[DOCBOT-2615 streaming]
  D3901 --> N10
  D3901 --> N11[DOCBOT-2616 tie-out download]
  D3901 --> D3902[DOCBOT-3902 async jobs]
  D3902 --> D3903[DOCBOT-3903 webhooks]
  D3902 --> D3904[DOCBOT-3904 MCP]
  D3901 --> D3906[DOCBOT-3906 Python SDK]
  D3901 --> D3907[DOCBOT-3907 TS SDK]
  N01 --> N13[DOCBOT-2617 connectors in v1]
```

## Test plan per story (minimum)

| Story | Tests (all `tests/unit/` unless noted) |
|---|---|
| DOCBOT-2591 | `test_v1_connectors_org_scope.py`: key of org A cannot list, sync, or see org B's connector |
| DOCBOT-2596 | `test_docs_exposure.py`: `/docs`, `/redoc`, `/openapi.json` 404 when `ENV!=dev`, 200 in dev |
| DOCBOT-2592 | `test_v1_hybrid_buffer.py`: stream with `replace` -> corrected answer; two `metadata` events -> intent preserved |
| DOCBOT-3901 | `test_v1_error_envelope.py`, `test_v1_rate_limit.py` (429 + headers, shared store), `test_v1_openapi_snapshot.py`, `test_route_guards.py` stays green |
| DOCBOT-2907 | extend `test_api_key_service.py`: test/live prefix, expiry 401, backfill, alias scope |
| DOCBOT-2597 | `test_v1_sandbox.py`: test key only sees its sandbox, allowance exhaustion 429, live org balance untouched |
| DOCBOT-2598 | `test_developer_routes.py`: unverified user 403, max 2 keys, raw key returned once |
| DOCBOT-3905 | Playwright smoke (manual regression list until e2e exists): sign in, get key, run quickstart |
| DOCBOT-2611 | `test_v1_pii_masking.py`: seeded PII masked in `/v1/chat`, `/v1/db/query`, `/v1/hybrid/chat` |
| DOCBOT-2614 | `test_v1_idempotency.py`: replay, mismatch 400, in-flight 409, no double charge |
