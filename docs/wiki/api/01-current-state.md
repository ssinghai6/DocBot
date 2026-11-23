# Developer API: Current State

Sources: `api/api_key_service.py`, `api/index.py`, `api/resource_scope_service.py`, `api/credit_service.py`, `api/billing_models.py`, `api/utils/request_guard.py`, `api/utils/security_headers.py`, `api/connector_store.py`, `api/routes/connector_routes.py`, `api/rbac_service.py`, `api/utils/caller_scope.py`, `next.config.ts`, `src/components/AdminPanel.tsx`, `src/hooks/useChatHandlers.ts`, `src/app/**`, `tests/unit/test_api_key_service.py`, `tests/unit/test_v1_billing.py`, `tests/unit/test_route_guards.py`, `project-tasks/docbot-master-backlog.md`

Last verified against code: 2026-11-23 (origin/main at `8a07189`)

## Short answer

**Partly.** DocBot already has a developer API: org-scoped API keys (DOCBOT-1530) and 6 routes under `/v1/*`, reachable through the production domain because Next.js proxies `/v1/*` to the backend (`next.config.ts`, rewrites, DOCBOT-2339). What is **missing** is everything a developer needs to "test and play":

- There is no `docbot.space/apis` page, no developer docs, no playground. `src/app/` contains only `chat`, `onboarding`, `privacy`, `signup`, `terms`, `verify` and the landing page.
- There is no test key or sandbox mode. Every key is a `dk_live_` key that touches real org data and real credits.
- A key cannot create anything to query. No `/v1` route creates a session, so a developer must first sign in, upload in the `/chat` UI, and copy a `session_id` (`api/resource_scope_service.py:25-31`).
- Only org admins can mint keys (`api/index.py:3335-3348`, `require_admin()`).

## 1. API key model

| Property | Current behavior | Evidence |
|---|---|---|
| Format | `dk_live_` + `secrets.token_urlsafe(32)` (43 url-safe chars, about 256 bits) | `api/api_key_service.py:237`, `:92` |
| Test/live split | None. Only `dk_live_` is ever generated | `api/api_key_service.py:237` |
| Storage | SHA-256 hex of the raw key only (`key_hash`, unique, indexed). Raw key returned once at creation | `api/api_key_service.py:183-184`, `:111`, `:238` |
| Display prefix | First 16 chars stored in clear (`key_prefix`, e.g. `dk_live_a1b2c3d4`) for the admin list | `api/api_key_service.py:91`, `:239` |
| Table | `api_keys(id, org_id, key_hash, key_prefix, name, scopes JSON, created_at, last_used_at, revoked_at)` | `api/api_key_service.py:106-118` |
| Org binding | Mandatory `org_id`, set from the admin's org at creation | `api/api_key_service.py:110`, `api/index.py:3343` |
| User binding | None. No `created_by` column; the key acts as the org, not as a person | `api/api_key_service.py:106-118` |
| Scopes | Fixed set: `hybrid:read`, `lineage:read`, `autopilot:run`, `connectors:manage`. Exact set membership, no hierarchy | `api/api_key_service.py:84-89`, `:394-398` |
| Expiry | **None.** No `expires_at` column. Keys live until revoked (planned in DOCBOT-2907) | `api/api_key_service.py:106-118`; backlog line 218 |
| Revocation | Soft revoke via `revoked_at`, idempotent, org-scoped | `api/api_key_service.py:274-304` |
| Validation | Hash lookup, reject if revoked, update `last_used_at` on every call | `api/api_key_service.py:307-334` |
| Logging | Only key id and prefix are logged, never the raw key | `api/api_key_service.py:259`, `:303` |

## 2. How a key is created today

| Route | Auth | Body | Returns |
|---|---|---|---|
| `GET /admin/api-keys` | Logged-in **admin** session (`require_admin()`; never honors open mode or the owner key) | none | `{org_id, keys: [ {id, org_id, key_prefix, name, scopes, created_at, last_used_at, revoked_at} ]}` |
| `POST /admin/api-keys` | Admin session | `{name: str, scopes: [str]}` | `{key: {...}, raw_key: "dk_live_..."}`, 400 on unknown/empty scopes |
| `DELETE /admin/api-keys/{key_id}` | Admin session | none | `{status: "revoked", key_id}`, 404 if not in caller's org |

Evidence: `api/index.py:3318-3359`, `api/rbac_service.py:129-141`.

UI: the Admin panel in `/chat` has an "API Keys" section with a scope picker for the four scopes, one-time raw key display and copy, list, and revoke (`src/components/AdminPanel.tsx:98`, `:652-666`, `:778-786`; fetches in `src/hooks/useChatHandlers.ts:540-581`). Viewers and analysts cannot create keys.

## 3. Which routes accept a key

A key is accepted **only** on the `/v1/*` routes. Every `/api/*` route uses the session cookie (`docbot_session`) or the anonymous cookie (`docbot_anon`) (`api/utils/caller_scope.py:70-71`). The two auth paths are parallel; neither falls back to the other.

| Method | Path | Scope | Wraps | Notes | Evidence |
|---|---|---|---|---|---|
| POST | `/v1/hybrid/chat` | `hybrid:read` | `hybrid_service.hybrid_chat` (same as `/api/hybrid/chat`) | SSE buffered into one JSON object | `api/index.py:3659-3753` |
| POST | `/v1/autopilot/run` | `autopilot:run` | `autopilot_service.run_autopilot` | Buffered; plan, steps, answer, citations, warnings | `api/index.py:3768-3879` |
| GET | `/v1/lineage/{run_id}` | `lineage:read` | `lineage_service.get_lineage` | 404 unless run belongs to a user-owned session or connection in the key's org | `api/index.py:3882-3897` |
| GET | `/v1/connectors` | `connectors:manage` | `connector_store.list_active_connectors()` | **Bug: called without `org_id`, so it lists every org's connectors** | `api/index.py:3905-3914`, `api/connector_store.py:308-313` |
| POST | `/v1/connectors` | `connectors:manage` | connector registry + `save_connector` | **Bug: `save_connector` called without `org_id`**, so the row is not owned by the key's org | `api/index.py:3917-3960`, `api/connector_store.py:140-145` |
| POST | `/v1/connectors/{id}/sync` | `connectors:manage` | `commerce_service.sync_connector_data` | **Bug: `_get_connector` has no org check**, any key with this scope can sync any org's connector by id | `api/index.py:3445-3451`, `:3963-3981` |

The `/api/connectors*` equivalents were org-scoped in DOCBOT-1602 (`api/routes/connector_routes.py:47-86`, `:191`, `:205`); the `/v1` copies were not. This is a cross-tenant gap and must be fixed before any public launch (see backlog, `NEW-01`).

### Resource scoping for `/v1/hybrid/chat` and `/v1/autopilot/run`

`_require_v1_resources` (`api/index.py:3610-3622`) calls `v1_resources_in_org` (`api/resource_scope_service.py:45-67`): the `session_id` must belong to the key's org **and** have a non-null `user_id`; the optional `connection_id` must belong to the org. Anonymous and demo sessions are deliberately unreachable (`api/resource_scope_service.py:25-31`, DOCBOT-1635). Failure is a 404 `"Session or connection not found."`.

### Correctness bugs in the buffered `/v1/hybrid/chat` response

The `/v1` handler collapses the internal SSE stream by event type (`api/index.py:3728-3740`) but the stream has grown since DOCBOT-1530:

1. **`replace` events are dropped.** When the discrepancy check rewrites the answer, `hybrid_service` emits `{"type": "replace", "content": <full corrected answer>}` (`api/hybrid_service.py:946-954`, DOCBOT-2004). The `/v1` loop has no `replace` branch, so API callers get the **original, uncorrected** answer while the UI and lineage get the corrected one.
2. **`intent` / `has_sql` / `has_docs` can come back `null`.** A second `metadata` event carrying SQL details (`api/hybrid_service.py:714-723`, no `intent` key) overwrites the first (`api/index.py:3728-3729`), so whenever SQL ran these fields are `null` in the JSON response.

Both are fixed by NEW-12 (merge metadata events, honor `replace`, add a regression test).

## 4. How the key is sent

```
Authorization: Bearer dk_live_<rest-of-key>
```

Parsed in `require_api_key` (`api/api_key_service.py:377-392`). Missing or non-Bearer header gives 401 `"Missing or malformed Authorization header. Expected 'Bearer <api_key>'."`; unknown or revoked key gives 401 `"Invalid or revoked API key."`; missing scope gives 403 `"API key missing required scope: <scope>."`. There is no query-string key parameter (good), but nothing explicitly rejects one either.

CORS `allow_headers=["Content-Type"]` (`api/index.py:57-63`), so a third-party browser page cannot send `Authorization` cross-origin. Calls from `docbot.space` itself are same-origin through the Vercel rewrite and are not affected. Server-to-server callers (curl, Python) are not subject to CORS.

## 5. Metering and credits per key

| Aspect | Current behavior | Evidence |
|---|---|---|
| Who pays | The key's org. `_begin_v1_billing` builds `CallerScope(user_id=None, org_id=key.org_id)` | `api/index.py:3625-3636` |
| Reservation | Up-front hold per route: `chat` 20, `db_chat` 30, `hybrid` 40, `sandbox` 50, `autopilot` 150 credits (1 credit = $0.001). Override with `BILLING_EST_CREDITS_<ROUTE>` | `api/credit_service.py:124-132`, `:185-187` |
| Actual debit | Metered provider cost (micro-USD) times `BILLING_MARKUP`, rounded up | `api/credit_service.py:190-194` |
| Enforcement | Shadow mode unless `BILLING_ENFORCE=true`; then 402 `insufficient_credits` / `account_frozen` | `api/credit_service.py:149-151`, `api/billing_models.py:115-121` |
| Per-key attribution | **None in the ledger.** `api_key_id` is only written into audit-log metadata | `api/index.py:3688`, `:3797` |
| Other caps | Platform spend cap (`_spend_cap`) and per-session budget (`charge_budget`) also apply | `api/index.py:463`, `:481`, `:3676`, `:3785` |
| Balance endpoint | `CreditBalance` model exists for `GET /api/billing/balance`, but **no such route is registered** | `api/billing_models.py:133-150`; no route in `api/routes/billing_routes.py` |

## 6. Rate limits

| Limiter | Scope | Behavior | Evidence |
|---|---|---|---|
| Per-key token bucket | 5 calls/s per key id | **Sleeps rather than rejects** (no 429), lives in process memory (per worker, lost on restart), and the dict grows with every key used | `api/api_key_service.py:129-132`, `:355-360`, `:400-401`; `api/connectors/rate_limiter.py:26-31` |
| Per-IP limiter | `RATE_LIMIT_PER_MIN` (default 60) for POST/DELETE on `/api/*` and `/v1/*` | 429 with `Retry-After`; uses the shared counter store | `api/index.py:101-119`, `api/utils/request_guard.py:105`, `:385-395` |
| Write freeze | 503 on writes to `/api/*` and `/v1/*` during cutover | | `api/utils/request_guard.py:366-368` |

No `X-RateLimit-*` headers are emitted. GET `/v1/*` calls are not IP-limited (only POST/DELETE are).

## 7. FastAPI interactive docs in production

`app = FastAPI(title="DocBot API", version="2.0.0")` (`api/index.py:50`) does not override `docs_url`, `redoc_url` or `openapi_url`, so FastAPI's defaults apply: `/docs`, `/redoc`, `/openapi.json` are served by the backend **with no auth**. The CSP middleware exempts `/docs` and `/redoc` so they render (`api/utils/security_headers.py:38-39`). The route-guard test skips them as built-ins (`tests/unit/test_route_guards.py:131`).

Who can reach them:

- Through `docbot.space`: **no**. Vercel only rewrites `/api/*` and `/v1/*` (`next.config.ts`, `rewrites()`), so `docbot.space/docs` hits Next.js and 404s (inferred from the absence of a `src/app/docs` page).
- Directly on the backend host (Railway URL today): **yes, anyone who knows the host** (inferred; no network restriction is configured in code). `/openapi.json` lists every route, including `/admin/*`, with request schemas. That is an information leak, not an auth bypass, because every route is guarded, but it should be closed.

## 8. Frontend playground

None. No `/apis`, `/docs` or `/developers` route under `src/app/`. No OpenAPI renderer (Swagger UI, Redoc, Scalar) is in `package.json`. No Zod schema exists for `/v1` payloads in `src/lib/schemas/` (files present: `autopilotStep.ts`, `billing.ts`, `emailVerify.ts`, `ownerUnlock.ts`, `signup.ts`, `tieout.ts`, `waitlist.ts`).

## 9. Tests that cover keys today

- `tests/unit/test_api_key_service.py`: 21 tests (create, hash-only storage, revoke, validate, scope checks).
- `tests/unit/test_v1_billing.py`: 6 tests (credit reservation on `/v1` routes, DOCBOT-2330).
- `tests/unit/test_route_guards.py`: recognises `require_api_key` as a guard (`:77`) and samples `POST /v1/hybrid/chat` (`:280`).

No test asserts org scoping on `/v1/connectors*` (which is why the gap above survived).

## 10. Curl examples that work today

Prerequisites: an org admin creates a key in `/chat` > Admin > API Keys, and a signed-in user of that org uploads a document in `/chat` so a user-owned `session_id` exists. Use `https://<api-host>` = `https://docbot.space` (proxied) or the backend host directly.

```bash
export DOCBOT_API_KEY="dk_live_..."      # never commit, never put in a URL
export HOST="https://<api-host>"
```

Hybrid question over the uploaded docs (and optional DB connection):

```bash
curl -sS -X POST "$HOST/v1/hybrid/chat" \
  -H "Authorization: Bearer $DOCBOT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "question": "What was Q4 net income, and does the database agree?",
        "session_id": "<session-id-from-the-ui>",
        "connection_id": "<optional-connection-id>",
        "persona": "Finance Expert"
      }'
```

Response shape (`api/index.py:3745-3753`):

```json
{
  "answer": "Q4 net income was $325M per the filing; the ledger shows $330M ...",
  "citations": [{"source": "10-K.pdf", "page": 42, "text": "..."}],
  "intent": "hybrid",
  "has_sql": true,
  "has_docs": true,
  "run_id": "<run-uuid>",
  "trace_id": "<trace-uuid>"
}
```

Autopilot run:

```bash
curl -sS -X POST "$HOST/v1/autopilot/run" \
  -H "Authorization: Bearer $DOCBOT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<session-id>", "question": "Analyze revenue trend and flag anomalies"}'
```

Returns `{plan, steps, answer, citations, warnings}` (`api/index.py:3873-3879`).

Lineage for an answer:

```bash
curl -sS "$HOST/v1/lineage/<run-id>" -H "Authorization: Bearer $DOCBOT_API_KEY"
```

Error you will see with a bad key:

```json
{"detail": "Invalid or revoked API key."}
```

Do not use the `/v1/connectors*` routes until `NEW-01` ships (see section 3). Until `NEW-12` ships, `intent`/`has_sql`/`has_docs` may be `null` and `answer` may be the pre-correction text (section 3).

## Gap table: what "docbot.space/apis" needs vs what exists

| Need | Present? | Gap | Backlog |
|---|---|---|---|
| Key-gated programmatic API | Yes, 7 `/v1` routes | Connectors routes leak across orgs | NEW-01 |
| Correct hybrid answers over `/v1` | Partly | `replace` dropped, metadata overwritten | NEW-12 |
| Versioned namespace | Yes, `/v1` | Routes live in `index.py`, inline models, no Zod mirror (violates F-4) | DOCBOT-3901 |
| Key types test vs live | No | Only `dk_live_` | DOCBOT-2907 + NEW-03 |
| Key expiry | No | No `expires_at` | DOCBOT-2907 |
| Self-serve key for non-admins | No | Admin-only mint | NEW-04 |
| Safe sandbox data to play with | No | `/v1` cannot reach demo sessions by design | NEW-03 |
| Create sessions/upload via API | No | No `/v1` session or upload route | NEW-05, NEW-07 |
| Plain doc chat and DB query via API | No | Only hybrid and autopilot | NEW-06 |
| Per-key rate limit with 429 + headers | No | Sleeping in-memory bucket | DOCBOT-3901 |
| Per-key usage and credit view | No | `api_key_id` only in audit metadata; no balance route | NEW-08 |
| Consistent error envelope | No | Plain `{"detail": ...}`; 402 uses `QuotaErrorPayload` | DOCBOT-3901 |
| Idempotency for POST | No | Retries double-charge | NEW-09 |
| Optional streaming | No | `/v1` is buffered only | NEW-10 |
| Public OpenAPI for `/v1` only | No | Default `/openapi.json` exposes all routes | NEW-02, DOCBOT-3901 |
| Docs + playground page | No | No `/apis` page, no renderer | DOCBOT-3905 |
| Audit of key usage | Yes | `api_key_use` event per call | none |
| SSRF on DB connect via API | N/A | No `/v1` DB connect; keep it that way in v1 | none |
| SDKs, MCP, webhooks | No | Parked in EPIC-39 | DOCBOT-3903/3904/3906/3907 |
