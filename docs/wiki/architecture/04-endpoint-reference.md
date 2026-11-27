# Endpoint Reference

Sources: `api/index.py`, `api/routes/billing_routes.py`, `api/routes/connector_routes.py`, `api/routes/email_verify_routes.py`, `api/routes/onboarding_routes.py`, `api/routes/tieout_routes.py`, `api/routes/waitlist_routes.py`, `api/utils/request_guard.py`, `api/utils/spend_guard.py`, `api/utils/caller_scope.py`, `api/rbac_service.py`, `api/api_key_service.py`, `tests/unit/test_route_guards.py`, `src/app/`, `src/proxy.ts`, `next.config.ts`

Last verified against code: 2026-11-23

## Inventory method and totals

**Total: 82 HTTP routes** (63 in `api/index.py`, 19 in 6 `APIRouter` modules under `api/routes/`; AST count of route decorators). No WebSocket routes. FastAPI's built-in `/docs`, `/redoc` and `/openapi.json` are not counted; `FastAPI(title="DocBot API", version="2.0.0")` at `api/index.py:50` does not disable them, so they are served (inferred from defaults).

How the inventory was produced (two independent methods, same count):

1. **Runtime import.** `api.index` was imported with throwaway `DATABASE_URL` and `DB_ENCRYPTION_KEY` env values (the same trick `tests/unit/test_route_guards.py:39-43` uses; the lifespan never runs, so nothing connects). Routes were flattened with that test's `_api_routes()` (it handles FastAPI >= 0.162 `_IncludedRouter` wrappers). The dependency tree of every route was walked with `_walk()`, and guard factories (`require_role`, `require_api_key`, `edgar_guard`) were inspected through their closure cells to recover `min_role`, `enforce_in_open_mode` and `min_scope`. Result: 82 routes, classified as 53 guarded, 21 `PUBLIC_ROUTES`, 2 handler-guarded, 3 `DEV_ONLY`, 3 `KNOWN_UNGUARDED`, 0 unclassified.
2. **AST parse.** Every `@app.<verb>` / `@router.<verb>` decorator in `api/index.py` and `api/routes/*_routes.py` was parsed (router `prefix=` applied). Also 82.

The "Service call(s)" column came from an AST walk of each handler body for calls into `api.*` modules, then hand-checked. The `/api/billing/webhook` path comes from the constant `billing_service.WEBHOOK_PATH` (`api/routes/billing_routes.py:60`).

## Cross-cutting request pipeline (applies to every route)

Order, outermost first (`api/index.py:50-136`):

1. `SecurityHeadersMiddleware` (`api/utils/security_headers.py`, added at `api/index.py:136`) sets CSP/HSTS/frame/nosniff/referrer headers on every response, including guard short-circuits.
2. `CORSMiddleware` (`api/index.py:57-63`): origins from `ALLOWED_ORIGINS` via `cors_config.parse_allowed_origins`; credentials on; methods `GET, POST, DELETE`; header `Content-Type` only. `PATCH /admin/users/{user_id}/role` therefore only works same-origin (through the Next.js rewrite), not cross-origin.
3. `abuse_guard` HTTP middleware (`api/index.py:72-128`):
   - `WRITE_FREEZE` -> 503 on write methods for `/api` and `/v1` (owner included) (`request_guard.is_write_frozen`).
   - Owner bypass: a valid `docbot_owner` HMAC cookie skips every guard below (`request_guard.is_owner`, `request_guard.py:149-152`).
   - `PUBLIC_DEMO_MODE` -> 403 for POST prefixes `/api/upload`, `/api/db/connect`, `/api/db/upload`, `/api/connectors/register`, `/api/edgar/ingest` (`request_guard.py:156-162`). Called "demo-blocked" below.
   - **Global per-IP limiter** on every `POST`/`DELETE` under `/api/` and `/v1/`, except `/api/billing/webhook`: `RATE_LIMIT_PER_MIN` (default 60) per 60 s window, counters in Postgres `rate_limit_counters` (`request_guard.py:104-105, 382`). GET and PATCH are not IP-limited.
   - Sets the `docbot_anon` owner cookie for new anonymous browsers on `/api/*` (`caller_scope.prime_anon_cookie`).
4. Route dependencies (table columns below), then the handler.

Shared dependency vocabulary (`api/index.py:419-489`):

| Name | Callable | Effect |
|---|---|---|
| principal | `caller_scope.require_principal` | Resolves `CallerScope(user_id, org_id, anon_token)`. Open mode: anonymous allowed, scoped to the `docbot_anon` cookie hash. `AUTH_REQUIRED=true`: 401 unless logged in or the resource is a server-created demo session/connection (DOCBOT-1711). |
| `require_role(r)` | `rbac_service.require_role` | `docbot_session` cookie -> user row -> role >= r. With `enforce_in_open_mode=False` it only enforces when `AUTH_REQUIRED` is on; with `True` it always enforces. |
| `require_admin()` / `require_platform_admin()` | `rbac_service` | Admin in every mode; platform admin also needs the user id in `PLATFORM_ADMIN_USER_IDS`. |
| `require_api_key(scope)` | `api_key_service.require_api_key` | `Authorization: Bearer` API key with the scope; principal is the key's org. Documented in detail in `docs/wiki/api/`. |
| `_spend_cap` | `spend_guard.check_platform_capacity` | 503 for free-tier callers once the platform daily USD cap is spent (`spend_guard.py:136-147`). |
| `demo_spend_guard` | `request_guard.demo_spend_guard` | Anonymous demo chats hit `DEMO_DAILY_USD_CAP` (default 10) (`request_guard.py:448, 464-475`). |
| `gate_uploads` | `api/index.py:444` | Not an auth guard: only enforces sign-in when `GATE_UPLOADS=true`. |
| `charge_budget` | `api/index.py:463` | In-handler: per-user daily budget (`USER_DAILY_BUDGET`, default 500) or per-session 7-day budget (`SESSION_BUDGET`, default 100); weights chat 1, hybrid 2, sandbox 2, autopilot 5 (`request_guard.py:106-107, 140-142, 168-173`). |
| `credit_service.begin_billing` | `api/credit_service.py:1016` | Reserves credits before streaming; 402 only when `BILLING_ENFORCE` is on (see [Request Flows](architecture-02-request-flows)). |
| `_require_session` | `api/index.py:489` | 404 unless the caller owns the `session_id` (foreign id looks missing). |

Guard class column values come from `tests/unit/test_route_guards.py`: `PUBLIC_ROUTES` (`:145`), `HANDLER_GUARDED` (`:181`), `DEV_ONLY` (`:188`), `KNOWN_UNGUARDED` (`:196`). That test fails CI if a new route has no guard and is not listed.

## Backend routes by area

### Health and catalogue

| Method | Path | Guard class | Auth / role | Request | Response | Service call(s) | Abuse guards (beyond global IP limiter) | Source |
|---|---|---|---|---|---|---|---|---|
| GET | `/api/health` | PUBLIC_ROUTES | none | - | JSON (untyped dict) | inline liveness | - | `api/index.py:889` |
| GET | `/api/personas` | PUBLIC_ROUTES | none | - | JSON (untyped dict) | `personas.EXPERT_PERSONAS` (api/personas.py) | - | `api/index.py:907` |
| GET | `/api/tools` | PUBLIC_ROUTES | none | - | `ToolsResponse` | `tools.registry.list_tools` | - | `api/index.py:943` |

### Demo

| Method | Path | Guard class | Auth / role | Request | Response | Service call(s) | Abuse guards (beyond global IP limiter) | Source |
|---|---|---|---|---|---|---|---|---|
| POST | `/api/demo/init` | PUBLIC_ROUTES | none | - | JSON (untyped dict) | `demo_service.init_demo_session` | `check_demo_init_limit` (per-IP/user/global hourly), `_spend_cap` | `api/index.py:969` |

### Documents, chat and sessions

| Method | Path | Guard class | Auth / role | Request | Response | Service call(s) | Abuse guards (beyond global IP limiter) | Source |
|---|---|---|---|---|---|---|---|---|
| POST | `/api/upload` | Guarded | principal (`require_principal`: user, or anon on demo resources; 401 under AUTH_REQUIRED) | multipart/form-data | JSON (untyped dict) | `upload_limits.spool_upload` -> `pdf_io.extract_pdf_pages` -> `chunker.chunk_document` -> `vector_store.create_store`; `document_extractor` (LangExtract) when financial; `usage_service.meter_upload` | `_spend_cap`, `gate_uploads`, demo-blocked, upload caps | `api/index.py:992` |
| POST | `/api/chat` | Guarded | principal (`require_principal`: user, or anon on demo resources; 401 under AUTH_REQUIRED) | `ChatRequest` | SSE `text/event-stream` | `persona_router.route_persona`, `query_expansion.expand_query`, Chroma retrieval (no rerank on this path), `prompt_guard.scan_chunks`, `llm_provider.achat_completion_stream`; `credit_service.begin_billing` | `_spend_cap`, `demo_spend_guard`, `charge_budget` (session/user budget, weight 1) | `api/index.py:1202` |
| GET | `/api/sessions` | Guarded | principal (`require_principal`: user, or anon on demo resources; 401 under AUTH_REQUIRED) | - | `SessionListResponse` | `session_service.list_sessions` | - | `api/index.py:1585` |
| GET | `/api/session/{session_id}` | Guarded | principal (`require_principal`: user, or anon on demo resources; 401 under AUTH_REQUIRED) | - | `SessionDetailResponse` | `session_service.get_session_detail` | - | `api/index.py:1592` |
| DELETE | `/api/session/{session_id}` | Guarded | principal (`require_principal`: user, or anon on demo resources; 401 under AUTH_REQUIRED) | - | `SessionDeleteResponse` | `session_service.delete_owned_session` | - | `api/index.py:1603` |
| GET | `/api/export/{session_id}` | Guarded | principal (`require_principal`: user, or anon on demo resources; 401 under AUTH_REQUIRED) | - | file download (txt / markdown, `session_service.render_export`) | `session_service.render_export` | - | `api/index.py:1616` |
| POST | `/api/session/{session_id}/update-persona` | Guarded | principal (`require_principal`: user, or anon on demo resources; 401 under AUTH_REQUIRED) | - | JSON (untyped dict) | inline UPDATE sessions.persona | - | `api/index.py:1636` |
| POST | `/api/test-persona-switch` | DEV_ONLY (404 unless ENV=dev) | none | `PersonaTestRequest` | JSON (untyped dict) | inline persona debug | - | `api/index.py:1671` |
| GET | `/api/test-personas` | DEV_ONLY (404 unless ENV=dev) | none | - | JSON (untyped dict) | inline persona debug | - | `api/index.py:1729` |

### Live DB, SQLite and CSV

| Method | Path | Guard class | Auth / role | Request | Response | Service call(s) | Abuse guards (beyond global IP limiter) | Source |
|---|---|---|---|---|---|---|---|---|
| POST | `/api/db/connect` | Guarded | analyst (AUTH_REQUIRED only) + principal | `DBConnectionRequest` | JSON (untyped dict) | `db_service.connect_database` (SSRF + encrypt) | `gate_uploads`, demo-blocked, SSRF validator | `api/index.py:1796` |
| DELETE | `/api/db/disconnect/{connection_id}` | Guarded | analyst (AUTH_REQUIRED only) + principal | `DisconnectRequest` | JSON (untyped dict) | `db_service.disconnect_database` | - | `api/index.py:1855` |
| GET | `/api/db/schema/{connection_id}` | Guarded | principal (`require_principal`: user, or anon on demo resources; 401 under AUTH_REQUIRED) | - | JSON (untyped dict) | `db_service.get_schema` | - | `api/index.py:1892` |
| POST | `/api/db/refresh-schema/{connection_id}` | Guarded | principal (`require_principal`: user, or anon on demo resources; 401 under AUTH_REQUIRED) | - | JSON (untyped dict) | `db_service.refresh_schema`, `ontology_service.build_ontology` | - | `api/index.py:1927` |
| POST | `/api/db/chat` | Guarded | principal (`require_principal`: user, or anon on demo resources; 401 under AUTH_REQUIRED) | `DBChatRequest` | SSE | `db_service.run_sql_pipeline` (CSV dialect -> `sandbox_service`); `credit_service.begin_billing` | `_spend_cap`, `demo_spend_guard` | `api/index.py:1964` |
| GET | `/api/db/sessions` | Guarded | principal (`require_principal`: user, or anon on demo resources; 401 under AUTH_REQUIRED) | - | JSON (untyped dict) | `db_service.list_connections_for_scope` | - | `api/index.py:2042` |
| GET | `/api/db/history/{connection_id}` | Guarded | principal (`require_principal`: user, or anon on demo resources; 401 under AUTH_REQUIRED) | - | JSON (untyped dict) | `db_service.get_query_history_for_scope` | - | `api/index.py:2061` |
| POST | `/api/db/upload` | Guarded | analyst (AUTH_REQUIRED only) + principal | multipart/form-data | JSON (untyped dict) | `file_upload_service.upload_sqlite` | `_spend_cap`, `gate_uploads`, demo-blocked, size cap | `api/index.py:2098` |
| POST | `/api/db/upload/csv` | Guarded | analyst (AUTH_REQUIRED only) + principal | multipart/form-data | JSON (untyped dict) | `file_upload_service.upload_csv` (DataProfile) | `_spend_cap`, `gate_uploads`, demo-blocked, size cap | `api/index.py:2135` |

### Sandbox and artifacts

| Method | Path | Guard class | Auth / role | Request | Response | Service call(s) | Abuse guards (beyond global IP limiter) | Source |
|---|---|---|---|---|---|---|---|---|
| POST | `/api/sandbox/execute` | Guarded | analyst+ (all modes; anon only with dev ALLOW_ANONYMOUS_SANDBOX) | `SandboxExecuteRequest` | `SandboxResult` | `sandbox_service.run_python` (E2B); `pii_masking.mask_pii_dataframe_output`; `request_guard.enforce_sandbox_quota` | `_spend_cap`, `enforce_sandbox_quota` (SANDBOX_RUNS_PER_HOUR) | `api/index.py:2184` |
| GET | `/api/artifacts/{session_id}` | Guarded | principal (`require_principal`: user, or anon on demo resources; 401 under AUTH_REQUIRED) | - | JSON (untyped dict) | `artifact_service.list_artifacts` | - | `api/index.py:2259` |
| GET | `/api/artifacts/item/{artifact_id}` | Guarded | principal (`require_principal`: user, or anon on demo resources; 401 under AUTH_REQUIRED) | - | JSON (untyped dict) | `artifact_service.get_artifact` | - | `api/index.py:2268` |

### Hybrid and Autopilot

| Method | Path | Guard class | Auth / role | Request | Response | Service call(s) | Abuse guards (beyond global IP limiter) | Source |
|---|---|---|---|---|---|---|---|---|
| POST | `/api/hybrid/chat` | Guarded | principal (`require_principal`: user, or anon on demo resources; 401 under AUTH_REQUIRED) | `HybridChatRequest` | SSE | `hybrid_service.hybrid_chat`; `credit_service.begin_billing` | `_spend_cap`, `demo_spend_guard`, `charge_budget` (weight 2) | `api/index.py:2349` |
| POST | `/api/autopilot/run` | Guarded | principal (`require_principal`: user, or anon on demo resources; 401 under AUTH_REQUIRED) | `AutopilotRequest` | SSE | `autopilot_sources.resolve_autopilot_sources`, `autopilot_service.run_autopilot`; `credit_service.begin_billing` | `_spend_cap`, `demo_spend_guard`, `charge_budget` (weight 5) | `api/index.py:2437` |

### Lineage, tie-out and traces

| Method | Path | Guard class | Auth / role | Request | Response | Service call(s) | Abuse guards (beyond global IP limiter) | Source |
|---|---|---|---|---|---|---|---|---|
| GET | `/api/lineage/{run_id}` | Guarded | viewer (AUTH_REQUIRED only) + principal | - | JSON (untyped dict) | `lineage_service.get_lineage` | - | `api/index.py:2296` |
| POST | `/api/traces/{trace_id}/feedback` | Guarded | viewer (AUTH_REQUIRED only) + principal | `TraceFeedbackRequest` | JSON (untyped dict) | `trace_service.record_feedback` | - | `api/index.py:2320` |
| GET | `/api/lineage/{run_id}/tieout.{fmt}` | Guarded | viewer (AUTH_REQUIRED only) + principal | - | xlsx / pdf download | `lineage_service.get_lineage` -> `tieout_export.render` (xlsx/pdf) | - | `api/routes/tieout_routes.py:37` |

### Auth and identity

| Method | Path | Guard class | Auth / role | Request | Response | Service call(s) | Abuse guards (beyond global IP limiter) | Source |
|---|---|---|---|---|---|---|---|---|
| GET | `/api/auth/saml/metadata` | PUBLIC_ROUTES | none | - | `application/xml` | `auth_service.get_sp_metadata` | - | `api/index.py:2550` |
| GET | `/api/auth/saml/login` | PUBLIC_ROUTES | none | - | JSON (untyped dict) | `auth_service.build_saml_auth` | - | `api/index.py:2566` |
| POST | `/api/auth/saml/acs` | PUBLIC_ROUTES | none | - | 302 redirect + `docbot_session` cookie | `auth_service.process_acs`, `identity_service.resolve_provider_login`, `auth_service.create_user_session` | - | `api/index.py:2589` |
| GET | `/api/auth/debug-urls` | DEV_ONLY (404 unless ENV=dev) | none | - | JSON (untyped dict) | `oauth_service._app_base_url/_frontend_url` | - | `api/index.py:2661` |
| GET | `/api/auth/config` | PUBLIC_ROUTES | none | - | JSON (untyped dict) | `oauth_service.is_*_configured`, `auth_service.is_saml_configured` | - | `api/index.py:2679` |
| POST | `/api/auth/owner-unlock` | PUBLIC_ROUTES | none | - | `OwnerUnlockResponse` | `request_guard.verify_owner_key` (X-Owner-Key header) -> sets `docbot_owner` HMAC cookie | - | `api/index.py:2696` |
| GET | `/api/auth/github` | PUBLIC_ROUTES | none | - | JSON (untyped dict) | `oauth_service.github_authorize_url` + state cookie | - | `api/index.py:2729` |
| GET | `/api/auth/github/callback` | PUBLIC_ROUTES | none | - | 302 redirect + cookie | `_github_callback` -> code exchange, `identity_service`, `create_user_session` | - | `api/index.py:2740` |
| GET | `/api/auth/google` | PUBLIC_ROUTES | none | - | JSON (untyped dict) | `oauth_service.google_authorize_url` (PKCE) + state cookie | - | `api/index.py:2800` |
| GET | `/api/auth/google/callback` | PUBLIC_ROUTES | none | - | 302 redirect + cookie | `_google_callback` | - | `api/index.py:2813` |
| POST | `/api/auth/register` | PUBLIC_ROUTES | none | `RegisterRequest` | JSON (untyped dict) | `email_verification_service.enforce_turnstile`, `oauth_service.validate_password_strength/is_password_breached/hash_password`, `identity_service.register_password_user` | Turnstile (when configured), breached-password check | `api/index.py:2896` |
| POST | `/api/auth/login` | PUBLIC_ROUTES | none | `LoginRequest` | JSON (untyped dict) | `request_guard.check_login_throttle`, `oauth_service.verify_password`, `auth_service.create_user_session` | `check_login_throttle` (5 fails / 15 min per email+IP) | `api/index.py:2978` |
| GET | `/api/auth/me` | Handler-guarded (401 in handler) | `docbot_session` cookie checked in handler | - | JSON (untyped dict) | `auth_service.get_user_from_session` | - | `api/index.py:3036` |
| POST | `/api/auth/logout` | PUBLIC_ROUTES | none | - | JSON (untyped dict) | `auth_service.delete_session` | - | `api/index.py:3055` |
| GET | `/api/auth/workspace` | Handler-guarded (401 in handler) | `docbot_session` cookie checked in handler | - | JSON (untyped dict) | inline query of caller's sessions/connections | - | `api/index.py:3082` |
| POST | `/api/auth/verify` | PUBLIC_ROUTES | none | `VerifyRequest` (optional) or `?token=` | `VerifyResponse` | `email_verification_service.consume_verification_token` | - | `api/routes/email_verify_routes.py:47` |
| POST | `/api/auth/onboarding` | Guarded | viewer (all modes) | `OnboardingRequest` | `OnboardingResponse` | inline UPDATE users.use_case | - | `api/routes/onboarding_routes.py:48` |

### Admin

| Method | Path | Guard class | Auth / role | Request | Response | Service call(s) | Abuse guards (beyond global IP limiter) | Source |
|---|---|---|---|---|---|---|---|---|
| GET | `/admin/audit-log` | Guarded | admin (all modes) | - | JSON or CSV (`format=csv`) | inline select on `audit_log` scoped by `rbac_service.require_admin_org_id`; CSV via StreamingResponse | - | `api/index.py:3152` |
| GET | `/admin/users` | Guarded | admin (all modes) | - | JSON (untyped dict) | inline select scoped to admin org | - | `api/index.py:3255` |
| PATCH | `/admin/users/{user_id}/role` | Guarded | admin (all modes) | `RoleUpdateRequest` | JSON (untyped dict) | `rbac_service.update_user_role_in_org` | - | `api/index.py:3289` |
| GET | `/admin/api-keys` | Guarded | admin (all modes) | - | JSON (untyped dict) | `api_key_service.list_api_keys` | - | `api/index.py:3323` |
| POST | `/admin/api-keys` | Guarded | admin (all modes) | `APIKeyCreateRequest` | JSON (untyped dict) | `api_key_service.create_api_key` | - | `api/index.py:3335` |
| DELETE | `/admin/api-keys/{key_id}` | Guarded | admin (all modes) | - | JSON (untyped dict) | `api_key_service.revoke_api_key` | - | `api/index.py:3351` |
| GET | `/admin/metrics` | Guarded | platform admin | - | JSON (untyped dict) | `metrics_service.get_platform_metrics` | - | `api/index.py:3366` |
| GET | `/admin/funnel` | Guarded | platform admin | - | `FunnelReport` | `funnel_service.get_funnel_report` | - | `api/index.py:3389` |
| GET | `/api/admin/waitlist` | Guarded | platform admin | - | `WaitlistList` | `waitlist_service.list_entries` | - | `api/routes/waitlist_routes.py:47` |
| POST | `/admin/credits/grant` | Guarded | platform admin | `AdminCreditGrantRequest` | `AdminCreditGrantResponse` | `credit_service.grant_purchase` | - | `api/routes/billing_routes.py:91` |

### SEC EDGAR

| Method | Path | Guard class | Auth / role | Request | Response | Service call(s) | Abuse guards (beyond global IP limiter) | Source |
|---|---|---|---|---|---|---|---|---|
| POST | `/api/edgar/search` | KNOWN_UNGUARDED (DOCBOT-1623) | none | `EdgarSearchRequest` | JSON (untyped dict) | `EdgarConnector.search_company` (live SEC call) | `edgar_guard` (per-IP/user hourly) | `api/index.py:3485` |
| POST | `/api/edgar/filings` | KNOWN_UNGUARDED (DOCBOT-1623) | none | `EdgarFilingsRequest` | JSON (untyped dict) | `EdgarConnector.list_filings` (live SEC call) | `edgar_guard` | `api/index.py:3495` |
| POST | `/api/edgar/ingest` | Guarded | principal (`require_principal`: user, or anon on demo resources; 401 under AUTH_REQUIRED) | `EdgarIngestRequest` | JSON (untyped dict) | `edgar_service.ingest_edgar_filing` | `_spend_cap`, `edgar_guard`, demo-blocked | `api/index.py:3505` |
| POST | `/api/edgar/ingest-batch` | Guarded | principal (`require_principal`: user, or anon on demo resources; 401 under AUTH_REQUIRED) | `EdgarBatchIngestRequest` | JSON (untyped dict) | `edgar_service.ingest_multiple_filings` | `_spend_cap`, `check_edgar_limit`, EDGAR_BATCH_MAX, demo-blocked | `api/index.py:3532` |
| GET | `/api/edgar/cache` | KNOWN_UNGUARDED (DOCBOT-1623) | none | - | JSON (untyped dict) | `connector_store.list_cached_filings` | - | `api/index.py:3570` |

### Marketplace connectors and commerce

| Method | Path | Guard class | Auth / role | Request | Response | Service call(s) | Abuse guards (beyond global IP limiter) | Source |
|---|---|---|---|---|---|---|---|---|
| POST | `/api/connectors/register` | Guarded | none | `ConnectorRegisterRequest` | `ConnectorRegisterResponse` | `registry.get_connector_class`, `connector_store.save_connector`, `sync_jobs.register_sync_schedules` | demo-blocked | `api/routes/connector_routes.py:162` |
| GET | `/api/connectors` | Guarded | none | - | `ConnectorListResponse` | `connector_store.list_active_connectors` | - | `api/routes/connector_routes.py:200` |
| GET | `/api/connectors/types` | Guarded | none | - | `ConnectorTypesResponse` | `registry` type list | - | `api/routes/connector_routes.py:208` |
| DELETE | `/api/connectors/{connector_id}` | Guarded | none | - | `ConnectorDisconnectResponse` | `connector_store.delete_connector`, `sync_jobs.deregister_sync_schedules` | - | `api/routes/connector_routes.py:216` |
| POST | `/api/connectors/{connector_id}/orders` | Guarded | none | `ConnectorDateRangeRequest` | `ConnectorOrdersResponse` | connector `.fetch_orders` (live API) | - | `api/routes/connector_routes.py:234` |
| POST | `/api/connectors/{connector_id}/financials` | Guarded | none | `ConnectorDateRangeRequest` | `ConnectorFinancialsResponse` | connector `.fetch_financials` (live API) | - | `api/routes/connector_routes.py:251` |
| POST | `/api/connectors/{connector_id}/sync` | Guarded | none | `ConnectorDateRangeRequest` | JSON (untyped dict) | `commerce_service.sync_connector_data` | - | `api/routes/connector_routes.py:268` |
| GET | `/api/commerce/{connector_id}/orders` | Guarded | none | - | `ConnectorOrdersResponse` | `commerce_service.query_orders` | - | `api/routes/connector_routes.py:294` |
| GET | `/api/commerce/{connector_id}/financials` | Guarded | none | - | `ConnectorFinancialsResponse` | `commerce_service.query_financials` | - | `api/routes/connector_routes.py:310` |
| POST | `/api/marketplace/webhook/shopify` | PUBLIC_ROUTES | none | - | `WebhookAck` | `shopify_connector.validate_shop_domain`, `verify_shopify_hmac`, `commerce_service.sync_connector_data` | HMAC-SHA256 + `*.myshopify.com` domain check | `api/routes/connector_routes.py:345` |

### Billing

| Method | Path | Guard class | Auth / role | Request | Response | Service call(s) | Abuse guards (beyond global IP limiter) | Source |
|---|---|---|---|---|---|---|---|---|
| GET | `/api/billing/packs` | PUBLIC_ROUTES | none | - | `CreditPackCatalog` | `billing_service.catalog` | - | `api/routes/billing_routes.py:40` |
| POST | `/api/billing/checkout` | Guarded | admin (all modes) | `CheckoutSessionRequest` | `CheckoutSessionResponse` | `billing_service.checkout_url` (Stripe Checkout) | - | `api/routes/billing_routes.py:45` |
| POST | `/api/billing/webhook` | PUBLIC_ROUTES | none | - | JSON (untyped dict) | `billing_service.verify_webhook` (Stripe-Signature) -> `credit_service.grant_purchase`, `funnel_service.record_paid` | exempt from IP limiter (`RATE_LIMIT_EXEMPT_PATHS`) | `api/routes/billing_routes.py:60` |

### Waitlist

| Method | Path | Guard class | Auth / role | Request | Response | Service call(s) | Abuse guards (beyond global IP limiter) | Source |
|---|---|---|---|---|---|---|---|---|
| POST | `/api/waitlist` | PUBLIC_ROUTES | none | `WaitlistRequest` | `WaitlistResponse` | `waitlist_service.upsert_entry`, `notify_founder` | honeypot + length caps | `api/routes/waitlist_routes.py:33` |

### Developer API (/v1, API-key auth)

| Method | Path | Guard class | Auth / role | Request | Response | Service call(s) | Abuse guards (beyond global IP limiter) | Source |
|---|---|---|---|---|---|---|---|---|
| POST | `/v1/hybrid/chat` | Guarded | API key scope `hybrid:read` | `V1HybridChatRequest` | JSON (untyped dict) | `hybrid_service.hybrid_chat` (SSE aggregated to JSON) | `_spend_cap`, `charge_budget`, credit reservation on key org | `api/index.py:3659` |
| POST | `/v1/autopilot/run` | Guarded | API key scope `autopilot:run` | `V1AutopilotRequest` | JSON (untyped dict) | `autopilot_service.run_autopilot` (SSE aggregated to JSON) | `_spend_cap`, `charge_budget`, credit reservation on key org | `api/index.py:3768` |
| GET | `/v1/lineage/{run_id}` | Guarded | API key scope `lineage:read` | - | JSON (untyped dict) | `lineage_service.get_lineage` | - | `api/index.py:3882` |
| GET | `/v1/connectors` | Guarded | API key scope `connectors:manage` | - | JSON (untyped dict) | `connector_store.list_active_connectors` | - | `api/index.py:3905` |
| POST | `/v1/connectors` | Guarded | API key scope `connectors:manage` | `V1ConnectorRegisterRequest` | JSON (untyped dict) | `registry.get_connector_class`, `connector_store.save_connector`, `sync_jobs.register_sync_schedules` | - | `api/index.py:3917` |
| POST | `/v1/connectors/{connector_id}/sync` | Guarded | API key scope `connectors:manage` | `ConnectorDateRangeRequest` | JSON (untyped dict) | `commerce_service.sync_connector_data` | - | `api/index.py:3963` |

## Known gaps worth tracking

| Route | Gap | Source |
|---|---|---|
| `POST /api/edgar/search`, `POST /api/edgar/filings`, `GET /api/edgar/cache` | No auth guard (ratchet `KNOWN_UNGUARDED`, DOCBOT-1623). Search/filings are rate-limited by `edgar_guard`; `GET /api/edgar/cache` has no per-route limit and the global IP limiter skips GET. | `tests/unit/test_route_guards.py:196-201`, `api/index.py:3485-3575` |
| `/api/connectors*`, `/v1/connectors*` | Live connector instances live in a per-process dict `app.state.connectors`, rehydrated from `marketplace_connections` at startup (`api/index.py:685-694`). With more than one backend replica, a connector registered on one replica is unknown to the other until restart (inferred). | `api/routes/connector_routes.py:62-66`, `api/index.py:3444-3451` |
| `GET/POST /v1/connectors`, `POST /v1/connectors/{connector_id}/sync` | Not org-scoped: list has no `org_id` filter, register saves without `org_id`, sync looks up any connector by id. A key with `connectors:manage` reaches other orgs' connectors. | `api/index.py:3905-3975` |
| `/admin/*` from the browser | The admin panel calls `fetch("/admin/users")` and `fetch("/admin/metrics")` on the Vercel origin (`src/hooks/useChatHandlers.ts:460, 478`), but Next only rewrites `/api/*` and `/v1/*` (`next.config.ts:36-43`). On Vercel these requests most likely 404 (inferred, not tested against production). `/api/admin/waitlist` is unaffected. | `next.config.ts:36-43` |
| `PATCH /admin/users/{user_id}/role` | Not in CORS `allow_methods`, so it only works same-origin; also not covered by the global IP limiter (POST/DELETE only). | `api/index.py:57-63, 103-107` |

## Next.js routes (frontend, Vercel)

There are **no Next.js API route handlers** (`find src/app -name 'route.ts'` returns nothing). All `/api/*` and `/v1/*` requests are proxied to the FastAPI backend by a rewrite (`next.config.ts:36-43`; `http://127.0.0.1:8000` in development, `BACKEND_URL` / legacy `RAILWAY_BACKEND_URL` in production, build fails without it, `next.config.ts:7-15`).

| Path | File | Kind | Notes |
|---|---|---|---|
| `/` | `src/app/page.tsx` | Page | Landing page. |
| `/landing` | `next.config.ts:31-33` | Redirect (308) | Permanent redirect to `/` (DOCBOT-2537). |
| `/chat` | `src/app/chat/page.tsx` | Page (client) | Main app. See [Frontend Architecture](architecture-08-frontend-architecture). |
| `/signup` | `src/app/signup/page.tsx` | Page | Sign-up. |
| `/verify` | `src/app/verify/page.tsx` | Page | Consumes email token via `POST /api/auth/verify`. |
| `/onboarding` | `src/app/onboarding/page.tsx` | Page | Calls `POST /api/auth/onboarding`. |
| `/privacy`, `/terms` | `src/app/privacy/page.tsx`, `src/app/terms/page.tsx` | Page | Legal. |
| `/robots.txt` | `src/app/robots.ts` | Metadata route | |
| `/sitemap.xml` | `src/app/sitemap.ts` | Metadata route | |
| `/api/:path*`, `/v1/:path*` | `next.config.ts:36-43` | Rewrite | Proxy to backend. |
| all non-API pages | `src/proxy.ts` | Next 16 proxy (middleware) | Per-request CSP nonce; matcher excludes `api/`, `v1/`, `_next/static`, `_next/image`, `favicon.ico` and prefetches (`src/proxy.ts:5-33`). |
