# YC W27 Plan v2 — User-Ready DocBot (2026-10-24 → 2026-11-02)

**Supersedes v1 (2026-10-23).** Merged on 2026-10-24 from six parallel read-only runs: Product Manager (epics, pricing), Senior Developer (signup/billing architecture), Security Engineer (77-route audit + threat model), Frontend Developer (rendering audit), UI Designer (visual spec), AI Engineer (bug hunt — feeds DOCBOT-1613). Code refs are DocBot @ `1f2ade3`.

**Deadline:** Mon 2026-11-02 8pm PT. **Internal submit: Sun 11-01 (D9).**

---

## 0. Founder decisions (locked 2026-10-24)

| # | Decision |
|---|---|
| F-1 | **Infra stays on Railway + Vercel.** AWS migration + Terraform collapsed into ONE parked epic (EPIC-21), post-YC. |
| F-2 | **Goal: user-ready self-serve.** Signup → verified → free monthly credits → pay (Stripe prepaid credit packs) when over quota. Usage = traction signal for YC. |
| F-3 | **BYOK deferred to after submit** (EPIC-19). Pricing page says "BYOK: contact us". Reason: ICP lacks LLM keys; key-leak surface (Sentry, logs, cache) not hardened yet. |
| F-4 | **New routes live in `api/routes/*_routes.py` (FastAPI `APIRouter`)**; `index.py` only `include_router`. CLAUDE.md updated in DOCBOT-1800. Enables parallel agents. |
| F-5 | **Scope = security gate + signup + credits/Stripe + formatting P0.** Defer password reset, portal, BYOK, UI polish, invites. |
| F-6 | **Positioning unchanged:** "AI that ties out financial reports against your books." Self-serve is the funnel; tie-out is the headline. Outreach (EPIC-22) is never cut. |

---

## 1. Critical findings (fix before anything public)

| Ref | Finding | Where |
|---|---|---|
| **SEC-01** | If prod runs without `AUTH_REQUIRED=true`, `require_role()` returns `None` → anonymous callers can **mint `/v1` API keys** and **promote any user to admin**. | `rbac_service.py:75`, `index.py:3052`, `:2974` |
| **ORG-BUG** | `jit_provision_user` inserts no `org_id`; `backfill_default_organization` puts every signup in ONE shared org (shared admin views, API keys, DEK). | `auth_service.py:342-353`, `org_service.py:96` |
| SEC-02 | Anonymous list/read/export/delete of any chat session; IDOR on chat, artifacts, persona. | `index.py:1364-1556, 2157-2165` |
| SEC-04 | Anonymous unlimited E2B Python with egress; timeout doesn't kill VM → thread-pool exhaustion. | `index.py:2097`, `request_guard.py:171`, `sandbox_service.py:458,747` |
| SEC-05 | DB URL injection: `dbname="x?host=postgres.railway.internal"` overrides SSRF-validated host. | `db_service.py:461-470` |
| SEC-09 | Account pre-hijack: email-only identity linking; Google `email_verified` unchecked. | `auth_service.py:318-338`, `oauth_service.py:189` |
| FMT-01 | Discrepancy (the demo "aha") never renders: backend emits unclosed `[DISCREPANCY]`, frontend regex needs a closing tag. | `discrepancy_detector.py:68,74` vs `ChatMessage.tsx:327` |
| FMT-03 | PII masker corrupts financial numbers: `4523000000` → `***-***-0000`. | `pii_masking.py:39-50` |

**D1 action for founder:** check Railway `AUTH_REQUIRED`. Or: `curl -s -o /dev/null -w "%{http_code}" https://<railway-url>/admin/metrics` → `200` = exposed.

---

## 2. Epics and stories

Legend — **Win**: `Y` committed in window · `S` in window, cut-first · `N` deferred. **Pri**: P0 pre-submit · P1 next · P2 later.

### EPIC-16 — Security Hardening (extended)

| ID | Title | Pts | Pri | Win | Acceptance criteria | Refs |
|---|---|---|---|---|---|---|
| DOCBOT-1610 | Admin routes require auth in every mode + `MULTI_TENANT` boot guard | 2 | P0 | Y | `require_admin` ignores open mode; anonymous `/admin/*` → 401. Boot exits if `MULTI_TENANT=true` without auth. Test per admin route. | SEC-01 |
| DOCBOT-1612 | Prod abuse lockdown (config) | 1 | P0 | Y | Railway: `RATE_LIMIT_PER_MIN`, `SESSION_BUDGET`, `GATE_UPLOADS=true`, `GEMINI_API_KEY`. `/api/demo/init` IP-limited. | v1 ENG-01 |
| DOCBOT-1601 | `CallerScope` on session routes | 5 | P0 | Y | List/get/delete/export scoped to owner+org; non-owner 404; authz before delete; inserts stamp `org_id`/`user_id`. Anon-A vs anon-B tests. | SEC-02 |
| DOCBOT-1611 | Session-bound routes | 3 | P0 | Y | `/api/chat`, `update-persona`, `/api/artifacts/*`, hybrid, autopilot reject foreign `session_id`. IDOR tests. | SEC-02 |
| DOCBOT-1603 | Connection / schema-cache / history scoping | 3 | P0 | Y | Org+owner predicate at `db_service.py:729/744/782/1045/1207/1735`. `/api/db/sessions` removed or caller-only. `/v1` body ids checked against `_key.org_id`. | SEC-06 |
| DOCBOT-1602 | Connector + commerce RBAC, Shopify SSRF | 4 | P0 | Y | All `/api/connectors*`, `/api/commerce/*` require auth + org scope. `shop_domain` must match `^[a-z0-9-]+\.myshopify\.com$`. No `{exc}` echo. | SEC-03 |
| DOCBOT-1607 | Sandbox lockdown | 4 | P0 | Y | `sandbox/execute` requires account; metered by org (no session_id bypass). `allow_internet_access=False`. `Sandbox.create(timeout=60)` + kill in `finally`. | SEC-04 |
| DOCBOT-1608 | SSRF + connection-URL injection | 4 | P0 | Y | `URL.create()` for all dialects; reject `?;=/` in dbname/host; ODBC brace-escape; only `ip.is_global` (incl. `::ffff:` mapped); pin resolved IP. | SEC-05, SEC-13 |
| DOCBOT-1604 | Org in LLM cache key | 2 | P0 | Y | `llm_cache.py:136` key includes `org_id`; cross-org cache-hit test. | SEC-25 |
| DOCBOT-1605 | OWNER_KEY hardening | 2 | P0 | Y | Header only, `hmac.compare_digest`, cookie stores HMAC; frontend unlock → POST. | SEC-12 |
| DOCBOT-1606 | Durable rate limiter + XFF | 3 | P0 | Y | Postgres-backed (or Redis) limiter survives deploy; safe non-zero defaults; client IP = right-most trusted hop. | SEC-07 |
| DOCBOT-1616 | Upload hardening | 3 | P0 | Y | 25 MB / 10 files / 500 pages; `%PDF` magic check; `tempfile` + cleanup; PDF parse off event loop with timeout; caps enforced while streaming, not after full read. | SEC-08 |
| DOCBOT-1617 | Identity linking | 3 | P0 | Y | `user_identities(provider, provider_sub)` table; link only on verified email; Google `email_verified` required; SAML cannot claim existing email unverified. | SEC-09 |
| DOCBOT-1618 | Sentry scrubbing | 1 | P0 | Y | `include_local_variables=False`, `max_request_body_size="never"`, `before_send` redacts `gsk_`/`sk-`/`AIza`/passwords. Canary test. | SEC-10 |
| DOCBOT-1619 | CSP + security headers + safe markdown media | 2 | P0 | Y | CSP, HSTS, `frame-ancestors`, `nosniff` (Next + FastAPI). Markdown `img` allows only `data:`/`blob:`; links `rel="noopener noreferrer"`. | SEC-11, FMT-05 |
| DOCBOT-1620 | Org filter: admin role PATCH, lineage, traces | 2 | P0 | Y | Role PATCH checks target org, protects last admin. `lineage/{run_id}`, `traces/{id}/feedback`, `/v1/lineage` org-filtered. | SEC-15, SEC-16 |
| DOCBOT-1621 | Session token hardening | 1 | P0 | Y | Tokens stored hashed; TTL from `SESSION_TTL_HOURS`; revoke-all on password/email change. | SEC-17 |
| DOCBOT-1622 | Login abuse controls | 1 | P0 | Y | Generic register/login errors; per-account throttle; breached-password check (k-anon HIBP). | SEC-18 |
| DOCBOT-1623 | Demo + EDGAR metering, vector-store cap | 1 | P0 | Y | `demo/init` + EDGAR rate-limited and metered; LRU cap on `VECTOR_STORES`. | SEC-21 |
| DOCBOT-1624 | CORS boot guard | 1 | P0 | Y | Origins stripped; refuse boot on `*` with credentials. | SEC-22 |
| DOCBOT-1625 | CI security jobs + route-guard sweep | 3 | P0 | Y | `bandit -ll`, `pip-audit`, `npm audit --audit-level=high`, `gitleaks`, Dependabot. `tests/unit/test_route_guards.py`: every route has guard or is in `PUBLIC_ROUTES`; anon → 401 under required mode. | §4 Security |
| DOCBOT-1615 | Global daily spend kill switch | 2 | P0 | Y | `PLATFORM_DAILY_USD_CAP` (LLM+E2B); breach → free-tier 503, paid continues, Sentry alert. | PM |
| DOCBOT-1613 | Prod smoke regression | 2 | P0 | Y | Scripted smoke on prod: PDF, CSV, SQL, hybrid, autopilot, demo, EDGAR, signup→pay. Run at Gate A and Gate B. Bug fixes themselves live in EPIC-23. | AI Eng |
| DOCBOT-1614 | Security review of new surfaces | 2 | P0 | Y | Signup, verification, webhook, ledger reviewed against §4 gates; findings closed before Stripe live. | PM |
| DOCBOT-1609 | SQL validator: INTO/locking, Azure read-only | 2 | P1 | N | Reject `SELECT … INTO`, `FOR UPDATE`; function denylist; Azure read-only DB user. | SEC-14 |
| DOCBOT-1626 | SAML replay protection | 1 | P1 | N | Reject unsolicited responses; assertion-ID replay cache. | SEC-19 |
| DOCBOT-1627 | Shopify webhook fix | 1 | P1 | N | Fix undefined `_connectors`; dedupe on `X-Shopify-Webhook-Id`; queue sync. | SEC-20 |
| DOCBOT-1628 | Stop echoing exceptions to clients | 1 | P1 | N | `safe_error_message()` everywhere (`index.py:3136, 3209-3255`). | SEC-23 |
| DOCBOT-1629 | Dependency pinning | 2 | P1 | N | Hash-locked `requirements.txt`; drop/vendor `langchain-experimental` (SemanticChunker only). | SEC-24 |

### EPIC-17 — Self-Serve Onboarding & Accounts

| ID | Title | Pts | Pri | Win | Acceptance criteria | Refs |
|---|---|---|---|---|---|---|
| DOCBOT-1702 | Personal org per signup + remediation | 3 | P0 | Y | `jit_provision_user` creates org, creator = admin. One-time script moves post-1524 signups out of default org. Backfill only touches legacy rows. Test: two signups → two orgs. | ORG-BUG, T1 |
| DOCBOT-1713 | Email verification + bot defense | 3 | P0 | Y | Resend magic link (httpx, hashed single-use token, 24h). Turnstile on register. Disposable-domain block. Gmail dot/`+` normalization. Unverified = demo only, no credits. | SEC threat model |
| DOCBOT-1711 | Auth mode switch (open → required) | 3 | P0 | Y | `APP_AUTH_MODE=open\|required`; `require_principal()`: user → org, anon + demo session → demo principal, else 401. Demo sessions tagged `source='demo'`. `DEMO_DAILY_USD_CAP` circuit breaker. | T6 |
| DOCBOT-1707 | Legal pages | 1 | P0 | Y | ToS, Privacy, data retention; signup checkbox. (Stripe prerequisite.) | PM |
| DOCBOT-1706 | Funnel events | 2 | P0 | Y | Server-side `funnel_events`: signup → verified → first upload → first answer → first tie-out → paywall → checkout → paid. Admin query. | PM |
| DOCBOT-1712 | Signup / verify / onboarding UI | 3 | P0 | Y | `/signup` split layout (form + sample tie-out card), "check inbox" + resend, `src/app/verify`, `src/app/onboarding` (workspace name, use case). Zod schemas. | T7, UI spec |
| DOCBOT-1704 | Tie-out onboarding checklist | 3 | P1 | S | Post-verify: sample tie-out → upload draft + TB → first flagged result. | PM |
| DOCBOT-1703 | Password reset | 2 | P1 | N | Single-use 30-min hashed token; generic response; revoke all sessions. | SEC |
| DOCBOT-1708 | Account deletion + export | 2 | P1 | N | Hard-delete sessions, connections, vectors; audit log. | PM |
| DOCBOT-1709 | Team invites | 3 | P2 | N | Invite by email with role; seat count. | PM |
| DOCBOT-1710 | Lifecycle emails | 2 | P2 | N | Welcome, 80% quota, day-3 inactive. | PM |

### EPIC-18 — Billing & Usage Metering (Stripe prepaid credits)

Model: **1 credit = $0.001 retail**, debit = `ceil(cost_usd × MARKUP × 1000)`. Free monthly grant only after email verify. Prepaid packs via Checkout `mode=payment`; hard stop at zero (no overage, no bad debt).

| ID | Title | Pts | Pri | Win | Acceptance criteria | Refs |
|---|---|---|---|---|---|---|
| DOCBOT-1800 | Billing contracts + routes scaffold | 2 | P0 | Y | `api/utils/billing_context.py` (ContextVars `Principal`, `KeyContext` masked repr, `RequestMeter`, `copy_ctx_run`), `api/billing_models.py` (all Pydantic), `src/lib/schemas/billing.ts` (Zod), `api/routes/` package, CLAUDE.md route rule updated (F-4). | T0 |
| DOCBOT-1801 | Usage ledger + metering hooks | 5 | P0 | Y | `usage_events` table; `llm_calls` + `org_id`/`user_id`/`key_source`. `_log_llm_call` enriched from ContextVar (never key). E2B seconds in `finally`; uploads metered. All `run_in_executor` sites use `copy_ctx_run`; SSE generators rebind context. | T2 |
| DOCBOT-1802 | Credit balance, reserve/settle, free grant | 5 | P0 | Y | `org_balances` + append-only `credit_ledger` `UNIQUE(kind,ref)`. Atomic reserve `UPDATE … WHERE balance-reserved>=:est` → else 402 `insufficient_credits`. Settle in `finally`. Lazy monthly grant. Stale-reservation reaper. **Weights calibrated from p95 cost: direct cost ≤30% of price; free-tier run ceiling $0.15.** 50-concurrent test → exactly one success at limit−1. | T3, PM margin |
| DOCBOT-1803 | Stripe Checkout | 3 | P0 | Y | `GET /api/billing/packs` (env `STRIPE_PACKS_JSON`), `POST /api/billing/checkout` (admin) → hosted URL; `client_reference_id` + `metadata.org_id` set server-side. E2E in test mode. | T4 |
| DOCBOT-1804 | Stripe webhook (idempotent) | 3 | P0 | Y | Raw-body `construct_event`; one txn: `stripe_events` insert ON CONFLICT → `credit_ledger(purchase, checkout_session_id)` → balance. Handles `checkout.session.completed` (paid only), `async_payment_succeeded`, `charge.refunded` (negative), `charge.dispute.created` (freeze). Bad sig 400; replay no-op; 2xx only after commit. | T4, SEC |
| DOCBOT-1805 | Paywall + 402/429 handling | 2 | P0 | Y | `useChatSubmit` maps 402 → `OutOfCreditsModal` (buy pack); demo 429 → signup prompt; resume blocked action after purchase. | T8 |
| DOCBOT-1811 | Billing settings + usage meter UI | 3 | P0 | Y | `UsageMeter` in sidebar (amber 80%, danger 100%); `/settings/billing` balance, packs, usage table (DataTable spec). | T8, UI spec |
| DOCBOT-1808 | Pricing page | 1 | P0 | Y | Free / Credit pack / Pro / Firm; "BYOK: contact us". | PM |
| DOCBOT-1806 | Stripe Customer Portal link | 1 | P1 | N | Invoices, receipts, payment method. | PM |
| DOCBOT-1807 | Revenue + margin in admin metrics | 2 | P1 | N | Signups, active orgs, credits used, revenue, est. gross margin. | PM |
| DOCBOT-1809 | Metered overage (Stripe Meters) | 5 | P2 | N | Pay-after-use for Firm plans. | PM |
| DOCBOT-1810 | Dunning + subscription lifecycle | 3 | P2 | N | Pro subscription states, failed-payment downgrade. | PM |

**Pricing (to validate on calls):** Free 100 credits-equivalent/mo · Pack $20 · Pro $49/mo · Firm $500–2,000/mo via LOI (design-partner rate $500/mo × 6 mo).

### EPIC-19 — BYOK & Key Management (deferred, post-submit)

Every story inherits §4 gates: canary-key never in logs/Sentry/traces/cache/responses; no user-supplied `base_url`; key bound to org via AAD.

| ID | Title | Pts | Pri | Win | Acceptance criteria |
|---|---|---|---|---|---|
| DOCBOT-1901 | Store org LLM keys | 3 | P1 | N | `org_llm_keys` encrypted via per-org DEK (1529); last4 only; validated by one cheap call. |
| DOCBOT-1902 | Per-request key resolution | 5 | P1 | N | `resolve_api_key(provider)` reads `KeyContext` then env; migrate 15 env-read sites; BYOK → `key_source=byok`, 0 LLM credits; defined fallback rule. |
| DOCBOT-1903 | OpenAI-compatible provider (allowlisted) | 3 | P2 | N | Fixed endpoint allowlist; JSON-output capability check. |
| DOCBOT-1904 | BYOK settings UI | 2 | P1 | N | `/settings/keys`: add/test/rotate/delete; dev API keys (1530) move here. |
| DOCBOT-1905 | Key redaction everywhere | 2 | P1 | N | Root-logger + Sentry + `_sanitize_error` patterns; tests. |
| DOCBOT-1906 | Formatting regression per BYOK model | 3 | P2 | N | EPIC-20 golden suite passes per model. |

### EPIC-20 — Response Formatting Quality

| ID | Title | Pts | Pri | Win | Acceptance criteria | Refs |
|---|---|---|---|---|---|---|
| DOCBOT-2002 | Discrepancy rendering from structured lineage | 3 | P0 | Y | Discrepancies rendered from `lineage.discrepancies` as `TieOutCard` (report vs books, badge, Δ, both citations); inline marker → badge; no literal `[DISCREPANCY]`. | FMT-01, UI §4 |
| DOCBOT-2003 | PII masker stops corrupting numbers | 3 | P0 | Y | Phone match needs separators/context; line-buffered masking (not per token). `mask_pii("4523000000")` unchanged; real phones masked. | FMT-03 |
| DOCBOT-2001 | Shared format contract + prompt dedupe | 3 | P0 | Y | `api/utils/response_format.py`: `FORMAT_CONTRACT_V1` (GFM only, no HTML/LaTeX/images, ≤6-col tables, `$4.2M`, citation syntax per source present), `build_format_block()`, `normalize_markdown()`. 5 persona copies + 2 autopilot copies removed; conflicting rules resolved. | FMT-08, FMT-10 |
| DOCBOT-2004 | No double-streamed hybrid answers | 2 | P0 | Y | Correction path emits SSE `replace` event; frontend swaps content. | FMT-02 |
| DOCBOT-2005 | Clean Autopilot step output | 3 | P0 | Y | Human `step.summary` separate from LLM `result`; no JSON / `<untrusted_…>` tags in UI; rendered via shared renderer; `safe_error_message` on failures. | FMT-04, FMT-13 |
| DOCBOT-2006 | `MarkdownRenderer` component | 2 | P0 | Y | Single renderer: table scroll wrapper (`role=region`, `tabIndex=0`), Prism code + copy, `skipHtml`; used by chat, steps, inspector. | FMT-06, FMT-15 |
| DOCBOT-2008 | SQL answer truncation | 1 | P0 | Y | `max_tokens` 800 → 1500; on `finish_reason=length` close open fence + "_(truncated)_". | FMT-09 |
| DOCBOT-2007 | Streaming-safe markdown | 3 | P1 | S | `prepareStreamingMarkdown()` holds incomplete table/line, hides partial markers; rAF token batching; memoized blocks. | FMT-07 |
| DOCBOT-2009 | CSV/sandbox output as tables | 2 | P1 | S | Validate pipe tables before passthrough; DataFrames → formatted tables; friendly sandbox errors. | FMT-12 |
| DOCBOT-2012 | Frontend test harness + fixtures | 3 | P1 | S | `vitest` + RTL + jsdom; shared `tests/fixtures/responses/*`; snapshot + streaming + security assertions in CI. | FE test plan |
| DOCBOT-2010 | Citation chips (doc/sql/csv/edgar) | 2 | P1 | N | Inline `[Source:…]`/`[DB:…]` → superscript chips; no "pundefined". | FMT-11, UI §5 |
| DOCBOT-2011 | Unified SSE parser + Zod | 2 | P1 | N | `src/lib/sse.ts` replaces 4 parsers; flushes tail buffer; Zod discriminated union. | FMT-14 |
| DOCBOT-2013 | UI spec polish | 5 | P1 | N | Light theme, contrast fix (`text-quaternary` 2.25:1), `AutopilotTimeline`, `AnswerError` via `Message.kind`, touch-visible actions, 10px floor. | UI gaps |
| DOCBOT-2014 | Tie-out export (XLSX/PDF) | 3 | P2 | N | Flagged lines with both citations. | PM |

### EPIC-23 — Pre-Launch Reliability (AI Engineer bug hunt)

Bug-hunt run 2026-10-24 on the Desktop checkout: **1,128 tests pass**; `tsc` fails and `npm run lint` crashes (see 2302). Bugs already covered by other tickets: BUG-01/02 → 1601/1611/1603 · BUG-03/04 → 1608 · BUG-06 → 1606/1607 · BUG-15 → 1616 · BUG-16 (sandbox) → 1607 · BUG-17 → 1609 · BUG-20 → 1801/1802 · BUG-21 → 2003 · BUG-22 → 2004 · BUG-25 → 2001 · BUG-27 → 2005/1628.

| ID | Title | Pts | Pri | Win | Acceptance criteria | Refs |
|---|---|---|---|---|---|---|
| DOCBOT-2302 | Green build: lockfile + lint | 2 | P0 | Y | Uncommitted `remark-math`/`rehype-katex` either installed + lockfile committed, or reverted. `eslint.config` ignores `.venv/**`; 3 `src` lint errors fixed (`ChatMessage.tsx:377`, `CommandPalette.tsx:176`, `ToolPicker.tsx:96`). `tsc` + lint green in CI. | BUG-07, BUG-29 |
| DOCBOT-2301 | Unblock the event loop | 5 | P0 | Y | Sync `chat_completion` callsites → async (`AsyncGroq`/`to_thread`); PDF parse + embedding + Chroma writes off loop; uvicorn workers ≥2 (or gunicorn). Test: `/health` <500ms during 2 concurrent autopilots. | BUG-05 |
| DOCBOT-2307 | LLM fallback correctness | 5 | P0 | Y | Intent classifier uses async provider, `max_tokens≥200` (`hybrid_service.py:581,181`). Fallback only before first token (no duplicated text). Doc chat via provider with Gemini fallback; gate on "any key". `content or ""`; no fallback on 400/401; `max_retries=1, timeout=20`. | BUG-09/10/11/28 |
| DOCBOT-2311 | Documents survive redeploy | 2 | P0 | Y | Chroma on Railway volume (`CHROMA_DIR`); lazy-load on miss; missing vectors → explicit "documents expired" SSE event, never silent no-docs hybrid answer. | BUG-14 |
| DOCBOT-2310 | Autopilot timeout still answers | 2 | P0 | Y | Wall-clock timeout routes to synthesizer with partial results; per-step `wait_for`; `done` event always emitted. | BUG-19 |
| DOCBOT-2306 | Upload crash chain | 1 | P0 | Y | Page count read before `doc.close()`; audit uses `"filename"`; encrypted-PDF error message correct. | BUG-12 |
| DOCBOT-2303 | Surface backend errors in chat | 1 | P0 | Y | Hybrid + doc-chat `error` events shown (not empty bubble), matching DB path (`useChatSubmit.ts:559,640`). | BUG-08 |
| DOCBOT-2305 | Auth/quota responses + history trim | 2 | P0 | Y | 401/403 → login modal; 429 → message from `detail`; doc chat sends `recentHistory` (no chart base64). (402 handled in 1805.) | BUG-24 |
| DOCBOT-2313 | Logout clears state | 1 | P0 | Y | Session, connection, messages reset on logout. | BUG-26 |
| DOCBOT-2304 | Stop button + idle timeout | 3 | P1 | S | `AbortController` on all streams; idle timeout; Stop button; input never locks. | BUG-23 |
| DOCBOT-2312 | Correct page citations | 3 | P1 | S | Per-page chunking or char-offset tracking; multi-page fixture test asserts pages. | BUG-13 |
| DOCBOT-2308 | DB server-side timeouts | 2 | P1 | S | MySQL `max_execution_time`, SQLite progress handler, Azure query timeout; bounded executor. | BUG-16 |
| DOCBOT-2309 | Schema-drift retry actually fires | 1 | P1 | S | Typed `SchemaDriftError` raised before friendly rewrite. | BUG-18 |
| DOCBOT-2314 | Zod on remaining frontend calls | 2 | P2 | N | History, upload, connectors, demo init, auth config validated. | FE |

### EPIC-21 — AWS Migration & Terraform (PARKED, post-YC)

Consolidates all of `docbot-platform` (95 PLAT stories, 470 pts; story-level detail stays in `../docbot-platform/docs/EPICS.md`). Prerequisite: all of EPIC-16 merged. Rows are story groups, not single stories.

| ID | Group (PLAT range) | Stories | Pts |
|---|---|---|---|
| DOCBOT-2101 | Repo bootstrap + engine extraction (0101–0107) | 7 | 28 |
| DOCBOT-2102 | Tenancy foundation, FORCE RLS (0201–0212) | 12 | 60 |
| DOCBOT-2103 | Modules, AI routing, keys (0301–0311) | 11 | 52 |
| DOCBOT-2104 | Control plane + Terraform: ECS Fargate, RDS+pgvector, KMS, BYOC (0400–0420) | 21 | 125 |
| DOCBOT-2105 | Frontend platform / console (0501–0524) | 24 | 96 |
| DOCBOT-2106 | OpenCode analyst workbench (0601–0609) | 9 | 48 |
| DOCBOT-2107 | Per-tenant fine-tuning + governance (0701–0711) | 11 | 61 |

Overlap to drop at post-YC re-plan: PLAT-0305 ↔ EPIC-19, PLAT-0509 ↔ 1713/1702, PLAT-0420 ↔ 1801, PLAT-0210 ↔ 1604.

### EPIC-22 — YC Go-To-Market

| ID | Title | Pts | Pri | Win | Acceptance criteria |
|---|---|---|---|---|---|
| DOCBOT-2201 | Confirm deadline + form setup | 1 | P0 | Y | Date confirmed on yc.com; questions in working doc. |
| DOCBOT-2202 | Lock positioning + ICP | 2 | P0 | Y | One-liner, 50-word, ICP final; landing hero swapped (remove "only platform" claim at `page.tsx:413`). |
| DOCBOT-2203 | Lead list | 2 | P0 | Y | ≥100 named contacts (≥40 fractional CFO / small firms) in tracker. |
| DOCBOT-2204 | 90-sec Loom | 2 | P0 | Y | Discrepancy visible by 0:45 (after 2002, or on current demo with caveat). |
| DOCBOT-2205 | Outreach waves | 5 | P0 | Y | 150 messages, ~20/weekday from Mon 10-26; reply rate logged. |
| DOCBOT-2206 | Discovery + demo calls | 5 | P0 | Y | ≥15 calls; pain, tool, hours/close, WTP logged. |
| DOCBOT-2207 | Design-partner onboarding | 2 | P0 | Y | ≥3 partners ran tie-out; only after security gate is live. |
| DOCBOT-2208 | LOI template + collection | 2 | P0 | Y | 1-page non-binding LOI ($1,000 list / $500 design partner); ≥3 signed. Template on D1. |
| DOCBOT-2209 | Draft application answers | 3 | P0 | Y | All questions drafted; numbers filled D9. |
| DOCBOT-2210 | Founder + demo videos | 2 | P0 | Y | 1-min founder; ≤3-min demo, tie-out framing. |
| DOCBOT-2211 | Review + submit | 2 | P0 | Y | ≥2 outside readers; submitted D9. |
| DOCBOT-2212 | Tie-out demo dataset | 3 | P0 | Y | One click loads draft-10-K PDF + trial-balance CSV; ≥3 planted mismatches; 0 false flags. |
| DOCBOT-2216 | Public self-serve launch | 2 | P0 | Y | Show HN / LinkedIn / finance communities; non-call contacts routed to signup. Only after 1614 + Stripe live. |
| DOCBOT-2217 | Traction snapshot | 1 | P0 | Y | Signups, verified, activated, paid, LOI MRR, calls — from 1706 + Stripe. |
| DOCBOT-2213 | Tie-out eval harness | 3 | P1 | S | ≥40 labeled lines; precision / recall / FP rate in `tests/eval/README.md`. |
| DOCBOT-2214 | Demo resilience | 2 | P1 | S | Embeddings warmed at startup; demo assets rebuilt after redeploy; first click <5s. |
| DOCBOT-2215 | Prod latency measurement | 1 | P2 | N | TTFT p50/p95 recorded. |

---

## 3. Parallel execution — lanes (Agents Orchestrator)

One branch per ticket (`feature/DOCBOT-XXXX-…`), merged to an integration branch per wave, then `main`. Lanes own files exclusively to avoid conflicts.

| Lane | Agent type | Owns (exclusive) | Tickets in order |
|---|---|---|---|
| **S — Security routes** | Security Engineer + Senior Dev | `api/index.py`, `api/rbac_service.py`, `api/utils/request_guard.py`, `api/demo_service.py` | 1610 → 1601 → 1611 → 1602 → 1605/1606 → 1616 → 1620 → 1623/1624 → 1711 (include_router wiring last) |
| **D — Data layer** | Backend Architect | `api/db_service.py`, `api/utils/ssrf_validator.py`, `api/utils/sql_validator.py`, `api/utils/llm_cache.py`, `api/connectors/shopify_connector.py` | 1603 → 1608 → 1604 → 2008 → 1602 (shopify part) |
| **A — Accounts** | Senior Developer | `api/auth_service.py`, `api/oauth_service.py`, `api/org_service.py`, `api/signup_service.py`, `api/routes/onboarding_routes.py` | 1702 → 1617 → 1713 → 1621 → 1622 → 1706 → 1707 |
| **B — Billing** | Senior Developer | `api/utils/billing_context.py`, `api/billing_models.py`, `api/usage_service.py`, `api/ledger_service.py`, `api/billing_service.py`, `api/routes/{billing,usage}_routes.py`, `api/utils/llm_provider.py`, `api/sandbox_service.py`, `api/llm_trace_service.py`, `requirements.txt` | 1800 → 1801 → 1607 (sandbox part) → 1802 → 1803 → 1804 → 1615 |
| **F — Formatting backend** | AI Engineer | `api/utils/response_format.py`, `api/personas.py`, `api/utils/pii_masking.py`, `api/hybrid_service.py`, `api/autopilot_service.py`, `api/utils/discrepancy_detector.py` | 2003 → 2001 → 2002 (backend) → 2004 → 2005 → 2212 |
| **U — Frontend** | Frontend Developer (+ UI Designer spec) | `src/components/**`, `src/hooks/**`, `src/app/{signup,verify,onboarding,settings,pricing}/**`, `src/lib/**`, `next.config.ts`, `package.json` | 2006 → 2002 (UI) → 1619 → 1712 → 1805 → 1811 → 1808 → 2202 (hero) |
| **X — Cross-cutting** | DevOps Automator / Security Engineer | `.github/workflows/**`, `eslint.config.*`, `Dockerfile`, `tests/unit/test_route_guards.py`, `index.py` Sentry block via Lane S handoff | 2302 → 1625 → 1618 → 2301 (Dockerfile workers) → 1613 → 1614 review |
| **R — Reliability** (EPIC-23) | AI Engineer | none exclusive — each fix goes through the owning lane's queue: 2307 → Lane B (`llm_provider`) + Lane F (`hybrid_service`); 2301 callsites → owning lanes; 2311 → Lane D (`vector_store.py` added to D); 2310 → Lane F; 2306 → Lane S; 2303/2305/2313/2304 → Lane U | triage + PRs per owner |

**Uncommitted WIP warning:** the Desktop checkout (`~/Desktop/Personal_Projects/DocBot`) has uncommitted changes in `index.py`, chat page, hooks and `package.json` (remark-math/KaTeX). Commit, stash with a tag, or discard **before** lanes branch off `main` — otherwise agents build on a tree that differs from what was audited.
| **Founder** | you | — | EPIC-22 GTM, Railway/Stripe/Resend/Turnstile/DNS setup, PR review + merge |

**Contracts first:** 1800 (Lane B) lands D1 so Lanes A/U build against shared Pydantic/Zod types.

---

## 4. Security gates (attach to every signup / billing ticket)

1. Route listed in route-guard manifest with guard + scope (or explicit `PUBLIC_ROUTES`).
2. Two-org isolation test: org B → 404 on org A ids.
3. Canary secret never appears in logs, Sentry, traces, cache rows, API responses.
4. Webhook: bad sig 400; replayed `event.id` no-op; out-of-order converges.
5. Quota: 50 concurrent at limit−1 → exactly one success.
6. Email verified before credits or live-DB connect.
7. Re-auth (≤15 min) before billing or API-key changes.

**Gate A (partner-safe, target D5):** 1610, 1612, 1601, 1611, 1603, 1602, 1607, 1608, 1605, 1606 in prod.
**Gate B (public signup + Stripe live, target D7):** Gate A + all remaining EPIC-16 `Y` + 1702, 1713, 1711, 1801, 1802, 1804 + 1614 sign-off.

---

## 5. Day-by-day

| Day | Date | Lanes (agents) | Founder |
|---|---|---|---|
| D1 | Sat 10-24 | S: 1610, 1612 · B: 1800 · F: 2003 · A: 1702 · X: 1625 | **Check `AUTH_REQUIRED`.** Start Stripe activation, Resend domain DNS, Turnstile site key. 2201, 2202, 2203 start, 2208 LOI template. |
| D2 | Sun 10-25 | S: 1601 · D: 1603 · A: 1617 · B: 1801 · F: 2001 · U: 2006 | 2203 done, 2204 Loom script, schedule Monday wave. Review/merge. |
| D3 | Mon 10-26 | S: 1611, 1602 · D: 1608 · A: 1713 · B: 1607, 1802 · F: 2002 · U: 2002 UI, 1619 | Outreach wave 1 (7–8am recipient TZ). Replies + booking. |
| D4 | Tue 10-27 | S: 1605/1606, 1616 · D: 1604, 2008 · A: 1621, 1622 · B: 1802 · F: 2004 · U: 1712 · X: 1618 | Outreach, calls. |
| D5 | Wed 10-28 | S: 1620, 1623/1624 · B: 1803 · F: 2005 · U: 1805 · X: 1613 → **Gate A deploy (partner-safe)** | Calls; 2207 partner onboarding opens. 2209 drafting. |
| D6 | Thu 10-29 | S: 1711 · B: 1804, 1615 · A: 1706, 1707 · F: 2212 · U: 1811, 1808 · Stripe E2E in test mode | Calls, partners, drafting. |
| D7 | Fri 10-30 | X: 1614 review + 1613 sweep · stretch: 2007, 2009, 2213, 1704 → **Gate B: Stripe live, `APP_AUTH_MODE=required`, launch (2216)** | Launch posts, calls. |
| D8 | Sat 10-31 | Launch bug fixes · stretch 2214, 2012 | 2210 videos; drafting done. |
| D9 | Sun 11-01 | **Deploy freeze from morning** | 2217 numbers, 2211 reviews, **SUBMIT**. |
| D10 | Mon 11-02 | Freeze | Buffer to 8pm PT. Keep calling. |

**Cut order if behind:** 2214 → 2213 → 2007/2009/2012 → 1704 → 1811 (use Stripe receipt + simple balance text) → 1804 (fallback: Stripe Payment Link + manual credit grant). If Gate A slips past D5, drop EPIC-18 to Payment Link entirely.
**Never cut:** Gate A tickets, 1702, 1713, 1617, 2002, 2003, 2205, 2206, 2208.

---

## 6. Points (script-verified — see §7)

| Epic | Total | In window (Y) | Stretch (S) | Deferred (N) |
|---|---|---|---|---|
| EPIC-16 | 64 | 57 | 0 | 7 |
| EPIC-17 | 27 | 15 | 3 | 9 |
| EPIC-18 | 35 | 24 | 0 | 11 |
| EPIC-19 | 18 | 0 | 0 | 18 |
| EPIC-20 | 37 | 17 | 8 | 12 |
| EPIC-23 | 32 | 21 | 9 | 2 |
| EPIC-21 | 470 | 0 | 0 | 470 |
| EPIC-22 | 40 | 34 | 5 | 1 |
| **All** | **723** | **168** | **25** | **530** |

---

## 7. After submit

1. EPIC-19 BYOK; remaining EPIC-16 (1609, 1626–1629); EPIC-17/18/20 `N` stories.
2. Re-plan EPIC-21 against shipped EPIC-17/18/19 overlap; resume `docbot-platform` at S2.
3. Tracker: add DOCBOT-1512..1532 + EPIC-16..22 to `docbot-v2-project-tracking.md`; Jira sync per merged ticket.
4. CLAUDE.md: tests 1,123; default LLM gpt-oss-20b; demo dataset QuickBite; palette = `globals.css` cyan/amber tokens; routes rule (F-4).

---

## 8. Post-Gate-A hardening (added 2026-11-08)

Same PR flow as code: branch `feature/DOCBOT-XXXX-…` off `origin/main` in its own worktree → tests green → PR into `main` → orchestrator merges. One ticket per PR. Owners: SD = senior-developer, DO = devops-automator.

| ID | Title | Owner | Pts | Pri | Status | Acceptance criteria | Refs |
|---|---|---|---|---|---|---|---|
| DOCBOT-2334 | Stripe webhook grants credits | SD | 3 | P0 | Done (PR #89) | `POST` webhook route in `api/routes/billing_routes.py`. Verifies `Stripe-Signature` (stripe lib or HMAC-SHA256 per Stripe docs, 300s tolerance) with `STRIPE_WEBHOOK_SECRET`; bad/missing/stale sig → 400, no grant. `checkout.session.completed` → `credit_service.grant_purchase(..., ref=<cs_… id>)`; replayed event → no second grant (`UNIQUE(kind, ref)`), 200. Calls `funnel_service.record_paid`. Other event types → 200 no-op. Exempt from CSRF/rate limiter for this path only. Unset secret → route returns 503, never grants. Unit tests: valid, bad sig, stale ts, replay, unknown event. Env var documented. | DOCBOT-1804 |
| DOCBOT-2335 | Deterministic segment-revenue tie-out | SD | 3 | P0 | Done (PR #88) | Root cause of 2/6 prod failures (1 SQL error, 1 180s hang) written in PR. Segment-revenue suggested question correct ≥10/10 local demo runs (log attached). Any remaining failure path ends in a bounded, user-visible error (no hang past the existing timeout); regression test covers the failing SQL case. | DOCBOT-2212 |
| DOCBOT-2336 | Login throttle keyed on (email, IP) | SD | 2 | P1 | Done (PR #91) | `request_guard.check_login_throttle` keys on (email, client IP) (same client-IP helper as Turnstile). Failures from IP A do not lock out the same email from IP B. Unit tests: same email two IPs; same IP repeats still throttled. | `api/utils/request_guard.py:608` |
| DOCBOT-2337 | CSP nonces, drop script `'unsafe-inline'` | SD | 5 | P1 | Done (PR #93) | `script-src` in `security-headers.mjs` has no `'unsafe-inline'`; per-request nonce from Next.js middleware (`'strict-dynamic'` + nonce). Turnstile, Vercel Analytics, Next runtime work. `next build && next start`: zero CSP violations in browser console on `/`, `/chat`, login + Turnstile. `style-src` unchanged. | `security-headers.mjs:7,72` |
| DOCBOT-2338 | Entrypoint fixes /data ownership, then drops root | DO | 2 | P1 | Done (PR #94) | Dockerfile entrypoint runs as root only to `chown` `/data` to `appuser`, then execs app via gosu/su-exec/setpriv; app process is non-root (`ps`/`id` check). Tested: `docker build` + run with a root-owned volume on `/data` → app writes succeed. PR notes founder can remove `RAILWAY_RUN_UID=0`. | `Dockerfile:36` |
| DOCBOT-2339 | Proxy `/v1/*` to Railway | DO | 1 | P2 | Done (PR #92) | `next.config.ts` rewrites `/v1/:path*` to backend exactly like `/api/:path*` (dev → 127.0.0.1:8000, prod → Railway). Preview deploy: `/v1/…` returns backend response, not Next 404. | `next.config.ts:34` |
| DOCBOT-2340 | Unblock pip Dependabot (#79) | DO | 1 | P2 | Done (PR #90) | Either pins adjusted so `pip install -r requirements.txt` + full CI suite pass with the group bump (pandas 3.1 vs `numpy==1.26.4`), or #79 closed and `.github/dependabot.yml` ignores pandas 3.x minor bumps. `pip install` never left broken on `main`. | PR #79 |
| TRACK-86 | AWS dev env PR #86 (base `dev`) | DO | — | P3 | Open (PR #86, parked) | Tracking only (EPIC-21 parked; founder decisions pending). No merge, no plan/apply, no AWS calls. Only review-nit fixes; `terraform fmt -check` + `terraform validate` stay green. | PR #86 |

**Points:** 17 (SD 13, DO 4). TRACK-86 unpointed.

**Status 2026-11-17:** all 7 tickets (17 pts) merged to `main` via PRs #88-#94. TRACK-86 is still open and parked under EPIC-21.
