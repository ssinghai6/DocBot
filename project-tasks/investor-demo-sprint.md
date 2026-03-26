# Investor Demo Sprint Board

**Sprint Goal:** Demo-stable by end of sprint
**Total Effort:** ~6-8 days | **Done:** 5 | **In Progress:** 0 | **Backlog:** 2

## Dependency Graph

```
TASK-01 (LLM Fallback)     ──┐
TASK-02 (Landing Page)      ──┤
TASK-03 (PII Masking)       ──┼──► TASK-07 (85-Test Regression) ──► DEMO READY
TASK-04 (FinanceBench)      ──┤
TASK-05 (Commerce Schema)   ──┤
TASK-06 (page.tsx Refactor)  ──┘
```

Tasks #1-#6 are independent and parallelizable. Task #7 is the gate — depends on all others.

---

## TASK-01: Wire LLM Fallback into Prod Code Paths

| Field | Value |
|-------|-------|
| **Status** | DONE |
| **Assignee** | AI Engineer |
| **Priority** | P0 |
| **Completed** | 2026-03-26 |

**What shipped:**
- Enhanced `llm_provider.py` with `chat_completion()` and `chat_completion_stream()` — raw SDK drop-ins with Groq → Gemini fallback
- All 8 direct Groq callsites replaced across 6 service files:
  - `db_service.py` (3 callsites: table selector, SQL generator, answer streamer)
  - `deep_research_service.py` (1: `_get_llm()` → `get_llm()`)
  - `autopilot_service.py` (2: planner + synthesizer → `chat_completion()`)
  - `sandbox_service.py` (2: code gen → `chat_completion()` with `GROQ_CODE_MODEL`)
  - `hybrid_service.py` (1: synthesis streaming → `chat_completion_stream()`)
  - `index.py` (1: `ChatGroq()` → `get_llm()`)
- SSE streaming works through fallback path
- 503 tests passing

---

## TASK-02: Route Landing Page to /

| Field | Value |
|-------|-------|
| **Status** | DONE |
| **Assignee** | Senior Developer |
| **Priority** | P0 |
| **Completed** | 2026-03-26 |

**What shipped:**
- `/` serves the investor landing page
- `/chat` serves the chat app (moved from root)
- All CTA links updated to `/chat`
- Test count updated to 535+

---

## TASK-03: Fix PII Masking Gaps (CSV + LLM Output)

| Field | Value |
|-------|-------|
| **Status** | DONE |
| **Assignee** | Backend Architect |
| **Priority** | P1 |
| **Completed** | 2026-03-26 |

**What shipped:**
- PII masking wired at SSE response boundary in all streaming endpoints:
  - `index.py` main chat (per-chunk `mask_pii()`)
  - `hybrid_service.py` synthesis streaming
  - `deep_research_service.py` token emission
  - `db_service.py` answer streaming + `mask_rows()` on `result_preview`
  - `autopilot_service.py` final answer
- Sandbox result stdout masked via `mask_pii_dataframe_output()`
- Audit log `detail` field masked before persistence
- 503 tests passing

---

## TASK-04: Run FinanceBench + Document Accuracy

| Field | Value |
|-------|-------|
| **Status** | BACKLOG |
| **Assignee** | Unassigned |
| **Priority** | P2 |
| **Estimate** | 2-4h |

**Description:** Test suite written at `tests/external/test_financebench_accuracy.py` but never executed with live keys. Need a concrete accuracy number for investor pitch.

**Definition of Done:**
- [ ] All 20 FinanceBench questions run against live APIs
- [ ] Accuracy percentage documented
- [ ] Results captured in a report file

---

## TASK-05: DOCBOT-702 Commerce Schema + RLS

| Field | Value |
|-------|-------|
| **Status** | DONE |
| **Assignee** | Backend Architect |
| **Priority** | P2 |
| **Completed** | 2026-03-26 |

**What shipped:**
- `api/commerce_service.py` — unified commerce schema with 2 tables:
  - `commerce_orders` — normalized orders with connection_id isolation, upsert on conflict
  - `commerce_financials` — normalized financial periods with connection_id isolation
- Multi-tenant RLS: all query/persist functions require mandatory `connection_id` parameter — no cross-connection access path
- Persistence helpers: `persist_orders()` (PostgreSQL upsert), `persist_financials()`, `sync_connector_data()` orchestrator
- Query helpers: `query_orders()`, `query_financials()`, `get_order_count()` — all RLS-filtered
- 3 new API routes: `POST /api/connectors/{id}/sync`, `GET /api/commerce/{id}/orders`, `GET /api/commerce/{id}/financials`
- Wired into `index.py` init_db() + lifespan
- 31 unit tests in `tests/unit/test_commerce_service.py`
- 566 total tests passing

---

## TASK-06: page.tsx Under 800 Lines

| Field | Value |
|-------|-------|
| **Status** | DONE |
| **Assignee** | Senior Developer |
| **Priority** | P2 |
| **Completed** | 2026-03-26 |

**What shipped:**
- Chat page reduced from 2,426 → 512 lines (79% reduction, well under 800 target)
- Extracted 6 components + 2 custom hooks:
  - `Sidebar.tsx` (356 lines) — mobile toggle, auth widget, workspace, file upload, connections, personas
  - `ChatArea.tsx` (534 lines) — header, message list, progress indicators, input area
  - `AuthModal.tsx` (175 lines) — SSO/OAuth/email auth modal
  - `AdminPanel.tsx` (228 lines) — user management, audit log, metrics
  - `personas.tsx` (185 lines) — persona data and selector component
  - `useChatHandlers.ts` (576 lines) — all handler functions (DB, auth, file, export, etc.)
  - `useChatSubmit.ts` (463 lines) — handleSendMessage with all 4 chat paths
- page.tsx now contains only state declarations, useEffect hooks, and layout JSX

---

## TASK-07: 85-Test Manual Regression on Prod

| Field | Value |
|-------|-------|
| **Status** | BACKLOG |
| **Assignee** | Unassigned |
| **Priority** | P0 (gate) |
| **Estimate** | 4-6h |
| **Depends On** | TASK-01 through TASK-06 |

**Description:** Final gate for "demo-ready". No shortcuts. Must pass all 85 test scenarios on production.

**Definition of Done:**
- [ ] All 85 scenarios executed on prod deployment
- [ ] Zero P0/P1 failures
- [ ] Results documented with pass/fail counts

---

## Sprint Status Summary

| Task | Status | Priority | Notes |
|------|--------|----------|-------|
| TASK-01 LLM Fallback | DONE | P0 | 8/8 callsites wired |
| TASK-02 Landing Page | DONE | P0 | `/` → landing, `/chat` → app |
| TASK-03 PII Masking | DONE | P1 | All SSE + sandbox + audit masked |
| TASK-04 FinanceBench | BACKLOG | P2 | Needs live API keys |
| TASK-05 Commerce Schema | DONE | P2 | 2 tables, RLS, 31 tests, 3 API routes |
| TASK-06 page.tsx Refactor | DONE | P2 | 2426→512 lines (79% reduction) |
| TASK-07 Regression | BACKLOG | P0 gate | Depends on #1-#6 |

## Risk Register

| Risk | Impact | Status |
|------|--------|--------|
| Groq outage during demo | Critical | MITIGATED — Gemini fallback wired |
| PII leak in live demo | High | MITIGATED — all response paths masked |
| Landing page not at `/` | Medium | RESOLVED |
| page.tsx merge conflicts | Low | RESOLVED — refactor complete (512 lines) |
| FinanceBench accuracy < 70% | Medium | OPEN — not yet run |
| Commerce schema slips | Low | RESOLVED — commerce_service.py shipped with 31 tests |

## Exit Criteria (Demo-Ready)

1. ~~LLM fallback wired and tested (Groq down → Gemini serves)~~ DONE
2. ~~`/` shows investor landing page with CTA to `/chat`~~ DONE
3. ~~PII masking applied to all response paths~~ DONE
4. ~~page.tsx under 800 lines (code quality signal)~~ DONE (512 lines)
5. FinanceBench accuracy documented — BACKLOG
6. 85-test regression passes on prod — BACKLOG
