# Investor Demo Sprint Board

**Sprint Goal:** Demo-stable by end of sprint
**Total Effort:** ~6-8 days | **Done:** 3 | **In Progress:** 1 | **Backlog:** 3

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
| **Status** | BACKLOG |
| **Assignee** | Unassigned |
| **Priority** | P2 |
| **Estimate** | 2-3 days |

**Description:** Amazon connector works but data isn't persisted. Need unified commerce tables with multi-tenant RLS.

**Definition of Done:**
- [ ] Commerce schema migrations created
- [ ] RLS policies enforce tenant isolation
- [ ] Amazon connector data persists to schema
- [ ] Query pipeline can read commerce tables

---

## TASK-06: page.tsx Under 800 Lines

| Field | Value |
|-------|-------|
| **Status** | IN PROGRESS |
| **Assignee** | Senior Developer |
| **Priority** | P2 |
| **Estimate** | 4-6h (remaining: ~2-3h) |

**Progress:**
- Chat page reduced from 2,426 → 1,888 lines via component extraction
- Extracted: `AuthModal.tsx` (175 lines), `AdminPanel.tsx` (228 lines), `personas.tsx` (185 lines)
- Remaining: sidebar (~567 lines) and handleSubmit (~490 lines)

**Definition of Done:**
- [ ] `page.tsx` (chat) under 800 lines
- [ ] All extracted components render correctly
- [ ] No TypeScript `any` types
- [ ] App builds without errors

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
| TASK-05 Commerce Schema | BACKLOG | P2 | 2-3 day effort |
| TASK-06 page.tsx Refactor | IN PROGRESS | P2 | 2426→1888, target 800 |
| TASK-07 Regression | BACKLOG | P0 gate | Depends on #1-#6 |

## Risk Register

| Risk | Impact | Status |
|------|--------|--------|
| Groq outage during demo | Critical | MITIGATED — Gemini fallback wired |
| PII leak in live demo | High | MITIGATED — all response paths masked |
| Landing page not at `/` | Medium | RESOLVED |
| page.tsx merge conflicts | Low | N/A — working on main |
| FinanceBench accuracy < 70% | Medium | OPEN — not yet run |
| Commerce schema slips | Low | ACCEPTED — connector demo-able without persistence |

## Exit Criteria (Demo-Ready)

1. ~~LLM fallback wired and tested (Groq down → Gemini serves)~~ DONE
2. ~~`/` shows investor landing page with CTA to `/chat`~~ DONE
3. ~~PII masking applied to all response paths~~ DONE
4. page.tsx under 800 lines (code quality signal) — IN PROGRESS
5. FinanceBench accuracy documented — BACKLOG
6. 85-test regression passes on prod — BACKLOG
