# DOCBOT-2721 BUG: Autopilot planner falls back to a docs-only single step; empty answer bubble
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | S0 | P0 | 2 | senior-dev | 2026-11-30 | Testing (2026-11-27) |

## User story
As a fractional CFO asking Autopilot to check a 10-K against my client's database, I want every run to query both the document and the database, and to always show me an answer or an error, so that a deep-dive never quietly skips my books or ends in a blank reply.

## Why now
Prod 2026-11-27, QuickBite demo (PDF + SQLite, has_db and has_docs true). Question: "Deep dive: check every quarter's net income in the 10-K against the database, investigate the root cause of any mismatch ..., then forecast Q1 2026 net income using the database figures." Via the UI twice: the plan was 1 step (the question verbatim) run as doc_search only. Run 1 ended with an empty assistant bubble, no toast; run 2 said the database returned nothing and missed the known Q4 $330M vs $325M break. The same request POSTed directly got a 5-step plan with doc_search and sql_query. Investor call 2026-12-01.

## Root cause
1. Planner token budget. The planner called the primary model (gpt-oss on Groq/Bedrock, a reasoning model) with max_tokens=400. Reasoning tokens count against that cap; on a long hybrid question the model sometimes spent all of it reasoning and returned no JSON. The single retry used the same cap, then the planner fell back. DOCBOT-2585 raised every other short-JSON caller to 1000 for this reason; the planner was the one caller left below 1000.
2. Fallback plan. The fallback was `[question]`. The tool router sends a step that mentions a "report" / "10-K" to doc_search, so a hybrid question never touched the database.
3. Empty bubble. The UI only fills the bubble on an `answer` event. If the stream closes without one (connection dropped, or an empty answer), the bubble stays empty, the Autopilot card closes and no toast fires. The synthesizer could also pass an empty LLM answer through on a non-degraded run.

## Scope
In:
1. Planner max_tokens 400 to 2000 (`_PLANNER_MAX_TOKENS`). No reasoning_effort: no caller or provider path sets it today.
2. Deterministic fallback plan from the resolved sources: doc_search step when docs exist; sql_query step when a DB exists (python_analysis for CSV). Each step is pinned to its tool (`plan_tools`), so the executor skips the router for it. `planner_fallback_single_step` stays as the degradation reason.
3. Synthesizer: empty LLM content uses the deterministic step-summary answer.
4. UI: a stream that ends without a non-empty answer replaces the empty bubble with an error line and shows a toast.

Out:
- Why the stream closed early in UI run 1 (not traced; possibly the proxy dropping a long stream). Autopilot has no answer-recovery poll like hybrid's DOCBOT-2636.

## Acceptance criteria
- [x] Given the planner returns empty / junk twice with docs and DB connected, then the plan has a doc_search step and a sql_query step and is labelled degraded.
- [x] Given a pinned fallback step, then the executor runs that tool without calling the router.
- [x] Given the planner call, then max_tokens >= 1000.
- [x] Given the synthesizer LLM returns "", then the answer is the step summary, not empty.
- [x] Given the stream ends without an answer, then the bubble shows an error and a toast fires (code path; no frontend test harness).

## Deliverable and approach
`api/autopilot_service.py`, `src/hooks/useChatSubmit.ts`. Tests: `tests/unit/test_autopilot_planner_fallback.py`.

## Test / verification
CI suite, `npx tsc --noEmit`, `npm run lint`. Live retest of the demo question on prod after deploy.

## Dependencies
Blocks: investor demo 2026-12-01.

## Definition of done
- [x] AC met
- [x] tests/CI or evidence
- [x] OpenProject ticket updated
- [x] tracker row updated
- [x] no secrets / no banned claims
