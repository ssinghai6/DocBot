# Decision: Multi-Agent Architecture for Autopilot (DOCBOT-1505)

**Date**: 2026-09-29
**Type**: Time-boxed evaluation spike — decision doc only, no production code
**Precedent**: mirrors the PageIndex evaluate-and-reject decision (EPIC-10, 2026-03-25, recorded in `project-tasks/docbot-v2-project-tracking.md`)

## Question

Should Autopilot's current single-agent LangGraph tool-orchestration evolve
into true multi-agent collaboration — specialized, persistent sub-agents
(e.g. a SQL-agent, a doc-agent) that hand off control and pass structured
findings between each other — or should the current architecture stay as-is?

## Decision

**Keep the current single-agent tool-orchestration architecture.** Do not
adopt agent-to-agent handoff. Revisit only if one of the trigger conditions
in "Revisit when" below actually occurs.

## What Autopilot actually is today (code-verified)

Read directly from `api/autopilot_service.py` (current `main`, includes
DOCBOT-1406's concurrent-wave dispatch):

- `_build_graph()` (line ~982): a 3-node `StateGraph` — `planner → executor
  → synthesizer`, with `executor` self-looping via a conditional edge
  (`_should_continue`) until the plan is exhausted or `MAX_ITERATIONS`/
  `TOTAL_TIMEOUT_S` is hit.
- **One shared state, no agent identity.** `AutopilotState` (TypedDict) is
  a single flat state object read and written by every node. There is no
  concept of "agent A's private context" vs. "agent B's private context" —
  everything lives in `steps_completed`, `citations`, etc., visible to
  whichever node runs next.
- **Tool selection is a heuristic, not an agent decision.**
  `_select_tool_heuristic()` (line 197) is plain keyword matching (verb
  detection like "fetch"/"query" → `sql_query`, chart keywords →
  `python_analysis`) — not an LLM call, not a reasoning agent choosing
  between tools. The planner LLM call only decomposes the question into
  step *strings*; a separate deterministic function then maps each string
  to a tool.
- **Tools are stateless function calls, not agents.** `sql_query` →
  `_collect_sql_result()` → `run_sql_pipeline()`. `doc_search` →
  `deep_retrieve()`. `python_analysis` → `generate_analysis_code()` +
  `run_python()`. None of these carry memory across steps beyond what's in
  the shared `AutopilotState`; each is a fresh, independent call.
- **DOCBOT-1406 added concurrency, not multi-agency.** The executor now
  dispatches an entire "wave" of independent steps via `asyncio.gather`
  when they have no data dependency on each other (`_next_wave_indices`).
  This is parallel execution of independent tool calls — there is still no
  negotiation, handoff, or agent-to-agent message passing between them.
- **No agent memory beyond conversation history.** Each `run_autopilot()`
  call is scoped to one investigation; there's no persistent "SQL agent"
  object that remembers past investigations or refines its own strategy
  session-over-session.

This is **single-agent tool orchestration** (a ReAct-style planner/executor
loop over typed tools), not multi-agent collaboration. The ticket's
ground-truth framing is accurate.

## What DocBot already has that a "specialist sub-agent" would supposedly add

The strongest case for multi-agent is "a dedicated SQL-agent could carry
its own retry/refinement loop distinct from a doc-agent's." Checking
whether that capability is actually missing:

| Capability a specialist agent would provide | Already exists, where |
|---|---|
| SQL-specific error recovery / iterative refinement | `db_service.run_sql_pipeline`'s schema-drift retry (invalidate cache → re-introspect → regenerate SQL → retry once) |
| Codegen-specific error recovery | `sandbox_service.generate_analysis_code`/`generate_csv_analysis_code`'s `error_context` corrective retry (feed the sandbox runtime error back to the LLM for one fix attempt) |
| Doc-search-specific refinement | `deep_research_service.deep_retrieve`'s sub-question decomposition + coverage-gap detection + gap-fill re-retrieval loop |
| Structured-output correctness | DOCBOT-1504 (this epic): Pydantic validation + one retry-with-feedback on the table-selector's JSON output |

Every "specialization" a hypothetical SQL-agent or doc-agent would bring is
**already implemented as a per-tool correction loop**, scoped to the
function that owns that data source. Wrapping these in an "agent" object
with a handoff protocol would mostly relabel existing behavior behind new
indirection (agent identity, a handoff message schema, coordination logic
to decide when to hand off), not add a capability the system lacks today.

## Analysis

### Arguments for keeping the current architecture

1. **Task shape fit.** Investigations are DAGs of independent, well-typed
   subtasks (fetch data → analyze/visualize) capped at `MAX_ITERATIONS`
   (5). A planner that decomposes into typed steps, routed to the right
   tool, is a good structural match. Multi-agent handoff earns its
   complexity at *much* higher step counts with genuinely conflicting
   sub-goals requiring negotiation — not five sequential/parallel fetches.
2. **Cost.** This is a solo-founder, cost-conscious deployment (see
   DOCBOT-1506 exact-match response cache and DOCBOT-1508 per-session cost
   ceiling, both in this same epic). Multi-agent handoff protocols
   typically cost *more* LLM round-trips per investigation (agent-to-agent
   negotiation, delegation acknowledgment, hand-back summarization) for a
   task shape that doesn't need the negotiation.
3. **Latency.** DocBot explicitly measures and markets TTFT / p95 latency
   (`tests/eval/eval_latency.py`, README "Investor Readiness" cost/latency
   framing). More agents in the loop is directly in tension with that.
4. **Maintainability for a solo developer.** A 3-node graph with typed
   tool dispatch is debuggable by reading one file. Agent-to-agent
   protocols introduce a new failure class (handoff loops, agents
   disagreeing about whose turn it is, partial-handoff state corruption)
   that's disproportionately expensive for one person to own and debug in
   production.
5. **No missing capability.** As shown above, the functional benefits
   specialist agents would bring (source-specific retry/refinement) are
   already present as targeted per-tool loops.

### Arguments for adopting specialist sub-agent handoff

1. **Context isolation.** Right now, doc-search retrieval and SQL results
   both compete for space in the same flat `steps_completed` list and the
   synthesizer's prompt. A dedicated doc-agent returning only a structured
   summary (not raw chunks) to a coordinator could reduce context bloat on
   large multi-source investigations.
2. **Per-source prompt specialization.** A persistent SQL-agent could
   carry a SQL-specific system prompt tuned differently from a doc-agent's,
   rather than sharing prompt-writing conventions across one planner.
3. **Extensibility as more connectors land.** EPIC-07/11 keep adding data
   sources (Amazon, Shopify, EDGAR). A per-source "agent" pattern could let
   each source own progressively more autonomous fetch-more-if-insufficient
   logic rather than extending one shared heuristic router indefinitely.
4. **Future step-count growth.** If investigations grow beyond ~5 steps
   with genuinely interdependent sub-goals (not just parallel independent
   fetches), a flatter shared-state graph may become harder to reason
   about than delegated sub-agents with narrower scopes.

### Why "keep as-is" wins today

The "for" case above is real but **speculative** — it describes problems
DocBot doesn't have yet (no evidence of context-bloat-driven quality
issues, no connector has needed autonomous multi-round fetching, no
investigation has needed >5 genuinely interdependent steps). The "against"
case (cost, latency, maintainability, no missing capability) is grounded
in what's actually measured and shipped today. Architecture investment
should follow evidence of a real limitation, not anticipate one — the same
standard applied to the PageIndex decision.

## Revisit when

- Investigations regularly need more than ~5–7 steps with genuinely
  **interdependent** sub-goals (not just more parallel independent
  fetches — DOCBOT-1406 already covers that case).
- A connector needs autonomous, multi-round fetch-and-evaluate behavior
  that the current heuristic tool router can't express without becoming
  unreadable (e.g. "keep pulling EDGAR filings and re-evaluate coverage
  until satisfied," which is a fundamentally different loop shape than
  "run this typed step once").
- Concrete evidence that shared-state context bloat degrades synthesis
  quality on real multi-source investigations (not a hypothetical).
- Cost/latency headroom exists to afford more LLM round-trips per
  investigation — meaning DOCBOT-1506 (response cache) and DOCBOT-1508
  (per-session cost ceiling) should land *first*, so adopting a more
  expensive pattern doesn't blow up spend before there's a guardrail.

## Marketing/copy reconciliation (AC #3)

Checked `README.md` and `src/app/page.tsx` for "multi-agent" claims:
**none found.** Existing copy already says "multi-step investigation
agent" (singular) and "Agentic Orchestration | LangGraph (StateGraph —
Planner -> Executor -> Synthesizer)" — both accurate descriptions of a
single-agent tool-orchestration system. No copy changes needed. Flagging
as a light guardrail: future copy should keep saying "multi-step agent" or
"agentic orchestration," not "multi-agent," until/unless this decision is
reversed.
