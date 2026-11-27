"""
DOCBOT-405: Analytical Autopilot — LangGraph-based multi-step investigation agent.
DOCBOT-1406: ExecutorNode now dispatches independent steps within a "wave"
concurrently instead of one step per LangGraph iteration (see _next_wave_indices).

Exposes a single async generator, run_autopilot(), that:
  - Plans an investigation into ≤5 steps via Groq Llama (PlannerNode)
  - Executes each step using the best tool: sql_query | doc_search | python_analysis
    (ExecutorNode, loops up to MAX_ITERATIONS *waves*, hard 90-second wall-clock guard).
    Consecutive data-fetch steps (sql_query/doc_search) have no ordering dependency
    on each other and run concurrently via asyncio.gather in a single wave.
    python_analysis steps read results accumulated in state["steps_completed"] from
    earlier steps (prior SQL rows / doc context), so they form a barrier: a
    python_analysis step never runs in the same wave as an unexecuted fetch step
    that precedes it in the plan, but consecutive python_analysis steps (which
    don't depend on each other, only on already-completed prior waves) do run
    concurrently with each other.
  - Synthesises all step findings into a final markdown answer (SynthesizerNode)
  - DOCBOT-1518: verifies the synthesized answer against the actual retrieved
    step data (VerifierNode) and triggers at most one corrective re-synthesis
    if a real numeric/factual contradiction is found — the "answer" SSE event
    and the persisted trace row both reflect this post-verification answer.
  - Yields SSE-formatted strings throughout for direct client streaming

SSE event types yielded:
  {type: "plan",    steps: [...], content: "N steps planned"}
  {type: "step",    step_num, tool, step_label, summary, content, artifact_id?, chart_b64?, error?}
                     — summary: short deterministic line (DOCBOT-2005);
                       content: cleaned raw tool output for the "details" toggle
  {type: "answer",  content: "<markdown answer>"}
  {type: "done",    citations: [...]}
  {type: "warning", content: "..."}  — non-fatal issues
  {type: "error",   content: "..."}  — fatal abort
"""

from __future__ import annotations

from api.utils.doc_refs import chunk_header, chunk_key, chunk_label, chunk_page  # DOCBOT-2637
from api.utils.yoy_facts import YOY_RULE, yoy_prompt_block  # DOCBOT-2637
import asyncio
import json
import logging
import operator
import os
import re
import time
from typing import Annotated, Any, AsyncGenerator, Optional

from typing_extensions import TypedDict

from api.utils.response_format import (
    SOURCE_CSV,
    SOURCE_DB,
    SOURCE_DOC,
    build_format_block,
)
from api.utils.sandbox_paths import sandbox_csv_path

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5  # cap on executor *waves*, not individual steps — see module docstring
# DOCBOT-2310: overall wall-clock budget; on expiry the synthesizer still
# answers from the steps finished so far. Each tool step gets its own budget.
TOTAL_TIMEOUT_S = float(os.getenv("AUTOPILOT_TIMEOUT_SECONDS", "90"))
STEP_TIMEOUT_S = float(os.getenv("AUTOPILOT_STEP_TIMEOUT_SECONDS", "45"))
_LIST_STATE_KEYS = ("steps_completed", "citations", "degradation_reasons")

# DOCBOT-1507: both prompts below are built dynamically per-request (tools
# list, data-fetch guidance), so each gets a version tag covering its
# prompt-assembly logic rather than a single static string constant. Bump
# manually when the wording/instructions change — see PROMPT_VERSION
# convention in api/utils/llm_provider.py.
PROMPT_VERSION_AUTOPILOT_PLANNER = "v1"
PROMPT_VERSION_AUTOPILOT_SYNTHESIZER = "v2"

# Internal step-result prefixes that signal a failed/empty step. These are
# scrubbed from user-facing output and replaced with a calm generic message —
# visitors should never see raw tracebacks or internal failure strings.
from api.utils.step_summary import (  # noqa: E402
    FAILURE_CONTENT_PREFIXES as _FAILURE_CONTENT_PREFIXES,
    build_step_summary,
    clean_step_details,
)
from api.utils.exceptions import safe_error_message  # noqa: E402

# P1: machine-readable degradation reason codes. The planner and executor
# append these to state["degradation_reasons"]; a non-empty set forces the
# synthesizer to label the final answer as degraded (P1c) so a LangGraph run
# can never silently collapse to a one-step / doc-only investigation.
_DEGR_PLANNER_FALLBACK = "planner_fallback_single_step"
_DEGR_DATA_TOOL_MISSING = "data_tool_missing"

# P1: the planner makes at most ONE bounded regeneration call (with the
# validation error appended to the prompt) when its first output fails the
# plan schema check. After that it falls back — visibly, never silently.
_PLANNER_REGENERATION_LIMIT = 1

# DOCBOT-2721: the primary model (gpt-oss on Groq/Bedrock) is a reasoning model
# whose reasoning tokens count against max_tokens. At 400 it intermittently
# spent the whole budget reasoning on hard hybrid questions, returned no JSON,
# failed the retry the same way and fell back. Same fix DOCBOT-2585 applied to
# the other short-JSON callers (raised to 1000); the planner prompt is longer.
_PLANNER_MAX_TOKENS = 2000

# P3(a): lowercase claim fragments that read as the answer apologising for
# missing data. Only actionable when data_results actually holds data —
# otherwise the guard leaves the answer untouched.
_APOLOGY_CLAIM_PATTERNS = (
    "no numeric data",
    "no numbers",
    "no data",
    "no available data",
    "data unavailable",
    "data is unavailable",
    "no quantitative",
    "insufficient data",
    "unable to compute",
    "cannot compute",
    "could not compute",
    "not enough data",
    "no relevant data",
)

# P3: defensive scrub for DB-URL-with-password fragments that could ride
# along in a tool error message surfaced through data summaries / refusal
# detail lines. Connection strings must never leak into surfaced output.
_DB_URL_SECRET_RE = re.compile(
    r"(?P<scheme>postgres(?:ql)?|mysql|mariadb|sqlite|azure|sqlserver|mssql|"
    r"oracle|db2|redshift|snowflake)://[^:/\s]+:[^@\s]+@",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


def _merge_dict_reducer(current: Optional[dict], update: Optional[dict]) -> dict:
    """LangGraph reducer for ``data_results`` — merges per-wave updates (each
    executor wave contributes its own "step:<plan index>:<tool>" entries)
    into one accumulated dict instead of replacing it."""
    merged = dict(current or {})
    if isinstance(update, dict):
        merged.update(update)
    return merged


class AutopilotState(TypedDict):
    question: str
    session_id: str
    connection_id: str
    persona: str
    # DOCBOT-1501: shared trace id for every LLM call made in this investigation
    run_id: str
    # steps from PlannerNode
    plan: list[str]
    # accumulated step results — operator.add reducer: each executor call appends
    # the results of one *wave* (one or more concurrently-run steps), always in
    # plan order regardless of which step finished first within the wave. Its
    # length also doubles as the "how many plan steps are done" pointer.
    steps_completed: Annotated[list[dict], operator.add]
    # number of executor *waves* dispatched so far (0-based) — NOT the number of
    # individual steps, since a wave may run several steps concurrently.
    iteration: int
    # final synthesised markdown answer
    final_answer: str
    # citations accumulated across doc_search steps
    citations: Annotated[list[dict], operator.add]
    # set True when the wall-clock guard fires
    timed_out: bool
    # DOCBOT-1508: set True when the session's soft LLM cost ceiling was hit
    # mid-investigation — the executor loop stops dispatching new waves and
    # the synthesizer still runs on whatever steps_completed has so far.
    budget_exceeded: bool
    # data-source availability flags
    has_docs: bool
    has_db: bool
    has_csv: bool
    # P2: normalized data outputs from executed tool steps, keyed by
    # "step:<plan index>:<tool>". Each value carries {tool, step_label,
    # row_count, sample_rows (max 10, deterministic), sql, explanation,
    # result_excerpt, provenance}. Consumed by the synthesizer and the P3
    # claim guard so the final answer can never claim "no numbers" while
    # tool data exists.
    data_results: Annotated[dict[str, Any], _merge_dict_reducer]
    # P1: degradation reason codes accumulated across planner/executor
    # nodes (e.g. "planner_fallback_single_step", "data_tool_missing:step:2:sql_query").
    # Non-empty → the synthesizer labels the answer degraded.
    degradation_reasons: Annotated[list[str], operator.add]
    # DOCBOT-2721: step text -> tool pinned by the planner's deterministic
    # fallback plan, so the executor skips the router for those steps.
    plan_tools: dict[str, str]


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, default=str)}\n\n"


# ---------------------------------------------------------------------------
# PlannerNode
# ---------------------------------------------------------------------------


def _fallback_plan(state: AutopilotState) -> dict:
    """DOCBOT-2721: deterministic plan used when the LLM planner fails.

    One fetch step per connected source, each pinned to its tool, so a hybrid
    question never silently drops the database (the old fallback was the raw
    question as one step, which the router sent to doc_search only). The
    synthesizer still runs over whatever these steps return.
    """
    q = state["question"]
    steps: dict[str, str] = {}
    if state.get("has_docs"):
        steps[f"Search the uploaded documents for: {q}"] = "doc_search"
    if state.get("has_csv"):
        steps[f"Analyse the uploaded CSV data to answer: {q}"] = "python_analysis"
    elif state.get("has_db"):
        steps[f"Query the database for the figures needed to answer: {q}"] = "sql_query"
    if not steps:
        # No source: the executor turns this into a clean "unsupported" refusal.
        return {"plan": [q], "plan_tools": {}}
    return {"plan": list(steps), "plan_tools": steps}


async def _planner_node(state: AutopilotState) -> dict:
    """Decompose the question into ≤5 investigation steps via Groq Llama.

    P1 hardening:
    - The model's plan output is schema-validated (a JSON array of non-empty
      step strings, capped at MAX_ITERATIONS). On failure the planner makes
      at most ONE bounded regeneration call with the validation error
      appended to the prompt, then falls back to a single-step plan.
    - Every fallback path records ``planner_fallback_single_step`` in
      ``degradation_reasons`` so the synthesizer labels the final answer
      degraded (P1c) — the run can never *silently* collapse to a one-step
      (or doc-only) investigation.
    """
    from api.utils.llm_provider import llm_configured
    if not llm_configured():
        logger.warning("planner_node: no LLM key set — source fallback plan (degraded)")
        return {
            **_fallback_plan(state),
            "iteration": 0,
            "degradation_reasons": [_DEGR_PLANNER_FALLBACK],
        }

    # Build dynamic tools list based on available data sources
    tools_available: list[str] = []
    if state.get("has_db"):
        tools_available.append("  - sql_query: run a SQL query against the connected database")
    if state.get("has_docs"):
        tools_available.append("  - doc_search: search uploaded PDF documents for relevant information")
    tools_available.append("  - python_analysis: run Python / generate charts — ONLY after a data-fetch step has provided data")

    # P1: never invent a doc_search tool line when no sources are resolved —
    # the executor would refuse the step anyway ("unsupported"), so the
    # planner should plan honestly for that outcome instead of steering the
    # whole investigation into doc-only wording.
    tools_str = "\n".join(tools_available)

    # DOCBOT-1517: cheap ontology nearest-neighbor pre-check — if a data
    # ontology has already been built for this connection, bias the planner
    # with a short "relevant tables" hint so its sql_query steps are more
    # likely to name real tables/columns. Strictly additive: ontology_lookup()
    # returning None (no ontology built yet) means `ontology_hint` stays "",
    # so the prompt is byte-for-byte identical to before this ticket.
    ontology_hint = ""
    if state.get("has_db") and state.get("connection_id"):
        try:
            from api import ontology_service
            _ontology_hits = await ontology_service.ontology_lookup(
                state["connection_id"], state["question"], top_k=5
            )
        except Exception:
            _ontology_hits = None
        if _ontology_hits:
            _lines = []
            for hit in _ontology_hits:
                _cols = ", ".join(c.get("name", "") for c in (hit.get("column_metadata") or [])[:8])
                _lines.append(f"  - {hit['table_name']}: {_cols}")
            ontology_hint = (
                "\n\nRelevant tables for this question (from the connection's data ontology, "
                "for reference only — confirm against the real schema during execution):\n"
                + "\n".join(_lines) + "\n"
            )

    # DOCBOT-2911: cross-source column mappings for the org (non-fatal; ""
    # unless the org has >= 2 connected sources).
    from api.cross_source_service import cross_source_context
    from api.utils.billing_context import get_principal

    _xs_principal = get_principal()
    _xsource_block, _ = await cross_source_context(
        _xs_principal.org_id if _xs_principal else None, state["question"], state.get("connection_id")
    )
    ontology_hint += _xsource_block

    # Build data-fetch guidance based on available sources
    if state.get("has_db"):
        fetch_guidance = (
            "1. If the question asks for a chart, heatmap, plot, or visualisation, the step BEFORE it "
            "MUST be a sql_query step that fetches the required data. Never start with python_analysis.\n"
            "2. Write each step as a clear action verb phrase, e.g. 'Fetch revenue by region for Q3.' "
            "or 'Generate a heatmap of the correlation matrix.' — NOT 'Query data for heatmap generation'.\n"
            "3. For chart/visualisation questions with no other analysis needed, use exactly 2 steps: "
            "first a sql_query fetch step, then a python_analysis visualisation step."
        )
    elif state.get("has_docs"):
        fetch_guidance = (
            "1. Use doc_search steps to retrieve information from the uploaded documents.\n"
            "2. Write each step as a clear action verb phrase, e.g. 'Search for revenue figures in the report.' "
            "or 'Find the risk factors section in the filing.'\n"
            "3. python_analysis can be used AFTER a doc_search step to further process or visualise the extracted data."
        )
    else:
        fetch_guidance = (
            "1. Write each step as a clear action verb phrase.\n"
            "2. Use python_analysis for computation and visualisation tasks."
        )

    system_prompt = (
        "You are a senior data analyst. Decompose the business question into a list of "
        "≤5 concrete investigation steps. Each step must be answerable with ONE tool:\n"
        f"{tools_str}\n\n"
        "CRITICAL RULES:\n"
        f"{fetch_guidance}\n"
        f"{ontology_hint}\n"
        "Return ONLY a JSON array of short step strings (no keys, no explanation), e.g.:\n"
        '["Fetch total revenue by region for Q3.", "Identify top 3 products by revenue.", '
        '"Generate a bar chart of revenue trend."]'
    )

    def _parse_plan_response(raw: str) -> list[str]:
        """Normalize + schema-validate a planner response. Raises ValueError
        on any invalid tool map (not JSON, not an array, empty) so the caller
        can trigger its bounded regeneration."""
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
        parsed = json.loads(stripped)
        if not isinstance(parsed, list):
            raise ValueError("Plan is not a JSON array")
        plan = [str(s).strip() for s in parsed if str(s).strip()]
        if not plan:
            raise ValueError("Plan array is empty")
        return plan[:MAX_ITERATIONS]

    def _attempt(error_context: Optional[str]) -> str:
        from api.utils.llm_provider import chat_completion

        user_content = f"Question: {state['question']}"
        if error_context:
            user_content += (
                "\n\nYour previous response was invalid: "
                f"{error_context}\nReturn ONLY a corrected JSON array of short step strings. "
                "No explanation, no markdown fences."
            )
        return chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=_PLANNER_MAX_TOKENS,
            caller="autopilot_planner",
            run_id=state.get("run_id"),
            prompt_version=PROMPT_VERSION_AUTOPILOT_PLANNER,
        )

    degradation_reasons: list[str] = []
    plan_update: dict = {}
    try:
        plan_update = {"plan": _parse_plan_response(await asyncio.to_thread(_attempt, None))}
    except Exception as first_exc:
        # P1: bounded retry — exactly one regeneration call with the
        # validation error appended, then a visible (non-silent) fallback.
        logger.warning(
            "planner_node: plan schema validation failed (%s) — retrying once",
            first_exc,
        )
        for _attempt_no in range(_PLANNER_REGENERATION_LIMIT):
            try:
                plan_update = {"plan": _parse_plan_response(
                    await asyncio.to_thread(_attempt, f"Attempt failed: {first_exc}")
                )}
                break
            except Exception as retry_exc:
                logger.warning(
                    "planner_node: regeneration attempt %d failed (%s)",
                    _attempt_no + 1, retry_exc,
                )
        else:
            logger.warning("planner_node: planning failed — source fallback plan (degraded)")
            plan_update = _fallback_plan(state)
            degradation_reasons.append(_DEGR_PLANNER_FALLBACK)

    return {
        **plan_update,
        "iteration": 0,
        "degradation_reasons": degradation_reasons,
    }


# ---------------------------------------------------------------------------
# Tool selection heuristic
# ---------------------------------------------------------------------------


def _select_tool_heuristic(step: str, has_db: bool = True, has_docs: bool = False, has_csv: bool = False) -> str:
    """Choose sql_query | doc_search | python_analysis | unsupported based on step wording.

    Data-fetch verbs (query/fetch/get/retrieve/select/find) always win over
    visualisation keywords so "Fetch data for heatmap" stays sql_query.

    When ``has_csv`` is True, CSV data is analysed via pandas in E2B sandbox,
    so data-fetch steps route to ``"python_analysis"`` instead of ``"sql_query"``.

    When ``has_db`` is False the function never returns ``"sql_query"`` —
    data-fetch verbs fall back to ``"doc_search"`` (if docs available) or
    ``"python_analysis"``.

    Returns ``"unsupported"`` when no data source is available (no DB, no docs,
    no CSV). Autopilot cannot fetch external data, so steps that need data
    cannot be executed — the executor surfaces this as a refusal.
    """
    if not has_db and not has_docs and not has_csv:
        return "unsupported"
    s = step.lower()
    # Data-fetch verbs take priority — these are SQL steps even if chart words present
    DATA_FETCH = ("fetch ", "retrieve ", "get ", "select ", "find ", "query ", "calculate ",
                  "compute ", "count ", "sum ", "aggregate ", "identify ")
    PYTHON_KEYWORDS = ("chart", "plot", "visuali", "correlat", "distribut",
                       "python", "scatter", "heatmap", "regression", "generate a ",
                       "create a ", "draw ", "predict", "forecast", "model", "seasonal",
                       "trend", "time series", "anomal", "cluster")
    DOC_KEYWORDS = ("document", "report", "pdf", "file", "upload",
                    "manual", "policy", "contract", "according to")

    has_chart = any(k in s for k in PYTHON_KEYWORDS)
    has_fetch = any(s.startswith(k) or f" {k}" in s for k in DATA_FETCH)

    # CSV connections always use python_analysis (pandas sandbox), never sql_query
    if has_csv and not any(k in s for k in DOC_KEYWORDS):
        return "python_analysis"

    if has_chart and not has_fetch:
        return "python_analysis"
    if any(k in s for k in DOC_KEYWORDS):
        return "doc_search"

    # Data-fetch step — route to sql_query if DB available, else doc_search or python
    if has_fetch or not has_chart:
        if has_db and not has_csv:
            return "sql_query"
        if has_csv:
            return "python_analysis"
        if has_docs:
            return "doc_search"
        return "python_analysis"

    if has_db and not has_csv:
        return "sql_query"
    return "doc_search" if has_docs else "python_analysis"


# DOCBOT-1513: the exact tool-name strings _select_tool_heuristic (and its
# model-driven replacement below) may return — kept in sync with
# api/tools/builtin.py's ToolSpec registrations, which cite this set.
_VALID_TOOL_SELECTIONS = {"sql_query", "doc_search", "python_analysis", "unsupported"}


def _tool_available_for_sources(
    tool: str, *, has_db: bool, has_docs: bool, has_csv: bool
) -> bool:
    """P1: True when ``tool`` can actually run given the resolved source flags.

    ``_select_tool_llm`` validates membership in ``_VALID_TOOL_SELECTIONS``
    but NOT availability — an LLM could return ``sql_query`` with no database
    connected (doc-only degradation) or ``unsupported`` with sources present.
    The executor calls this before advancing; on an invalid selection it
    retries ONCE (bounded, error appended), then deterministically re-routes
    with the flag-aware heuristic, which can never produce an unsupportable
    tool.
    """
    any_source = has_db or has_docs or has_csv
    if tool == "sql_query":
        return has_db and not has_csv
    if tool == "doc_search":
        return has_docs
    if tool == "python_analysis":
        return any_source
    if tool == "unsupported":
        return not any_source
    return False

# DOCBOT-1507: bump manually ("v1" -> "v2") whenever the system prompt below
# changes — see PROMPT_VERSION convention in api/utils/llm_provider.py.
PROMPT_VERSION_AUTOPILOT_TOOL_ROUTER = "v1"


async def _select_tool_llm(
    step: str,
    has_db: bool = True,
    has_docs: bool = False,
    has_csv: bool = False,
    persona_tool_preference: str | None = None,
    run_id: str | None = None,
    error_context: str | None = None,
) -> str:
    """Model-driven replacement for ``_select_tool_heuristic`` — DOCBOT-1513.

    Makes ONE structured-JSON Groq call (cheap general model, not the code
    model) to pick exactly one of ``sql_query`` / ``doc_search`` /
    ``python_analysis`` / ``unsupported`` for a single investigation step,
    given which data sources are available and an optional persona
    tool-preference hint (``EXPERT_PERSONAS[persona]["tool_preference"]`` —
    ``"balanced"`` / ``"rag_first"`` / ``"sql_first"``).

    ``error_context`` (P1) folds a previous selection's validation failure
    into the prompt so the executor's bounded one-retry loop can ask for a
    corrected tool map instead of a blind repeat.

    On ANY failure — the LLM call raising, the response not parsing as JSON,
    or the parsed value falling outside ``_VALID_TOOL_SELECTIONS`` — falls
    back to ``_select_tool_heuristic`` unchanged, so a transient LLM outage
    never breaks step execution. This function never raises.
    """
    if not has_db and not has_docs and not has_csv:
        return "unsupported"

    available: list[str] = []
    if has_db and not has_csv:
        available.append("sql_query — run a SQL query against the connected database")
    if has_docs:
        available.append("doc_search — search uploaded PDF documents")
    if has_csv or has_db or has_docs:
        available.append(
            "python_analysis — run Python/pandas (charts, forecasting, stats) on data "
            "already fetched by a prior step, or directly on an uploaded CSV"
        )
    if not available:
        available.append("python_analysis — run Python/pandas analysis")

    tools_str = "\n".join(f"  - {a}" for a in available)
    pref_hint = (
        f"\nThe active persona's soft tool preference is \"{persona_tool_preference}\" "
        "— use it only as a light tie-breaker, never override an obvious data-fetch vs. "
        "analysis distinction because of it."
        if persona_tool_preference
        else ""
    )

    system_prompt = (
        "You are a tool router for a multi-step AI investigation agent. Given ONE "
        "investigation step, choose exactly one tool to execute it, from:\n"
        f"{tools_str}\n"
        f"{pref_hint}\n\n"
        "RULES:\n"
        "- A step that fetches/queries/retrieves/counts/aggregates data should use "
        "sql_query (when a database is available) even if it also mentions a chart — "
        "the chart itself is a later step.\n"
        "- Use python_analysis only for a step that itself asks to visualise, plot, "
        "forecast, model, or compute on data a prior step already fetched (or any step "
        "at all when the only source is an uploaded CSV).\n"
        "- Use doc_search for steps about documents, reports, PDFs, contracts, filings, "
        "or policies.\n\n"
        'Respond with ONLY a JSON object, e.g. {"tool": "sql_query"}. No explanation, '
        "no markdown fences."
    )

    user_content = f"Step: {step}"
    if error_context:
        user_content += (
            "\n\nYour previous selection was invalid: "
            f"{error_context}\nRespond with ONLY a corrected JSON object "
            'like {"tool": "<one of the available tools>"}.'
        )

    try:
        # DOCBOT-1513: use the async call_llm (LangChain .ainvoke under the
        # hood), NOT the synchronous chat_completion — _select_tool_llm runs
        # concurrently for every step in a wave via asyncio.gather (see
        # executor_node), and a blocking sync HTTP call here would serialize
        # the whole wave on the event loop, silently defeating DOCBOT-1406's
        # concurrent-wave dispatch.
        from api.utils.llm_provider import call_llm

        raw = await call_llm(
            f"{system_prompt}\n\n{user_content}",
            temperature=0,
            caller="autopilot_tool_router",
            run_id=run_id,
            prompt_version=PROMPT_VERSION_AUTOPILOT_TOOL_ROUTER,
        )
        # The model may wrap the JSON in markdown fences or add stray text —
        # extract the first {...} object rather than assuming the whole
        # response is bare JSON.
        start_idx = raw.index("{")
        end_idx = raw.rindex("}") + 1
        parsed = json.loads(raw[start_idx:end_idx])
        tool = str(parsed.get("tool", "")).strip()
        if tool not in _VALID_TOOL_SELECTIONS:
            raise ValueError(f"LLM returned an out-of-set tool: {tool!r}")
        return tool
    except Exception as exc:
        logger.warning(
            "_select_tool_llm failed (%s) — falling back to heuristic for step: %s",
            exc, step[:80],
        )
        return _select_tool_heuristic(step, has_db=has_db, has_docs=has_docs, has_csv=has_csv)


# ---------------------------------------------------------------------------
# Wave grouping — decide which plan steps can run concurrently
# ---------------------------------------------------------------------------

# Tool categories that never read another step's output. sql_query and
# doc_search each fetch from an external source (DB / vector store) using only
# the step's own text — they have no data dependency on sibling steps.
_FETCH_TOOLS = ("sql_query", "doc_search")


def _wave_tool_hint(step: str, state: AutopilotState) -> str:
    """Cheap, side-effect-free tool guess used only to group steps into waves.

    This intentionally skips the runtime "demote python_analysis if no prior
    data exists" logic in ``_run_single_step`` — that check depends on the
    *results* of earlier steps, which aren't known until the barrier those
    results imply has already been respected by wave grouping below. Wave
    grouping only needs the fetch vs. non-fetch category, not the exact tool.

    DOCBOT-1513: deliberately still uses ``_select_tool_heuristic`` rather
    than the model-driven ``_select_tool_llm`` — this function's only job is
    a *fetch-vs-non-fetch* grouping guess, called once per step on every wave
    boundary, and its own docstring already commits to being "cheap,
    side-effect-free". ``_run_single_step`` makes the authoritative,
    model-driven tool choice for each step right before executing it (see
    below), so a wrong grouping guess here only risks suboptimal
    concurrency batching, never an incorrect tool execution.
    """
    return _select_tool_heuristic(
        step,
        has_db=state.get("has_db", True),
        has_docs=state.get("has_docs", False),
        has_csv=state.get("has_csv", False),
    )


def _next_wave_indices(plan: list[str], start: int, state: AutopilotState) -> list[int]:
    """Return the plan indices (from ``start``) that can be dispatched together.

    A wave is a maximal run of consecutive steps that are all fetch-type
    (sql_query/doc_search — no inter-step dependency) or all non-fetch
    (python_analysis/unsupported — each may read every step completed in an
    earlier wave, but not a sibling in the same wave). This keeps a
    python_analysis step from ever running before a fetch step that precedes
    it in the plan, while still letting independent fetches — or independent
    analyses that both only read already-completed data — run concurrently.
    """
    if start >= len(plan):
        return []

    is_fetch = _wave_tool_hint(plan[start], state) in _FETCH_TOOLS
    indices = [start]
    i = start + 1
    while i < len(plan):
        step_is_fetch = _wave_tool_hint(plan[i], state) in _FETCH_TOOLS
        if step_is_fetch != is_fetch:
            break
        indices.append(i)
        i += 1
    return indices


# ---------------------------------------------------------------------------
# ExecutorNode factory
# ---------------------------------------------------------------------------


async def _persist_analysis_artifact(
    *,
    session_id: str,
    step_idx: int,
    step: str,
    output: str,
    session_artifacts_table: Any,
    async_session_factory: Any,
) -> Optional[str]:
    """P2: persist a python/CSV analysis step's visible stdout as a session
    artifact so the run's visible outputs survive the turn. Charts were
    already persisted at the sandbox call sites; bare-text analysis outputs
    were not. Non-fatal: any failure logs and returns None (save_artifact
    itself is also failure-tolerant internally)."""
    if session_artifacts_table is None or not (output or "").strip():
        return None
    try:
        from api.artifact_service import save_artifact

        artifact_id = await save_artifact(
            session_id=session_id,
            turn_id=step_idx + 1,
            artifact_type="analysis",
            name=f"Analysis: {step[:60]}",
            result_dicts=[{"output": output[:2000]}],
            session_artifacts_table=session_artifacts_table,
            async_session_factory=async_session_factory,
        )
        return artifact_id
    except Exception as exc:
        logger.warning(
            "_persist_analysis_artifact failed (non-fatal) for step '%s': %s",
            step[:60], exc,
        )
        return None


def _sql_data_result(
    *, step: str, row_count: int, preview: Any, sql: Any, explanation: Any, artifact_id: Optional[str]
) -> dict:
    """P2: build the normalized data_results entry for a sql_query step."""
    sample_rows: list = list(preview)[:10] if isinstance(preview, list) else []
    return {
        "tool": "sql_query",
        "step_label": step,
        "row_count": row_count,
        "sample_rows": sample_rows,
        "sql": sql if isinstance(sql, str) else None,
        "explanation": explanation if isinstance(explanation, str) else None,
        "provenance": "sql_pipeline",
        "artifact_id": artifact_id,
    }


def _python_data_result(
    *,
    step: str,
    provenance: str,
    result_excerpt: str,
    row_count: Optional[int] = None,
    artifact_id: Optional[str] = None,
) -> dict:
    """P2: build the normalized data_results entry for a python_analysis step."""
    return {
        "tool": "python_analysis",
        "step_label": step,
        "row_count": row_count,
        "sample_rows": [],
        "sql": None,
        "explanation": None,
        "result_excerpt": result_excerpt,
        "provenance": provenance,
        "artifact_id": artifact_id,
    }


def make_executor_node(
    *,
    db_connections_table: Any,
    schema_cache_table: Any,
    query_history_table: Any,
    query_embeddings_table: Any,
    session_artifacts_table: Any,
    table_embeddings_table: Any,
    async_session_factory: Any,
    expert_personas: dict,
    vector_stores: dict,
):
    """Return an async executor node with all DB tables captured via closure."""

    async def _run_single_step(step_idx: int, step: str, state: AutopilotState) -> tuple[dict, list[dict]]:
        """Execute one plan step and return (result_entry, new_citations).

        Never raises — every branch below is wrapped in the try/except at the
        bottom, which captures the error into ``result_entry["error"]`` instead
        of propagating. This is what makes it safe to run several of these
        concurrently via asyncio.gather: one step's failure cannot cancel or
        corrupt its siblings' results.
        """
        # DOCBOT-1513: model-driven tool selection (falls back to
        # _select_tool_heuristic internally on any LLM/parse failure).
        persona_name = state.get("persona", "Generalist")
        persona_pref = (
            expert_personas.get(persona_name, {}).get("tool_preference")
            if isinstance(expert_personas.get(persona_name), dict)
            else None
        )
        source_flags = {
            "has_db": state.get("has_db", True),
            "has_docs": state.get("has_docs", False),
            "has_csv": state.get("has_csv", False),
        }
        # DOCBOT-2721: fallback-plan steps carry a pinned tool; skip the router.
        tool = (state.get("plan_tools") or {}).get(step) or await _select_tool_llm(
            step,
            persona_tool_preference=persona_pref,
            run_id=state.get("run_id"),
            **source_flags,
        )

        # P1: validate the selected tool map against the resolved source
        # flags before advancing. An out-of-map selection (e.g. the LLM
        # picked sql_query with no database, or unsupported with sources
        # present) gets exactly ONE bounded regeneration with the error
        # appended; if that is still invalid, the flag-aware heuristic
        # re-routes deterministically so an unsupportable tool can never
        # execute and the run can never silently degrade to doc-only.
        if not _tool_available_for_sources(tool, **source_flags):
            logger.warning(
                "executor_node: tool selection %r invalid for sources "
                "(db=%s docs=%s csv=%s) — retrying once for step: %s",
                tool, source_flags["has_db"], source_flags["has_docs"],
                source_flags["has_csv"], step[:80],
            )
            tool = await _select_tool_llm(
                step,
                persona_tool_preference=persona_pref,
                run_id=state.get("run_id"),
                error_context=(
                    f"Tool {tool!r} is not available: "
                    f"db={source_flags['has_db']}, docs={source_flags['has_docs']}, "
                    f"csv={source_flags['has_csv']}. Choose one of the listed available tools."
                ),
                **source_flags,
            )
            if not _tool_available_for_sources(tool, **source_flags):
                tool = _select_tool_heuristic(step, **source_flags)
                logger.warning(
                    "executor_node: regeneration still invalid — deterministic "
                    "heuristic re-route → %r for step: %s", tool, step[:80],
                )

        # Bug #12: python_analysis needs prior data (a sql_query row-set or a
        # doc_search context). If the planner picked it as the very first step,
        # or if no prior step produced usable data, demote it to the best
        # data-fetch alternative for the available source. CSV is exempt —
        # its python_analysis branch loads the CSV directly.
        if tool == "python_analysis" and not state.get("has_csv"):
            _prior_has_data = any(
                p.get("tool") in {"sql_query", "doc_search"} and not p.get("error")
                for p in state.get("steps_completed", [])
            )
            if not _prior_has_data:
                if state.get("has_db", True):
                    tool = "sql_query"
                    logger.info(
                        "executor_node: demoted python_analysis → sql_query "
                        "(no prior data for step: %s)", step[:80]
                    )
                elif state.get("has_docs"):
                    tool = "doc_search"
                    logger.info(
                        "executor_node: demoted python_analysis → doc_search "
                        "(no prior data for step: %s)", step[:80]
                    )
                else:
                    tool = "unsupported"
                    logger.info(
                        "executor_node: demoted python_analysis → unsupported "
                        "(no data source for step: %s)", step[:80]
                    )
        session_id = state["session_id"]
        connection_id = state["connection_id"]
        persona = state.get("persona", "Generalist")

        result_entry: dict = {
            "step": step,
            "tool": tool,
            "result": "",
            "artifact_id": None,
            "chart_b64": None,
            "error": None,
            # P1: machine-readable failure code (e.g. data_tool_missing);
            # folded into state["degradation_reasons"] by executor_node.
            "failure_reason": None,
            # P2: normalized data output, popped into state["data_results"]
            # by executor_node (never doubled in steps_completed).
            "data_result": None,
        }
        new_citations: list[dict] = []

        try:
            # ── unsupported ────────────────────────────────────────────────
            # No data source (no DB, no docs, no CSV) — autopilot can't fetch
            # external data. Surface as a clean refusal instead of silently
            # running python on an undefined `df`.
            if tool == "unsupported":
                msg = (
                    "This step needs a data source. Connect a database, "
                    "upload a document (PDF), or upload a CSV to proceed."
                )
                result_entry["result"] = msg
                result_entry["error"] = "No data source available"
                # P1: the planned data tool was lost — record it so the run
                # is marked degraded instead of silently continuing.
                result_entry["failure_reason"] = _DEGR_DATA_TOOL_MISSING

            # ── sql_query ──────────────────────────────────────────────────
            elif tool == "sql_query":
                from api.hybrid_service import _collect_sql_result

                sql_meta = await _collect_sql_result(
                    connection_id=connection_id,
                    question=step,
                    persona=persona,
                    db_connections_table=db_connections_table,
                    schema_cache_table=schema_cache_table,
                    query_history_table=query_history_table,
                    query_embeddings_table=query_embeddings_table,
                    async_session_factory=async_session_factory,
                    expert_personas=expert_personas,
                )
                if sql_meta:
                    row_count = sql_meta.get("row_count", 0)
                    # run_sql_pipeline yields the preview under "result_preview";
                    # tolerate "preview" too for any older callers. The prior
                    # `sql_meta.get("preview")` always missed and left prior_rows
                    # empty, so python_analysis reported "No prior data".
                    preview = sql_meta.get("result_preview") or sql_meta.get("preview") or []
                    result_entry["result"] = (
                        f"{row_count} rows returned. "
                        f"Preview: {json.dumps(preview[:3], default=str)}"
                    )
                    result_entry["row_count"] = row_count
                    result_entry["sql"] = sql_meta.get("sql_query") or sql_meta.get("sql")
                    result_entry["explanation"] = sql_meta.get("explanation")
                    # P2: normalized data entry — the synthesizer and P3 claim
                    # guard consume this instead of relying on LLM recall.
                    result_entry["data_result"] = _sql_data_result(
                        step=step,
                        row_count=row_count if isinstance(row_count, (int, float)) else 0,
                        preview=preview,
                        sql=result_entry.get("sql"),
                        explanation=result_entry.get("explanation"),
                        artifact_id=None,
                    )
                    # DOCBOT-2002: structured inputs for the tie-out detection
                    # (rows mirror data_result sample_rows so both detection
                    # passes agree on item ids; sources/sql feed the db citation).
                    result_entry["tieout_sql"] = {
                        "result_preview": [
                            r for r in (preview[:10] if isinstance(preview, list) else [])
                            if isinstance(r, dict)
                        ],
                        "sources": [str(s) for s in (sql_meta.get("sources") or []) if s],
                        "sql_query": result_entry.get("sql"),
                    }
                    # Persist artifact for potential python_analysis in a later step
                    if preview and session_artifacts_table is not None:
                        from api.artifact_service import save_artifact

                        artifact_id = await save_artifact(
                            session_id=session_id,
                            turn_id=step_idx + 1,
                            artifact_type="sql_result",
                            name=step[:80],
                            result_dicts=preview,
                            session_artifacts_table=session_artifacts_table,
                            async_session_factory=async_session_factory,
                        )
                        result_entry["artifact_id"] = artifact_id
                        if isinstance(result_entry.get("data_result"), dict):
                            result_entry["data_result"]["artifact_id"] = artifact_id
                else:
                    result_entry["result"] = "No SQL result returned."
                    # P1: the routed data tool produced nothing at all while
                    # the connection flags said data should exist — mark it
                    # failed with data_tool_missing instead of auto-continuing.
                    result_entry["failure_reason"] = _DEGR_DATA_TOOL_MISSING

            # ── doc_search ─────────────────────────────────────────────────
            elif tool == "doc_search":
                if session_id not in vector_stores:
                    result_entry["result"] = "No relevant documents found."
                    # P1: has_docs was resolved upstream, but the session's
                    # vector store is gone — the planned data tool is lost.
                    result_entry["failure_reason"] = _DEGR_DATA_TOOL_MISSING
                else:
                    from api.deep_research_service import deep_retrieve

                    docs, sub_questions = await deep_retrieve(
                        question=step,
                        vector_store=vector_stores[session_id],
                        run_id=state.get("run_id"),
                    )

                    if docs:
                        # DOCBOT-1527: heuristic injection-pattern scan on
                        # retrieved chunks before they are folded into the
                        # investigation's step results — flag-and-continue,
                        # see api/utils/prompt_guard.py.
                        from api.utils.prompt_guard import scan_chunks, wrap_untrusted

                        scan_chunks(
                            [
                                (chunk_label(doc.metadata), doc.page_content)
                                for doc in docs
                            ],
                            session_id=session_id,
                            run_id=state.get("run_id"),
                        )

                        context = "\n\n".join(
                            f"{chunk_header(doc.metadata)}\n{doc.page_content}"
                            for doc in docs
                        )
                        result_entry["result"] = wrap_untrusted(context[:800])
                        # DOCBOT-2005: counts for the human step summary.
                        result_entry["passages"] = len(docs)
                        result_entry["doc_sources"] = list(dict.fromkeys(
                            str(doc.metadata.get("source", "Unknown")) for doc in docs
                        ))
                        # DOCBOT-2002: full (untruncated) chunk text with its
                        # "Source: X, Page N" headers — the doc side of the
                        # tie-out detection (doc citations + period extraction
                        # need the real headers, not the 800-char preview).
                        result_entry["tieout_doc_context"] = context

                        # DOCBOT-1510: keep the sub-questions deep_retrieve used
                        result_entry["sub_questions"] = list(sub_questions or [])

                        # Build citations from retrieved documents
                        from api.utils.lineage import safe_snippet
                        seen_keys: set[str] = set()
                        for doc in docs:
                            key = chunk_key(doc.metadata)
                            if key not in seen_keys:
                                seen_keys.add(key)
                                new_citations.append({
                                    "source": doc.metadata.get("source", "Unknown"),
                                    "page": chunk_page(doc.metadata),
                                    # PII-masked excerpt + owning step for lineage
                                    "text": safe_snippet(doc.page_content),
                                    "step_label": step,
                                })
                    else:
                        result_entry["result"] = "No relevant documents found."

            # ── python_analysis ────────────────────────────────────────────
            elif tool == "python_analysis":
                # ── CSV path (Bug #2 fix) ─────────────────────────────────
                # For CSV-backed sessions, load the CSV creds/profile directly
                # and run pandas analysis on E2B. This bypasses the SQL-artifact
                # lookup path which would otherwise yield empty prior_rows.
                handled_csv = False
                if state.get("has_csv") and connection_id:
                    try:
                        from sqlalchemy import select as _select
                        from api.utils.encryption import decrypt_credentials
                        from api.sandbox_service import (
                            generate_csv_analysis_code,
                            _run_csv_in_sandbox,
                        )
                        import base64 as _base64

                        async with async_session_factory() as _sess:
                            _conn_result = await _sess.execute(
                                _select(db_connections_table).where(
                                    db_connections_table.c.id == connection_id
                                )
                            )
                            _conn_row = _conn_result.fetchone()

                        if _conn_row and _conn_row.dialect == "csv":
                            handled_csv = True
                            _creds = decrypt_credentials(_conn_row.credentials_blob)
                            _table_name = _creds.get("table_name", "data")
                            _column_names = _creds.get("columns", [])
                            _csv_b64 = _creds.get("csv_content", "")
                            _section_manifest = _creds.get("section_manifest")
                            _sections = _creds.get("sections")
                            _is_multi_section = bool(_sections and len(_sections) > 1)

                            _data_profile = _creds.get("data_profile")
                            _data_profile_str: Optional[str] = None
                            if _data_profile and isinstance(_data_profile, dict):
                                _data_profile_str = json.dumps(
                                    _data_profile, indent=2, default=str
                                )

                            persona_data = expert_personas.get(
                                persona, expert_personas.get("Generalist", {})
                            )
                            persona_def = (
                                persona_data.get("persona_def", "")
                                if isinstance(persona_data, dict)
                                else ""
                            )

                            _csv_path = sandbox_csv_path(_table_name)
                            _csv_code = await generate_csv_analysis_code(
                                csv_path_in_sandbox=_csv_path,
                                column_names=_column_names,
                                question=step,
                                persona_def=persona_def,
                                chart_type="auto",
                                section_manifest=_section_manifest,
                                is_multi_section=_is_multi_section,
                                data_profile=_data_profile_str,
                                chat_history=None,
                            )

                            if not _csv_code:
                                result_entry["result"] = (
                                    "CSV code generation failed."
                                )
                                result_entry["error"] = "CSV code generation failed"
                                # P1: the analysis tool over the CSV data
                                # source is lost — mark the run degraded.
                                result_entry["failure_reason"] = _DEGR_DATA_TOOL_MISSING
                            else:
                                # Build preamble that defines `df` — same logic as
                                # run_csv_query_on_e2b. Without this, the sandbox never
                                # defines `df` and every generated reference raises
                                # NameError: name 'df' is not defined.
                                if _sections:
                                    from api.utils.csv_preprocessor import (
                                        dicts_to_sections,
                                        generate_section_preamble,
                                    )
                                    _csv_preamble = generate_section_preamble(
                                        dicts_to_sections(_sections), _csv_path
                                    )
                                else:
                                    _csv_preamble = (
                                        "import pandas as pd\n"
                                        "import re as _re\n"
                                        "import numpy as np\n"
                                        "\n"
                                        f"df = pd.read_csv('{_csv_path}')\n"
                                        "df.columns = [_re.sub(r'[^a-z0-9_]', '_', str(c).strip().lower().replace(' ', '_')) for c in df.columns]\n"
                                        "df = df.dropna(how='all').dropna(axis=1, how='all')\n"
                                        "for _c in df.columns:\n"
                                        "    _v = pd.to_numeric(df[_c], errors='coerce')\n"
                                        "    if _v.notna().sum() / max(df[_c].notna().sum(), 1) > 0.5:\n"
                                        "        df[_c] = _v\n"
                                        "# --- end preamble ---\n\n"
                                    )
                                from api.sandbox_service import _prepend_csv_preamble
                                _csv_code = _prepend_csv_preamble(_csv_code, _csv_preamble)
                                # Expose the pandas/ML code for the UI (demo:
                                # "DocBot writes and runs real code on demand").
                                result_entry["code"] = _csv_code

                                try:
                                    _csv_bytes = _base64.b64decode(_csv_b64)
                                except Exception:
                                    _csv_bytes = b""
                                sandbox_result = await _run_csv_in_sandbox(
                                    code=_csv_code,
                                    csv_path=_csv_path,
                                    csv_bytes=_csv_bytes,
                                    timeout_seconds=60,
                                )

                                # One corrective retry: feed the runtime error
                                # back to the LLM so it can fix common generated-
                                # code mistakes (bad pandas freq strings, wrong
                                # statsmodels kwargs) instead of surfacing a raw
                                # traceback. Mirrors run_csv_query_on_e2b.
                                if sandbox_result.error:
                                    _retry_code = await generate_csv_analysis_code(
                                        csv_path_in_sandbox=_csv_path,
                                        column_names=_column_names,
                                        question=step,
                                        persona_def=persona_def,
                                        chart_type="auto",
                                        section_manifest=_section_manifest,
                                        is_multi_section=_is_multi_section,
                                        data_profile=_data_profile_str,
                                        chat_history=None,
                                        error_context=(
                                            "Your previous code failed at runtime with:\n"
                                            f"{sandbox_result.error}\n\n"
                                            "Return corrected code that fixes this error. "
                                            "Common fixes: use fixed pandas offsets like "
                                            "'MS'/'ME'/'D' (not '3M') for asfreq/resample; "
                                            "for statsmodels ExponentialSmoothing use "
                                            "seasonal_periods=, not period=."
                                        ),
                                    )
                                    if _retry_code:
                                        # Re-attach the df-binding preamble (the
                                        # generator returns bare analysis code)
                                        # and expose it for the UI.
                                        _retry_code = _prepend_csv_preamble(
                                            _retry_code, _csv_preamble
                                        )
                                        result_entry["code"] = _retry_code
                                        sandbox_result = await _run_csv_in_sandbox(
                                            code=_retry_code,
                                            csv_path=_csv_path,
                                            csv_bytes=_csv_bytes,
                                            timeout_seconds=60,
                                        )

                                if sandbox_result.error:
                                    result_entry["error"] = sandbox_result.error
                                    # P1: failed analysis over the CSV source —
                                    # mark degraded, don't auto-continue.
                                    result_entry["failure_reason"] = _DEGR_DATA_TOOL_MISSING
                                    # Surface error in result so synthesizer + UI
                                    # see the failure, not a misleading "complete".
                                    result_entry["result"] = (
                                        sandbox_result.stdout[:500]
                                        if sandbox_result.stdout
                                        else f"Step failed: {sandbox_result.error}"
                                    )
                                else:
                                    result_entry["result"] = (
                                        sandbox_result.stdout[:500]
                                        if sandbox_result.stdout
                                        else "Analysis complete."
                                    )
                                    # P2: normalized data output for the claim
                                    # guard / synthesizer.
                                    result_entry["data_result"] = _python_data_result(
                                        step=step,
                                        provenance="e2b_csv",
                                        result_excerpt=result_entry["result"],
                                    )
                                    _analysis_artifact_id = await _persist_analysis_artifact(
                                        session_id=session_id,
                                        step_idx=step_idx,
                                        step=step,
                                        output=result_entry["result"],
                                        session_artifacts_table=session_artifacts_table,
                                        async_session_factory=async_session_factory,
                                    )
                                    if _analysis_artifact_id and not result_entry["artifact_id"]:
                                        result_entry["artifact_id"] = _analysis_artifact_id
                                if (
                                    sandbox_result.charts
                                    and session_artifacts_table is not None
                                ):
                                    from api.artifact_service import save_artifact

                                    artifact_id = await save_artifact(
                                        session_id=session_id,
                                        turn_id=step_idx + 1,
                                        artifact_type="chart",
                                        name=f"Chart: {step[:60]}",
                                        chart_b64=sandbox_result.charts[0],
                                        session_artifacts_table=session_artifacts_table,
                                        async_session_factory=async_session_factory,
                                    )
                                    result_entry["artifact_id"] = artifact_id
                                    result_entry["chart_b64"] = sandbox_result.charts[0]
                    except Exception as csv_exc:
                        logger.warning(
                            "executor_node CSV python_analysis failed: %s", csv_exc
                        )
                        result_entry["result"] = f"CSV analysis failed: {csv_exc}"
                        result_entry["failure_reason"] = _DEGR_DATA_TOOL_MISSING
                        handled_csv = True  # don't fall through to SQL-artifact path

                if handled_csv:
                    pass  # already populated result_entry above
                else:
                    prior_rows: list[dict] = []

                    # ── Predictive / ML path (fuel forecast fix) ──────────────
                    # For forecast/predict/model steps on a SQL source, the
                    # 10-row preview artifact is useless — a time-series model
                    # needs the FULL history. Re-run the prior step's validated
                    # SELECT with a raised row cap so the sandbox fits on all
                    # rows (e.g. 675 monthly crude-oil points), not a truncation.
                    from api.sandbox_service import _COMPLEX_QUERY_RE

                    is_ml_step = bool(_COMPLEX_QUERY_RE.search(step))
                    if is_ml_step and state.get("has_db") and connection_id:
                        for prior in reversed(state.get("steps_completed", [])):
                            if prior.get("tool") == "sql_query" and prior.get("sql"):
                                try:
                                    from api.db_service import fetch_full_series

                                    full_rows = await fetch_full_series(
                                        connection_id=connection_id,
                                        sql=prior["sql"],
                                        db_connections_table=db_connections_table,
                                        async_session_factory=async_session_factory,
                                    )
                                    if full_rows:
                                        prior_rows = full_rows
                                        logger.info(
                                            "executor_node: fetched %d full rows "
                                            "for ML step (%s)",
                                            len(full_rows), step[:60],
                                        )
                                except Exception as _ml_exc:
                                    logger.warning(
                                        "executor_node: fetch_full_series failed "
                                        "(%s) — falling back to preview artifact",
                                        _ml_exc,
                                    )
                                break

                    # Load the most recent sql_result artifact as input data
                    # (fallback for non-ML steps or if the full fetch failed).
                    if not prior_rows:
                        for prior in reversed(state.get("steps_completed", [])):
                            if prior.get("tool") == "sql_query" and prior.get("artifact_id"):
                                if session_artifacts_table is not None:
                                    from api.artifact_service import get_artifact

                                    art = await get_artifact(
                                        prior["artifact_id"],
                                        session_artifacts_table=session_artifacts_table,
                                        async_session_factory=async_session_factory,
                                    )
                                    if art and art.data_json:
                                        prior_rows = json.loads(art.data_json)
                                        break

                    # Fallback: if no SQL artifact, try doc_search results as context
                    if not prior_rows:
                        doc_context_parts: list[str] = []
                        for prior in reversed(state.get("steps_completed", [])):
                            if prior.get("tool") == "doc_search" and prior.get("result"):
                                doc_context_parts.append(prior["result"])
                        if doc_context_parts:
                            # Synthesise doc results into a pseudo-data context for code gen
                            prior_rows = [{"document_context": "\n\n".join(doc_context_parts)}]

                    if not prior_rows:
                        result_entry["result"] = "No prior data available for analysis."
                        # P1: the analysis tool has no data source to work on
                        # — the planned data tool is lost, mark degraded.
                        result_entry["failure_reason"] = _DEGR_DATA_TOOL_MISSING
                    else:
                        from api.sandbox_service import generate_analysis_code, run_python, _FORECAST_RE

                        persona_data = expert_personas.get(persona, expert_personas.get("Generalist", {}))
                        persona_def = persona_data.get("persona_def", "") if isinstance(persona_data, dict) else ""

                        # The plan step may read "analyze trends" while the user's
                        # actual goal is a forecast. Fold the original question in
                        # so the forecast guidance (fit a model, PRINT the
                        # predicted values) triggers and numbers are surfaced.
                        _orig_q = state.get("question", "")
                        analysis_question = step
                        if _FORECAST_RE.search(_orig_q) and not _FORECAST_RE.search(step):
                            analysis_question = f"{step}\n\nOverall goal: {_orig_q}"

                        code = await generate_analysis_code(
                            result_dicts=prior_rows,
                            question=analysis_question,
                            persona_def=persona_def,
                            chart_type="auto",
                        )
                        if code:
                            # Surface the exact code we run in E2B so the UI can
                            # show "Generated Python" — a core demo selling point.
                            result_entry["code"] = code
                            # Deterministic preamble — bind `df` from prior_rows so the
                            # LLM-generated code never hits NameError on `df`.
                            preamble = (
                                "import json as _json\n"
                                "import pandas as pd\n"
                                f"data = _json.loads({json.dumps(json.dumps(prior_rows, default=str))})\n"
                                "df = pd.DataFrame(data)\n"
                            )
                            sandbox_result = await run_python(preamble + code, allow_pip_egress=True)

                            # One corrective retry: feed the runtime error back so
                            # the generator can fix bad literals / API misuse
                            # instead of surfacing a raw traceback.
                            if sandbox_result.error:
                                _retry_code = await generate_analysis_code(
                                    result_dicts=prior_rows,
                                    question=analysis_question,
                                    persona_def=persona_def,
                                    chart_type="auto",
                                    error_context=(
                                        "Your previous code failed with:\n"
                                        f"{sandbox_result.error}\n\n"
                                        "Return corrected code. Do not write unit-suffixed "
                                        "or comma-separated numbers as literals; parse them "
                                        "from the data instead."
                                    ),
                                )
                                if _retry_code:
                                    result_entry["code"] = _retry_code
                                    sandbox_result = await run_python(preamble + _retry_code, allow_pip_egress=True)

                            if sandbox_result.error:
                                result_entry["error"] = sandbox_result.error
                                # P1: failed analysis → mark degraded.
                                result_entry["failure_reason"] = _DEGR_DATA_TOOL_MISSING
                                result_entry["result"] = (
                                    sandbox_result.stdout[:500] if sandbox_result.stdout
                                    else f"Step failed: {sandbox_result.error}"
                                )
                            else:
                                result_entry["result"] = (
                                    sandbox_result.stdout[:500] if sandbox_result.stdout
                                    else "Analysis complete."
                                )
                                # P2: normalized data output for the claim
                                # guard / synthesizer.
                                result_entry["data_result"] = _python_data_result(
                                    step=step,
                                    provenance="e2b_python",
                                    result_excerpt=result_entry["result"],
                                    row_count=len(prior_rows),
                                )
                                _analysis_artifact_id = await _persist_analysis_artifact(
                                    session_id=session_id,
                                    step_idx=step_idx,
                                    step=step,
                                    output=result_entry["result"],
                                    session_artifacts_table=session_artifacts_table,
                                    async_session_factory=async_session_factory,
                                )
                                if _analysis_artifact_id and not result_entry["artifact_id"]:
                                    result_entry["artifact_id"] = _analysis_artifact_id
                            # Persist first chart as an artifact
                            if sandbox_result.charts and session_artifacts_table is not None:
                                from api.artifact_service import save_artifact

                                artifact_id = await save_artifact(
                                    session_id=session_id,
                                    turn_id=step_idx + 1,
                                    artifact_type="chart",
                                    name=f"Chart: {step[:60]}",
                                    chart_b64=sandbox_result.charts[0],
                                    session_artifacts_table=session_artifacts_table,
                                    async_session_factory=async_session_factory,
                                )
                                result_entry["artifact_id"] = artifact_id
                                result_entry["chart_b64"] = sandbox_result.charts[0]
                        else:
                            result_entry["result"] = "Code generation failed or insufficient data."
                            # P1: the analysis generator produced nothing —
                            # the data tool's output is lost, mark degraded.
                            result_entry["failure_reason"] = _DEGR_DATA_TOOL_MISSING

        except Exception as exc:
            logger.warning("executor_node step %d ('%s') failed: %s", step_idx, tool, exc)
            result_entry["error"] = str(exc)

        return result_entry, new_citations

    async def executor_node(state: AutopilotState) -> dict:
        """Dispatch the next wave of independent plan steps concurrently.

        A "wave" is a run of steps returned by ``_next_wave_indices`` — either
        several fetch steps (sql_query/doc_search) with no dependency on each
        other, or several python_analysis steps that only depend on waves
        already reflected in ``state["steps_completed"]``. Steps within a wave
        are dispatched together via asyncio.gather(return_exceptions=True):
        each step already contains its own try/except (see _run_single_step),
        so return_exceptions is a defense-in-depth backstop, not the primary
        error path — it exists so that if a step coroutine somehow raises
        before reaching its own try/except (e.g. a bug in argument setup), one
        broken step still can't cancel its siblings or abort the whole wave.
        A raised sibling is converted into a normal error result_entry rather
        than surfaced to the graph.
        """
        plan = state.get("plan", [])
        pointer = len(state.get("steps_completed", []))

        if pointer >= len(plan):
            # Nothing left to run — keep the wave counter as-is so
            # _should_continue routes straight to the synthesizer.
            return {"iteration": state.get("iteration", 0)}

        # DOCBOT-1508: soft per-session LLM cost ceiling. Checked before
        # dispatching another wave (each wave can trigger several LLM calls —
        # SQL gen, codegen, deep_retrieve's planner, etc.) rather than before
        # every individual call inside a step, since a wave's steps run
        # concurrently and any one of them could already be mid-flight by
        # the time its own internal check would fire. Stops the loop
        # gracefully — partial results already in steps_completed still get
        # synthesized — instead of erroring.
        from api.utils.llm_provider import is_session_budget_exceeded, SESSION_COST_CEILING_USD

        run_id = state.get("run_id")
        if is_session_budget_exceeded(run_id):
            logger.warning(
                "executor_node: session cost ceiling ($%.4f) reached for run_id=%s "
                "— stopping investigation early with %d/%d steps completed",
                SESSION_COST_CEILING_USD, run_id, pointer, len(plan),
            )
            return {
                "iteration": state.get("iteration", 0) + 1,
                "budget_exceeded": True,
            }

        wave_indices = _next_wave_indices(plan, pointer, state)

        # DOCBOT-2310: per-step budget — a hung tool becomes a failed step.
        results = await asyncio.gather(
            *(
                asyncio.wait_for(_run_single_step(idx, plan[idx], state), STEP_TIMEOUT_S)
                for idx in wave_indices
            ),
            return_exceptions=True,
        )

        step_results: list[dict] = []
        all_new_citations: list[dict] = []
        for idx, outcome in zip(wave_indices, results):
            if isinstance(outcome, asyncio.TimeoutError):
                logger.warning("executor_node: step %d timed out after %ss", idx, STEP_TIMEOUT_S)
                step_results.append({
                    "step": plan[idx],
                    "tool": "unknown",
                    "result": "",
                    "artifact_id": None,
                    "chart_b64": None,
                    "error": f"Step timed out after {STEP_TIMEOUT_S:g}s",
                    "failure_reason": "step_timeout",
                })
                continue
            if isinstance(outcome, BaseException):
                logger.warning(
                    "executor_node: step %d raised outside its own try/except: %s",
                    idx, outcome,
                )
                step_results.append({
                    "step": plan[idx],
                    "tool": "unknown",
                    "result": "",
                    "artifact_id": None,
                    "chart_b64": None,
                    "error": str(outcome),
                    "failure_reason": None,
                })
                continue
            entry, citations = outcome
            step_results.append(entry)
            if citations:
                all_new_citations.extend(citations)

        # P1/P2 fan-in: normalize the wave's outputs into
        # (a) ``data_results`` — one entry per step keyed
        #     "step:<plan index>:<tool>" (bulk payloads are popped out of the
        #     step entries so steps_completed stays lean), and
        # (b) ``degradation_reasons`` — marks for steps that lost their
        #     planned data tool or crashed, which force the synthesizer to
        #     label the final answer degraded (P1c).
        data_results_update: dict[str, Any] = {}
        degradation_update: list[str] = []
        for wave_pos, idx in enumerate(wave_indices):
            entry = step_results[wave_pos]
            tool_name = str(entry.get("tool", "unknown"))
            key = f"step:{idx + 1}:{tool_name}"
            data_result = entry.pop("data_result", None)
            if isinstance(data_result, dict):
                data_result.setdefault("step_label", entry.get("step", ""))
                data_results_update[key] = data_result
            if entry.get("failure_reason"):
                degradation_update.append(f"{entry['failure_reason']}:{key}")
            elif entry.get("error"):
                # A step that crashed for any other reason also means part of
                # the plan did not run — keep the degraded labeling honest.
                degradation_update.append(f"step_execution_error:{key}")

        # Dedupe while preserving order (a wave may repeat a reason pattern).
        seen_reasons: set[str] = set()
        degradation_update = [
            r for r in degradation_update if not (r in seen_reasons or seen_reasons.add(r))
        ]

        update: dict = {
            "iteration": state.get("iteration", 0) + 1,
            "steps_completed": step_results,
        }
        if all_new_citations:
            update["citations"] = all_new_citations
        if data_results_update:
            update["data_results"] = data_results_update
        if degradation_update:
            update["degradation_reasons"] = degradation_update
        return update

    return executor_node


# ---------------------------------------------------------------------------
# SynthesizerNode
# ---------------------------------------------------------------------------


def _autopilot_sources(state: AutopilotState, steps: list[dict]) -> set[str]:
    """Source keys (for ``build_format_block``) the synthesizer can cite.

    Derived from the tools that actually produced a result; ``python_analysis``
    runs over the uploaded CSV when one is attached.
    """
    sources: set[str] = set()
    for s in steps:
        if s.get("error") and not s.get("result"):
            continue
        tool = s.get("tool")
        if tool == "doc_search":
            sources.add(SOURCE_DOC)
        elif tool == "sql_query":
            csv_only = state.get("has_csv") and not state.get("has_db")
            sources.add(SOURCE_CSV if csv_only else SOURCE_DB)
        elif tool == "python_analysis" and state.get("has_csv"):
            sources.add(SOURCE_CSV)
    return sources


def _scrub_db_url_secrets(text: str) -> str:
    """P3: defensive scrub for DB-URL-with-password fragments in text that
    gets surfaced to users (data summaries, refusal details)."""
    if not text:
        return text
    return _DB_URL_SECRET_RE.sub(r"\g<scheme>://[redacted]@", text)


def _build_data_results_summary(state: AutopilotState) -> tuple[str, int, bool]:
    """P2(2): deterministic summary of ``state['data_results']`` for the
    synthesizer prompt and the P3 claim guard.

    Returns ``(text, rows_total, has_real_data)``:
    - ``text``: markdown blocks per step (row count + deterministic sample
      capped at 10 rows + generated-SQL/tool provenance), "" when empty.
    - ``rows_total``: sum of reported row counts (python entries count their
      input rows; CSV entries report None and don't add rows but still set
      ``has_real_data`` via their excerpt).
    - ``has_real_data``: True if any entry carries actual data (non-zero
      rows, sample rows, or a tool output excerpt) — i.e. the final answer
      has numbers available and may not claim "no data".
    """
    data_results = state.get("data_results")
    if not isinstance(data_results, dict):
        return "", 0, False

    blocks: list[str] = []
    rows_total = 0
    has_real_data = False
    for key in sorted(data_results.keys()):
        entry = data_results.get(key)
        if not isinstance(entry, dict):
            continue
        step_label = str(entry.get("step_label") or key)
        entry_tool = str(entry.get("tool", "unknown"))
        entry_lines = [f"### Data from {key} (tool: {entry_tool}, step: {step_label})"]

        row_count = entry.get("row_count")
        if isinstance(row_count, (int, float)):
            rows_total += int(row_count or 0)
            if entry_tool == "sql_query" and int(row_count) == 0:
                entry_lines.append(
                    "- Rows returned: 0 — the query executed successfully "
                    "and returned NO rows (empty result set, not missing data)."
                )
            else:
                entry_lines.append(f"- Rows returned: {int(row_count)}")
                has_real_data = has_real_data or int(row_count) > 0

        sql = entry.get("sql")
        if isinstance(sql, str) and sql.strip():
            entry_lines.append(
                f"- Generated SQL (provenance: {entry.get('provenance', 'unknown')}): "
                f"{_scrub_db_url_secrets(sql)[:400]}"
            )
        explanation = entry.get("explanation")
        if isinstance(explanation, str) and explanation.strip():
            entry_lines.append(f"- Query explanation: {explanation[:300]}")

        excerpt = entry.get("result_excerpt")
        if isinstance(excerpt, str) and excerpt.strip():
            entry_lines.append(f"- Tool output (excerpt): {_scrub_db_url_secrets(excerpt)[:600]}")
            has_real_data = True

        sample_rows = entry.get("sample_rows")
        if isinstance(sample_rows, list) and sample_rows:
            entry_lines.append(
                "- Deterministic sample (max 10 rows): "
                + _scrub_db_url_secrets(json.dumps(sample_rows[:10], default=str))
            )
            has_real_data = True

        blocks.append("\n".join(entry_lines))

    if not blocks:
        return "", 0, False
    return _scrub_db_url_secrets("\n\n".join(blocks)), rows_total, has_real_data


def _collect_doc_context(state: AutopilotState) -> str:
    """Join successful doc_search step outputs — the document side for the
    P2(3) doc-vs-data discrepancy cross-check.

    DOCBOT-2002: prefer the full untruncated ``tieout_doc_context`` (with its
    "Source: X, Page N" headers) so detection sees every value and can build
    doc citations; fall back to the 800-char preview for older steps."""
    doc_parts = []
    for s in state.get("steps_completed", []):
        if s.get("tool") != "doc_search" or s.get("error"):
            continue
        part = str(s.get("tieout_doc_context") or s.get("result") or "")
        if part and part != "No relevant documents found.":
            doc_parts.append(part)
    return "\n\n".join(doc_parts)


def _executed_rows_preview(state: AutopilotState) -> list[dict]:
    """All sample rows actually returned by executed data tools."""
    rows: list[dict] = []
    data_results = state.get("data_results")
    if not isinstance(data_results, dict):
        return rows
    for key in sorted(data_results.keys()):
        entry = data_results.get(key)
        if isinstance(entry, dict) and isinstance(entry.get("sample_rows"), list):
            rows.extend(r for r in entry["sample_rows"] if isinstance(r, dict))
    return rows


def _detect_data_doc_discrepancies(state: AutopilotState):
    """P2(3): cross-check doc-search context against data_results rows with
    the hybrid discrepancy detector (api/utils/discrepancy_detector.py).
    Deterministic, no LLM. Returns None on failure/empty inputs — never raises."""
    doc_context = _collect_doc_context(state)
    rows = _executed_rows_preview(state)
    if not doc_context or not rows:
        return None
    try:
        from api.utils.discrepancy_detector import detect_discrepancies

        return detect_discrepancies(
            doc_context,
            {"result_preview": rows},
        )
    except Exception as exc:
        logger.warning("autopilot discrepancy detection failed (non-fatal): %s", exc)
        return None


def _ensure_discrepancies_in_answer(answer: str, report) -> str:
    """P2(3) deterministic backstop: if pre-computed discrepancies exist, the
    final answer MUST reference them — even when the LLM omits them.

    DOCBOT-2002: the backstop appends a one-line "Tie-out:" footer carrying
    each item's inline marker token instead of a literal "[DISCREPANCY]"
    bullet block. The interface renders a full TieOutCard from the structured
    lineage payload, so the numbers live on the card and the answer only
    carries the badge."""
    if report is None or not getattr(report, "has_discrepancies", False):
        return answer
    if "discrepan" in (answer or "").lower():
        return answer
    from api.utils.tieout import missing_marker_tail

    return (answer or "").rstrip() + missing_marker_tail("", report.discrepancies)


def _degradation_label(state: AutopilotState) -> str:
    """P1(c): the deterministic degraded-answer label, driven purely by
    machine-readable degradation reasons (never LLM discretion)."""
    reasons = state.get("degradation_reasons") or []
    if not reasons:
        return ""
    return (
        "\n\n---\n**Note:** This analysis is **degraded** — not all planned "
        "investigation steps could run ("
        + ", ".join(sorted(set(str(r) for r in reasons)))
        + "). Findings may be incomplete."
    )


def _ensure_degradation_label(answer: str, label: str) -> str:
    if not label:
        return answer
    if "analysis is **degraded**" in (answer or ""):
        return answer
    return (answer or "") + label


async def _synthesizer_node(state: AutopilotState) -> dict:
    """Synthesise all step results into a final markdown answer via Groq Llama.

    P1/P2 hardening:
    - The prompt carries a deterministic ground-truth summary of
      ``data_results`` (row counts, ≤10-row samples, SQL/tool provenance) so
      the answer cannot claim "no numbers" while tool data exists.
    - Pre-computed doc-vs-data discrepancies are injected with exact numbers.
    - The returned answer is deterministically backstopped: discrepancies
      forced in (P2) and a degraded-label appended when any planned tool was
      skipped (P1c) — never left to LLM discretion.
    """
    steps = state.get("steps_completed", [])

    if not steps:
        return {"final_answer": "No investigation steps were completed.", "citations": []}

    steps_text = "\n\n".join(
        f"Step {i + 1} [{s['tool']}]: {s['step']}\n"
        f"Result: {s.get('result', 'N/A')}"
        + (f"\nError: {s['error']}" if s.get("error") else "")
        for i, s in enumerate(steps)
    )

    # P2(2): ground-truth data summary (also feeds the P3 refusal guard).
    data_summary_text, _rows_total, _has_real_data = _build_data_results_summary(state)

    # P2(3): pre-computed doc-vs-data discrepancy cross-check.
    discrepancy_report = _detect_data_doc_discrepancies(state)
    discrepancy_instruction = (
        discrepancy_report.to_prompt_block()
        if discrepancy_report is not None and discrepancy_report.has_discrepancies
        else ""
    )

    degradation_reasons = [str(r) for r in (state.get("degradation_reasons") or [])]

    from api.utils.llm_provider import llm_configured
    if not llm_configured():
        base = f"Investigation complete.\n\n{steps_text}"
        if data_summary_text:
            base = (
                f"{base}\n\n## Available data from executed tools\n{data_summary_text}"
            )
        return {
            "final_answer": _ensure_degradation_label(
                _ensure_discrepancies_in_answer(base, discrepancy_report),
                _degradation_label(state),
            ),
            "citations": [],
        }

    from api.utils.prompt_guard import PROMPT_INJECTION_DEFENSE_NOTICE

    system_prompt = (
        "You are a senior data analyst and financial modeler. Based on the investigation "
        "steps and results below, write a comprehensive answer to the original question. "
        "Use specific numbers and findings from the results. "
        "When the question asks for calculations (DCF, valuations, projections, comparisons), "
        "you MUST perform them step by step using the data and assumptions provided. "
        "Show intermediate calculations clearly. Never say 'insufficient data' when the "
        "question or investigation results provide the necessary inputs.\n\n"
        f"{PROMPT_INJECTION_DEFENSE_NOTICE}\n\n"
        f"{build_format_block(_autopilot_sources(state, steps))}"
    )
    if data_summary_text:
        system_prompt += (
            "GROUND-TRUTH DATA RETURNED BY THE INVESTIGATION TOOLS — use these "
            "numbers where relevant. You MUST NOT claim that no data or no "
            "numbers are available:\n\n"
            f"{data_summary_text}\n\n"
        )
    if discrepancy_instruction:
        system_prompt += f"{discrepancy_instruction}\n\n"
    # DOCBOT-2637: year-over-year changes computed in code, never by the model
    yoy_block = yoy_prompt_block(_collect_doc_context(state))
    if yoy_block:
        system_prompt += f"{yoy_block.lstrip()}\n{YOY_RULE}\n"
    if degradation_reasons:
        system_prompt += (
            "This investigation is DEGRADED — the following planned steps could "
            f"not run: {', '.join(sorted(set(degradation_reasons)))}. End your "
            "answer with a short note stating that the analysis is degraded and "
            "that findings may be incomplete.\n\n"
        )
    user_content = (
        f"Original question: {state['question']}\n\n"
        f"Investigation results:\n{steps_text}"
    )

    try:
        from api.utils.llm_provider import achat_completion

        final_answer = await achat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=2000,
            caller="autopilot_synthesizer",
            run_id=state.get("run_id"),
            prompt_version=PROMPT_VERSION_AUTOPILOT_SYNTHESIZER,
        )
        # DOCBOT-2721: a reasoning model can spend the budget and return no
        # content; use the deterministic fallback below, never an empty answer.
        if not (final_answer or "").strip():
            raise ValueError("synthesizer returned empty content")
    except Exception as exc:
        logger.warning("synthesizer_node LLM call failed: %s", exc)
        final_answer = f"Investigation complete.\n\n{steps_text}"
        if data_summary_text:
            final_answer = (
                f"{final_answer}\n\n## Available data from executed tools\n{data_summary_text}"
            )

    # Deterministic backstops — never trust the LLM to label these itself.
    final_answer = _ensure_discrepancies_in_answer(final_answer, discrepancy_report)
    final_answer = _ensure_degradation_label(final_answer, _degradation_label(state))

    return {"final_answer": final_answer}


# ---------------------------------------------------------------------------
# VerifierNode — DOCBOT-1518
# ---------------------------------------------------------------------------

# DOCBOT-1507: bump manually ("v1" -> "v2") whenever either prompt below
# changes — see PROMPT_VERSION convention in api/utils/llm_provider.py.
PROMPT_VERSION_AUTOPILOT_VERIFIER = "v1"


def _has_apology_claim(answer: str) -> bool:
    """P3(a): cheap deterministic scan for "no data / no numbers / unable to
    compute"-style claims in a generated answer."""
    lowered = (answer or "").lower()
    return any(pattern in lowered for pattern in _APOLOGY_CLAIM_PATTERNS)


def _zero_row_sql_entries(state: AutopilotState) -> list[dict]:
    """P3(b): normalized data_results entries for sql_query steps that
    executed successfully but returned zero rows — the feedstock for the
    explicit "this query ran and returned empty" refusal clarity."""
    data_results = state.get("data_results")
    if not isinstance(data_results, dict):
        return []
    entries: list[dict] = []
    for key in sorted(data_results.keys()):
        entry = data_results.get(key)
        if not isinstance(entry, dict) or entry.get("tool") != "sql_query":
            continue
        row_count = entry.get("row_count")
        if isinstance(row_count, (int, float)) and int(row_count) == 0:
            entries.append(entry)
    return entries


def _refusal_detail(zero_entries: list[dict]) -> str:
    """P3(b): deterministic refusal-clarification block naming exactly which
    query ran and returned empty — replaces blanket data-absence claims."""
    lines = [
        "**Data availability detail:** the following queries executed "
        "successfully and returned zero rows — this is an empty result set, "
        "not missing data:"
    ]
    for entry in zero_entries:
        step_label = str(entry.get("step_label") or "unnamed step")
        sql = str(entry.get("sql") or "").strip()
        if sql:
            lines.append(
                f'- Step "{step_label}": `{_scrub_db_url_secrets(sql)}` returned 0 rows.'
            )
        else:
            lines.append(
                f'- Step "{step_label}" (tool: sql_query) executed against the '
                "connected database and returned 0 rows."
            )
    return "\n".join(lines)


async def _force_fixup_with_data(
    state: AutopilotState, answer: str, data_summary_text: str
) -> str:
    """P3(a): ONE bounded regeneration with the explicit ground-truth data
    context, replacing an answer that claims no data exists while data does.
    If the regenerated text still denies the data — or the call fails — a
    deterministic data block is appended instead so numbers always surface.
    Never raises."""
    try:
        from api.utils.llm_provider import achat_completion

        system_prompt = (
            "You are a senior data analyst. The draft answer below wrongly claims "
            "that no data or no numbers are available, but the investigation tools "
            "DID return the ground-truth data listed below. Rewrite the FULL "
            "answer so it uses those actual numbers. Do NOT claim data is "
            "unavailable. Keep any correct structure or formatting from the draft."
        )
        user_content = (
            f"Original question: {state.get('question', '')}\n\n"
            f"Ground-truth data returned by the tools:\n{data_summary_text}\n\n"
            f"Draft answer to rewrite:\n{answer}"
        )
        fixed = await achat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=2000,
            caller="autopilot_claim_guard",
            run_id=state.get("run_id"),
            prompt_version=PROMPT_VERSION_AUTOPILOT_VERIFIER,
        )
        fixed_text = (fixed or "").strip() if isinstance(fixed, str) else ""
        if fixed_text:
            if _has_apology_claim(fixed_text):
                return fixed_text + (
                    "\n\n---\n**Available data from executed tools:**\n"
                    + data_summary_text
                )
            return fixed_text
        return answer + (
            "\n\n---\n**Available data from executed tools:**\n" + data_summary_text
        )
    except Exception as exc:
        logger.warning(
            "claim-guard fix-up regeneration failed (non-fatal): %s", exc
        )
        return answer + (
            "\n\n---\n**Available data from executed tools:**\n" + data_summary_text
        )


async def _apply_claim_guard(state: AutopilotState, answer: str) -> str:
    """P3 final-answer output guard. Runs once at the very end of the
    verifier node, on the post-verification answer:

    (a) If the tools returned real data but the answer contains a
        "no data / no numbers / unable to compute"-style claim → force ONE
        fix-up regeneration with the explicit data context.
    (b) If queries executed and returned empty sets → the answer must state
        specifically which query ran and returned empty (deterministic
        append; no extra LLM cost).

    Vacuous when data_results is empty. Never raises, never blocks."""
    if not answer:
        return answer
    try:
        data_summary_text, rows_total, has_real_data = _build_data_results_summary(state)
        zero_entries = _zero_row_sql_entries(state)

        if _has_apology_claim(answer):
            if has_real_data and data_summary_text:
                answer = await _force_fixup_with_data(state, answer, data_summary_text)
            elif rows_total == 0 and zero_entries:
                answer = answer.rstrip() + "\n\n" + _refusal_detail(zero_entries)
        return answer
    except Exception as exc:
        logger.warning("claim guard failed (non-fatal): %s", exc)
        return answer


async def _finalize_answer(state: AutopilotState, answer: str) -> str:
    """Shared verifier tail (P1c + P3): claim guard first, then re-stamp the
    degraded label (a verifier correction may have dropped the label the
    synthesizer appended). Never raises, never blocks."""
    guarded = await _apply_claim_guard(state, answer)
    return _ensure_degradation_label(guarded, _degradation_label(state))


async def _verifier_node(state: AutopilotState) -> dict:
    """DOCBOT-1518: active discrepancy gate on the synthesized answer.

    Runs once, sequentially, after ``_synthesizer_node`` has already
    produced ``final_answer`` and after the executor wave loop has fully
    finished (see ``_build_graph`` — this node has a single inbound edge
    from "synthesizer", no fan-out). It is never invoked concurrently with
    itself or with any other node, unlike ``_select_tool_llm`` which runs
    once per step inside an ``asyncio.gather`` wave. Even so, this uses the
    async ``call_llm`` helper (LangChain ``.ainvoke``) rather than the
    synchronous ``chat_completion`` the synthesizer uses, for consistency
    with DOCBOT-1513/1517's async LLM call sites and so a future change that
    moves verification earlier (e.g. per-wave) doesn't silently reintroduce
    the event-loop-blocking bug DOCBOT-1513 hit.

    Makes ONE LLM call asking whether the synthesized answer's numeric/
    factual claims are contradicted by the data actually retrieved during
    the investigation. If a real contradiction is flagged (not hedging or
    missing detail), triggers exactly ONE corrective re-synthesis — mirrors
    the single-retry pattern used for sandbox codegen failures in
    api/sandbox_service.py (feed the failure/contradiction back to the LLM
    once, use the corrected output if it comes back, otherwise keep the
    original).

    On ANY failure (LLM call errors, response unparseable, correction call
    fails) — never raises, never blocks the response — passes the original
    synthesized answer through unchanged. This is a quality gate, not a new
    point of failure; mirrors `_select_tool_llm`'s try/except/fallback shape.

    P3: after the contradiction gate, the FINAL answer (original or
    corrected) is passed through ``_apply_claim_guard`` — the deterministic
    "no data" claim guard — and re-stamped with the P1c degraded label if
    degradation reasons exist (correction may have dropped it). All return
    paths funnel through this tail so the guard always runs.
    """
    original_answer = state.get("final_answer", "")
    steps = [s for s in state.get("steps_completed", []) if not s.get("error")]

    if not original_answer or not steps:
        return {"final_answer": original_answer}

    steps_text = "\n\n".join(
        f"Step {i + 1} [{s.get('tool', '')}]: {s.get('step', '')}\n"
        f"Actual data retrieved: {s.get('result', 'N/A')}"
        for i, s in enumerate(steps)
    )

    from api.utils.prompt_guard import PROMPT_INJECTION_DEFENSE_NOTICE

    verifier_system_prompt = (
        "You are a fact-checking verifier for an AI investigation agent. You are "
        "given the ACTUAL DATA retrieved during an investigation and a DRAFT ANSWER "
        "synthesized from it. Check whether the draft answer's numeric and factual "
        "claims are actually supported by the retrieved data.\n\n"
        "Only flag a REAL contradiction — a specific number, name, or fact in the "
        "answer that directly conflicts with what the retrieved data shows. Do NOT "
        "flag hedging, missing detail, rounding differences under 1%, or stylistic "
        "choices as contradictions.\n\n"
        f"{PROMPT_INJECTION_DEFENSE_NOTICE}\n\n"
        'Respond with ONLY a JSON object: {"contradiction_found": true or false, '
        '"explanation": "<specific description of the contradiction, or empty '
        'string if none>"}. No markdown fences, no other text.'
    )
    verifier_user_content = (
        f"Original question: {state.get('question', '')}\n\n"
        f"Retrieved data:\n{steps_text}\n\n"
        f"Draft answer:\n{original_answer}"
    )

    try:
        from api.utils.llm_provider import call_llm

        raw = await call_llm(
            f"{verifier_system_prompt}\n\n{verifier_user_content}",
            temperature=0,
            caller="autopilot_verifier",
            run_id=state.get("run_id"),
            prompt_version=PROMPT_VERSION_AUTOPILOT_VERIFIER,
        )
        start_idx = raw.index("{")
        end_idx = raw.rindex("}") + 1
        parsed = json.loads(raw[start_idx:end_idx])
        contradiction_found = bool(parsed.get("contradiction_found"))
        explanation = str(parsed.get("explanation", "")).strip()
    except Exception as exc:
        logger.warning(
            "verifier_node LLM call/parse failed (%s) — passing answer through unchanged",
            exc,
        )
        final_answer = original_answer
        # P3+P1c tail still runs: the claim guard is independent of the
        # verifier's own LLM health.
        return {"final_answer": await _finalize_answer(state, final_answer)}

    if not contradiction_found or not explanation:
        return {"final_answer": await _finalize_answer(state, original_answer)}

    # One corrective re-synthesis — same correction prompt shape as
    # _synthesizer_node, with the verifier's specific contradiction folded in.
    try:
        from api.utils.llm_provider import achat_completion

        correction_system_prompt = (
            "You are a senior data analyst and financial modeler. A previous draft "
            "answer to the question below was found by a verifier to contain a "
            "factual/numeric contradiction with the actual retrieved data. Rewrite "
            "the FULL answer so it accurately reflects the retrieved data, fixing "
            "the specific contradiction described below, while keeping everything "
            "else that was already correct.\n"
            f"{build_format_block(_autopilot_sources(state, steps))}"
        )
        correction_user_content = (
            f"Original question: {state.get('question', '')}\n\n"
            f"Retrieved data:\n{steps_text}\n\n"
            f"Draft answer:\n{original_answer}\n\n"
            f"Contradiction found by verifier:\n{explanation}\n\n"
            "Rewrite the full answer, correcting this contradiction."
        )
        corrected_answer = await achat_completion(
            [
                {"role": "system", "content": correction_system_prompt},
                {"role": "user", "content": correction_user_content},
            ],
            temperature=0.3,
            max_tokens=2000,
            caller="autopilot_verifier_correction",
            run_id=state.get("run_id"),
            prompt_version=PROMPT_VERSION_AUTOPILOT_VERIFIER,
        )
    except Exception as exc:
        logger.warning(
            "verifier_node correction LLM call failed (%s) — passing original answer through unchanged",
            exc,
        )
        return {"final_answer": await _finalize_answer(state, original_answer)}

    if not corrected_answer or not corrected_answer.strip():
        return {"final_answer": await _finalize_answer(state, original_answer)}

    logger.info("verifier_node: corrected final answer after contradiction: %s", explanation[:200])
    return {"final_answer": await _finalize_answer(state, corrected_answer)}


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------


def _should_continue(state: AutopilotState) -> str:
    """Conditional edge: loop executor until plan is exhausted or limits hit.

    Plan exhaustion is measured by ``len(steps_completed)`` — the number of
    individual steps actually finished — since a wave can complete more than
    one step per executor invocation. ``iteration`` (wave count) is still
    capped separately against MAX_ITERATIONS as a hard ceiling on how many
    executor invocations (LLM/tool round-trips worth of latency) an
    investigation can take, independent of how many steps each wave resolved.
    """
    iteration = state.get("iteration", 0)
    plan = state.get("plan", [])
    steps_done = len(state.get("steps_completed", []))
    timed_out = state.get("timed_out", False)
    budget_exceeded = state.get("budget_exceeded", False)
    if timed_out or budget_exceeded or steps_done >= len(plan) or iteration >= MAX_ITERATIONS:
        return "synthesize"
    return "execute"


def _merge_state_update(state: dict, updates: Any) -> None:
    """Apply one node's update to ``state`` with the graph's reducers."""
    for key, value in (updates or {}).items():
        if key in _LIST_STATE_KEYS:
            state[key] = list(state.get(key) or []) + list(value or [])
        elif key == "data_results":
            state[key] = _merge_dict_reducer(state.get(key), value)
        else:
            state[key] = value


async def _synthesize_partial(state: dict, done: int, total: int) -> str:
    """DOCBOT-2310: answer from the steps finished before the overall timeout.
    Bounded by STEP_TIMEOUT_S; falls back to the step summaries. Never raises."""
    from api.utils.pii_masking import mask_pii
    from api.utils.response_format import normalize_markdown

    note = f"_Stopped early after {done} of {total} steps — answer is based on partial results._"
    try:
        result = await asyncio.wait_for(
            _synthesizer_node({**state, "timed_out": True}), STEP_TIMEOUT_S
        )
        answer = str(result.get("final_answer") or "")
    except Exception as exc:  # incl. TimeoutError — the user still gets an answer
        logger.warning("autopilot partial synthesis failed (%s)", type(exc).__name__)
        answer = ""
    if not answer.strip():
        answer = "\n".join(
            f"- {build_step_summary(s)}" for s in state.get("steps_completed") or []
        ) or "No investigation steps were completed."
    return normalize_markdown(mask_pii(f"{note}\n\n{answer}"))


def _build_graph(executor_node):
    graph = StateGraph(AutopilotState)
    graph.add_node("planner", _planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("synthesizer", _synthesizer_node)
    # DOCBOT-1518: verifier runs after the synthesizer, once, sequentially —
    # single inbound/outbound edge, never part of the executor's concurrent
    # wave dispatch.
    graph.add_node("verifier", _verifier_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges(
        "executor",
        _should_continue,
        {"execute": "executor", "synthesize": "synthesizer"},
    )
    graph.add_edge("synthesizer", "verifier")
    graph.add_edge("verifier", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Public async generator
# ---------------------------------------------------------------------------


async def run_autopilot(
    question: str,
    session_id: str,
    connection_id: str,
    persona: str,
    *,
    db_connections_table: Any,
    schema_cache_table: Any,
    query_history_table: Any,
    query_embeddings_table: Any,
    session_artifacts_table: Any,
    table_embeddings_table: Any,
    async_session_factory: Any,
    expert_personas: dict,
    vector_stores: dict,
    has_docs: bool = False,
    has_db: bool = False,
    has_csv: bool = False,
    chat_history: Optional[list[dict[str, str]]] = None,
) -> AsyncGenerator[str, None]:
    """Run the Autopilot investigation and yield SSE-formatted strings.

    Yields in order:
      - plan event (steps list from PlannerNode)
      - step events (one per ExecutorNode iteration)
      - answer event (synthesis from SynthesizerNode)
      - done event (citations)
    Never raises — errors are emitted as SSE error events.
    """
    start_time = time.monotonic()

    # DOCBOT-1501: one run_id groups every LLM call made anywhere in this
    # investigation (planner, each executor wave, synthesizer, and any
    # nested deep_retrieve / SQL-pipeline / codegen calls) in the persisted
    # call log — see api/utils/llm_provider.py's run_trace().
    from api.utils.llm_provider import (
        SESSION_COST_CEILING_USD,
        current_run_id,
        new_run_id,
        run_trace,
    )
    run_id = current_run_id() or new_run_id()

    # DOCBOT-2311: lazy-load the session's vectors from CHROMA_DIR; if they
    # are gone, say so explicitly instead of investigating without the docs.
    if has_docs:
        from api.utils.vector_store import (
            DOCUMENTS_EXPIRED_MESSAGE,
            documents_expired_sse,
            get_or_load_store,
        )

        store = await asyncio.to_thread(get_or_load_store, session_id, vector_stores)
        if store is None:
            logger.info("autopilot: documents expired for session %s", session_id)
            yield documents_expired_sse(session_id)
            has_docs = False
            if not (has_db or has_csv):
                yield f"data: {json.dumps({'type': 'answer', 'content': DOCUMENTS_EXPIRED_MESSAGE})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'citations': []})}\n\n"
                return
            yield f"data: {json.dumps({'type': 'warning', 'content': DOCUMENTS_EXPIRED_MESSAGE})}\n\n"

    # Rephrase follow-up questions into standalone queries using chat history
    if chat_history:
        from api.db_service import _rephrase_with_history
        try:
            question = await _rephrase_with_history(question, chat_history)
        except Exception as exc:
            logger.warning("Autopilot rephrase failed, using original: %s", exc)

    executor_node = make_executor_node(
        db_connections_table=db_connections_table,
        schema_cache_table=schema_cache_table,
        query_history_table=query_history_table,
        query_embeddings_table=query_embeddings_table,
        session_artifacts_table=session_artifacts_table,
        table_embeddings_table=table_embeddings_table,
        async_session_factory=async_session_factory,
        expert_personas=expert_personas,
        vector_stores=vector_stores,
    )

    app = _build_graph(executor_node)

    initial_state: AutopilotState = {
        "question": question,
        "session_id": session_id,
        "connection_id": connection_id,
        "persona": persona,
        "run_id": run_id,
        "plan": [],
        "steps_completed": [],
        "iteration": 0,
        "final_answer": "",
        "citations": [],
        "timed_out": False,
        "budget_exceeded": False,
        "has_docs": has_docs,
        "has_db": has_db,
        "has_csv": has_csv,
        # P1/P2: normalized tool data + degradation marks (populated by
        # executor waves; consumed by synthesizer + claim guard).
        "data_results": {},
        "degradation_reasons": [],
    }

    step_num = 0
    all_citations: list[dict] = []
    # DOCBOT-1512: every completed step's result dict, kept for the trace row's
    # tool_chosen/plan_steps fields (step_by_label only maps label -> step_num).
    all_step_results: list[dict] = []

    # DOCBOT-1510: per-answer lineage
    from api.utils.lineage import LineageCollector
    lineage = LineageCollector(run_id, "autopilot", question=question)
    lineage.set_persona(persona)
    lineage.set_pii(True)
    step_by_label: dict[str, int] = {}

    # DOCBOT-1501: bind run_id for the duration of the investigation. Entered
    # manually (not `with`) so the existing try/except body below doesn't
    # need re-indenting; asyncio.Task copies the current context at creation
    # time, so this also covers concurrently-dispatched executor waves.
    # DOCBOT-1509: name opens the LangSmith root for this investigation.
    _trace_cm = run_trace(run_id, name="autopilot")
    _trace_cm.__enter__()
    done_sent = False
    # DOCBOT-2310: mirror of the graph state, so a timeout can still synthesize.
    partial_state: dict = dict(initial_state)
    stream = app.astream(initial_state, stream_mode="updates")
    try:
        while True:
            try:
                state_update = await asyncio.wait_for(
                    stream.__anext__(), max(start_time + TOTAL_TIMEOUT_S - time.monotonic(), 0)
                )
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                await stream.aclose()
                total = len(partial_state.get("plan") or [])
                done = len(partial_state.get("steps_completed") or [])
                yield _sse({
                    "type": "warning",
                    "content": (
                        f"Investigation reached the {TOTAL_TIMEOUT_S:g}s time limit. "
                        "Showing partial results."
                    ),
                })
                answer = await _synthesize_partial(partial_state, done, total)
                yield _sse({"type": "answer", "content": answer})
                yield _sse({"type": "done", "citations": all_citations, "run_id": run_id})
                done_sent = True
                break

            for node_name, updates in state_update.items():
                _merge_state_update(partial_state, updates)
                if node_name == "planner":
                    plan = updates.get("plan", [])
                    yield _sse({
                        "type": "plan",
                        "content": f"Investigation plan: {len(plan)} step(s)",
                        "steps": plan,
                    })

                elif node_name == "executor":
                    if updates.get("budget_exceeded"):
                        yield _sse({
                            "type": "warning",
                            "content": (
                                f"This investigation reached its session cost ceiling "
                                f"(${SESSION_COST_CEILING_USD:.2f}). Showing results from the "
                                "steps completed so far."
                            ),
                        })
                    completed = updates.get("steps_completed", [])
                    for step_result in completed:
                        step_num += 1

                        # Never surface raw tracebacks / internal failure strings
                        # to the user. Log the real error server-side and show a
                        # calm, generic message so a failed step reads as "skipped
                        # and continued" rather than a scary Python error.
                        raw_error = step_result.get("error")
                        content = step_result.get("result", "") or ""
                        friendly_error = None
                        if raw_error:
                            logger.warning(
                                "autopilot step %d (%s) failed: %s",
                                step_num, step_result.get("tool", ""), raw_error,
                            )
                            friendly_error = "Couldn't complete this step — continued with the available data."
                        if content.startswith(_FAILURE_CONTENT_PREFIXES) or (raw_error and not content.strip()):
                            content = "This step didn't return usable data, so the investigation continued with the other findings."
                        content = clean_step_details(content)

                        # Expose the exact code executed in E2B (python_analysis
                        # / forecast steps) so the UI shows real on-the-fly code,
                        # not just results. Errors stay sanitized above; code is
                        # not an error string, so it's safe to surface. Cap the
                        # length to keep the SSE payload reasonable.
                        step_code = step_result.get("code")
                        if isinstance(step_code, str) and len(step_code) > 6000:
                            step_code = step_code[:6000] + "\n# … (truncated)"

                        step_by_label[step_result.get("step", "")] = step_num
                        all_step_results.append(step_result)
                        lineage.add_step(
                            step_result.get("step", f"step {step_num}") or f"step {step_num}",
                            tool=step_result.get("tool", ""),
                            status="error" if raw_error else "ok",
                            detail=friendly_error,
                        )
                        if step_result.get("sub_questions"):
                            lineage.set_sub_questions(step_result["sub_questions"])
                        if step_result.get("sql"):
                            lineage.set_sql(sql=step_result.get("sql"))

                        yield _sse({
                            "type": "step",
                            "step_num": step_num,
                            "tool": step_result.get("tool", ""),
                            "step_label": step_result.get("step", ""),
                            "summary": build_step_summary(step_result),
                            "content": content,
                            "artifact_id": step_result.get("artifact_id"),
                            "chart_b64": step_result.get("chart_b64"),
                            "sql": step_result.get("sql"),
                            "explanation": step_result.get("explanation"),
                            "code": step_code,
                            "error": friendly_error,
                        })
                    # Accumulate citations from doc_search steps
                    new_cits = updates.get("citations", [])
                    if new_cits:
                        all_citations.extend(new_cits)

                elif node_name == "synthesizer":
                    # DOCBOT-1518: the synthesizer's output is now a draft —
                    # _verifier_node runs next and may replace it with a
                    # corrected answer. Record the synthesis step in the
                    # lineage here (it's the node that actually did the
                    # synthesis work); the user-facing "answer" SSE event,
                    # citations, emit_lineage(), and log_trace() all moved to
                    # the "verifier" branch below so they capture the FINAL
                    # (possibly corrected) answer, never the pre-verification
                    # draft.
                    lineage.add_step("synthesize", tool="llm")

                elif node_name == "verifier":
                    from api.utils.pii_masking import mask_pii
                    from api.utils.response_format import normalize_markdown
                    from api.utils.tieout import (
                        detect_from_autopilot_steps,
                        missing_marker_tail,
                        replace_discrepancy_tags,
                    )
                    final_answer = normalize_markdown(mask_pii(updates.get("final_answer", "")))

                    # DOCBOT-2002: turn the doc-vs-data cross-check into
                    # structured TieOutCards. Detection is deterministic (no
                    # LLM) and re-runs over the full stashed doc context +
                    # sql metadata so the cards carry both citations; item ids
                    # match the markers the synthesizer prompt placed because
                    # both passes see identical inputs. Legacy "[DISCREPANCY]"
                    # tags the model echoed are rewritten to markers, and a
                    # tail badge is appended for cards the answer never
                    # referenced — every card gets a badge in the text.
                    try:
                        _tieout_report = detect_from_autopilot_steps(all_step_results)
                    except Exception as exc:
                        logger.warning("autopilot tie-out detection failed (non-fatal): %s", exc)
                        _tieout_report = None
                    if _tieout_report is not None and _tieout_report.has_discrepancies:
                        for item in _tieout_report.discrepancies:
                            lineage.add_discrepancy(item)
                        final_answer = replace_discrepancy_tags(final_answer, _tieout_report.discrepancies)
                        _tail = missing_marker_tail(final_answer, _tieout_report.discrepancies)
                        if _tail:
                            final_answer += _tail

                    yield _sse({"type": "answer", "content": final_answer})
                    for cit in all_citations:
                        lineage.add_source(
                            kind="pdf",
                            label=str(cit.get("source", "Unknown")),
                            page=cit.get("page"),
                            snippet=cit.get("text"),
                            step_num=step_by_label.get(cit.get("step_label", "")),
                        )
                    lineage.add_step("verify", tool="llm")
                    from api.lineage_service import emit_lineage
                    _lin_evt = emit_lineage(lineage, session_id, connection_id)
                    if _lin_evt:
                        yield _lin_evt

                    # DOCBOT-1512: fire-and-forget agent trace row for this run.
                    # DOCBOT-1518: `final_answer` here is the post-verification
                    # value read from the verifier node's own state update —
                    # see run_autopilot's docstring/report for how this was
                    # confirmed to be the corrected (not draft) text.
                    from api.trace_service import log_trace
                    _tools_used = sorted({
                        s.get("tool", "") for s in all_step_results if s.get("tool")
                    })
                    trace_id = await log_trace(
                        run_id=run_id,
                        session_id=session_id,
                        connection_id=connection_id,
                        pipeline="autopilot",
                        question=question,
                        tool_chosen=", ".join(_tools_used) or None,
                        plan_steps=list(step_by_label.keys()),
                        retrieved_refs=all_citations,
                        final_answer=final_answer,
                        latency_ms=(time.monotonic() - start_time) * 1000,
                    )
                    yield _sse({
                        "type": "done",
                        "citations": all_citations,
                        "run_id": run_id,
                        "trace_id": trace_id,
                    })
                    done_sent = True

    except Exception as exc:
        logger.error("run_autopilot failed: %s", exc)
        yield _sse({"type": "error", "content": f"Autopilot error: {safe_error_message(exc)}"})
    finally:
        _trace_cm.__exit__(None, None, None)
    # DOCBOT-2310: the stream always ends with `done`, even after an error.
    if not done_sent:
        yield _sse({"type": "done", "citations": all_citations, "run_id": run_id})
