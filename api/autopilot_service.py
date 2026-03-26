"""
DOCBOT-405: Analytical Autopilot — LangGraph-based multi-step investigation agent.

Exposes a single async generator, run_autopilot(), that:
  - Plans an investigation into ≤5 steps via Groq Llama (PlannerNode)
  - Executes each step using the best tool: sql_query | doc_search | python_analysis
    (ExecutorNode, loops up to MAX_ITERATIONS times, hard 90-second wall-clock guard)
  - Synthesises all step findings into a final markdown answer (SynthesizerNode)
  - Yields SSE-formatted strings throughout for direct client streaming

SSE event types yielded:
  {type: "plan",    steps: [...], content: "N steps planned"}
  {type: "step",    step_num, tool, step_label, content, artifact_id?, chart_b64?, error?}
  {type: "answer",  content: "<markdown answer>"}
  {type: "done",    citations: [...]}
  {type: "warning", content: "..."}  — non-fatal issues
  {type: "error",   content: "..."}  — fatal abort
"""

from __future__ import annotations

import asyncio
import json
import logging
import operator
import os
import time
from typing import Annotated, Any, AsyncGenerator, Optional

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5
TOTAL_TIMEOUT_S = 90


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class AutopilotState(TypedDict):
    question: str
    session_id: str
    connection_id: str
    persona: str
    # steps from PlannerNode
    plan: list[str]
    # accumulated step results — operator.add reducer: each executor call appends
    steps_completed: Annotated[list[dict], operator.add]
    # which plan step we are about to execute (0-based)
    iteration: int
    # final synthesised markdown answer
    final_answer: str
    # citations accumulated across doc_search steps
    citations: Annotated[list[dict], operator.add]
    # set True when the wall-clock guard fires
    timed_out: bool


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, default=str)}\n\n"


# ---------------------------------------------------------------------------
# PlannerNode
# ---------------------------------------------------------------------------


async def _planner_node(state: AutopilotState) -> dict:
    """Decompose the question into ≤5 investigation steps via Groq Llama."""
    api_key = os.getenv("groq_api_key")
    if not api_key:
        logger.warning("planner_node: groq_api_key not set — single-step fallback")
        return {"plan": [state["question"]], "iteration": 0}

    import groq as groq_module

    system_prompt = (
        "You are a senior data analyst. Decompose the business question into a list of "
        "≤5 concrete investigation steps. Each step must be answerable with ONE tool:\n"
        "  - sql_query: run a SQL query against the connected database\n"
        "  - doc_search: search uploaded PDF documents\n"
        "  - python_analysis: run Python / generate charts — ONLY after a sql_query step has fetched data\n\n"
        "CRITICAL RULES:\n"
        "1. If the question asks for a chart, heatmap, plot, or visualisation, the step BEFORE it "
        "MUST be a sql_query step that fetches the required data. Never start with python_analysis.\n"
        "2. Write each step as a clear action verb phrase, e.g. 'Fetch revenue by region for Q3.' "
        "or 'Generate a heatmap of the correlation matrix.' — NOT 'Query data for heatmap generation'.\n"
        "3. For chart/visualisation questions with no other analysis needed, use exactly 2 steps: "
        "first a sql_query fetch step, then a python_analysis visualisation step.\n\n"
        "Return ONLY a JSON array of short step strings (no keys, no explanation), e.g.:\n"
        '["Fetch total revenue by region for Q3.", "Identify top 3 products by revenue.", '
        '"Generate a bar chart of revenue trend."]'
    )

    try:
        from api.utils.llm_provider import chat_completion

        raw = chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {state['question']}"},
            ],
            temperature=0,
            max_tokens=400,
        )
        # Strip markdown fences if the model adds them
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

        plan: list[str] = json.loads(raw)
        if not isinstance(plan, list):
            raise ValueError("Plan is not a list")
        plan = [str(s).strip() for s in plan[:MAX_ITERATIONS] if str(s).strip()]
    except Exception as exc:
        logger.warning("planner_node LLM call failed (%s) — single-step fallback", exc)
        plan = [state["question"]]

    return {"plan": plan, "iteration": 0}


# ---------------------------------------------------------------------------
# Tool selection heuristic
# ---------------------------------------------------------------------------


def _select_tool(step: str) -> str:
    """Choose sql_query | doc_search | python_analysis based on step wording.

    Data-fetch verbs (query/fetch/get/retrieve/select/find) always win over
    visualisation keywords so "Fetch data for heatmap" stays sql_query.
    """
    s = step.lower()
    # Data-fetch verbs take priority — these are SQL steps even if chart words present
    DATA_FETCH = ("fetch ", "retrieve ", "get ", "select ", "find ", "query ", "calculate ",
                  "compute ", "count ", "sum ", "aggregate ", "identify ")
    if any(s.startswith(k) or f" {k}" in s for k in DATA_FETCH):
        # Only override to python_analysis if the step is PURELY about visualisation
        # (no data-fetch verb AND chart keyword present)
        pass  # fall through to explicit chart check below with fetch-verb guard
    PYTHON_KEYWORDS = ("chart", "plot", "visuali", "correlat", "distribut",
                       "python", "scatter", "heatmap", "regression", "generate a ",
                       "create a ", "draw ")
    has_chart = any(k in s for k in PYTHON_KEYWORDS)
    has_fetch = any(s.startswith(k) or f" {k}" in s for k in DATA_FETCH)
    if has_chart and not has_fetch:
        return "python_analysis"
    if any(k in s for k in ("document", "report", "pdf", "file", "upload",
                             "manual", "policy", "contract", "according to")):
        return "doc_search"
    return "sql_query"


# ---------------------------------------------------------------------------
# ExecutorNode factory
# ---------------------------------------------------------------------------


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

    async def executor_node(state: AutopilotState) -> dict:
        iteration = state["iteration"]
        plan = state.get("plan", [])

        if iteration >= len(plan):
            return {"iteration": iteration}

        step = plan[iteration]
        tool = _select_tool(step)
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
        }
        new_citations: list[dict] = []

        try:
            # ── sql_query ──────────────────────────────────────────────────
            if tool == "sql_query":
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
                    preview = sql_meta.get("preview", [])
                    result_entry["result"] = (
                        f"{row_count} rows returned. "
                        f"Preview: {json.dumps(preview[:3], default=str)}"
                    )
                    result_entry["sql"] = sql_meta.get("sql")
                    # Persist artifact for potential python_analysis in a later step
                    if preview and session_artifacts_table is not None:
                        from api.artifact_service import save_artifact

                        artifact_id = await save_artifact(
                            session_id=session_id,
                            turn_id=iteration + 1,
                            artifact_type="sql_result",
                            name=step[:80],
                            result_dicts=preview,
                            session_artifacts_table=session_artifacts_table,
                            async_session_factory=async_session_factory,
                        )
                        result_entry["artifact_id"] = artifact_id
                else:
                    result_entry["result"] = "No SQL result returned."

            # ── doc_search ─────────────────────────────────────────────────
            elif tool == "doc_search":
                from api.hybrid_service import rag_retrieve

                context, citations = await rag_retrieve(
                    question=step,
                    session_id=session_id,
                    vector_stores=vector_stores,
                )
                result_entry["result"] = (context[:800] if context
                                          else "No relevant documents found.")
                new_citations = citations

            # ── python_analysis ────────────────────────────────────────────
            elif tool == "python_analysis":
                # Load the most recent sql_result artifact as input data
                prior_rows: list[dict] = []
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

                if not prior_rows:
                    result_entry["result"] = "No prior SQL data available for analysis."
                else:
                    from api.sandbox_service import generate_analysis_code, run_python

                    persona_data = expert_personas.get(persona, expert_personas.get("Generalist", {}))
                    persona_def = persona_data.get("persona_def", "") if isinstance(persona_data, dict) else ""
                    code = await generate_analysis_code(
                        result_dicts=prior_rows,
                        question=step,
                        persona_def=persona_def,
                        chart_type="auto",
                    )
                    if code:
                        sandbox_result = await run_python(code)
                        result_entry["result"] = (
                            sandbox_result.stdout[:500] if sandbox_result.stdout
                            else "Analysis complete."
                        )
                        if sandbox_result.error:
                            result_entry["error"] = sandbox_result.error
                        # Persist first chart as an artifact
                        if sandbox_result.charts and session_artifacts_table is not None:
                            from api.artifact_service import save_artifact

                            artifact_id = await save_artifact(
                                session_id=session_id,
                                turn_id=iteration + 1,
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

        except Exception as exc:
            logger.warning("executor_node step %d ('%s') failed: %s", iteration, tool, exc)
            result_entry["error"] = str(exc)

        update: dict = {
            "iteration": iteration + 1,
            "steps_completed": [result_entry],
        }
        if new_citations:
            update["citations"] = new_citations
        return update

    return executor_node


# ---------------------------------------------------------------------------
# SynthesizerNode
# ---------------------------------------------------------------------------


async def _synthesizer_node(state: AutopilotState) -> dict:
    """Synthesise all step results into a final markdown answer via Groq Llama."""
    steps = state.get("steps_completed", [])

    if not steps:
        return {"final_answer": "No investigation steps were completed.", "citations": []}

    steps_text = "\n\n".join(
        f"Step {i + 1} [{s['tool']}]: {s['step']}\n"
        f"Result: {s.get('result', 'N/A')}"
        + (f"\nError: {s['error']}" if s.get("error") else "")
        for i, s in enumerate(steps)
    )

    api_key = os.getenv("groq_api_key")
    if not api_key:
        return {"final_answer": f"Investigation complete.\n\n{steps_text}", "citations": []}

    system_prompt = (
        "You are a senior data analyst. Based on the investigation steps and results "
        "below, write a clear, concise final answer to the original question. "
        "Use specific numbers and findings from the results. "
        "Format as markdown with headers and bullet points. Max 400 words."
    )
    user_content = (
        f"Original question: {state['question']}\n\n"
        f"Investigation results:\n{steps_text}"
    )

    try:
        from api.utils.llm_provider import chat_completion

        final_answer = chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=600,
        )
    except Exception as exc:
        logger.warning("synthesizer_node LLM call failed: %s", exc)
        final_answer = f"Investigation complete.\n\n{steps_text}"

    return {"final_answer": final_answer}


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------


def _should_continue(state: AutopilotState) -> str:
    """Conditional edge: loop executor until plan is exhausted or limits hit."""
    iteration = state.get("iteration", 0)
    plan = state.get("plan", [])
    timed_out = state.get("timed_out", False)
    if timed_out or iteration >= len(plan) or iteration >= MAX_ITERATIONS:
        return "synthesize"
    return "execute"


def _build_graph(executor_node):
    graph = StateGraph(AutopilotState)
    graph.add_node("planner", _planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("synthesizer", _synthesizer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges(
        "executor",
        _should_continue,
        {"execute": "executor", "synthesize": "synthesizer"},
    )
    graph.add_edge("synthesizer", END)
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
        "plan": [],
        "steps_completed": [],
        "iteration": 0,
        "final_answer": "",
        "citations": [],
        "timed_out": False,
    }

    step_num = 0
    all_citations: list[dict] = []

    try:
        async for state_update in app.astream(initial_state, stream_mode="updates"):
            # Wall-clock guard — emit warning but continue to synthesizer
            if time.monotonic() - start_time > TOTAL_TIMEOUT_S:
                yield _sse({
                    "type": "warning",
                    "content": (
                        f"Investigation reached the {TOTAL_TIMEOUT_S}s time limit. "
                        "Showing partial results."
                    ),
                })
                break

            for node_name, updates in state_update.items():
                if node_name == "planner":
                    plan = updates.get("plan", [])
                    yield _sse({
                        "type": "plan",
                        "content": f"Investigation plan: {len(plan)} step(s)",
                        "steps": plan,
                    })

                elif node_name == "executor":
                    completed = updates.get("steps_completed", [])
                    for step_result in completed:
                        step_num += 1
                        yield _sse({
                            "type": "step",
                            "step_num": step_num,
                            "tool": step_result.get("tool", ""),
                            "step_label": step_result.get("step", ""),
                            "content": step_result.get("result", ""),
                            "artifact_id": step_result.get("artifact_id"),
                            "chart_b64": step_result.get("chart_b64"),
                            "error": step_result.get("error"),
                        })
                    # Accumulate citations from doc_search steps
                    new_cits = updates.get("citations", [])
                    if new_cits:
                        all_citations.extend(new_cits)

                elif node_name == "synthesizer":
                    from api.utils.pii_masking import mask_pii
                    final_answer = mask_pii(updates.get("final_answer", ""))
                    yield _sse({"type": "answer", "content": final_answer})
                    yield _sse({"type": "done", "citations": all_citations})

    except Exception as exc:
        logger.error("run_autopilot failed: %s", exc)
        yield _sse({"type": "error", "content": f"Autopilot error: {exc}"})
