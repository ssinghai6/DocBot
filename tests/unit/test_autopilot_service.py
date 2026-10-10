"""Unit tests for api/autopilot_service.py — DOCBOT-405.

Covers:
- _select_tool_heuristic() heuristic routing
- _should_continue() graph edge logic (max iterations, plan exhaustion, timed_out flag)
- _sse() serialisation helper
- AutopilotState TypedDict field defaults
- PlannerNode fallback when groq_api_key is absent
- SynthesizerNode fallback when groq_api_key is absent
- Hard iteration limit enforced by _should_continue
- Wall-clock timeout flag respected by _should_continue
- make_executor_node() returns a callable
"""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.autopilot_service import (
    MAX_ITERATIONS,
    TOTAL_TIMEOUT_S,
    AutopilotState,
    _next_wave_indices,
    _planner_node,
    _select_tool_heuristic,
    _select_tool_llm,
    _should_continue,
    _sse,
    _synthesizer_node,
    _verifier_node,
    make_executor_node,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_max_iterations_is_five(self):
        assert MAX_ITERATIONS == 5

    def test_timeout_is_ninety_seconds(self):
        assert TOTAL_TIMEOUT_S == 90


# ---------------------------------------------------------------------------
# _sse()
# ---------------------------------------------------------------------------


class TestSse:
    def test_produces_data_prefix(self):
        result = _sse({"type": "done"})
        assert result.startswith("data: ")
        assert result.endswith("\n\n")

    def test_payload_is_valid_json(self):
        result = _sse({"type": "step", "step_num": 1})
        payload = json.loads(result[6:].strip())
        assert payload["type"] == "step"
        assert payload["step_num"] == 1

    def test_handles_non_serializable_with_default_str(self):
        from datetime import datetime
        result = _sse({"ts": datetime(2026, 1, 1)})
        assert "2026" in result


# ---------------------------------------------------------------------------
# _select_tool_heuristic()
# ---------------------------------------------------------------------------


class TestSelectTool:
    def test_sql_is_default(self):
        assert _select_tool_heuristic("Query total revenue by region") == "sql_query"

    def test_chart_keywords_route_to_python(self):
        assert _select_tool_heuristic("Create a bar chart of sales") == "python_analysis"

    def test_plot_keyword(self):
        assert _select_tool_heuristic("Plot the distribution") == "python_analysis"

    def test_visualize_keyword(self):
        assert _select_tool_heuristic("Visualise revenue trend") == "python_analysis"

    def test_correlat_keyword(self):
        assert _select_tool_heuristic("Correlation between price and volume") == "python_analysis"

    def test_analys_keyword(self):
        assert _select_tool_heuristic("Analyse the Q3 data with Python") == "python_analysis"

    def test_document_keyword_routes_to_doc_search(self):
        assert _select_tool_heuristic("Search the uploaded document for policy details") == "doc_search"

    def test_pdf_keyword(self):
        assert _select_tool_heuristic("Find the clause in the PDF") == "doc_search"

    def test_contract_keyword(self):
        assert _select_tool_heuristic("According to the contract agreement") == "doc_search"

    def test_generic_query_is_sql(self):
        assert _select_tool_heuristic("What was Q3 revenue by region?") == "sql_query"

    def test_case_insensitive(self):
        assert _select_tool_heuristic("CHART of top products") == "python_analysis"

    def test_fetch_verb_beats_chart_keyword(self):
        """'Fetch data for heatmap' should stay sql_query — fetch verb wins."""
        assert _select_tool_heuristic("Fetch data for heatmap generation") == "sql_query"

    def test_retrieve_with_chart_keyword_is_sql(self):
        assert _select_tool_heuristic("Retrieve revenue data for scatter plot") == "sql_query"

    def test_generate_heatmap_without_fetch_is_python(self):
        """'Generate a heatmap' with no fetch verb → python_analysis."""
        assert _select_tool_heuristic("Generate a heatmap of the correlation matrix") == "python_analysis"

    def test_create_chart_without_fetch_is_python(self):
        assert _select_tool_heuristic("Create a line chart of the results") == "python_analysis"


# ---------------------------------------------------------------------------
# _should_continue()
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> AutopilotState:
    """Build a minimal AutopilotState for testing _should_continue."""
    base: AutopilotState = {
        "question": "test",
        "session_id": "s1",
        "connection_id": "c1",
        "persona": "Generalist",
        "plan": ["step 1", "step 2", "step 3"],
        "steps_completed": [],
        "iteration": 0,
        "final_answer": "",
        "citations": [],
        "timed_out": False,
        "has_docs": False,
        "has_db": True,
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


class TestShouldContinue:
    """DOCBOT-1406: exhaustion is measured by len(steps_completed) — the number
    of individual steps actually finished, since one wave can finish several
    steps at once. `iteration` now counts executor *waves* and is checked
    separately against MAX_ITERATIONS as a hard ceiling on round-trips."""

    def test_continue_when_steps_remain(self):
        state = _make_state(plan=["a", "b"], iteration=0, steps_completed=[])
        assert _should_continue(state) == "execute"

    def test_synthesize_when_plan_exhausted(self):
        state = _make_state(
            plan=["a", "b"],
            iteration=1,
            steps_completed=[{"step": "a", "tool": "sql_query"}, {"step": "b", "tool": "doc_search"}],
        )
        assert _should_continue(state) == "synthesize"

    def test_synthesize_when_wave_completes_multiple_steps_at_once(self):
        """A single wave can finish 2 steps concurrently — plan is exhausted
        after 1 wave even though there are 2 steps, proving the exhaustion
        check no longer assumes 1 step per iteration."""
        state = _make_state(
            plan=["fetch a", "fetch b"],
            iteration=1,
            steps_completed=[
                {"step": "fetch a", "tool": "sql_query"},
                {"step": "fetch b", "tool": "sql_query"},
            ],
        )
        assert _should_continue(state) == "synthesize"

    def test_synthesize_at_max_iterations(self):
        long_plan = [f"step {i}" for i in range(10)]
        state = _make_state(plan=long_plan, iteration=MAX_ITERATIONS, steps_completed=[])
        assert _should_continue(state) == "synthesize"

    def test_continue_at_max_minus_one(self):
        long_plan = [f"step {i}" for i in range(10)]
        state = _make_state(plan=long_plan, iteration=MAX_ITERATIONS - 1, steps_completed=[])
        assert _should_continue(state) == "execute"

    def test_synthesize_when_timed_out(self):
        state = _make_state(plan=["a", "b"], iteration=0, timed_out=True, steps_completed=[])
        assert _should_continue(state) == "synthesize"

    def test_synthesize_when_empty_plan(self):
        state = _make_state(plan=[], iteration=0, steps_completed=[])
        assert _should_continue(state) == "synthesize"

    def test_synthesize_when_steps_completed_exceeds_plan(self):
        state = _make_state(
            plan=["only one"],
            iteration=1,
            steps_completed=[{"step": "only one", "tool": "sql_query"}],
        )
        assert _should_continue(state) == "synthesize"


# ---------------------------------------------------------------------------
# _planner_node() — no API key fallback
# ---------------------------------------------------------------------------


class TestPlannerNodeFallback:
    def test_single_step_fallback_when_no_api_key(self):
        state = _make_state(question="What is revenue?")
        with patch.dict(os.environ, {}, clear=True):
            # Ensure groq_api_key is absent
            os.environ.pop("groq_api_key", None)
            result = asyncio.run(_planner_node(state))
        assert result["plan"] == ["What is revenue?"]
        assert result["iteration"] == 0

    def test_plan_never_exceeds_max_iterations(self):
        """Even if API returns more steps, we cap at MAX_ITERATIONS."""
        long_plan = [f"step {i}" for i in range(10)]
        state = _make_state(question="Big question")
        # groq is imported inline inside _planner_node; patch the top-level module
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(long_plan)
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("groq.Groq", return_value=mock_client), \
             patch.dict(os.environ, {"groq_api_key": "fake"}):
            result = asyncio.run(_planner_node(state))
        assert len(result["plan"]) <= MAX_ITERATIONS


# ---------------------------------------------------------------------------
# _synthesizer_node() — no API key fallback
# ---------------------------------------------------------------------------


class TestSynthesizerNodeFallback:
    def test_no_steps_returns_empty_message(self):
        state = _make_state()
        result = asyncio.run(_synthesizer_node(state))
        assert "No investigation" in result["final_answer"]

    def test_fallback_includes_step_text_when_no_api_key(self):
        state = _make_state(
            steps_completed=[
                {"step": "Query revenue", "tool": "sql_query", "result": "42 rows", "error": None}
            ]
        )
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("groq_api_key", None)
            result = asyncio.run(_synthesizer_node(state))
        assert "42 rows" in result["final_answer"] or "Investigation complete" in result["final_answer"]


# ---------------------------------------------------------------------------
# make_executor_node()
# ---------------------------------------------------------------------------


class TestMakeExecutorNode:
    def test_returns_callable(self):
        node = make_executor_node(
            db_connections_table=MagicMock(),
            schema_cache_table=MagicMock(),
            query_history_table=MagicMock(),
            query_embeddings_table=MagicMock(),
            session_artifacts_table=MagicMock(),
            table_embeddings_table=MagicMock(),
            async_session_factory=MagicMock(),
            expert_personas={"Generalist": {"persona_def": "test"}},
            vector_stores={},
        )
        assert callable(node)

    def test_executor_increments_iteration(self):
        """When plan is exhausted (iteration >= len(plan)), returns same iteration unchanged."""
        node = make_executor_node(
            db_connections_table=MagicMock(),
            schema_cache_table=MagicMock(),
            query_history_table=MagicMock(),
            query_embeddings_table=MagicMock(),
            session_artifacts_table=MagicMock(),
            table_embeddings_table=MagicMock(),
            async_session_factory=MagicMock(),
            expert_personas={"Generalist": {"persona_def": ""}},
            vector_stores={},
        )
        state = _make_state(plan=[], iteration=0)
        result = asyncio.run(node(state))
        # Plan is empty so we return without executing — iteration unchanged
        assert result == {"iteration": 0}


# ---------------------------------------------------------------------------
# _next_wave_indices() — DOCBOT-1406 wave grouping
# ---------------------------------------------------------------------------


class TestNextWaveIndices:
    def test_consecutive_sql_steps_form_one_wave(self):
        plan = ["Fetch revenue by region", "Fetch costs by region"]
        state = _make_state(plan=plan)
        assert _next_wave_indices(plan, 0, state) == [0, 1]

    def test_consecutive_doc_steps_form_one_wave(self):
        plan = ["Search the document for revenue", "Find the report's risk factors"]
        state = _make_state(plan=plan, has_db=False, has_docs=True)
        assert _next_wave_indices(plan, 0, state) == [0, 1]

    def test_fetch_then_analysis_is_two_waves(self):
        plan = ["Fetch revenue by region", "Generate a bar chart of revenue"]
        state = _make_state(plan=plan)
        assert _next_wave_indices(plan, 0, state) == [0]
        assert _next_wave_indices(plan, 1, state) == [1]

    def test_consecutive_analysis_steps_after_fetch_form_one_wave(self):
        plan = [
            "Fetch revenue by region",
            "Generate a bar chart of revenue",
            "Plot the cost distribution",
        ]
        state = _make_state(plan=plan)
        assert _next_wave_indices(plan, 0, state) == [0]
        # Both analysis steps depend only on the completed fetch wave, not on
        # each other, so they batch into one wave.
        assert _next_wave_indices(plan, 1, state) == [1, 2]

    def test_mixed_sql_and_doc_fetch_steps_still_batch(self):
        plan = ["Fetch revenue from the database", "Search the filing for risk factors"]
        state = _make_state(plan=plan, has_db=True, has_docs=True)
        assert _next_wave_indices(plan, 0, state) == [0, 1]

    def test_alternating_fetch_analysis_stays_sequential(self):
        plan = [
            "Fetch revenue by region",
            "Generate a bar chart of revenue",
            "Fetch costs by region",
            "Plot the cost distribution",
        ]
        state = _make_state(plan=plan)
        assert _next_wave_indices(plan, 0, state) == [0]
        assert _next_wave_indices(plan, 1, state) == [1]
        assert _next_wave_indices(plan, 2, state) == [2]
        assert _next_wave_indices(plan, 3, state) == [3]

    def test_start_past_end_returns_empty(self):
        plan = ["Fetch revenue"]
        state = _make_state(plan=plan)
        assert _next_wave_indices(plan, 1, state) == []


# ---------------------------------------------------------------------------
# Executor concurrency — DOCBOT-1406
# ---------------------------------------------------------------------------


class TestExecutorConcurrency:
    """Verifies the executor dispatches an independent wave of steps via
    asyncio.gather rather than one step per LangGraph iteration."""

    @pytest.fixture(autouse=True)
    def _mock_tool_router(self):
        """DOCBOT-1513: _run_single_step now calls the model-driven
        _select_tool_llm instead of the heuristic directly. Mock call_llm to
        mirror the heuristic's own answer (these tests all use has_db=True,
        has_docs=False) so the asserted tool ("sql_query") and the
        concurrency/fan-in behavior under test are unaffected, without ever
        making a real network call."""

        async def _fake_call_llm(prompt, **kwargs):
            step_text = prompt.rsplit("Step:", 1)[-1].strip()
            tool = _select_tool_heuristic(step_text, has_db=True, has_docs=False)
            return json.dumps({"tool": tool})

        with patch(
            "api.utils.llm_provider.call_llm",
            new=AsyncMock(side_effect=_fake_call_llm),
        ):
            yield

    def _node(self):
        return make_executor_node(
            db_connections_table=MagicMock(),
            schema_cache_table=MagicMock(),
            query_history_table=MagicMock(),
            query_embeddings_table=MagicMock(),
            session_artifacts_table=None,
            table_embeddings_table=MagicMock(),
            async_session_factory=MagicMock(),
            expert_personas={"Generalist": {"persona_def": ""}},
            vector_stores={},
        )

    def test_two_sql_steps_run_concurrently_not_sequentially(self):
        """Two independent sql_query steps should overlap in wall-clock time —
        if the executor still ran them one-at-a-time, total time would be
        ~2x a single step's sleep instead of ~1x."""
        node = self._node()
        state = _make_state(
            plan=["Fetch revenue by region", "Fetch costs by region"],
            has_db=True, has_docs=False,
        )

        async def slow_sql_result(**kwargs):
            await asyncio.sleep(0.15)
            return {"row_count": 1, "result_preview": [{"x": 1}], "sql_query": "SELECT 1"}

        async def _run():
            with patch("api.hybrid_service._collect_sql_result", side_effect=slow_sql_result):
                start = asyncio.get_event_loop().time()
                result = await node(state)
                elapsed = asyncio.get_event_loop().time() - start
            return result, elapsed

        result, elapsed = asyncio.run(_run())
        assert len(result["steps_completed"]) == 2
        # Sequential execution would take >= 0.30s; concurrent should be well under.
        assert elapsed < 0.25, f"steps did not run concurrently (took {elapsed:.3f}s)"

    def test_wave_results_fan_in_by_plan_order_not_completion_order(self):
        """The step that finishes LAST (index 0, longer sleep) must still land
        first in steps_completed, matching plan order — proving fan-in doesn't
        depend on which coroutine happened to complete first."""
        node = self._node()
        state = _make_state(
            plan=["Fetch revenue by region", "Fetch costs by region"],
            has_db=True, has_docs=False,
        )

        call_count = {"n": 0}

        async def variable_sql_result(**kwargs):
            call_count["n"] += 1
            # First call (plan index 0) sleeps longer than the second, so it
            # completes AFTER the second call if they're running concurrently.
            delay = 0.12 if call_count["n"] == 1 else 0.02
            await asyncio.sleep(delay)
            return {
                "row_count": 1,
                "result_preview": [{"x": call_count["n"]}],
                "sql_query": "SELECT 1",
            }

        with patch("api.hybrid_service._collect_sql_result", side_effect=variable_sql_result):
            result = asyncio.run(node(state))

        steps = result["steps_completed"]
        assert steps[0]["step"] == "Fetch revenue by region"
        assert steps[1]["step"] == "Fetch costs by region"

    def test_one_failing_step_does_not_drop_or_crash_siblings(self):
        """A step whose tool call raises should surface as an error entry for
        that step only — the sibling step's result must still be present."""
        node = self._node()
        state = _make_state(
            plan=["Fetch revenue by region", "Fetch costs by region"],
            has_db=True, has_docs=False,
        )

        call_count = {"n": 0}

        async def flaky_sql_result(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated DB connection failure")
            return {"row_count": 1, "result_preview": [{"x": 1}], "sql_query": "SELECT 1"}

        with patch("api.hybrid_service._collect_sql_result", side_effect=flaky_sql_result):
            result = asyncio.run(node(state))

        steps = result["steps_completed"]
        assert len(steps) == 2
        errored = [s for s in steps if s.get("error")]
        ok = [s for s in steps if not s.get("error")]
        assert len(errored) == 1
        assert len(ok) == 1

    def test_iteration_increments_by_one_wave_regardless_of_step_count(self):
        """A 2-step wave still only costs 1 unit of the iteration/wave budget."""
        node = self._node()
        state = _make_state(
            plan=["Fetch revenue by region", "Fetch costs by region"],
            iteration=0, has_db=True, has_docs=False,
        )

        async def fast_sql_result(**kwargs):
            return {"row_count": 1, "result_preview": [{"x": 1}], "sql_query": "SELECT 1"}

        with patch("api.hybrid_service._collect_sql_result", side_effect=fast_sql_result):
            result = asyncio.run(node(state))

        assert result["iteration"] == 1
        assert len(result["steps_completed"]) == 2

    def test_wave_barrier_blocks_analysis_step_until_fetch_wave_done(self):
        """A plan of [fetch, fetch, analyze] must only dispatch the two fetch
        steps in the first executor call — analyze must wait for a second
        executor invocation once steps_completed reflects the fetch results."""
        node = self._node()
        state = _make_state(
            plan=[
                "Fetch revenue by region",
                "Fetch costs by region",
                "Generate a bar chart of margin",
            ],
            has_db=True, has_docs=False,
        )

        async def fast_sql_result(**kwargs):
            return {"row_count": 1, "result_preview": [{"x": 1}], "sql_query": "SELECT 1"}

        with patch("api.hybrid_service._collect_sql_result", side_effect=fast_sql_result):
            result = asyncio.run(node(state))

        assert len(result["steps_completed"]) == 2
        assert all(s["tool"] == "sql_query" for s in result["steps_completed"])


# ---------------------------------------------------------------------------
# _select_tool_heuristic() with data-source flags
# ---------------------------------------------------------------------------


class TestSelectToolDataSourceFlags:
    """Test _select_tool_heuristic() respects has_db / has_docs flags."""

    def test_no_db_never_returns_sql_query(self):
        """When has_db=False, _select_tool_heuristic never returns sql_query."""
        steps = [
            "Query total revenue by region",
            "Fetch all customer records",
            "What was Q3 revenue by region?",
            "Select top products",
            "Count active users",
        ]
        for step in steps:
            result = _select_tool_heuristic(step, has_db=False, has_docs=True)
            assert result != "sql_query", f"Step '{step}' returned sql_query with has_db=False"

    def test_no_db_data_fetch_falls_back_to_doc_search(self):
        """Data-fetch verbs with has_db=False and has_docs=True → doc_search."""
        assert _select_tool_heuristic("Fetch revenue data", has_db=False, has_docs=True) == "doc_search"
        assert _select_tool_heuristic("Query total by region", has_db=False, has_docs=True) == "doc_search"

    def test_no_db_no_docs_no_csv_returns_unsupported(self):
        """No data source at all → unsupported (autopilot can't fetch external data)."""
        assert _select_tool_heuristic("Fetch the data", has_db=False, has_docs=False, has_csv=False) == "unsupported"

    def test_has_docs_returns_doc_search_for_document_steps(self):
        """Doc keywords route to doc_search regardless of flags."""
        assert _select_tool_heuristic("Search the uploaded document", has_db=True, has_docs=True) == "doc_search"
        assert _select_tool_heuristic("Find in the PDF report", has_db=False, has_docs=True) == "doc_search"

    def test_chart_keywords_still_python_with_no_db(self):
        """Chart/viz keywords → python_analysis when at least one data source exists."""
        assert _select_tool_heuristic("Create a bar chart of results", has_db=False, has_docs=True) == "python_analysis"
        # No data source at all → unsupported, even for chart keywords
        assert _select_tool_heuristic("Plot the distribution", has_db=False, has_docs=False, has_csv=False) == "unsupported"
        # CSV alone is enough to keep python_analysis
        assert _select_tool_heuristic("Plot the distribution", has_db=False, has_docs=False, has_csv=True) == "python_analysis"

    def test_default_flags_preserve_existing_behavior(self):
        """With default flags (has_db=True, has_docs=False), behavior is unchanged."""
        assert _select_tool_heuristic("Query total revenue") == "sql_query"
        assert _select_tool_heuristic("Create a bar chart") == "python_analysis"
        assert _select_tool_heuristic("Search the uploaded document") == "doc_search"

    def test_generic_step_with_docs_only(self):
        """A generic question with only docs → doc_search."""
        result = _select_tool_heuristic("What was the revenue?", has_db=False, has_docs=True)
        assert result == "doc_search"


# ---------------------------------------------------------------------------
# AutopilotState with optional connection_id
# ---------------------------------------------------------------------------


class TestAutopilotStateOptionalConnection:
    def test_empty_connection_id_accepted(self):
        """AutopilotState accepts empty string for connection_id (doc-only sessions)."""
        state = _make_state(connection_id="", has_db=False, has_docs=True)
        assert state["connection_id"] == ""
        assert state["has_docs"] is True
        assert state["has_db"] is False

    def test_should_continue_works_without_connection(self):
        """_should_continue works normally with empty connection_id."""
        state = _make_state(connection_id="", has_db=False, has_docs=True, plan=["a", "b"], iteration=0)
        assert _should_continue(state) == "execute"

    def test_planner_fallback_with_doc_only_state(self):
        """Planner falls back to single-step when no API key, even for doc-only."""
        state = _make_state(connection_id="", has_db=False, has_docs=True, question="Summarize the report")
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("groq_api_key", None)
            result = asyncio.run(_planner_node(state))
        assert result["plan"] == ["Summarize the report"]


# ---------------------------------------------------------------------------
# DOCBOT-1508: per-session soft LLM cost ceiling gating
# ---------------------------------------------------------------------------


class TestSessionCostCeilingGate:
    def test_should_continue_synthesize_when_budget_exceeded(self):
        """budget_exceeded=True routes to synthesize even with steps remaining
        and iteration well under MAX_ITERATIONS."""
        state = _make_state(
            plan=["a", "b", "c"], iteration=1, steps_completed=[{"step": "a", "tool": "sql_query"}],
            budget_exceeded=True,
        )
        assert _should_continue(state) == "synthesize"

    def test_should_continue_execute_when_budget_not_exceeded(self):
        state = _make_state(
            plan=["a", "b"], iteration=0, steps_completed=[], budget_exceeded=False,
        )
        assert _should_continue(state) == "execute"

    def test_should_continue_defaults_budget_exceeded_to_false(self):
        """A state dict without the key (e.g. older callers) behaves as if unset."""
        state = _make_state(plan=["a", "b"], iteration=0, steps_completed=[])
        state.pop("budget_exceeded", None)
        assert _should_continue(state) == "execute"

    def test_executor_stops_wave_dispatch_when_budget_exceeded(self):
        """When the session cost ceiling has been hit, executor_node returns a
        budget_exceeded update and does not dispatch a new wave of steps —
        no sql_query/doc_search/python_analysis side effects should run."""
        node = make_executor_node(
            db_connections_table=MagicMock(),
            schema_cache_table=MagicMock(),
            query_history_table=MagicMock(),
            query_embeddings_table=MagicMock(),
            session_artifacts_table=MagicMock(),
            table_embeddings_table=MagicMock(),
            async_session_factory=MagicMock(),
            expert_personas={"Generalist": {"persona_def": ""}},
            vector_stores={},
        )
        state = _make_state(
            plan=["Fetch revenue by region", "Fetch costs by region"],
            steps_completed=[],
            iteration=0,
            run_id="over-budget-run",
        )

        with patch(
            "api.utils.llm_provider.is_session_budget_exceeded", return_value=True
        ):
            result = asyncio.run(node(state))

        assert result["budget_exceeded"] is True
        assert result["iteration"] == 1
        assert "steps_completed" not in result

    def test_executor_dispatches_normally_when_under_budget(self):
        """Sanity check: with the ceiling check mocked to False, the executor
        still dispatches a wave as before (no change to the happy path)."""
        node = make_executor_node(
            db_connections_table=MagicMock(),
            schema_cache_table=MagicMock(),
            query_history_table=MagicMock(),
            query_embeddings_table=MagicMock(),
            session_artifacts_table=MagicMock(),
            table_embeddings_table=MagicMock(),
            async_session_factory=MagicMock(),
            expert_personas={"Generalist": {"persona_def": ""}},
            vector_stores={},
        )
        state = _make_state(
            plan=["Fetch revenue by region"],
            steps_completed=[],
            iteration=0,
            has_db=False, has_docs=False, has_csv=False,
            run_id="under-budget-run",
        )

        with patch(
            "api.utils.llm_provider.is_session_budget_exceeded", return_value=False
        ):
            result = asyncio.run(node(state))

        assert "steps_completed" in result
        assert "budget_exceeded" not in result




# ---------------------------------------------------------------------------
# _select_tool_llm() — DOCBOT-1513 model-driven tool router
#
# call_llm() is async (LangChain .ainvoke) rather than the synchronous
# chat_completion() — _select_tool_llm runs concurrently for every step in a
# wave via asyncio.gather, and a blocking sync call would serialize the
# whole wave on the event loop. Tests mock call_llm with AsyncMock.
# ---------------------------------------------------------------------------


def _mock_call_llm_json(tool: str):
    """AsyncMock replacement for llm_provider.call_llm returning a JSON
    tool-selection payload."""
    return AsyncMock(return_value=json.dumps({"tool": tool}))


class TestSelectToolLlm:
    def test_valid_tool_is_used_as_is(self):
        """A well-formed LLM response is trusted without falling back."""
        with patch(
            "api.utils.llm_provider.call_llm",
            _mock_call_llm_json("sql_query"),
        ) as mock_cl:
            result = asyncio.run(
                _select_tool_llm("Fetch revenue by region", has_db=True, has_docs=False)
            )
        assert result == "sql_query"
        mock_cl.assert_called_once()

    def test_llm_exception_falls_back_to_heuristic(self):
        """If the LLM call raises, the heuristic's result is used instead."""
        heuristic_result = _select_tool_heuristic(
            "Fetch revenue by region", has_db=True, has_docs=False
        )
        with patch(
            "api.utils.llm_provider.call_llm",
            new=AsyncMock(side_effect=RuntimeError("groq exploded")),
        ):
            result = asyncio.run(
                _select_tool_llm("Fetch revenue by region", has_db=True, has_docs=False)
            )
        assert result == heuristic_result == "sql_query"

    def test_invalid_tool_string_falls_back_to_heuristic(self):
        """An out-of-set tool name from the LLM is rejected and the heuristic
        result is used instead."""
        heuristic_result = _select_tool_heuristic(
            "Search the uploaded document", has_db=False, has_docs=True
        )
        with patch(
            "api.utils.llm_provider.call_llm",
            _mock_call_llm_json("totally_made_up_tool"),
        ):
            result = asyncio.run(
                _select_tool_llm(
                    "Search the uploaded document", has_db=False, has_docs=True
                )
            )
        assert result == heuristic_result == "doc_search"

    def test_malformed_json_falls_back_to_heuristic(self):
        """Non-JSON LLM output (no braces at all) is treated as a parse
        failure, not trusted."""
        heuristic_result = _select_tool_heuristic(
            "Create a bar chart of sales", has_db=True, has_docs=False
        )
        with patch(
            "api.utils.llm_provider.call_llm",
            new=AsyncMock(return_value="sure, the answer is sql_query"),
        ):
            result = asyncio.run(
                _select_tool_llm(
                    "Create a bar chart of sales", has_db=True, has_docs=False
                )
            )
        assert result == heuristic_result

    def test_no_data_source_short_circuits_without_llm_call(self):
        """When no data source is available, 'unsupported' is returned
        immediately without ever calling the LLM."""
        with patch("api.utils.llm_provider.call_llm", new=AsyncMock()) as mock_cl:
            result = asyncio.run(
                _select_tool_llm(
                    "Fetch the data", has_db=False, has_docs=False, has_csv=False
                )
            )
        assert result == "unsupported"
        mock_cl.assert_not_called()

    def test_strips_markdown_fences_before_parsing(self):
        """A fenced JSON response (```json ... ```) is still parsed correctly."""
        fenced = '```json\n{"tool": "doc_search"}\n```'
        with patch(
            "api.utils.llm_provider.call_llm", new=AsyncMock(return_value=fenced)
        ):
            result = asyncio.run(
                _select_tool_llm("Find the clause in the PDF", has_db=False, has_docs=True)
            )
        assert result == "doc_search"

    def test_persona_preference_is_passed_without_overriding_response(self):
        """persona_tool_preference is accepted and threaded into the prompt,
        but the LLM's own tool choice still wins."""
        with patch(
            "api.utils.llm_provider.call_llm",
            _mock_call_llm_json("doc_search"),
        ) as mock_cl:
            result = asyncio.run(
                _select_tool_llm(
                    "Search the filing",
                    has_db=True,
                    has_docs=True,
                    persona_tool_preference="sql_first",
                )
            )
        assert result == "doc_search"
        # The preference hint should appear in the prompt sent to the LLM.
        sent_prompt = mock_cl.call_args.args[0]
        assert "sql_first" in sent_prompt


# ---------------------------------------------------------------------------
# _verifier_node() — DOCBOT-1518 active discrepancy gate
# ---------------------------------------------------------------------------


def _verifier_state(**overrides) -> AutopilotState:
    state = _make_state(
        final_answer="Revenue grew to $330M in Q4.",
        steps_completed=[
            {
                "step": "Fetch Q4 revenue",
                "tool": "sql_query",
                "result": "Q4 revenue was $325M per the database.",
                "error": None,
            }
        ],
    )
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


class TestVerifierNode:
    def test_real_contradiction_triggers_one_resynthesis(self):
        """A flagged contradiction causes exactly one corrective re-synthesis
        call, and its output becomes the final answer."""
        state = _verifier_state()
        verify_response = json.dumps({
            "contradiction_found": True,
            "explanation": "Answer says $330M but retrieved data shows $325M.",
        })
        with patch(
            "api.utils.llm_provider.call_llm",
            new=AsyncMock(return_value=verify_response),
        ) as mock_call_llm, patch(
            "api.utils.llm_provider.chat_completion",
            new=MagicMock(return_value="Revenue grew to $325M in Q4 (corrected)."),
        ) as mock_chat_completion:
            result = asyncio.run(_verifier_node(state))

        mock_call_llm.assert_called_once()
        mock_chat_completion.assert_called_once()
        assert result["final_answer"] == "Revenue grew to $325M in Q4 (corrected)."

    def test_no_contradiction_passes_through_unchanged(self):
        """No contradiction found -> original answer unchanged, no
        corrective re-synthesis call made."""
        state = _verifier_state()
        verify_response = json.dumps({"contradiction_found": False, "explanation": ""})
        with patch(
            "api.utils.llm_provider.call_llm",
            new=AsyncMock(return_value=verify_response),
        ), patch(
            "api.utils.llm_provider.chat_completion",
            new=MagicMock(return_value="should not be called"),
        ) as mock_chat_completion:
            result = asyncio.run(_verifier_node(state))

        mock_chat_completion.assert_not_called()
        assert result["final_answer"] == state["final_answer"]

    def test_verifier_llm_failure_falls_back_to_original_answer(self):
        """If the verifier's own LLM call raises, the original answer passes
        through unchanged and no exception propagates."""
        state = _verifier_state()
        with patch(
            "api.utils.llm_provider.call_llm",
            new=AsyncMock(side_effect=RuntimeError("groq exploded")),
        ), patch(
            "api.utils.llm_provider.chat_completion",
            new=MagicMock(return_value="should not be called"),
        ) as mock_chat_completion:
            result = asyncio.run(_verifier_node(state))

        mock_chat_completion.assert_not_called()
        assert result["final_answer"] == state["final_answer"]

    def test_unparseable_verifier_response_falls_back_to_original_answer(self):
        """A response with no JSON object at all is treated as a parse
        failure, never raises, and keeps the original answer."""
        state = _verifier_state()
        with patch(
            "api.utils.llm_provider.call_llm",
            new=AsyncMock(return_value="not json at all"),
        ):
            result = asyncio.run(_verifier_node(state))

        assert result["final_answer"] == state["final_answer"]

    def test_correction_call_failure_falls_back_to_original_answer(self):
        """Contradiction found, but the corrective re-synthesis call itself
        raises -> original answer is kept, never raises."""
        state = _verifier_state()
        verify_response = json.dumps({
            "contradiction_found": True,
            "explanation": "mismatch",
        })
        with patch(
            "api.utils.llm_provider.call_llm",
            new=AsyncMock(return_value=verify_response),
        ), patch(
            "api.utils.llm_provider.chat_completion",
            new=MagicMock(side_effect=RuntimeError("groq exploded")),
        ):
            result = asyncio.run(_verifier_node(state))

        assert result["final_answer"] == state["final_answer"]

    def test_no_steps_or_no_answer_short_circuits_without_llm_call(self):
        """Nothing to verify (no final_answer, or no completed steps) means
        the verifier never calls the LLM at all."""
        state = _verifier_state(final_answer="", steps_completed=[])
        with patch("api.utils.llm_provider.call_llm", new=AsyncMock()) as mock_call_llm:
            result = asyncio.run(_verifier_node(state))

        mock_call_llm.assert_not_called()
        assert result["final_answer"] == ""
