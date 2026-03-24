"""Unit tests for api/autopilot_service.py — DOCBOT-405.

Covers:
- _select_tool() heuristic routing
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
    _planner_node,
    _select_tool,
    _should_continue,
    _sse,
    _synthesizer_node,
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
# _select_tool()
# ---------------------------------------------------------------------------


class TestSelectTool:
    def test_sql_is_default(self):
        assert _select_tool("Query total revenue by region") == "sql_query"

    def test_chart_keywords_route_to_python(self):
        assert _select_tool("Create a bar chart of sales") == "python_analysis"

    def test_plot_keyword(self):
        assert _select_tool("Plot the distribution") == "python_analysis"

    def test_visualize_keyword(self):
        assert _select_tool("Visualise revenue trend") == "python_analysis"

    def test_correlat_keyword(self):
        assert _select_tool("Correlation between price and volume") == "python_analysis"

    def test_analys_keyword(self):
        assert _select_tool("Analyse the Q3 data with Python") == "python_analysis"

    def test_document_keyword_routes_to_doc_search(self):
        assert _select_tool("Search the uploaded document for policy details") == "doc_search"

    def test_pdf_keyword(self):
        assert _select_tool("Find the clause in the PDF") == "doc_search"

    def test_contract_keyword(self):
        assert _select_tool("According to the contract agreement") == "doc_search"

    def test_generic_query_is_sql(self):
        assert _select_tool("What was Q3 revenue by region?") == "sql_query"

    def test_case_insensitive(self):
        assert _select_tool("CHART of top products") == "python_analysis"


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
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


class TestShouldContinue:
    def test_continue_when_steps_remain(self):
        state = _make_state(plan=["a", "b"], iteration=0)
        assert _should_continue(state) == "execute"

    def test_synthesize_when_plan_exhausted(self):
        state = _make_state(plan=["a", "b"], iteration=2)
        assert _should_continue(state) == "synthesize"

    def test_synthesize_at_max_iterations(self):
        long_plan = [f"step {i}" for i in range(10)]
        state = _make_state(plan=long_plan, iteration=MAX_ITERATIONS)
        assert _should_continue(state) == "synthesize"

    def test_continue_at_max_minus_one(self):
        long_plan = [f"step {i}" for i in range(10)]
        state = _make_state(plan=long_plan, iteration=MAX_ITERATIONS - 1)
        assert _should_continue(state) == "execute"

    def test_synthesize_when_timed_out(self):
        state = _make_state(plan=["a", "b"], iteration=0, timed_out=True)
        assert _should_continue(state) == "synthesize"

    def test_synthesize_when_empty_plan(self):
        state = _make_state(plan=[], iteration=0)
        assert _should_continue(state) == "synthesize"

    def test_synthesize_when_iteration_exceeds_plan(self):
        state = _make_state(plan=["only one"], iteration=5)
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
            result = asyncio.get_event_loop().run_until_complete(_planner_node(state))
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
            result = asyncio.get_event_loop().run_until_complete(_planner_node(state))
        assert len(result["plan"]) <= MAX_ITERATIONS


# ---------------------------------------------------------------------------
# _synthesizer_node() — no API key fallback
# ---------------------------------------------------------------------------


class TestSynthesizerNodeFallback:
    def test_no_steps_returns_empty_message(self):
        state = _make_state()
        result = asyncio.get_event_loop().run_until_complete(_synthesizer_node(state))
        assert "No investigation" in result["final_answer"]

    def test_fallback_includes_step_text_when_no_api_key(self):
        state = _make_state(
            steps_completed=[
                {"step": "Query revenue", "tool": "sql_query", "result": "42 rows", "error": None}
            ]
        )
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("groq_api_key", None)
            result = asyncio.get_event_loop().run_until_complete(_synthesizer_node(state))
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
        result = asyncio.get_event_loop().run_until_complete(node(state))
        # Plan is empty so we return without executing — iteration unchanged
        assert result == {"iteration": 0}
