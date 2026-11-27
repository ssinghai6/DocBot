"""DOCBOT-2721: Autopilot planner budget + source-aware fallback plan.

Prod: the planner (max_tokens=400 on a reasoning model) intermittently
returned no JSON, the retry failed the same way, and the fallback plan was
the raw question as one step, which the router sent to doc_search only. A
hybrid PDF + DB question then never touched the database.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

from api import autopilot_service as ap

Q = "Check every quarter's net income in the 10-K against the database and forecast Q1 2026."


def _state(**flags) -> dict:
    base = {
        "question": Q, "session_id": "s1", "connection_id": "c1", "persona": "Generalist",
        "run_id": "r1", "plan": [], "steps_completed": [], "iteration": 0,
        "has_docs": False, "has_db": False, "has_csv": False,
    }
    base.update(flags)
    return base


def _run_planner(state: dict, side_effect) -> tuple[dict, MagicMock]:
    mock = MagicMock(side_effect=side_effect)
    with patch.dict(os.environ, {"groq_api_key": "fake"}), \
         patch("api.utils.llm_provider.chat_completion", new=mock), \
         patch("api.cross_source_service.cross_source_context", new=AsyncMock(return_value=("", None))), \
         patch("api.ontology_service.ontology_lookup", new=AsyncMock(return_value=None)):
        return asyncio.run(ap._planner_node(state)), mock


class TestPlannerFallbackPlan:
    def test_hybrid_junk_output_falls_back_to_doc_and_sql_steps(self):
        # Empty string is what a reasoning model returns when it spends the budget reasoning.
        out, mock = _run_planner(_state(has_docs=True, has_db=True), ["", "not json"])
        assert mock.call_count == 2  # one bounded retry, then fallback
        assert sorted(out["plan_tools"].values()) == ["doc_search", "sql_query"]
        assert out["plan"] == list(out["plan_tools"])
        assert all(Q in step for step in out["plan"])
        assert out["degradation_reasons"] == [ap._DEGR_PLANNER_FALLBACK]

    def test_llm_raising_falls_back_with_data_step(self):
        out, _ = _run_planner(_state(has_db=True), RuntimeError("provider down"))
        assert list(out["plan_tools"].values()) == ["sql_query"]
        assert out["degradation_reasons"] == [ap._DEGR_PLANNER_FALLBACK]

    def test_csv_fallback_uses_python_analysis_not_sql(self):
        out, _ = _run_planner(_state(has_docs=True, has_db=True, has_csv=True), ["", ""])
        assert sorted(out["plan_tools"].values()) == ["doc_search", "python_analysis"]

    def test_no_sources_keeps_raw_question_for_unsupported_refusal(self):
        out, _ = _run_planner(_state(), ["", ""])
        assert out["plan"] == [Q]
        assert out["plan_tools"] == {}

    def test_no_llm_key_uses_same_fallback(self):
        with patch("api.utils.llm_provider.llm_configured", return_value=False):
            out = asyncio.run(ap._planner_node(_state(has_docs=True, has_db=True)))
        assert sorted(out["plan_tools"].values()) == ["doc_search", "sql_query"]
        assert out["degradation_reasons"] == [ap._DEGR_PLANNER_FALLBACK]

    def test_valid_plan_carries_no_pinned_tools(self):
        out, _ = _run_planner(_state(has_docs=True, has_db=True), ['["Fetch net income by quarter."]'])
        assert out["plan"] == ["Fetch net income by quarter."]
        assert "plan_tools" not in out
        assert out["degradation_reasons"] == []


class TestPlannerTokenBudget:
    def test_planner_budget_leaves_reasoning_headroom(self):
        # DOCBOT-2585 raised every short-JSON caller to 1000; the planner was missed at 400.
        assert ap._PLANNER_MAX_TOKENS >= 1000
        _, mock = _run_planner(_state(has_db=True), ['["Fetch revenue."]'])
        assert mock.call_args.kwargs["max_tokens"] == ap._PLANNER_MAX_TOKENS


class TestExecutorHonoursPinnedTool:
    def test_pinned_step_skips_router(self):
        step = "Query the database for the figures needed to answer: " + Q
        state = _state(has_docs=True, has_db=True, plan=[step], plan_tools={step: "sql_query"})
        node = ap.make_executor_node(
            db_connections_table=None, schema_cache_table=None, query_history_table=None,
            query_embeddings_table=None, session_artifacts_table=None, table_embeddings_table=None,
            async_session_factory=None, expert_personas={}, vector_stores={},
        )
        router = AsyncMock(return_value="doc_search")
        sql = AsyncMock(return_value={"row_count": 4, "result_preview": [{"q": "Q4", "net_income": 330}]})
        with patch.object(ap, "_select_tool_llm", new=router), \
             patch("api.hybrid_service._collect_sql_result", new=sql), \
             patch("api.utils.llm_provider.is_session_budget_exceeded", return_value=False):
            out = asyncio.run(node(state))
        router.assert_not_called()
        assert out["steps_completed"][0]["tool"] == "sql_query"
        assert sql.call_args.kwargs["question"] == step


class TestSynthesizerEmptyContent:
    def test_empty_llm_answer_uses_deterministic_fallback(self):
        state = _state(has_db=True, steps_completed=[
            {"step": "Fetch net income", "tool": "sql_query", "result": "4 rows returned."}
        ])
        with patch.dict(os.environ, {"groq_api_key": "fake"}), \
             patch("api.utils.llm_provider.achat_completion", new=AsyncMock(return_value="")):
            out = asyncio.run(ap._synthesizer_node(state))
        assert out["final_answer"].strip()
        assert "4 rows returned." in out["final_answer"]
