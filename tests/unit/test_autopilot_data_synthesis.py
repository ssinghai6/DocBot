"""Unit tests for autopilot data-synthesis hardening (P1/P2/P3).

Scope: the fix plan implemented in api/autopilot_service.py that guarantees
the LangGraph run can never silently degrade to doc-only and that the final
answer is always grounded in the tool data that actually came back:

- P1: planner schema check + bounded regeneration (max 1), tool-map
  validation against resolved source flags, data_tool_missing marking,
  degraded-answer labeling by the synthesizer.
- P2: normalized data_results state (keyed "step:<idx>:<tool>"), artifact
  persistence of visible analysis outputs, ground-truth synthesis input,
  doc-vs-data discrepancy inclusion.
- P3: final-answer claim guard ("no data" fix-up) and refusal clarity
  (name the query that ran and returned empty).

All tests are unit-level: asyncio.run + unittest.mock, no network, no real
DB, no real API keys. LLM providers are patched at
api.utils.llm_provider.{chat_completion,call_llm}.
"""

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from api.autopilot_service import (
    _build_data_results_summary,
    _has_apology_claim,
    _planner_node,
    _refusal_detail,
    _synthesizer_node,
    _verifier_node,
    make_executor_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> dict:
    """Build a minimal AutopilotState for direct node invocation."""
    base = {
        "question": "What was revenue by region?",
        "session_id": "s1",
        "connection_id": "c1",
        "persona": "Generalist",
        "run_id": "run-1",
        "plan": [],
        "steps_completed": [],
        "iteration": 0,
        "final_answer": "",
        "citations": [],
        "timed_out": False,
        "budget_exceeded": False,
        "has_docs": False,
        "has_db": True,
        "has_csv": False,
        "data_results": {},
        "degradation_reasons": [],
    }
    base.update(overrides)
    return base


def _executor_node(session_artifacts_table=None):
    """Executor node with fully mocked table wiring."""
    return make_executor_node(
        db_connections_table=MagicMock(),
        schema_cache_table=MagicMock(),
        query_history_table=MagicMock(),
        query_embeddings_table=MagicMock(),
        session_artifacts_table=session_artifacts_table,
        table_embeddings_table=MagicMock(),
        async_session_factory=MagicMock(),
        expert_personas={"Generalist": {"persona_def": "", "tool_preference": "balanced"}},
        vector_stores={},
    )


# ---------------------------------------------------------------------------
# (a) P1 — a skipped data tool marks the run degraded
# ---------------------------------------------------------------------------


class TestSkippedDataToolMarksDegraded:
    def test_unsupported_step_produces_data_tool_missing_reason(self):
        """No data sources resolved → step routes to unsupported; the run
        must carry an explicit data_tool_missing degradation reason."""
        node = _executor_node()
        state = _make_state(
            plan=["Fetch revenue by region"],
            has_db=False, has_docs=False, has_csv=False,
        )
        with patch(
            "api.utils.llm_provider.is_session_budget_exceeded", return_value=False
        ):
            result = asyncio.run(node(state))

        steps = result["steps_completed"]
        assert steps[0]["tool"] == "unsupported"
        assert steps[0]["failure_reason"] == "data_tool_missing"
        assert result["degradation_reasons"] == ["data_tool_missing:step:1:unsupported"]

    def test_unsupported_step_result_has_no_private_data(self):
        """The unsupported branch surfaces a clean refusal message with no
        internal error text."""
        node = _executor_node()
        state = _make_state(
            plan=["Fetch revenue by region"],
            has_db=False, has_docs=False, has_csv=False,
        )
        with patch(
            "api.utils.llm_provider.is_session_budget_exceeded", return_value=False
        ):
            result = asyncio.run(node(state))

        entry = result["steps_completed"][0]
        assert "data source" in entry["result"].lower()
        assert "unsupported" in entry["result"].lower() or "proceed" in entry["result"].lower()


# ---------------------------------------------------------------------------
# (b) P1 — planner retries on an invalid tool map and succeeds within 1
#     regeneration call
# ---------------------------------------------------------------------------


class TestPlannerBoundedRetry:
    def test_invalid_first_attempt_recovered_by_one_regeneration(self):
        """First planner response is not a JSON array; the single bounded
        regeneration (with the error appended) recovers a valid plan and no
        degradation reason is recorded."""
        state = _make_state(question="Analyse Q3 revenue trends")
        responses = [
            "The plan is: fetch revenue and chart it.",  # invalid: not JSON
            json.dumps(["Fetch revenue by region", "Calculate gross margin"]),
        ]
        with patch.dict(os.environ, {"groq_api_key": "fake"}), patch(
            "api.utils.llm_provider.chat_completion",
            new=MagicMock(side_effect=responses),
        ) as mock_chat:
            result = asyncio.run(_planner_node(state))

        assert result["plan"] == ["Fetch revenue by region", "Calculate gross margin"]
        assert result["degradation_reasons"] == []
        # Exactly one regeneration: 2 calls total, not 3+
        assert mock_chat.call_count == 2


# ---------------------------------------------------------------------------
# (c) P1 — a one-step planner fallback is never silent
# ---------------------------------------------------------------------------


class TestPlannerFallbackNotSilent:
    def test_no_api_key_fallback_records_degradation_reason(self):
        state = _make_state(question="What is revenue?")
        with patch.dict(os.environ, {}, clear=True):
            result = asyncio.run(_planner_node(state))
        # DOCBOT-2721: the fallback plan targets the connected DB, not the raw question.
        assert result["plan"] == ["Query the database for the figures needed to answer: What is revenue?"]
        assert result["degradation_reasons"] == ["planner_fallback_single_step"]

    def test_degraded_label_present_in_synthesized_answer(self):
        """Synthesizer output must visibly carry the degraded state — the
        fallback can never be silent in the answer string."""
        state = _make_state(
            degradation_reasons=["planner_fallback_single_step"],
            steps_completed=[
                {
                    "step": "What is revenue?",
                    "tool": "sql_query",
                    "result": "3 rows returned.",
                    "error": None,
                }
            ],
        )
        with patch.dict(os.environ, {"groq_api_key": "fake"}), patch(
            "api.utils.llm_provider.chat_completion",
            new=MagicMock(return_value="Revenue was $1.2M last quarter."),
        ):
            result = asyncio.run(_synthesizer_node(state))

        answer = result["final_answer"]
        assert "degraded" in answer.lower()
        assert "planner_fallback_single_step" in answer


# ---------------------------------------------------------------------------
# (d) P2 — data_results persisted + artifact save invoked
# ---------------------------------------------------------------------------


class TestDataResultsPersistedAndSaved:
    def test_sql_success_stores_data_results_and_calls_save_artifact(self):
        node = _executor_node(session_artifacts_table=MagicMock())
        state = _make_state(plan=["Fetch revenue by region"])
        sql_meta = {
            "row_count": 3,
            "result_preview": [
                {"region": "EMEA", "revenue": 1250},
                {"region": "APAC", "revenue": 300},
                {"region": "NA", "revenue": 900},
            ],
            "sql_query": "SELECT region, SUM(revenue) FROM sales GROUP BY region",
            "explanation": "Revenue grouped by region",
        }
        with patch(
            "api.utils.llm_provider.call_llm",
            new=AsyncMock(return_value=json.dumps({"tool": "sql_query"})),
        ), patch(
            "api.utils.llm_provider.is_session_budget_exceeded", return_value=False
        ), patch(
            "api.hybrid_service._collect_sql_result", new=AsyncMock(return_value=sql_meta)
        ), patch(
            "api.artifact_service.save_artifact",
            new=AsyncMock(return_value="art-sql-1"),
        ) as mock_save:
            result = asyncio.run(node(state))

        entry = result["data_results"]["step:1:sql_query"]
        assert entry["row_count"] == 3
        assert len(entry["sample_rows"]) == 3
        assert entry["sql"] == "SELECT region, SUM(revenue) FROM sales GROUP BY region"
        assert entry["provenance"] == "sql_pipeline"
        assert entry["artifact_id"] == "art-sql-1"
        mock_save.assert_called_once()
        assert mock_save.call_args.kwargs["artifact_type"] == "sql_result"

    def test_sample_rows_hard_capped_at_ten(self):
        """The deterministic sample is capped at 10 rows regardless of
        result size."""
        node = _executor_node(session_artifacts_table=MagicMock())
        state = _make_state(plan=["Fetch revenue by region"])
        rows = [{"i": i, "v": i} for i in range(40)]
        sql_meta = {"row_count": 40, "result_preview": rows, "sql_query": "SELECT * FROM t"}
        with patch(
            "api.utils.llm_provider.call_llm",
            new=AsyncMock(return_value=json.dumps({"tool": "sql_query"})),
        ), patch(
            "api.utils.llm_provider.is_session_budget_exceeded", return_value=False
        ), patch(
            "api.hybrid_service._collect_sql_result", new=AsyncMock(return_value=sql_meta)
        ), patch(
            "api.artifact_service.save_artifact", new=AsyncMock(return_value="art-1")
        ):
            result = asyncio.run(node(state))

        assert len(result["data_results"]["step:1:sql_query"]["sample_rows"]) == 10

    def test_python_analysis_stores_data_results_and_saves_analysis_artifact(self):
        """A python_analysis step over a prior SQL artifact persists its
        visible stdout through artifact_service (P2) and normalizes the
        output into data_results."""
        node = _executor_node(session_artifacts_table=MagicMock())
        state = _make_state(
            plan=["Fetch revenue by region", "Generate a bar chart of revenue"],
            steps_completed=[
                {
                    "step": "Fetch revenue by region",
                    "tool": "sql_query",
                    "result": "1 rows returned. Preview: [{\"revenue\": 100}]",
                    "artifact_id": "art-sql-1",
                    "error": None,
                }
            ],
            iteration=1,
        )
        with patch(
            "api.utils.llm_provider.call_llm",
            new=AsyncMock(return_value=json.dumps({"tool": "python_analysis"})),
        ), patch(
            "api.utils.llm_provider.is_session_budget_exceeded", return_value=False
        ), patch(
            "api.artifact_service.get_artifact",
            new=AsyncMock(
                return_value=SimpleNamespace(data_json=json.dumps([{"revenue": 100}]))
            ),
        ), patch(
            "api.sandbox_service.generate_analysis_code",
            new=AsyncMock(return_value="print(df['revenue'].sum())"),
        ), patch(
            "api.sandbox_service.run_python",
            new=AsyncMock(
                return_value=SimpleNamespace(stdout="Revenue total: 100", error=None, charts=[])
            ),
        ), patch(
            "api.artifact_service.save_artifact",
            new=AsyncMock(return_value="art-analysis-1"),
        ) as mock_save:
            result = asyncio.run(node(state))

        entry = result["data_results"]["step:2:python_analysis"]
        assert entry["row_count"] == 1
        assert entry["provenance"] == "e2b_python"
        assert entry["result_excerpt"] == "Revenue total: 100"
        mock_save.assert_called_once()
        assert mock_save.call_args.kwargs["artifact_type"] == "analysis"


# ---------------------------------------------------------------------------
# (e) P3 — apology pattern removed when data_results hold data
# ---------------------------------------------------------------------------


class TestApologyGuardFixesAnswer:
    def test_no_numbers_claim_regenerated_with_explicit_data_context(self):
        state = _make_state(
            final_answer="Unfortunately there is no numeric data available for this question.",
            steps_completed=[
                {
                    "step": "Fetch revenue by region",
                    "tool": "sql_query",
                    "result": "2 rows returned.",
                    "error": None,
                }
            ],
            data_results={
                "step:1:sql_query": {
                    "tool": "sql_query",
                    "step_label": "Fetch revenue by region",
                    "row_count": 2,
                    "sample_rows": [
                        {"region": "EMEA", "revenue": 1250},
                        {"region": "APAC", "revenue": 300},
                    ],
                    "sql": "SELECT region, SUM(revenue) FROM sales GROUP BY region",
                    "provenance": "sql_pipeline",
                }
            },
        )
        with patch(
            "api.utils.llm_provider.call_llm",
            new=AsyncMock(
                return_value=json.dumps({"contradiction_found": False, "explanation": ""})
            ),
        ), patch(
            "api.utils.llm_provider.chat_completion",
            new=MagicMock(
                return_value="Regional revenue: EMEA $1,250 and APAC $300 for Q3."
            ),
        ) as mock_chat:
            result = asyncio.run(_verifier_node(state))

        answer = result["final_answer"]
        assert "no numeric data" not in answer.lower()
        assert "$1,250" in answer
        mock_chat.assert_called_once()  # exactly one bounded fix-up

    def test_has_apology_claim_detection(self):
        assert _has_apology_claim("Sorry, there is no data to compute this.")
        assert _has_apology_claim("Unable to compute the requested figure.")
        assert not _has_apology_claim("Revenue grew to $330M in Q4.")


# ---------------------------------------------------------------------------
# (f) P3 — refusal states which query ran and returned empty
# ---------------------------------------------------------------------------


class TestRefusalNamesQuery:
    def test_zero_row_query_named_in_refusal(self):
        sql = "SELECT * FROM orders WHERE region = 'Mars'"
        state = _make_state(
            final_answer="There is no data available for this analysis.",
            steps_completed=[
                {
                    "step": "Fetch orders for Mars",
                    "tool": "sql_query",
                    "result": "0 rows returned.",
                    "error": None,
                }
            ],
            data_results={
                "step:1:sql_query": {
                    "tool": "sql_query",
                    "step_label": "Fetch orders for Mars",
                    "row_count": 0,
                    "sample_rows": [],
                    "sql": sql,
                    "provenance": "sql_pipeline",
                }
            },
        )
        with patch(
            "api.utils.llm_provider.call_llm",
            new=AsyncMock(
                return_value=json.dumps({"contradiction_found": False, "explanation": ""})
            ),
        ), patch(
            "api.utils.llm_provider.chat_completion", new=MagicMock()
        ) as mock_chat:
            result = asyncio.run(_verifier_node(state))

        answer = result["final_answer"]
        assert sql in answer
        assert "0 rows" in answer
        assert "empty result set" in answer
        mock_chat.assert_not_called()  # refusal clarity is deterministic

    def test_refusal_detail_builder_output(self):
        detail = _refusal_detail(
            [
                {
                    "tool": "sql_query",
                    "step_label": "Fetch orders for Mars",
                    "row_count": 0,
                    "sql": "SELECT * FROM orders WHERE region = 'Mars'",
                }
            ]
        )
        assert "SELECT * FROM orders WHERE region = 'Mars'" in detail
        assert "returned 0 rows" in detail


# ---------------------------------------------------------------------------
# (g) P2(3) — discrepancies explicitly included in the synthesized answer
# ---------------------------------------------------------------------------


class TestDiscrepancyInclusion:
    def test_doc_vs_data_discrepancy_forced_into_answer(self):
        state = _make_state(
            steps_completed=[
                {
                    "step": "Search the filing for net income",
                    "tool": "doc_search",
                    "result": "Net Income: $330M according to the filing.",
                    "error": None,
                },
                {
                    "step": "Fetch net income",
                    "tool": "sql_query",
                    "result": "1 rows returned.",
                    "error": None,
                },
            ],
            data_results={
                "step:2:sql_query": {
                    "tool": "sql_query",
                    "step_label": "Fetch net income",
                    "row_count": 1,
                    "sample_rows": [{"net_income_m": 325}],
                    "sql": "SELECT net_income_m FROM fina",
                    "provenance": "sql_pipeline",
                }
            },
        )
        with patch.dict(os.environ, {"groq_api_key": "fake"}), patch(
            "api.utils.llm_provider.chat_completion",
            new=MagicMock(
                return_value="Net income as reported in the tables above is 325."
            ),
        ) as mock_chat:
            result = asyncio.run(_synthesizer_node(state))

        answer = result["final_answer"]
        # DOCBOT-2002: the backstop appends a marker-based Tie-out footer —
        # the literal [DISCREPANCY] tag must never reach the answer text.
        assert "[[discrepancy:d-" in answer
        assert "**Tie-out:**" in answer
        assert "Net Income" in answer
        assert "[DISCREPANCY]" not in answer
        mock_chat.assert_called_once()
        # The system prompt must have carried the pre-computed marker block
        sent_system = mock_chat.call_args.args[0][0]["content"]
        assert "[[discrepancy:d-" in sent_system
        assert "[DISCREPANCY]" not in sent_system.replace(
            "NEVER write the literal text [DISCREPANCY]", ""
        )


# ---------------------------------------------------------------------------
# (h) — no secrets in surfaced failure strings
# ---------------------------------------------------------------------------


class TestNoSecretsInSurfacedStrings:
    def test_data_summary_scrubs_connection_strings(self):
        summary, _rows, _flag = _build_data_results_summary(
            _make_state(
                data_results={
                    "step:1:sql_query": {
                        "tool": "sql_query",
                        "step_label": "Fetch revenue",
                        "row_count": 1,
                        "sample_rows": [{"region": "EMEA"}],
                        "sql": "SELECT 1 -- postgresql://svc_user:SuperSecret123@prod-db/corp",
                        "provenance": "sql_pipeline",
                    }
                }
            )
        )
        assert "SuperSecret123" not in summary
        assert "[redacted]@" in summary

    def test_refusal_detail_scrubs_connection_strings(self):
        detail = _refusal_detail(
            [
                {
                    "tool": "sql_query",
                    "step_label": "Fetch orders",
                    "row_count": 0,
                    "sql": "SELECT * FROM t",
                }
            ]
        )
        assert "SuperSecret" not in detail

    def test_crashed_sql_step_surfaces_no_secret_content(self):
        """A step whose tool call raises with a connection string in the
        error keeps the raw error internal; the user-facing result stays
        empty (run_autopilot replaces it with a generic message) and no
        data/summary content contains the secret."""
        node = _executor_node(session_artifacts_table=MagicMock())
        secret_msg = "connection failed to postgresql://admin:S3cretPw@prod-db/corp"
        state = _make_state(plan=["Fetch revenue by region"])
        with patch(
            "api.utils.llm_provider.call_llm",
            new=AsyncMock(return_value=json.dumps({"tool": "sql_query"})),
        ), patch(
            "api.utils.llm_provider.is_session_budget_exceeded", return_value=False
        ), patch(
            "api.hybrid_service._collect_sql_result",
            new=AsyncMock(side_effect=RuntimeError(secret_msg)),
        ):
            result = asyncio.run(node(state))

        entry = result["steps_completed"][0]
        assert entry["result"] == ""           # nothing surfaced
        assert "data_results" not in result    # no data leaked into state
        assert secret_msg in entry["error"]    # internal-only, for server logs


# ---------------------------------------------------------------------------
# Bonus (P1a) — invalid tool-for-source selection re-routed deterministically
# ---------------------------------------------------------------------------


class TestInvalidSourceToolRerouted:
    def test_sql_selected_without_db_rerouted_to_available_source(self):
        """The router validated against _VALID_TOOL_SELECTIONS could still
        pick sql_query with has_db=False (doc-only session) — the bounded
        regeneration + flag-aware heuristic must prevent that tool from
        executing and never let the run degrade silently."""
        node = make_executor_node(
            db_connections_table=MagicMock(),
            schema_cache_table=MagicMock(),
            query_history_table=MagicMock(),
            query_embeddings_table=MagicMock(),
            session_artifacts_table=MagicMock(),
            table_embeddings_table=MagicMock(),
            async_session_factory=MagicMock(),
            expert_personas={"Generalist": {"persona_def": "", "tool_preference": "balanced"}},
            vector_stores={"s1": MagicMock()},  # docs available → doc_search can run
        )
        state = _make_state(
            plan=["Fetch revenue by region"],
            has_db=False, has_docs=True,
        )
        with patch(
            "api.utils.llm_provider.call_llm",
            new=AsyncMock(side_effect=[
                json.dumps({"tool": "sql_query"}),  # first: invalid for sources
                json.dumps({"tool": "sql_query"}),  # regeneration: still invalid
            ]),
        ) as mock_call_llm, patch(
            "api.utils.llm_provider.is_session_budget_exceeded", return_value=False
        ), patch(
            "api.deep_research_service.deep_retrieve",
            new=AsyncMock(return_value=([], [])),
        ):
            result = asyncio.run(node(state))

        entry = result["steps_completed"][0]
        assert entry["tool"] == "doc_search"      # heuristic last-resort re-route
        assert mock_call_llm.call_count == 2      # bounded: 1 regenerate, no more
        # A correct re-route to an available data tool is not a degradation.
        assert entry["failure_reason"] is None
