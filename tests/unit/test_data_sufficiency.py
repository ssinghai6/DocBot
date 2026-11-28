"""DOCBOT-2724/2725/2726: insufficient-data gate, NaN detector, honest synthesis.

No network: LLM, E2B and artifact calls are mocked.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from api.autopilot_service import _synthesizer_node, _apply_claim_guard, make_executor_node
from api.utils.data_sufficiency import (
    count_year_value_pairs,
    detect_insufficient_output,
    guess_metric,
    insufficient_history_message,
)
from api.utils.step_summary import build_step_summary

FUEL_TEXT = (
    "Fuel prices are influenced by crude oil supply, refinery capacity, taxes "
    "and seasonal demand. In 2022 prices rose sharply after supply disruptions; "
    "analysts expect volatility to continue."
)
SERIES_TEXT = (
    "Average retail diesel price by year: 2019: $3.06, 2020: $2.55, "
    "2021 | 3.29\n2022\n5.00\nIn 2023 the average was 4.21 per gallon."
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_count_pairs_fuel_text_has_none():
    assert count_year_value_pairs(FUEL_TEXT) == 0
    assert count_year_value_pairs("") == 0


def test_count_pairs_series_text():
    assert count_year_value_pairs(SERIES_TEXT) == 5


def test_guess_metric_and_message():
    assert guess_metric("predict fuel prices next 10 years") == "fuel prices"
    assert guess_metric("") == "the requested metric"
    msg = insufficient_history_message(0, "fuel prices")
    assert msg == (
        "INSUFFICIENT_DATA: found 0 historical data points; "
        "need at least 3 historical values of fuel prices"
    )


def test_detect_marker_and_nan():
    assert detect_insufficient_output(
        "INSUFFICIENT_DATA: no price series in df", False
    ) == "INSUFFICIENT_DATA: no price series in df"
    nan_out = "Forecast:\n2025: nan\n2026: nan\nCHART_META:{\"title\": \"x\"}"
    assert detect_insufficient_output(nan_out, True).startswith("INSUFFICIENT_DATA:")
    assert detect_insufficient_output("", True).startswith("INSUFFICIENT_DATA:")
    # Real forecast output and non-forecast output are left alone.
    assert detect_insufficient_output("2025-01: $3.41\n2025-02: $3.52", True) is None
    assert detect_insufficient_output("", False) is None
    assert detect_insufficient_output("mean: nan, latest 2.50", True) is None


def test_step_summary_label():
    assert build_step_summary(
        {"tool": "python_analysis", "failure_reason": "insufficient_data"}
    ) == "Insufficient data"


# ---------------------------------------------------------------------------
# Executor integration (docs-only and SQL python_analysis paths)
# ---------------------------------------------------------------------------


def _state(question, steps_completed, *, has_db=False, has_docs=True):
    return {
        "question": question,
        "session_id": "s1",
        "connection_id": "c1" if has_db else None,
        "persona": "Generalist",
        "run_id": "run-1",
        "plan": [s["step"] for s in steps_completed] + ["Fit a regression and forecast"],
        "steps_completed": steps_completed,
        "iteration": 1,
        "final_answer": "",
        "citations": [],
        "timed_out": False,
        "budget_exceeded": False,
        "has_docs": has_docs,
        "has_db": has_db,
        "has_csv": False,
        "data_results": {},
        "degradation_reasons": [],
    }


def _doc_step(text):
    return {"step": "Search docs for prices", "tool": "doc_search", "result": text,
            "artifact_id": None, "error": None}


def _node(artifacts_table=None):
    return make_executor_node(
        db_connections_table=MagicMock(),
        schema_cache_table=MagicMock(),
        query_history_table=MagicMock(),
        query_embeddings_table=MagicMock(),
        session_artifacts_table=artifacts_table,
        table_embeddings_table=MagicMock(),
        async_session_factory=MagicMock(),
        expert_personas={"Generalist": {"persona_def": "", "tool_preference": "balanced"}},
        vector_stores={},
    )


def _run(state, *, stdout="", charts=None, code="print(1)", get_artifact=None):
    gen = AsyncMock(return_value=code)
    run_py = AsyncMock(return_value=SimpleNamespace(stdout=stdout, error=None, charts=charts or []))
    save = AsyncMock(return_value="art-1")
    with patch(
        "api.utils.llm_provider.call_llm",
        new=AsyncMock(return_value=json.dumps({"tool": "python_analysis"})),
    ), patch("api.utils.llm_provider.is_session_budget_exceeded", return_value=False), patch(
        "api.sandbox_service.generate_analysis_code", new=gen
    ), patch("api.sandbox_service.run_python", new=run_py), patch(
        "api.artifact_service.save_artifact", new=save
    ), patch(
        "api.artifact_service.get_artifact", new=get_artifact or AsyncMock(return_value=None)
    ):
        result = asyncio.run(_node(MagicMock())(state))
    return result, gen, run_py, save


def test_docs_without_numbers_skips_e2b():
    state = _state("predict fuel prices next 10 years", [_doc_step(FUEL_TEXT)])
    result, gen, run_py, save = _run(state)
    entry = result["steps_completed"][0]
    assert entry["failure_reason"] == "insufficient_data"
    assert entry["result"].startswith("INSUFFICIENT_DATA: found 0 historical data points")
    assert "fuel prices" in entry["result"]
    assert entry["chart_b64"] is None and entry["artifact_id"] is None
    gen.assert_not_called()
    run_py.assert_not_called()
    save.assert_not_called()
    assert "data_results" not in result
    assert "insufficient_data:step:2:python_analysis" in result["degradation_reasons"]


def test_docs_with_series_proceeds():
    state = _state("predict diesel prices next 3 years", [_doc_step(SERIES_TEXT)])
    result, gen, run_py, _ = _run(state, stdout="2024: $4.40\n2025: $4.62")
    entry = result["steps_completed"][0]
    assert entry["failure_reason"] is None
    run_py.assert_called_once()
    assert entry["result"].startswith("2024: $4.40")


def test_nan_forecast_output_marked_and_charts_dropped():
    state = _state("predict diesel prices next 3 years", [_doc_step(SERIES_TEXT)])
    result, _, run_py, save = _run(
        state, stdout="2024: nan\n2025: nan", charts=["b64chart"]
    )
    entry = result["steps_completed"][0]
    run_py.assert_called_once()
    assert entry["failure_reason"] == "insufficient_data"
    assert entry["result"].startswith("INSUFFICIENT_DATA:")
    assert entry["chart_b64"] is None
    save.assert_not_called()


def test_marker_in_stdout_marked():
    state = _state("predict diesel prices next 3 years", [_doc_step(SERIES_TEXT)])
    result, _, _, save = _run(
        state, stdout="INSUFFICIENT_DATA: only 2 yearly prices found", charts=["c"]
    )
    entry = result["steps_completed"][0]
    assert entry["result"] == "INSUFFICIENT_DATA: only 2 yearly prices found"
    assert entry["failure_reason"] == "insufficient_data"
    save.assert_not_called()


def test_sql_forecast_path_unchanged():
    rows = [{"year": 2019 + i, "price": 2.5 + i * 0.1} for i in range(6)]
    sql_step = {"step": "Fetch prices", "tool": "sql_query", "result": "6 rows",
                "artifact_id": "art-sql", "error": None}
    state = _state("forecast price next 3 years", [sql_step], has_db=True, has_docs=False)
    with patch("api.db_service.fetch_full_series", new=AsyncMock(return_value=None)):
        result, gen, run_py, save = _run(
            state,
            stdout="2025: $3.10\n2026: $3.20",
            charts=["chart"],
            get_artifact=AsyncMock(return_value=SimpleNamespace(data_json=json.dumps(rows))),
        )
    entry = result["steps_completed"][0]
    assert entry["failure_reason"] is None
    assert entry["chart_b64"] == "chart"
    run_py.assert_called_once()
    assert result["data_results"]["step:2:python_analysis"]["row_count"] == 6


# ---------------------------------------------------------------------------
# Synthesizer / claim guard (DOCBOT-2726)
# ---------------------------------------------------------------------------


def _insufficient_state():
    return {
        "question": "predict fuel prices next 10 years",
        "run_id": "r",
        "steps_completed": [
            _doc_step(FUEL_TEXT),
            {"step": "Forecast", "tool": "python_analysis",
             "result": insufficient_history_message(0, "fuel prices"),
             "failure_reason": "insufficient_data", "error": None},
        ],
        "data_results": {"step:9:sql_query": {"tool": "sql_query", "row_count": 4,
                                              "sample_rows": [{"a": 1}]}},
        "degradation_reasons": ["insufficient_data:step:2:python_analysis"],
    }


def test_synthesizer_prompt_carries_rule():
    captured = {}

    async def fake(messages, **kw):
        captured["system"] = messages[0]["content"]
        return "Not enough data to forecast fuel prices."

    with patch("api.utils.llm_provider.llm_configured", return_value=True), patch(
        "api.utils.llm_provider.achat_completion", new=fake
    ):
        out = asyncio.run(_synthesizer_node(_insufficient_state()))
    assert "INSUFFICIENT DATA RULE" in captured["system"]
    assert "NEVER invent coefficients" in captured["system"]
    assert out["final_answer"].startswith("Not enough data to forecast fuel prices.")


def test_claim_guard_keeps_honest_answer():
    fixup = AsyncMock(return_value="rewritten")
    with patch("api.utils.llm_provider.achat_completion", new=fixup):
        out = asyncio.run(_apply_claim_guard(_insufficient_state(), "Not enough data to forecast."))
    assert out == "Not enough data to forecast."
    fixup.assert_not_called()
