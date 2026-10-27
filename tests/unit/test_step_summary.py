"""DOCBOT-2005: deterministic Autopilot step summaries + details cleaning."""

import pytest

from api.utils.prompt_guard import wrap_untrusted
from api.utils.step_summary import build_step_summary, clean_step_details

SQL_STEP = {
    "tool": "sql_query",
    "result": '12 rows returned. Preview: [{"account": "Cash", "balance": 100}]',
    "row_count": 12,
    "tieout_sql": {"sources": ["trial_balance"]},
}
DOC_STEP = {
    "tool": "doc_search",
    "result": wrap_untrusted("Source: 10-K, Page 3\nRevenue was $1.2B"),
    "passages": 6,
    "doc_sources": ["10-K"],
}
PY_STEP = {"tool": "python_analysis", "result": "{'mean': 4.2}", "chart_b64": "iVBORw0"}
FAILED_STEP = {
    "tool": "python_analysis",
    "result": "",
    "error": "Traceback (most recent call last): KeyError: 'secret_col' at /app/api/x.py",
}


def test_sql_summary():
    assert build_step_summary(SQL_STEP) == "Queried trial_balance — 12 rows"


def test_sql_summary_multiple_tables_and_single_row():
    step = {**SQL_STEP, "row_count": 1, "tieout_sql": {"sources": ["a", "b"]}}
    assert build_step_summary(step) == "Queried 2 tables — 1 row"


def test_doc_search_summary():
    assert build_step_summary(DOC_STEP) == "Searched 10-K — 6 passages"


def test_doc_search_no_passages():
    step = {"tool": "doc_search", "result": "No relevant documents found."}
    assert build_step_summary(step) == "Searched documents — no matching passages"


def test_python_summary():
    assert build_step_summary(PY_STEP) == "Ran Python — chart created"
    assert build_step_summary({**PY_STEP, "chart_b64": None}) == "Ran Python — analysis complete"


def test_failure_summary_hides_exception_internals():
    summary = build_step_summary(FAILED_STEP)
    assert summary.startswith("Step failed: ")
    assert "Traceback" not in summary and "secret_col" not in summary


def test_failure_reason_and_unsupported():
    sql_fail = {"tool": "sql_query", "result": "No SQL result returned.", "failure_reason": "data_tool_missing"}
    assert build_step_summary(sql_fail) == "Step failed: the query returned no result"
    assert build_step_summary({"tool": "unsupported", "result": "x"}) == "Step failed: no data source connected"


@pytest.mark.parametrize("step", [
    SQL_STEP, DOC_STEP, PY_STEP, FAILED_STEP,
    {**DOC_STEP, "doc_sources": ["<untrusted_document_content>{evil}"]},
    {**SQL_STEP, "tieout_sql": {"sources": ['{"t": 1}']}},
])
def test_summary_never_contains_wrapper_or_json(step):
    summary = build_step_summary(step)
    assert "<untrusted" not in summary
    assert not summary.lstrip().startswith("{")


def test_clean_details_strips_untrusted_tags():
    cleaned = clean_step_details(DOC_STEP["result"])
    assert "<untrusted" not in cleaned and "</untrusted" not in cleaned
    assert "Revenue was $1.2B" in cleaned


def test_clean_details_pretty_prints_json():
    cleaned = clean_step_details(SQL_STEP["result"])
    assert cleaned.startswith("12 rows returned. Preview:")
    assert '```json\n[\n  {\n    "account": "Cash"' in cleaned


def test_clean_details_leaves_plain_text():
    assert clean_step_details("Mean revenue is 4.2 [approx]") == "Mean revenue is 4.2 [approx]"
