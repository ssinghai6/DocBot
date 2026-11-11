"""DOCBOT-2542: "Does segment revenue in Note 12 agree with the books?" must
flag exactly Lab Services, whatever reasonable SQL the generator writes, and
must always take the hybrid route. No network.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import AsyncMock, patch

import pytest

from api import demo_service as d
from api.hybrid_service import classify_intent
from api.utils.discrepancy_detector import detect_discrepancies
from api.utils.sql_validator import validate_and_sanitize_sql

QUESTION = "Does segment revenue in Note 12 agree with the books?"
DOC_CONTEXT = "\n\n".join(
    f"Source: {c.metadata['source']}, Page {c.metadata['page']}\n{c.page_content}"
    for c in d.TIEOUT_DOCUMENT_CHUNKS
)

# SQL shapes an LLM plausibly writes for QUESTION. Before the fix the last
# four lost Lab Services: the column alias or the ledger account name added
# tokens ("segment", "4100", "revenue") that pushed label Jaccard under 0.55.
SQL_VARIANTS = {
    "segment_amount": "SELECT segment, amount FROM revenue_by_segment WHERE period = 'FY2025'",
    "with_period": "SELECT period, segment, amount FROM revenue_by_segment",
    "as_revenue": "SELECT segment, amount AS revenue FROM revenue_by_segment WHERE period = 'FY2025'",
    "join_ledger": (
        "SELECT s.segment, s.amount AS segment_amount, t.credit - t.debit AS ledger_amount "
        "FROM revenue_by_segment s JOIN trial_balance t ON t.account LIKE '%' || s.segment"
    ),
    "as_segment_revenue": "SELECT segment, amount AS segment_revenue FROM revenue_by_segment WHERE period = 'FY2025'",
    "sum_as_segment_revenue": (
        "SELECT segment, SUM(amount) AS segment_revenue FROM revenue_by_segment GROUP BY segment"
    ),
    "ledger_net": (
        "SELECT account, credit - debit AS revenue FROM trial_balance "
        "WHERE account_type = 'revenue' AND period = 'FY2025'"
    ),
    "ledger_debit_credit": "SELECT account, debit, credit FROM trial_balance WHERE account_type = 'revenue'",
}


@pytest.fixture(scope="module")
def books():
    path = d._create_tieout_sqlite()
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()
    path.unlink(missing_ok=True)


@pytest.mark.parametrize("sql", SQL_VARIANTS.values(), ids=SQL_VARIANTS.keys())
def test_segment_question_flags_exactly_lab_services(books, sql):
    sql = validate_and_sanitize_sql(sql, "sqlite")
    rows = [dict(r) for r in books.execute(sql)]
    report = detect_discrepancies(
        DOC_CONTEXT, {"result_preview": rows, "sources": ["revenue_by_segment"], "sql_query": sql}
    )
    got = {i.label: (i.doc_value, i.db_value, i.doc_page) for i in report.discrepancies}
    assert got == {"Lab Services": (104_900_000.0, 102_700_000.0, 9)}


def test_entity_match_does_not_pull_in_unrelated_doc_figures():
    """A bare doc 'Revenue' must not match a segment row via its entity."""
    rows = [{"account": "4100 Revenue - Lab Services", "revenue": 300_000_000}]
    report = detect_discrepancies("Source: r.pdf, Page 7\nRevenue: $310M", {"result_preview": rows})
    assert report.discrepancies == [] and report.checked_pairs == 0


@pytest.mark.parametrize(
    "question",
    [
        QUESTION,
        "Does total revenue in the draft 10-K agree with the books?",
        "Does sales and marketing expense in the MD&A agree with the general ledger?",
        "Does the Q4 net income in the 10-K match the database?",
        "Reconcile the segment note to the trial balance",
        "Where do the report and the database disagree?",
    ],
)
async def test_tieout_questions_route_hybrid_without_llm(question):
    with patch("api.utils.llm_provider.achat_completion", new_callable=AsyncMock) as llm:
        intent = await classify_intent(question, has_db=True, has_docs=True, session_id="s",
                                       async_session_factory=None)
    assert intent == "hybrid"
    llm.assert_not_called()


@pytest.mark.parametrize("question", ["What are the main risk factors?", "Summarize the lease agreement terms"])
async def test_other_questions_still_use_llm_router(question):
    with (
        patch("api.utils.llm_provider.achat_completion", new_callable=AsyncMock, return_value="doc") as llm,
        patch("api.utils.llm_cache.get_cached_response", new_callable=AsyncMock, return_value=None),
        patch("api.utils.llm_cache.set_cached_response", new_callable=AsyncMock),
    ):
        intent = await classify_intent(question, has_db=True, has_docs=True,
                                       session_id="s", async_session_factory=None)
    assert intent == "doc"
    llm.assert_awaited_once()
