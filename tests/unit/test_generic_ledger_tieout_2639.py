"""DOCBOT-2639: tie-out on a filing the fixes were not tuned on (Chipotle FY2025 10-K
against a signed-amount Postgres ledger with a voided entry)."""
from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

from api.utils.discrepancy_detector import detect_discrepancies

INCOME_STATEMENT = (
    "Source: CMG-10-K-2026-02-04, Page 37\n"
    "CONSOLIDATED STATEMENTS OF INCOME AND COMPREHENSIVE INCOME\n"
    "2025 | 2024 | 2023\n"
    "Labor | 2,991,680 | 2,789,789 | 2,440,982\n"
    "Occupancy | 624,898 | 563,374 | 503,264\n"
    "Impairment, closure costs, and asset disposals | 27,503 | 26,949 | 38,370\n"
)
CASH_FLOW = (
    "Source: CMG-10-K-2026-02-04, Page 40\n"
    "CONSOLIDATED STATEMENTS OF CASH FLOWS\n"
    "2025 | 2024 | 2023\n"
    "Impairment, closure costs, and asset disposals | 26,576 | 25,213 | 35,986\n"
)
SEGMENT = (
    "Source: CMG-10-K-2026-02-04, Page 62\n"
    "2025 | 2024 | 2023\n"
    "Labor | 2,931,465 | 2,741,590 | 2,402,964\n"
)


def _books(labor: float) -> dict:
    rows = [
        {"report_line": "Labor", "expense_total": labor},
        {"report_line": "Occupancy", "expense_total": 624_898_000.0},
        {"report_line": "Impairment, closure costs, and asset disposals", "expense_total": 27_503_000.0},
    ]
    return {"result_preview": rows, "sql_query": "SELECT report_line, SUM(amount) FROM gl_line GROUP BY 1",
            "sources": ["gl_line"]}


def _flags(doc: str, labor: float) -> list[tuple[str, float, float]]:
    return [(d.label, d.doc_value, d.db_value) for d in detect_discrepancies(doc, _books(labor)).discrepancies]


def test_statement_cell_flags_a_break_under_half_a_percent():
    # $12.5M on $2.99B is 0.42%: under the prose default, but a statement cell is exact.
    assert _flags(INCOME_STATEMENT, 3_004_180_000.0) == [("Labor", 2_991_680_000.0, 3_004_180_000.0)]


def test_statement_cell_ties_out_within_rounding():
    assert _flags(INCOME_STATEMENT, 2_991_680_400.0) == []


def test_rounded_prose_keeps_the_default_tolerance():
    prose = "Source: CMG, Page 30\nFY2025 labor costs were $2.99 billion.\n"
    assert _flags(prose, 3_004_180_000.0) == []


def test_cash_flow_copy_does_not_flag_when_statement_ties_out():
    assert _flags(INCOME_STATEMENT + "\n" + CASH_FLOW, 2_991_680_000.0) == []


def test_segment_copy_does_not_flag_when_statement_ties_out():
    assert _flags(SEGMENT + "\n" + INCOME_STATEMENT, 2_991_680_000.0) == []


def test_only_the_closest_table_copy_is_the_break():
    flags = _flags(SEGMENT + "\n" + INCOME_STATEMENT, 3_004_180_000.0)
    assert flags == [("Labor", 2_991_680_000.0, 3_004_180_000.0)]


async def test_book_guided_retrieval_searches_deeper():
    from api import hybrid_service

    sbc_only = "Source: CMG, Page 45\n2025 | 2024\nLabor | 119,543 | 112,000\n"
    rag = AsyncMock(return_value=("", []))
    with patch.object(hybrid_service, "rag_retrieve", rag):
        await hybrid_service._book_guided_retrieval(
            sbc_only, [], _books(2_991_680_000.0), "s1", {}, MagicMock())
    assert rag.await_args.kwargs["k"] == hybrid_service.BOOK_GUIDED_SEARCH_K == 40


def test_sql_prompt_skips_void_entries_and_flips_credit_signs():
    from api import db_service

    src = inspect.getsource(db_service._generate_sql)
    assert "keep only posted ones" in src
    assert "-SUM(amount)" in src
    assert db_service.PROMPT_VERSION_SQL_GEN == "v8"


NET_INCOME_DOC = "Source: CMG, Page 37\n2025 | 2024\nNet income | $1,535,761 | $1,534,110\n"


def test_status_filter_and_case_when_do_not_rename_rows():
    sql = ("SELECT -SUM(CASE WHEN a.acct_class = 'revenue' THEN l.amount END) AS total_revenue, "
           "-SUM(l.amount) AS net_income FROM gl_line l JOIN gl_journal j ON l.journal_id = j.journal_id "
           "JOIN gl_account a ON l.acct_code = a.acct_code WHERE j.status = 'posted' LIMIT 500")
    meta = {"result_preview": [{"total_revenue": 11_925_601_000.0, "net_income": 1_523_261_000.0}], "sql_query": sql}
    flags = [(d.label, d.db_value) for d in detect_discrepancies(NET_INCOME_DOC, meta).discrepancies]
    assert flags == [("Net income", 1_523_261_000.0)]


def test_metric_tag_names_the_row_alongside_where_literals():
    sql = ("SELECT 'Net Income' AS metric, -SUM(l.amount) AS amount FROM gl_line l "
           "JOIN gl_account a ON l.acct_code = a.acct_code WHERE a.acct_class = 'revenue' LIMIT 500")
    meta = {"result_preview": [{"metric": "Net Income", "amount": 1_523_261_000.0}], "sql_query": sql}
    flags = [(d.doc_value, d.db_value) for d in detect_discrepancies(NET_INCOME_DOC, meta).discrepancies]
    assert flags == [(1_535_761_000.0, 1_523_261_000.0)]


def test_wide_row_reports_its_unpaired_lines_by_column():
    meta = {"result_preview": [{"total_revenue": 11_925_601_000.0, "net_income": 1_523_261_000.0}],
            "sql_query": "SELECT 1"}
    assert detect_discrepancies(NET_INCOME_DOC, meta).unpaired_db_lines == ["total revenue"]


def test_operating_income_matches_income_from_operations():
    doc = "Source: CMG, Page 37\n2025 | 2024\nIncome from operations | 1,935,798 | 1,916,333\n"
    sql = ("SELECT 'Operating Income' AS metric, -SUM(l.amount) AS amount FROM gl_line l "
           "JOIN gl_account a ON l.acct_code = a.acct_code WHERE a.acct_class = 'revenue' LIMIT 500")
    meta = {"result_preview": [{"metric": "Operating Income", "amount": 1_923_298_000.0}], "sql_query": sql}
    report = detect_discrepancies(doc, meta)
    assert [(d.doc_value, d.db_value) for d in report.discrepancies] == [(1_935_798_000.0, 1_923_298_000.0)]
    assert detect_discrepancies("Source: CMG, Page 45\nNothing here.\n", meta).unpaired_db_lines == ["Operating Income"]
