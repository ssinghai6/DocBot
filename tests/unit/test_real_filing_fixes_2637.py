"""DOCBOT-2637: fixes found by the real-filing (Etsy FY2025 10-K) test on prod."""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.utils.chunker import chunk_filing
from api.utils.discrepancy_detector import _doc_context_for
from api.utils.doc_refs import chunk_header, chunk_key, chunk_page
from api.utils.tieout_lines import add_skipped_lines

BOOKS = [
    "Revenue", "Cost of revenue", "Gross profit", "Marketing", "Product development",
    "General and administrative", "Asset impairment charges", "Total operating expenses",
    "Income from operations", "Net income",
]
SCHEMA = [{"name": "gl_fs_rollup", "columns": [
    {"name": "fiscal_year", "type": "INTEGER"},
    {"name": "acct_desc", "type": "TEXT", "sample_values": BOOKS, "values_complete": True},
    {"name": "amount_usd", "type": "INTEGER"},
]}]
STATEMENT = (
    "Source: ETSY-10-K-2026-02-19, Page 67\n"
    "Etsy, Inc.\nConsolidated Statements of Operations\n(In thousands)\n"
    "2025 | 2024\n"
    "Revenue | $2,883,501 | $2,808,332\n"
    "Cost of revenue | 817,800 | 774,554\n"
    "Gross profit | 2,065,701 | 2,033,778\n"
    "Operating expenses:\n"
    "Marketing | 914,830 | 856,565\n"
    "Product development | 450,192 | 443,056\n"
    "General and administrative | 332,766 | 353,949\n"
    "Asset impairment charges | 101,703 | —\n"
    "Total operating expenses | 1,799,491 | 1,653,570\n"
    "Income from operations | 266,210 | 380,208\n"
    "Interest expense | (18,509) | (13,806)\n"
    "Net income | $162,982 | $303,276\n"
)
OPEX_SQL = (
    "SELECT acct_desc, amount_usd FROM gl_fs_rollup WHERE fiscal_year = 2025 AND acct_desc IN "
    "('Marketing', 'Product development', 'General and administrative', 'Total operating expenses') LIMIT 500"
)


# ── 1. skipped filing lines come back into the books query ────────────────────

def test_skipped_line_between_selected_lines_is_added():
    sql, added = add_skipped_lines(OPEX_SQL, SCHEMA, STATEMENT, "sqlite")
    assert added == ["Asset impairment charges"]
    assert "'Asset impairment charges'" in sql and "fiscal_year = 2025" in sql


def test_far_apart_picks_are_left_alone():
    sql = ("SELECT acct_desc, amount_usd FROM gl_fs_rollup WHERE fiscal_year = 2025 "
           "AND acct_desc IN ('Revenue', 'Income from operations', 'Net income')")
    assert add_skipped_lines(sql, SCHEMA, STATEMENT, "sqlite") == (sql, [])


def test_line_the_books_do_not_hold_is_not_added():
    schema = [{"name": "gl_fs_rollup", "columns": [{
        "name": "acct_desc", "type": "TEXT", "values_complete": True,
        "sample_values": [v for v in BOOKS if v != "Asset impairment charges"],
    }]}]
    assert add_skipped_lines(OPEX_SQL, schema, STATEMENT, "sqlite")[1] == []


def test_needs_a_complete_value_list():
    schema = [{"name": "gl_fs_rollup", "columns": [
        {"name": "acct_desc", "type": "TEXT", "sample_values": BOOKS[:5], "values_complete": False},
    ]}]
    assert add_skipped_lines(OPEX_SQL, schema, STATEMENT, "sqlite")[1] == []


def test_not_in_filter_is_never_widened():
    sql = OPEX_SQL.replace("acct_desc IN", "acct_desc NOT IN")
    assert add_skipped_lines(sql, SCHEMA, STATEMENT, "sqlite")[1] == []


def test_matching_is_exact_not_synonyms():
    from api.utils.tieout_lines import _match

    # the detector's synonym matching reads "Gross profit" as "Net income"
    assert _match("Gross profit", ["Net income"]) is None
    assert _match("Operating expenses", ["Total operating expenses"]) is None
    assert _match("General and Administrative", BOOKS) == "General and administrative"


def test_plural_and_case_differences_still_match():
    doc = STATEMENT.replace("Asset impairment charges |", "ASSET IMPAIRMENT CHARGE |")
    assert add_skipped_lines(OPEX_SQL, SCHEMA, doc, "sqlite")[1] == ["Asset impairment charges"]


def test_rows_in_another_chunk_do_not_bridge():
    doc = STATEMENT.replace("Asset impairment charges", "Source: Y, Page 9\nAsset impairment charges")
    assert add_skipped_lines(OPEX_SQL, SCHEMA, doc, "sqlite")[1] == []


async def test_hybrid_updates_sql_result_and_lineage():
    from api import hybrid_service

    meta = {"sql_query": OPEX_SQL, "sources": ["gl_fs_rollup"], "result_preview": [{"a": 1}], "row_count": 1}
    rows = [{"acct_desc": "Asset impairment charges", "amount_usd": 101703000}]
    lineage = MagicMock()
    with patch("api.db_service.complete_tieout_lines",
               AsyncMock(return_value=("SELECT 2", rows, ["Asset impairment charges"]))) as done:
        await hybrid_service._complete_tieout_lines(meta, STATEMENT, "c1", None, None, None, lineage)
    assert done.await_args.args[:4] == ("c1", OPEX_SQL, STATEMENT, ["gl_fs_rollup"])
    assert meta["sql_query"] == "SELECT 2" and meta["result_preview"] == rows and meta["row_count"] == 1
    assert "Asset impairment charges" in lineage.add_step.call_args.kwargs["detail"]


async def test_hybrid_keeps_first_result_when_completion_fails():
    from api import hybrid_service

    meta = {"sql_query": OPEX_SQL, "sources": ["gl_fs_rollup"], "result_preview": [{"a": 1}], "row_count": 1}
    with patch("api.db_service.complete_tieout_lines", AsyncMock(side_effect=RuntimeError("db down"))):
        await hybrid_service._complete_tieout_lines(meta, STATEMENT, "c1", None, None, None, MagicMock())
    assert meta["sql_query"] == OPEX_SQL and meta["result_preview"] == [{"a": 1}]


@pytest.mark.parametrize("meta", [None, {"sql_query": "", "sources": []}, {"sql_query": OPEX_SQL, "csv_answer": "x"}])
async def test_hybrid_skips_without_a_sql_result(meta):
    from api import hybrid_service

    with patch("api.db_service.complete_tieout_lines", AsyncMock()) as done:
        await hybrid_service._complete_tieout_lines(meta, STATEMENT, "c1", None, None, None, MagicMock())
    done.assert_not_awaited()


def test_synthesis_never_calls_unqueried_lines_missing():
    from api import hybrid_service

    src = inspect.getsource(hybrid_service.hybrid_chat)
    assert "never say it is missing from the books" in src


# ── 2. filing citations use the printed page, never the chunk number ──────────

def _filing(pages: int) -> str:
    body = []
    for n in range(1, pages + 1):
        body.append(f"Prose on page {n}. " * 30)
        if n == 3:
            body.append("Etsy, Inc.\nConsolidated Statements of Operations\n\n"
                        "Revenue | 2,883,501\nNet income | 162,982")
        body.append(f"{n}\nTable of Contents")
    return "\n\n".join(body)


def test_filing_chunks_carry_printed_page():
    docs = chunk_filing(_filing(6))
    statement = next(d for d in docs if "Revenue | 2,883,501" in d.page_content)
    assert statement.metadata["page"] == 3
    pages = [d.metadata["page"] for d in docs]
    assert pages == sorted(pages) and pages[0] == 1 and set(pages) <= set(range(1, 8))


def test_filing_without_page_footers_has_no_page():
    docs = chunk_filing("ITEM 1.\n\n" + "Some prose. " * 400 + "\n\nRevenue | 10\nNet income | 2")
    assert all(d.metadata["page"] is None for d in docs)


def test_stray_number_lines_are_not_pages():
    text = "Intro.\n\n42\n\nMore prose.\n\n7\n\nRevenue | 10"
    assert all(d.metadata["page"] is None for d in chunk_filing(text))


def test_page_headers_and_keys():
    with_page = {"source": "ETSY-10-K", "page": 67, "section": "Item 8."}
    no_page = {"source": "ETSY-10-K", "page": None, "section": "Item 8."}
    assert chunk_header(with_page) == "Source: ETSY-10-K, Page 67"
    assert chunk_header(no_page) == "Source: ETSY-10-K, Section Item 8."
    assert chunk_header({"source": "X"}) == "Source: X"
    assert chunk_page(no_page) is None and chunk_page({"page": 0}) is None
    assert chunk_key(with_page) != chunk_key({**with_page, "page": 68})


def test_detector_reads_section_header_without_page():
    text = "Source: ETSY-10-K, Section Item 8.\nFY2025\nRevenue | 2,883,501\n"
    period, file, page = _doc_context_for(text, text.index("Revenue"), text.index("2,883"))
    assert (file, page) == ("ETSY-10-K", None)


def test_detector_still_reads_page_header():
    text = "Source: ETSY-10-K, Page 67\nRevenue | 2,883,501\n"
    _, file, page = _doc_context_for(text, text.index("Revenue"), text.index("2,883"))
    assert (file, page) == ("ETSY-10-K", 67)


def test_filing_cache_format_bumped():
    from api import edgar_service

    assert edgar_service.FILING_FORMAT_VERSION == "v3"


# ── 3. marketplace_connections created before DOCBOT-704 gets its columns ─────

def test_connector_migrations_cover_every_later_column():
    from sqlalchemy import MetaData

    from api.connector_store import CONNECTOR_COLUMN_MIGRATIONS, register_connector_tables

    table = register_connector_tables(MetaData())
    original = {"id", "connector_type", "encrypted_credentials", "is_active", "created_at"}
    handled_elsewhere = {"org_id"}  # org_service.ORG_ID_COLUMN_MIGRATIONS
    added = {ddl.split("IF NOT EXISTS ")[1].split()[0] for ddl in CONNECTOR_COLUMN_MIGRATIONS}
    assert set(table.c.keys()) - original - handled_elsewhere <= added
    assert all("ADD COLUMN IF NOT EXISTS" in ddl for ddl in CONNECTOR_COLUMN_MIGRATIONS)


def test_connector_migrations_run_on_startup():
    import api.index as idx

    assert "CONNECTOR_COLUMN_MIGRATIONS" in inspect.getsource(idx.init_db)


# ── 4. clearing the chat also clears the Inspector ────────────────────────────

def test_clear_session_and_chat_clear_inspector():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "src/hooks/useChatHandlers.ts").read_text()
    for fn in ("const clearChat = () => {", "const clearSession = () => {"):
        body = src[src.index(fn):src.index("};", src.index(fn))]
        assert "clearInspector();" in body
    helper = src[src.index("const clearInspector"):src.index("};", src.index("const clearInspector"))]
    assert "selectLineage(null)" in helper and "selectArtifact(null)" in helper
