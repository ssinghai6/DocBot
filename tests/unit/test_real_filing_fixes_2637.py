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


# ── 5. page footers: cross-references and the "Form 10-K | N" style ──────────

def test_cross_reference_number_does_not_break_page_run():
    from api.utils.chunker import _printed_pages

    pages = [f"Text of page {n}. " * 40 + f"\n{n}\nTable of Contents" for n in range(1, 8)]
    pages[2] = pages[2].replace("Text of page 3.", "See the table on page\n6\n. Text of page 3.", 1)
    text = "\n\n".join(pages) + "\n\nLast page text."
    page_at = _printed_pages(text)
    assert page_at(text.index("Text of page 4")) == 4
    assert page_at(text.index("Text of page 6")) == 6
    assert page_at(text.index("Last page text")) == 8


def test_form_10k_footer_rows_give_pages():
    pages = [f"Words on page {n}. " * 40 + f"\nApple Inc. | 2026 Form 10-K | {n}" for n in range(1, 6)]
    docs = chunk_filing("\n\n".join(pages))
    assert next(d for d in docs if "Words on page 3." in d.page_content).metadata["page"] == 3


def test_leading_footer_belongs_to_previous_page():
    text = "\n\n".join(["Page one prose. " * 60 + "\n1\nTable of Contents",
                        "Page two prose. " * 60 + "\n2\nTable of Contents",
                        "Page three prose. " * 60 + "\n3\nTable of Contents"])
    checked = 0
    for d in chunk_filing(text):
        first = next((ln for ln in d.page_content.splitlines()
                      if ln.strip() and not ln.strip().isdigit() and ln.strip() != "Table of Contents"), "")
        for word, page in (("Page one", 1), ("Page two", 2), ("Page three", 3)):
            if first.startswith(word):
                assert d.metadata["page"] == page
                checked += 1
    assert checked >= 2


# ── 6. year-over-year changes are computed in code ────────────────────────────

def test_yoy_facts_from_statement():
    from api.utils.yoy_facts import yoy_facts

    facts = {f.label: f for f in yoy_facts(STATEMENT.replace("2025 | 2024\n", "Year Ended December 31,\n2025 | 2024\n"))}
    assert facts["Total operating expenses"].line().endswith("higher by 145,921 (+8.8%)")
    assert facts["Income from operations"].line().endswith("lower by 113,998 (-30.0%)")
    impairment = facts["Asset impairment charges"].line()
    assert "FY2024 nil" in impairment and "no % change" in impairment
    interest = facts["Interest expense"].line()
    assert "larger by 4,703 (+34.1%)" in interest  # (18,509) vs (13,806): the expense grew
    assert "p67" in facts["Marketing"].line()


def test_yoy_units_percent_rows_and_header_with_label_column():
    from api.utils.yoy_facts import yoy_facts

    doc = ("Source: X, Page 5\n(In millions)\nMetric | 2025 | 2024 | % Change\n"
           "Take rate | 24.2% | 22.3% | 1.9\nRevenue | 100 | — | n/m\n")
    facts = {f.label: f.line() for f in yoy_facts(doc)}
    assert facts["Take rate"].endswith("higher by 1.9 percentage points")
    assert "in millions" in facts["Revenue"] and "nil in FY2024" in facts["Revenue"]


def test_yoy_ignores_tables_without_year_headers_and_prose():
    from api.utils.yoy_facts import yoy_prompt_block

    doc = "Source: X, Page 1\nSegment | Amount\nAds | 890\n\nRevenue grew 5% in 2025 vs 2024.\n"
    assert yoy_prompt_block(doc) == ""


def test_yoy_block_and_rule_reach_every_synthesis_prompt():
    import api.index as idx
    from api import autopilot_service, hybrid_service

    assert "yoy_prompt_block" in inspect.getsource(hybrid_service.hybrid_chat)
    assert "yoy_prompt_block" in inspect.getsource(autopilot_service._synthesizer_node)
    assert "yoy_prompt_block" in inspect.getsource(idx)


# ── 7. tie-out still works when the question asks for year-over-year ─────────

PIVOT_ROWS = [
    {"acct_desc": "Marketing", "amount_2025": 919_830_000, "amount_2024": 856_565_000, "change": 63_265_000},
    {"acct_desc": "Product development", "amount_2025": 450_192_000, "amount_2024": 443_056_000, "change": 7_136_000},
]
STATEMENT_DOC = STATEMENT.replace("2025 | 2024\n", "Year Ended December 31,\n2025 | 2024\n")


def test_pivoted_year_columns_carry_their_year():
    from api.utils.discrepancy_detector import _extract_from_rows

    vals = {(v.label, v.period): v.value for v in _extract_from_rows(PIVOT_ROWS)}
    assert vals[("marketing amount", "FY2025")] == 919_830_000
    assert vals[("marketing amount", "FY2024")] == 856_565_000


def test_pivoted_books_flag_the_planted_break_only():
    from api.utils.discrepancy_detector import detect_discrepancies

    report = detect_discrepancies(STATEMENT_DOC, {"result_preview": PIVOT_ROWS, "sql_query": "SELECT 1"})
    assert [(d.label, d.period) for d in report.discrepancies] == [("Marketing", "FY2025")]
    clean = [dict(r, amount_2025=914_830_000) if r["acct_desc"] == "Marketing" else r for r in PIVOT_ROWS]
    assert detect_discrepancies(STATEMENT_DOC, {"result_preview": clean, "sql_query": "SELECT 1"}).discrepancies == []


def test_filing_table_cells_take_the_header_year():
    from api.utils.discrepancy_detector import _extract_from_text

    vals = {v.label: v.period for v in _extract_from_text(STATEMENT_DOC, "doc")}
    assert vals["marketing"] == "FY2025"
    ascending = "Source: X, Page 3\nMetric | 2024 | 2025\nMarketing | 856,565 | 914,830\n"
    assert {v.label: v.period for v in _extract_from_text(ascending, "doc")}["marketing"] == "FY2024"


def test_unpaired_books_lines_are_reported():
    from api.utils.discrepancy_detector import detect_discrepancies

    sbc_only = "Source: X, Page 102\n2025 | 2024\nMarketing | 14,572 | 23,508\n"
    report = detect_discrepancies(sbc_only, {"result_preview": PIVOT_ROWS, "sql_query": "SELECT 1"})
    assert report.discrepancies == []
    assert report.unpaired_db_lines == ["Marketing", "Product development"]
    assert detect_discrepancies(STATEMENT_DOC, {"result_preview": PIVOT_ROWS, "sql_query": "SELECT 1"}).unpaired_db_lines == []


async def test_book_guided_retrieval_adds_the_missing_statement():
    from api import hybrid_service

    sbc_only = "Source: ETSY, Page 102\n2025 | 2024\nMarketing | 14,572 | 23,508\n"
    lineage = MagicMock()
    rag = AsyncMock(return_value=(STATEMENT_DOC.replace("ETSY-10-K-2026-02-19", "ETSY"),
                                  [{"source": "ETSY", "page": 67, "text": "Etsy, Inc."}]))
    with patch.object(hybrid_service, "rag_retrieve", rag):
        ctx, cits = await hybrid_service._book_guided_retrieval(
            sbc_only, [{"source": "ETSY", "page": 102, "text": "x"}],
            {"result_preview": PIVOT_ROWS, "sql_query": "SELECT 1"}, "s1", {}, lineage)
    assert rag.await_args.args[0] == "Marketing Product development"
    assert "Marketing | 914,830" in ctx and ctx.startswith(sbc_only)
    assert [c["page"] for c in cits] == [102, 67]
    assert "book_guided_retrieval" == lineage.add_step.call_args.args[0]


@pytest.mark.parametrize("doc", [
    STATEMENT_DOC,  # every books line already has a filing figure
    "Source: Demo.pdf, Page 4\nMarketing expense was $14.6M in FY2025.\n",  # no filing tables (demo PDFs)
])
async def test_book_guided_retrieval_leaves_answered_or_non_filing_context(doc):
    from api import hybrid_service

    with patch.object(hybrid_service, "rag_retrieve", AsyncMock()) as rag:
        ctx, _ = await hybrid_service._book_guided_retrieval(
            doc, [], {"result_preview": PIVOT_ROWS, "sql_query": "SELECT 1"}, "s1", {}, MagicMock())
    rag.assert_not_awaited()
    assert ctx == doc


def test_sql_generation_has_room_for_reasoning_tokens():
    from api import db_service

    src = inspect.getsource(db_service._generate_sql)
    assert src.count("max_tokens=SQL_GEN_MAX_TOKENS") == 2 and db_service.SQL_GEN_MAX_TOKENS >= 3000


def test_flag_column_does_not_rename_the_row():
    from api.utils.discrepancy_detector import detect_discrepancies

    rows = [dict(r, agree="No") for r in PIVOT_ROWS]
    report = detect_discrepancies(STATEMENT_DOC, {"result_preview": rows, "sql_query": "SELECT 1"})
    assert [d.label for d in report.discrepancies] == ["Marketing"]


def test_only_where_filters_are_widened():
    sql = ("SELECT acct_desc, (SELECT CASE WHEN SUM(CASE WHEN acct_desc IN ('Marketing', 'Product development', "
           "'General and administrative', 'Total operating expenses') THEN amount_usd END) > 0 THEN 'Yes' END) AS agree "
           "FROM gl_fs_rollup WHERE acct_desc IN ('Marketing', 'Product development', 'General and administrative', "
           "'Total operating expenses')")
    new, added = add_skipped_lines(sql, SCHEMA, STATEMENT, "sqlite")
    assert added == ["Asset impairment charges"]
    assert new.count("'Asset impairment charges'") == 1
    assert new.index("'Asset impairment charges'") > new.upper().index("WHERE")


# ── 8. answers are checked against the computed changes ───────────────────────

STATEMENT_WITH_SBC = (
    "Source: ETSY, Page 67\nConsolidated Statements of Operations\n(In thousands)\n2025 | 2024\n"
    "Revenue | 2,883,501 | 2,808,332\nCost of revenue | 817,800 | 774,554\nMarketing | 914,830 | 856,565\n"
    "General and administrative | 332,766 | 353,949\nProvision for income taxes | 83,683 | 107,494\n\n"
    "Source: ETSY, Page 102\nStock-based compensation expense included in the Consolidated Statements "
    "of Operations is as follows (in thousands):\n2025 | 2024\nCost of revenue | 28,004 | 32,575\n"
    "Marketing | 14,572 | 23,508\n"
)


def _wrong(answer: str) -> list[str]:
    from api.utils.yoy_facts import yoy_contradictions, yoy_facts

    return yoy_contradictions(answer, yoy_facts(STATEMENT_WITH_SBC))


@pytest.mark.parametrize("answer", [
    "Net income fell, driven by the impairment and a higher tax provision.",
    "- **Cost of revenue** fell 14% YoY.",
    "G&A rose to $332.8M.",
])
def test_contradicting_statements_are_caught(answer):
    assert len(_wrong(answer)) == 1


@pytest.mark.parametrize("answer", [
    "Marketing rose 6.8% while G&A fell 6.0%.",           # both right, split clauses
    "Cost of revenue increased 5.6% year over year.",     # "revenue" inside "cost of revenue"
    "Marketing rose despite lower ad spend.",             # two directions: left alone
    "Operating margin fell as revenue grew.",              # ratios are not the line
    "If revenue growth remains flat, margins could fall.",  # hypothetical
    "Marketing, G&A and revenue all match the books.",    # no direction word
])
def test_correct_or_unclear_statements_are_left_alone(answer):
    assert _wrong(answer) == []


def test_allocation_table_rows_are_not_the_line():
    from api.utils.yoy_facts import yoy_facts

    sbc = next(f for f in yoy_facts(STATEMENT_WITH_SBC) if f.source.endswith("p102") and f.label == "Marketing")
    assert sbc.table.startswith("Stock-based compensation expense included in")
    assert "[Stock-based compensation" in sbc.line()
    # "marketing fell" is checked against marketing expense (rose), not stock comp (fell)
    assert len(_wrong("Marketing spend fell this year.")) == 1


async def test_yoy_rewrite_prompt_lists_the_statements():
    from api import hybrid_service

    with patch("api.utils.llm_provider.achat_completion", AsyncMock(return_value="fixed")) as llm:
        out = await hybrid_service._correct_yoy_claims(
            "old answer", ['"higher tax provision" contradicts: ...'], "persona", "fmt", "\nFACTS\n", "q", "r1")
    prompt = llm.await_args.args[0][0]["content"]
    assert out == "fixed" and "higher tax provision" in prompt and "FACTS" in prompt
    assert "[[discrepancy:...]] marker" in prompt


def test_hybrid_checks_answers_against_computed_changes():
    from api import hybrid_service

    src = inspect.getsource(hybrid_service.hybrid_chat)
    assert "yoy_contradictions(" in src and "verify_yoy_claims" in src
