"""DOCBOT-2212: "tieout" demo dataset — draft 10-K vs trial balance.

No network: the vector store is mocked and the app DB is in-memory SQLite.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import api.index as idx
from api import demo_service as d
from api.utils.discrepancy_detector import detect_discrepancies

# Every currency figure stated in the draft report -> SQL that recomputes it
# from the books (trial_balance only; independent of the helper tables/view).
_TB = "SELECT SUM({expr}) FROM trial_balance WHERE period = 'FY2025' AND {where}"
_REV = _TB.format(expr="credit - debit", where="account_type = 'revenue'")
_COGS = _TB.format(expr="debit - credit", where="account_type = 'cogs'")
_OPEX = _TB.format(expr="debit - credit", where="account_type = 'opex'")
_PL = "SELECT SUM(credit - debit) FROM trial_balance WHERE period = 'FY2025' AND account_type IN ({})"


def _acct(name: str) -> str:
    return _TB.format(expr="ABS(debit - credit)", where=f"account LIKE '%{name}'")


REPORT_FIGURES: list[tuple[str, str, str]] = [
    # (label, figure as written in the draft, books SQL)
    ("Total revenue", "Total revenue: $418.2 million", _REV),
    ("Gross profit (MD&A)", "Gross profit: $241.3 million", _PL.format("'revenue','cogs'")),
    ("Operating income (MD&A)", "Operating income: $109.0 million", _PL.format("'revenue','cogs','opex'")),
    ("Net income (MD&A)", "Net income: $82.5 million", _PL.format("'revenue','cogs','opex','other','tax'")),
    ("R&D (MD&A)", "Research and development expense: $41.2 million", _acct("Research and development")),
    ("Sales and marketing expense", "Sales and marketing expense: $54.9 million", _acct("Sales and marketing")),
    ("G&A (MD&A)", "General and administrative expense: $35.9 million", _acct("General and administrative")),
    ("Revenue (IS)", "Revenue: $412.6M", _REV),
    ("Cost of revenue", "Cost of revenue: $171.3M", _COGS),
    ("Gross profit (IS)", "Gross profit: $241.3M", _PL.format("'revenue','cogs'")),
    ("R&D (IS)", "Research and development: $41.2M", _acct("Research and development")),
    ("S&M (IS)", "Sales and marketing: $55.2M", _acct("Sales and marketing")),
    ("G&A (IS)", "General and administrative: $35.9M", _acct("General and administrative")),
    ("Total opex", "Total operating expenses: $132.3M", _OPEX),
    ("Operating income (IS)", "Operating income: $109.0M", _PL.format("'revenue','cogs','opex'")),
    ("Interest expense", "Interest expense: $4.6M", _acct("Interest expense")),
    ("Pre-tax income", "Income before income taxes: $104.4M", _PL.format("'revenue','cogs','opex','other'")),
    ("Income tax", "Income tax expense: $21.9M", _acct("Income tax expense")),
    ("Net income (IS)", "Net income: $82.5M", _PL.format("'revenue','cogs','opex','other','tax'")),
    ("Commercial Instruments", "Commercial Instruments: $248.4M", _acct("Revenue - Commercial Instruments")),
    ("Lab Services", "Lab Services: $104.9M", _acct("Revenue - Lab Services")),
    ("Software and Analytics", "Software and Analytics: $61.5M", _acct("Revenue - Software and Analytics")),
]

_MONEY_RE = re.compile(r"\$([0-9][0-9,]*(?:\.[0-9]+)?)\s*(M\b|million)")


def _report_text() -> str:
    return "\n\n".join(c.page_content for c in d.TIEOUT_DOCUMENT_CHUNKS)


def _doc_context() -> str:
    # Same shape hybrid_service.rag_retrieve builds.
    return "\n\n".join(
        f"Source: {c.metadata['source']}, Page {c.metadata['page']}\n{c.page_content}"
        for c in d.TIEOUT_DOCUMENT_CHUNKS
    )


@pytest.fixture
def books():
    path = d._create_tieout_sqlite()
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()
    path.unlink(missing_ok=True)


def test_trial_balance_balances(books):
    dr, cr = books.execute("SELECT SUM(debit), SUM(credit) FROM trial_balance").fetchone()
    assert dr == cr


def test_every_report_figure_recomputed_and_only_planted_mismatch(books):
    text = _report_text()
    # Coverage: every $ figure in the draft is in REPORT_FIGURES (no unchecked figure).
    assert len(_MONEY_RE.findall(text)) == len(REPORT_FIGURES)
    mismatches = {}
    for label, written, sql in REPORT_FIGURES:
        assert written in text, written
        reported = float(_MONEY_RE.search(written).group(1)) * 1_000_000
        actual = books.execute(sql).fetchone()[0]
        if round(reported) != round(actual):
            mismatches[label] = (reported, actual)
    assert mismatches == {k: (float(a), float(b)) for k, (a, b) in d.TIEOUT_EXPECTED_DISCREPANCIES.items()}


def test_detector_flags_exactly_planted(books):
    rows = [
        dict(r)
        for q in ("SELECT * FROM income_statement", "SELECT * FROM revenue_by_segment", "SELECT * FROM opex")
        for r in books.execute(q)
    ]
    report = detect_discrepancies(_doc_context(), {"result_preview": rows, "sources": ["trial_balance"]})
    got = {i.label: (i.doc_value, i.db_value) for i in report.discrepancies}
    assert got == {k: (float(a), float(b)) for k, (a, b) in d.TIEOUT_EXPECTED_DISCREPANCIES.items()}
    small = next(i for i in report.discrepancies if i.label == "Sales and marketing expense")
    assert small.severity == "medium" and small.doc_page == 5


@pytest.fixture
async def app_db():
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    tables = [idx.sessions_table, idx.db_connections_table, idx.schema_cache_table]
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: idx.metadata.create_all(c, tables=tables))
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


async def test_init_tieout_dataset(app_db):
    owner = {"user_id": None, "org_id": None, "anon_owner": "anon-digest-x"}
    stores: dict = {}
    with patch("api.demo_service.create_store", new=MagicMock(return_value="store")):
        result = await d.init_demo_session(
            vector_stores=stores,
            get_embeddings_fn=MagicMock(),
            sessions_table=idx.sessions_table,
            db_connections_table=idx.db_connections_table,
            schema_cache_table=idx.schema_cache_table,
            async_session_factory=app_db,
            dataset="tieout",
            owner=owner,
        )
    assert result["dataset"] == "tieout"
    assert result["company"] == d.TIEOUT_COMPANY
    assert result["suggested_questions"][0] == "Tie out the draft 10-K against the trial balance"
    assert {"trial_balance", "revenue_by_segment", "opex"} <= set(result["schema_summary"]["tables"])
    assert stores[result["session_id"]] == "store"

    async with app_db() as s:
        sess = (await s.execute(select(idx.sessions_table).where(
            idx.sessions_table.c.session_id == result["session_id"]))).one()
        conn = (await s.execute(select(idx.db_connections_table).where(
            idx.db_connections_table.c.id == result["connection_id"]))).one()
    assert sess.anon_owner == "anon-digest-x" and sess.source == "demo"
    assert conn.anon_owner == "anon-digest-x" and conn.session_id == result["session_id"]
    db_path = Path(conn.dbname)
    assert db_path.exists()
    db_path.unlink()
