"""DOCBOT-2637: complete_tieout_lines re-runs a books query on a real SQLite file."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import api.index as idx
from api.db_service import complete_tieout_lines
from api.utils.encryption import encrypt_credentials

LINES = [
    ("Marketing", 914_830_000), ("Product development", 450_192_000),
    ("General and administrative", 332_766_000), ("Asset impairment charges", 101_703_000),
    ("Total operating expenses", 1_799_491_000), ("Net income", 162_982_000),
]
DOC = (
    "Source: ETSY-10-K-2026-02-19, Page 67\n"
    "Marketing | 914,830\nProduct development | 450,192\nGeneral and administrative | 332,766\n"
    "Asset impairment charges | 101,703\nTotal operating expenses | 1,799,491\n"
)
SQL = (
    "SELECT acct_desc, amount_usd FROM gl_fs_rollup WHERE fiscal_year = 2025 AND acct_desc IN "
    "('Marketing', 'Product development', 'General and administrative', 'Total operating expenses') LIMIT 500"
)


@pytest.fixture
async def conn(tmp_path):
    books = tmp_path / "gl_export.db"
    with sqlite3.connect(books) as db:
        db.execute("CREATE TABLE gl_fs_rollup (fiscal_year INTEGER, acct_desc TEXT, amount_usd INTEGER)")
        db.executemany("INSERT INTO gl_fs_rollup VALUES (2025, ?, ?)", LINES)

    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as c:
        await c.run_sync(lambda s: idx.metadata.create_all(
            s, tables=[idx.sessions_table, idx.db_connections_table, idx.schema_cache_table]))
    factory = async_sessionmaker(engine, expire_on_commit=False)
    creds = encrypt_credentials({"dialect": "sqlite", "host": "__local_file__", "port": 0,
                                 "dbname": str(books), "user": "", "password": ""})
    async with factory() as s, s.begin():
        await s.execute(insert(idx.db_connections_table).values(
            id="c1", session_id="s1", dialect="sqlite", host="__local_file__", port=0,
            dbname=str(books), credentials_blob=creds, created_at=datetime.now(timezone.utc)))
    yield factory
    await engine.dispose()


async def test_skipped_line_is_queried_from_the_books(conn):
    done = await complete_tieout_lines(
        "c1", SQL, DOC, ["gl_fs_rollup"], idx.db_connections_table, idx.schema_cache_table, conn)
    assert done is not None
    sql, rows, added = done
    assert added == ["Asset impairment charges"]
    assert {r["acct_desc"]: r["amount_usd"] for r in rows}["Asset impairment charges"] == 101_703_000
    assert len(rows) == 5 and "Net income" not in {r["acct_desc"] for r in rows}
    assert "LIMIT" in sql.upper()


async def test_nothing_skipped_returns_none(conn):
    doc = DOC.replace("Asset impairment charges | 101,703\n", "")
    assert await complete_tieout_lines(
        "c1", SQL, doc, ["gl_fs_rollup"], idx.db_connections_table, idx.schema_cache_table, conn) is None


async def test_numeric_amounts_come_back_as_numbers(conn, monkeypatch):
    """DOCBOT-2638: Postgres NUMERIC arrives as Decimal; the detector needs numbers."""
    from decimal import Decimal

    import api.db_service as dbs

    real = dbs._execute_query

    async def as_numeric(*a, **kw):
        rows, cols = await real(*a, **kw)
        return [tuple(Decimal(v) if isinstance(v, int) else v for v in r) for r in rows], cols

    monkeypatch.setattr(dbs, "_execute_query", as_numeric)
    _sql, rows, _added = await complete_tieout_lines(
        "c1", SQL, DOC, ["gl_fs_rollup"], idx.db_connections_table, idx.schema_cache_table, conn)
    amounts = {r["acct_desc"]: r["amount_usd"] for r in rows}
    assert amounts["Marketing"] == 914_830_000 and isinstance(amounts["Marketing"], float)
