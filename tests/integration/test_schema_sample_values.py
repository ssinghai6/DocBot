"""DOCBOT-2321: sample values for low-cardinality TEXT columns, computed at
introspection time and shown to the SQL generator."""

import sqlite3
from unittest.mock import AsyncMock, patch

import pytest

from api import db_service
from api.demo_service import DEMO_SQL_DATA, DEMO_SQL_SCHEMA, TIEOUT_SQL_DATA, TIEOUT_SQL_SCHEMA


def _db(tmp_path, name, *scripts):
    path = tmp_path / name
    conn = sqlite3.connect(path)
    for script in scripts:
        conn.executescript(script)
    conn.commit()
    conn.close()
    return f"sqlite:///{path}"


def _cols(schema, table):
    return {c["name"]: c for c in next(t for t in schema if t["name"] == table)["columns"]}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_quickbite_quarter_values_are_sampled(tmp_path):
    url = _db(tmp_path, "qb.sqlite", DEMO_SQL_SCHEMA, DEMO_SQL_DATA)
    schema = await db_service._introspect_schema_from_url(url, "sqlite")
    quarterly = _cols(schema, "quarterly")
    assert quarterly["quarter"]["sample_values"] == ["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"]
    assert "sample_values" not in quarterly["net_income_m"]  # numeric: never sampled
    assert len(_cols(schema, "segments")["segment_name"]["sample_values"]) == 4


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sqlite_view_text_columns_are_sampled(tmp_path):
    url = _db(tmp_path, "to.sqlite", TIEOUT_SQL_SCHEMA, TIEOUT_SQL_DATA)
    schema = await db_service._introspect_schema_from_url(url, "sqlite")
    line_items = _cols(schema, "income_statement")["line_item"]["sample_values"]
    assert "Net income" in line_items and len(line_items) > 5  # DOCBOT-2633: full list


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pii_high_cardinality_and_long_values_are_skipped(tmp_path):
    rows = "".join(
        f"INSERT INTO people VALUES ('u{i}@example.com', 'u{i}@example.com', "
        f"'{'gold' if i % 2 else 'silver'}', 'id-{i}', '{'x' * 70 if i == 0 else 'short'}');"
        for i in range(25)
    )
    url = _db(
        tmp_path, "pii.sqlite",
        "CREATE TABLE people (email TEXT, contact TEXT, tier TEXT, ref TEXT, note TEXT);",
        rows,
    )
    cols = _cols(await db_service._introspect_schema_from_url(url, "sqlite"), "people")
    assert "sample_values" not in cols["email"]      # PII by column name
    assert "sample_values" not in cols["contact"]    # PII by value (masker)
    assert len(cols["ref"]["sample_values"]) == 25  # <=50 distinct: full list (DOCBOT-2633)
    assert cols["tier"]["sample_values"] == ["gold", "silver"]
    assert cols["tier"]["values_complete"] is True
    assert cols["note"]["sample_values"] == ["short"]  # 70-char value dropped


@pytest.mark.integration
def test_column_budget_caps_sampling(tmp_path):
    from sqlalchemy import create_engine

    path = tmp_path / "b.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE t (a TEXT, b TEXT); INSERT INTO t VALUES ('x', 'y');")
    conn.close()
    cols = [{"name": "a", "type": "TEXT"}, {"name": "b", "type": "TEXT"}]
    db_service._attach_sample_values(create_engine(f"sqlite:///{path}"), "t", cols, [1])
    assert cols[0]["sample_values"] == ["x"] and "sample_values" not in cols[1]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sql_prompt_shows_sample_values():
    from api.utils import llm_cache

    subset = [{"name": "quarterly", "columns": [
        {"name": "quarter", "type": "TEXT", "sample_values": ["Q1 2025", "Q4 2025"]},
        {"name": "net_income_m", "type": "REAL"},
    ]}]
    with (
        patch.object(llm_cache, "get_cached_response", AsyncMock(return_value=None)),
        patch.object(llm_cache, "set_cached_response", AsyncMock()),
        patch("api.utils.llm_provider.chat_completion", return_value="SELECT 1") as cc,
    ):
        await db_service._generate_sql("What was Q4 net income?", subset, [])
    prompt = cc.call_args.args[0][0]["content"]
    assert "quarter (TEXT, e.g. 'Q1 2025', 'Q4 2025')" in prompt
    assert "net_income_m (REAL)" in prompt
    assert "LIKE 'Q4%'" in prompt


# ---------------------------------------------------------------------------
# DOCBOT-2633: complete value lists for <=50 distinct, frequency sample above
# ---------------------------------------------------------------------------

_ACCOUNTS = [
    "Revenue", "Cost of revenue", "Gross profit", "Marketing", "Product development",
    "General and administrative", "Asset impairment charges", "Total operating expenses",
    "Income from operations", "Net income", "Cash and cash equivalents", "Total assets",
    "Total liabilities", "Total stockholders' deficit",
]


def _gl_db(tmp_path):
    rows = "".join(
        f"INSERT INTO gl VALUES ({y}, '{a.replace(chr(39), chr(39) * 2)}', 1);"
        for y in (2024, 2025) for a in _ACCOUNTS
    )
    return _db(tmp_path, "gl.sqlite", "CREATE TABLE gl (fiscal_year INTEGER, acct_desc TEXT, amount_usd INTEGER);", rows)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_low_cardinality_column_gets_complete_list(tmp_path):
    schema = await db_service._introspect_schema_from_url(_gl_db(tmp_path), "sqlite")
    col = _cols(schema, "gl")["acct_desc"]
    assert col["sample_values"] == sorted(_ACCOUNTS)  # Marketing / Net income included
    assert col["values_complete"] is True
    assert all(t["schema_fmt"] == db_service._SCHEMA_FMT for t in schema)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_high_cardinality_column_sends_no_values(tmp_path):
    # 60 distinct values (> 50), some repeated: nothing goes to the prompt (no real values leak)
    rows = "".join(f"INSERT INTO t VALUES ('v{i:02d}');" for i in range(58))
    rows += "INSERT INTO t VALUES ('z_common');" * 5 + "INSERT INTO t VALUES ('a_mid');" * 3
    url = _db(tmp_path, "hc.sqlite", "CREATE TABLE t (cat TEXT);", rows)
    col = _cols(await db_service._introspect_schema_from_url(url, "sqlite"), "t")["cat"]
    assert "sample_values" not in col and "values_complete" not in col


@pytest.mark.integration
@pytest.mark.asyncio
async def test_char_budget_falls_back_to_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(db_service, "_SAMPLE_LIST_MAX_CHARS", 40)
    schema = await db_service._introspect_schema_from_url(_gl_db(tmp_path), "sqlite")
    col = _cols(schema, "gl")["acct_desc"]
    assert col["sample_values"] == sorted(_ACCOUNTS)[:5] and "values_complete" not in col


@pytest.mark.integration
@pytest.mark.asyncio
async def test_table_bigger_than_scan_is_not_called_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(db_service, "_SAMPLE_SCAN_ROWS", 10)
    col = _cols(await db_service._introspect_schema_from_url(_gl_db(tmp_path), "sqlite"), "gl")["acct_desc"]
    assert "values_complete" not in col and col["sample_values"]


async def _prompt_for(subset):
    from api.utils import llm_cache

    with (
        patch.object(llm_cache, "get_cached_response", AsyncMock(return_value=None)),
        patch.object(llm_cache, "set_cached_response", AsyncMock()),
        patch("api.utils.llm_provider.chat_completion", return_value="SELECT 1") as cc,
    ):
        await db_service._generate_sql("Do the opex lines agree?", subset, [])
    return cc.call_args.args[0][0]["content"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sql_prompt_lists_complete_values_and_caps_budget(monkeypatch):
    col = {"name": "acct_desc", "type": "TEXT", "sample_values": sorted(_ACCOUNTS), "values_complete": True}
    prompt = await _prompt_for([{"name": "gl", "columns": [col]}])
    assert "acct_desc (TEXT, values 'Asset impairment charges'," in prompt
    assert "'Marketing'" in prompt and "'Total stockholders'' deficit'" in prompt
    assert "EVERY value" in prompt

    # prompt-wide budget spent: later columns drop to 5 examples
    monkeypatch.setattr(db_service, "_PROMPT_VALUES_MAX_CHARS", 10)
    prompt = await _prompt_for([{"name": "gl", "columns": [dict(col), dict(col, name="acct2")]}])
    assert "acct_desc (TEXT, values " in prompt
    assert "acct2 (TEXT, e.g. 'Asset impairment charges', 'Cash and cash equivalents', " \
           "'Cost of revenue', 'General and administrative', 'Gross profit')" in prompt


@pytest.mark.integration
@pytest.mark.asyncio
async def test_old_format_cache_row_is_reintrospected():
    """A schema cached before DOCBOT-2633 (no schema_fmt) must not be served."""
    import json
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from sqlalchemy import Column, MetaData, String, Table

    tbl = Table("x", MetaData(), Column("id", String), Column("connection_id", String))
    fresh = SimpleNamespace(
        schema_json=json.dumps([{"name": "gl", "columns": []}]),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    rows = iter([fresh, None])  # cache row, then connection row (missing)
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.execute = AsyncMock(side_effect=lambda *_: MagicMock(fetchone=lambda: next(rows)))
    with pytest.raises(db_service.ConnectionNotFoundError):  # fell through past the cache
        await db_service.get_schema("c1", tbl, tbl, lambda: session)

    fresh.schema_json = json.dumps([{"name": "gl", "columns": [], "schema_fmt": db_service._SCHEMA_FMT}])
    rows = iter([fresh])
    assert (await db_service.get_schema("c1", tbl, tbl, lambda: session))[0]["name"] == "gl"
