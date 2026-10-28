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
    assert len(line_items) == 5 and "Net income" in line_items


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pii_high_cardinality_and_long_values_are_skipped(tmp_path):
    rows = "".join(
        f"INSERT INTO people VALUES ('u{i}@example.com', 'u{i}@example.com', "
        f"'{'gold' if i % 2 else 'silver'}', 'id-{i}', '{'x' * 50 if i == 0 else 'short'}');"
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
    assert "sample_values" not in cols["ref"]        # 25 distinct > 20
    assert cols["tier"]["sample_values"] == ["gold", "silver"]
    assert cols["note"]["sample_values"] == ["short"]  # 50-char value dropped


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
