"""DOCBOT-2720: the SQL prompt listed account_no and account_name as two
independently sorted lists; the model zipped them by position ('6200' read as
Food, '9100' as Revenue). Code/name columns that are 1:1 in the data are now
shown as real pairs."""

import re
import sqlite3
from unittest.mock import AsyncMock, patch

import pytest

from api import db_service

ACCOUNTS = [
    ("4000", "Revenue"),
    ("5100", "Food, beverage and packaging"),
    ("5200", "Labor"),
    ("5300", "Occupancy"),
    ("5400", "Other operating costs"),
    ("6100", "General and administrative expenses"),
    ("6200", "Depreciation and amortization"),
    ("6300", "Pre-opening costs"),
    ("6400", "Impairment, closure costs, and asset disposals"),
    ("8000", "Interest and other income, net"),
    ("9100", "Provision for income taxes"),
]
GL_DDL = (
    "CREATE TABLE gl_journal (entry_id INTEGER PRIMARY KEY, fiscal_year INTEGER, "
    "fiscal_quarter INTEGER, account_no TEXT, account_name TEXT, amount NUMERIC, "
    "status TEXT, memo TEXT);"
)


def _gl_db(tmp_path, accounts=ACCOUNTS, extra=()):
    path = tmp_path / "gl.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(GL_DDL)
    rows = [
        (2025, q, no, name, 1000.0 * (i + 1), "void" if (q == 4 and i == 3) else "posted", f"Q{q} {name}")
        for i, (no, name) in enumerate(accounts) for q in (1, 2, 3, 4)
    ]
    conn.executemany(
        "INSERT INTO gl_journal (fiscal_year, fiscal_quarter, account_no, account_name, amount, status, memo) "
        "VALUES (?,?,?,?,?,?,?)", list(rows) + list(extra),
    )
    conn.commit()
    conn.close()
    return f"sqlite:///{path}"


def _cols(schema):
    return {c["name"]: c for c in schema[0]["columns"]}


async def _prompt(schema):
    with patch("api.utils.llm_provider.achat_completion", new=AsyncMock(return_value="SELECT 1")) as llm, \
            patch("api.utils.llm_cache.get_cached_response", new=AsyncMock(return_value=None)), \
            patch("api.utils.llm_cache.set_cached_response", new=AsyncMock()):
        await db_service._generate_sql("Does FY2025 revenue tie out?", schema, [])
    return llm.await_args.args[0][0]["content"]


@pytest.mark.asyncio
async def test_code_name_columns_get_true_pairs(tmp_path):
    schema = await db_service._introspect_schema_from_url(_gl_db(tmp_path), "sqlite")
    cols = _cols(schema)
    assert cols["account_no"]["value_pairs"] == [list(p) for p in ACCOUNTS]
    assert cols["account_no"]["paired_with"] == "account_name"
    assert cols["account_name"]["paired_with"] == "account_no"
    assert "value_pairs" not in cols["account_name"]
    # status (2 values) and memo (44 values) are not paired
    assert "paired_with" not in cols["status"] and "paired_with" not in cols["memo"]
    # the plain lists stay (tie-out line recovery reads them)
    assert cols["account_name"]["sample_values"] == sorted(n for _, n in ACCOUNTS)
    assert schema[0]["schema_fmt"] == db_service._SCHEMA_FMT == 4


@pytest.mark.asyncio
async def test_prompt_shows_pairs_not_two_positional_lists(tmp_path):
    schema = await db_service._introspect_schema_from_url(_gl_db(tmp_path), "sqlite")
    prompt = await _prompt(schema)
    for no, name in ACCOUNTS:
        assert f"'{no}'='{name.replace(chr(39), chr(39) * 2)}'" in prompt
    # the misalignment from prod: 5100 paired with the alphabetically-first name
    assert "'5100'='Depreciation and amortization'" not in prompt
    # the name column no longer carries its own sorted list
    assert re.search(r"account_name \(TEXT, see account_no pairs\)", prompt)
    assert "'Depreciation and amortization', 'Food, beverage and packaging'" not in prompt
    assert "never infer which code goes with which name from list position" in prompt


@pytest.mark.asyncio
async def test_not_one_to_one_falls_back_to_separate_lists(tmp_path):
    # account 5200 also posted under a second name: no longer 1:1
    extra = [(2025, 1, "5200", "Labour", 1.0, "posted", "dup")]
    accounts = ACCOUNTS[:10]  # 10 numbers vs 11 names after the extra row
    schema = await db_service._introspect_schema_from_url(_gl_db(tmp_path, accounts, extra), "sqlite")
    cols = _cols(schema)
    assert "paired_with" not in cols["account_no"] and "paired_with" not in cols["account_name"]
    prompt = await _prompt(schema)
    assert "account_name (TEXT, values 'Depreciation and amortization'" in prompt


@pytest.mark.asyncio
async def test_same_count_but_not_bijective_falls_back(tmp_path):
    # 11 codes, 11 names, but 5200 has two names and 5300 has none of its own
    accounts = list(ACCOUNTS)
    extra = [(2025, 1, "5200", "Occupancy", 1.0, "posted", "misposted")]
    schema = await db_service._introspect_schema_from_url(_gl_db(tmp_path, accounts, extra), "sqlite")
    assert "paired_with" not in _cols(schema)["account_no"]


@pytest.mark.asyncio
async def test_old_cached_schema_without_pairs_still_renders():
    old = [{
        "name": "gl_journal", "schema_fmt": 2,
        "columns": [
            {"name": "account_no", "type": "TEXT", "sample_values": sorted(n for n, _ in ACCOUNTS), "values_complete": True},
            {"name": "account_name", "type": "TEXT", "sample_values": sorted(n for _, n in ACCOUNTS), "values_complete": True},
            {"name": "amount", "type": "NUMERIC"},
        ],
    }]
    prompt = await _prompt(old)
    assert "account_no (TEXT, values '4000'" in prompt
    assert "amount (NUMERIC)" in prompt


@pytest.mark.asyncio
async def test_over_budget_pairs_render_as_first_five_pairs():
    cols = [
        {"name": "account_no", "type": "TEXT", "sample_values": [n for n, _ in ACCOUNTS],
         "values_complete": True, "value_pairs": [list(p) for p in ACCOUNTS], "paired_with": "account_name"},
        {"name": "account_name", "type": "TEXT", "sample_values": sorted(n for _, n in ACCOUNTS),
         "values_complete": True, "paired_with": "account_no"},
    ]
    filler = {"name": "big", "columns": [
        {"name": "c", "type": "TEXT", "sample_values": ["x" * 59] * 110, "values_complete": True}]}
    prompt = await _prompt([filler, {"name": "gl_journal", "columns": cols}])
    assert "account_no (TEXT, e.g. paired account_no=account_name: '4000'='Revenue'" in prompt
    assert "'5400'='Other operating costs')" in prompt and "'6100'=" not in prompt
