"""DOCBOT-2731: SQL prompt grounded in the data's sign convention, the finer
(report-line) category column, and the revenue / expense side of a total."""

import sqlite3

import pytest

from api import db_service
from tests.unit.test_sql_grounding_2723 import _prompt, _schema


async def _signed_schema(tmp_path):
    """Chipotle-style journal: revenue rows stored negative."""
    path = tmp_path / "cmg.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE gl_journal (account_name TEXT, amount NUMERIC, status TEXT)")
    conn.executemany(
        "INSERT INTO gl_journal VALUES (?,?,'posted')",
        [("Revenue", -1000), ("Labor", 300), ("Occupancy", 100), ("Revenue", -900), ("Labor", 250)],
    )
    conn.commit()
    conn.close()
    return await db_service._introspect_schema_from_url(f"sqlite:///{path}", "sqlite")


def _col(schema, name):
    return next(c for c in schema[0]["columns"] if c["name"] == name)


# ── 2. sign convention from the data ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_positive_ledger_gets_no_negation_rule(tmp_path):
    schema = await _schema(tmp_path)
    assert _col(schema, "amount")["amount_sign"] == "positive"
    assert "amount_sign" not in _col(schema, "fiscal_year")  # not an amount column
    prompt = await _prompt("Does Transient Occupancy Taxes revenue agree with the books?", schema)
    assert "stored positive for revenue and expense rows alike" in prompt
    assert "never negate it" in prompt
    assert "-SUM(amount)" not in prompt


@pytest.mark.asyncio
async def test_credit_negative_ledger_keeps_negation_rule(tmp_path):
    schema = await _signed_schema(tmp_path)
    assert _col(schema, "amount")["amount_sign"] == "credits_negative"
    prompt = await _prompt("What was revenue?", schema)
    assert "signed: debits/expenses positive, credits/revenue negative" in prompt
    assert "a revenue line is -SUM(amount)" in prompt
    assert schema[0]["schema_fmt"] == db_service._SCHEMA_FMT == 5


def test_small_negative_share_is_still_positive():
    """San Diego: 1% of rows are small corrections; 0.06% of the value."""
    assert db_service._CREDIT_NEGATIVE_SHARE > 0.0006


# ── 3. finer category column ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_broad_type_hint_is_replaced_by_finer_class(tmp_path):
    schema = await _schema(tmp_path)
    q = "Do Personnel Services expenditures for FY2023 in the report agree with the books?"
    assert db_service._category_hints(q, schema) == [("gl_actuals", "account_class", "Personnel Cost")]
    prompt = await _prompt(q, schema)
    assert "- gl_actuals.account_class = 'Personnel Cost'" in prompt
    assert "filter the finer, report-line-level column" in prompt


@pytest.mark.asyncio
async def test_non_personnel_is_not_rewritten(tmp_path):
    schema = await _schema(tmp_path)
    assert db_service._category_hints("total Non-Personnel spending", schema) == [
        ("gl_actuals", "account_type", "Non-Personnel")
    ]


# ── 4. expense totals never take revenue categories ──────────────────────────

@pytest.mark.parametrize("value,side", [
    ("Transient Occupancy Taxes", "revenue"), ("Property Tax Revenue", "revenue"),
    ("Transfers In", "revenue"), ("Charges for Current Services", "revenue"),
    ("Non-Personnel", "expense"), ("Personnel", "expense"), ("Transfers Out", "expense"),
    ("Provision for income taxes", None), ("Mission Bay", None),
])
def test_value_side(value, side):
    assert db_service._value_side(value) == side


@pytest.mark.asyncio
async def test_expense_question_lists_revenue_values_to_exclude(tmp_path):
    schema = await _schema(tmp_path)
    q = "Do total General Fund expenditures for FY2023 in the report agree with the books?"
    asked, excl = db_service._side_exclusions(q, schema)
    assert asked == "expense"
    by_col = {c: vals for _, c, vals in excl}
    assert set(by_col["account_type"]) == {"Transient Occupancy Taxes", "Property Tax Revenue"}
    prompt = await _prompt(q, schema)
    assert "This question totals expenses. These listed values are revenue categories" in prompt
    assert "'Property Tax Revenue'" in prompt.split("never include them")[1]


@pytest.mark.asyncio
async def test_no_side_hint_when_question_mixes_or_omits_sides(tmp_path):
    schema = await _schema(tmp_path)
    assert db_service._side_exclusions("net of revenue and expenses", schema) == ("", [])
    assert db_service._side_exclusions("how many departments", schema) == ("", [])


def test_sql_gen_runs_at_temperature_zero():
    import inspect
    import re
    temps = re.findall(r"temperature=([\w.]+)", inspect.getsource(db_service._generate_sql))
    assert temps and set(temps) == {"0"}
