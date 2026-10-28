"""DOCBOT-2320: realistic LLM replies for broad tie-out questions, and the
single bounded corrective SQL retry."""

import logging
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.demo_service import TIEOUT_SQL_DATA, TIEOUT_SQL_SCHEMA
from api.utils.sql_validator import (
    QueryValidationError,
    extract_sql,
    validate_and_sanitize_sql,
)


@pytest.fixture(scope="module")
def tieout_db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(TIEOUT_SQL_SCHEMA)
    conn.executescript(TIEOUT_SQL_DATA)
    yield conn
    conn.close()


def _run(reply, db):
    sql = validate_and_sanitize_sql(extract_sql(reply, "sqlite"), "sqlite")
    return sql, db.execute(sql).fetchall()


CTE_UNION = """WITH rev AS (
  SELECT SUM(credit - debit) AS amt FROM trial_balance WHERE account_type = 'revenue'
),
opx AS (
  SELECT SUM(debit - credit) AS amt FROM trial_balance WHERE account_type = 'opex'
)
SELECT 'Total revenue' AS metric, amt FROM rev
UNION ALL
SELECT 'Total operating expenses', amt FROM opx
UNION ALL
SELECT 'Segment: ' || segment, amount FROM revenue_by_segment
ORDER BY metric"""


@pytest.mark.unit
@pytest.mark.parametrize(
    "reply, expected_rows",
    [
        (CTE_UNION, 5),
        ("```sql\n" + CTE_UNION + "\n```", 5),
        ("```sqlite\n" + CTE_UNION + "\n```", 5),
        ("Here is the tie-out query:\n\n" + CTE_UNION + ";\n\nThis compares the books.", 5),
        ("sql\n" + CTE_UNION, 5),
        ("Query: SELECT category, amount FROM opex", 3),
        ("SQL: SELECT category, amount FROM opex", 3),
        ("sql SELECT category, amount FROM opex", 3),
        ("SELECT category, amount FROM opex\nThis query lists every opex line.", 3),
        ("SELECT category, amount FROM opex\n\n**Explanation:** opex by category", 3),
        ("SELECT category, amount FROM opex -- all categories", 3),
        ("-- Tie-out: opex\nSELECT category, amount FROM opex", 3),
        ("SELECT 'a;b' AS x, amount FROM opex;", 3),
        # double-quoted identifiers, CASE / ROUND / printf / || / IIF / total()
        ('SELECT "category", ROUND("amount" / 1e6, 1) || \'M\' AS m, '
         "printf('%.1f', amount / 1e6), CASE WHEN amount > 4e7 THEN 'big' ELSE 'small' END, "
         "IIF(amount > 0, 'pos', 'neg'), IFNULL(amount, 0) FROM \"opex\"", 3),
        ("SELECT period, total(amount) FROM opex GROUP BY period", 1),
        ("SELECT account FROM trial_balance WHERE account GLOB '6*'", 3),
        # document figures embedded as a VALUES CTE: column list must survive
        ("WITH doc(line_item, amount) AS (VALUES ('Revenue', 412600000), "
         "('Total operating expenses', 132300000))\n"
         "SELECT d.line_item, d.amount AS doc_amount, i.amount AS books_amount, "
         "d.amount - i.amount AS diff FROM doc d JOIN income_statement i "
         "ON i.line_item = d.line_item", 2),
        # two statements, same width -> UNION ALL
        ("SELECT 'Revenue' AS metric, SUM(amount) AS v FROM revenue_by_segment;\n"
         "SELECT 'Opex' AS metric, SUM(amount) AS v FROM opex;", 2),
    ],
)
def test_broad_tieout_replies_validate_and_execute(reply, expected_rows, tieout_db):
    _sql, rows = _run(reply, tieout_db)
    assert len(rows) == expected_rows


@pytest.mark.unit
def test_parenthesized_first_union_branch_is_kept():
    # Used to start at the later "SELECT 'Opex'" line and silently drop branch 1.
    sql = extract_sql(
        "Query:\n(SELECT 'Revenue' AS m,\n  SUM(amount) AS v FROM revenue_by_segment)\n"
        "UNION ALL\nSELECT 'Opex', SUM(amount) FROM opex",
        "postgresql",
    )
    assert sql.startswith("(SELECT 'Revenue'") and "'Opex'" in sql
    validate_and_sanitize_sql(sql, "postgresql")


@pytest.mark.unit
def test_two_incompatible_statements_take_the_first(tieout_db):
    sql, rows = _run(
        "SELECT category, amount FROM opex;\nSELECT segment FROM revenue_by_segment;", tieout_db
    )
    assert "revenue_by_segment" not in sql and len(rows) == 3


@pytest.mark.unit
def test_comment_tail_does_not_swallow_injected_limit():
    sql = validate_and_sanitize_sql(extract_sql("SELECT 1 -- note", "sqlite"), "sqlite")
    assert sql.endswith("\nLIMIT 500")


@pytest.mark.unit
@pytest.mark.parametrize(
    "reply",
    [
        "SELECT 1; DROP TABLE trial_balance",
        "SELECT 1;\nDELETE FROM opex",
        "SELECT 1 AS a;\nINSERT INTO opex SELECT 1",
        "DROP TABLE opex",
        "WITH d AS (DELETE FROM opex RETURNING *) SELECT * FROM d",
    ],
)
def test_write_statements_never_survive(reply):
    try:
        sql = validate_and_sanitize_sql(extract_sql(reply, "postgresql"), "postgresql")
    except QueryValidationError:
        return
    up = sql.upper()
    assert not any(w in up for w in ("DROP", "DELETE", "INSERT"))


@pytest.mark.unit
@pytest.mark.parametrize(
    "reply",
    [
        "SELECT 'Revenue' AS m, SUM(amount) FROM revenue_by_segment WHERE period = 'FY20",
        "To tie out the draft 10-K, compare revenue and opex with the trial balance.",
        "",
    ],
)
def test_truncated_or_prose_replies_raise_validation_error(reply):
    with pytest.raises(QueryValidationError):
        validate_and_sanitize_sql(extract_sql(reply, "sqlite"), "sqlite")


@pytest.mark.unit
def test_azure_sql_dialect_is_understood():
    assert "TOP 500" in validate_and_sanitize_sql("SELECT a FROM t", "azure_sql")


# --------------------------- corrective retry ---------------------------------

BAD = "To tie out the 10-K I compared revenue 'FY2025' against the ledger."


@pytest.mark.unit
@pytest.mark.asyncio
class TestCorrectiveRetry:
    async def test_retry_once_with_feedback_and_logs_prefix(self, caplog):
        from api import db_service

        gen = AsyncMock(side_effect=[BAD, "SELECT 1"])
        lin = MagicMock()
        with patch.object(db_service, "_generate_sql", gen), caplog.at_level(logging.WARNING):
            sql, retried = await db_service._generate_validated_sql(
                "q", [], [], "sqlite", lin, allow_retry=True
            )
        assert retried and sql.startswith("SELECT 1")
        assert gen.await_count == 2
        feedback = gen.await_args_list[1].kwargs["feedback"]
        assert BAD[:300] in feedback
        assert "Return exactly one SELECT statement (UNION ALL allowed) and nothing else." in feedback
        rec = next(r for r in caplog.records if "SQL validation failed" in r.getMessage())
        msg = rec.getMessage()
        assert "QueryValidationError" in msg and "dialect=sqlite" in msg
        assert f"reply_len={len(BAD)}" in msg and "'FY2025'" in msg
        detail = lin.add_step.call_args.kwargs["detail"]
        assert detail.startswith("sql_gen_error: ") and "FY2025" not in detail

    async def test_second_failure_raises_after_exactly_two_calls(self):
        from api import db_service

        gen = AsyncMock(side_effect=[BAD, BAD])
        with patch.object(db_service, "_generate_sql", gen), pytest.raises(QueryValidationError):
            await db_service._generate_validated_sql("q", [], [], "sqlite", MagicMock(), True)
        assert gen.await_count == 2

    async def test_no_retry_when_not_allowed(self):
        from api import db_service

        gen = AsyncMock(side_effect=[BAD])
        with patch.object(db_service, "_generate_sql", gen), pytest.raises(QueryValidationError):
            await db_service._generate_validated_sql("q", [], [], "sqlite", MagicMock(), False)
        assert gen.await_count == 1

    async def test_retry_bypasses_llm_cache(self):
        from api import db_service
        from api.utils import llm_cache

        get = AsyncMock(return_value=None)
        put = AsyncMock()
        with (
            patch.object(llm_cache, "get_cached_response", get),
            patch.object(llm_cache, "set_cached_response", put),
            patch("api.utils.llm_provider.chat_completion", return_value="SELECT 1") as cc,
        ):
            out = await db_service._generate_sql("q", [], [], feedback="rejected")
        assert out == "SELECT 1"
        get.assert_not_awaited()
        put.assert_not_awaited()
        assert "rejected" in cc.call_args.args[0][0]["content"]
        assert cc.call_args.kwargs["caller"] == "sql_gen_retry"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_drift_retry_does_not_stack_on_corrective_retry():
    """Corrective retry used the one extra SQL-gen call -> a later schema-drift
    error is raised instead of regenerating a third time."""
    from api import db_service
    from tests.unit.test_db_service_helpers import (
        _FakeConnRow,
        _fake_db_connections_table,
        _make_conn_lookup_session_factory,
    )

    gen = AsyncMock(side_effect=[BAD, "SELECT x FROM orders", "SELECT 1"])
    schema = [{"name": "orders", "columns": [{"name": "id", "type": "INTEGER"}]}]
    with (
        patch.object(db_service, "get_schema", AsyncMock(return_value=schema)),
        patch.object(db_service, "_select_relevant_tables", AsyncMock(return_value=["orders"])),
        patch.object(db_service, "_retrieve_few_shot", AsyncMock(return_value=[])),
        patch.object(db_service, "_generate_sql", gen),
        patch.object(db_service, "_execute_query",
                     AsyncMock(side_effect=Exception("no such column: x"))),
        patch.object(db_service, "_get_embeddings_model", return_value=MagicMock()),
        patch.object(db_service, "_get_embedding", AsyncMock(return_value=[0.1])),
        patch.object(db_service, "decrypt_credentials", return_value={}),
        patch.object(db_service, "_resolve_connection", return_value=("sqlite:///:memory:", None)),
    ):
        pipeline = db_service.run_sql_pipeline(
            connection_id="conn-1",
            question="Tie out the draft 10-K against the trial balance",
            persona="Data Analyst",
            db_connections_table=_fake_db_connections_table(),
            schema_cache_table=MagicMock(),
            query_history_table=MagicMock(),
            query_embeddings_table=MagicMock(),
            async_session_factory=_make_conn_lookup_session_factory(_FakeConnRow()),
            expert_personas={"Data Analyst": {"persona_def": "x"}},
            table_embeddings_table=None,
        )
        try:
            _ = [c async for c in pipeline]
        except Exception:  # noqa: BLE001 — the pipeline may surface the drift error
            pass
    assert gen.await_count == 2
