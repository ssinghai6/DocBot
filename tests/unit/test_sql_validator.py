"""Unit tests for sqlglot AST SQL validator — DOCBOT-203."""

import pytest
from api.utils.sql_validator import validate_and_sanitize_sql, QueryValidationError


@pytest.mark.unit
class TestSQLValidator:

    # ── Valid SELECT queries ─────────────────────────────────────────────

    def test_simple_select_passes(self):
        sql = validate_and_sanitize_sql("SELECT id, name FROM users")
        assert "SELECT" in sql.upper()

    def test_limit_injected_when_missing(self):
        sql = validate_and_sanitize_sql("SELECT * FROM orders")
        assert "LIMIT" in sql.upper()
        assert "500" in sql

    def test_existing_limit_preserved(self):
        sql = validate_and_sanitize_sql("SELECT * FROM orders LIMIT 10")
        assert "10" in sql
        assert sql.upper().count("LIMIT") == 1

    def test_aggregate_select_passes(self):
        sql = validate_and_sanitize_sql(
            "SELECT customer, SUM(amount) FROM orders GROUP BY customer"
        )
        assert "SUM" in sql.upper()

    def test_join_select_passes(self):
        sql = validate_and_sanitize_sql(
            "SELECT o.id, p.name FROM orders o JOIN products p ON o.product_id = p.id"
        )
        assert "JOIN" in sql.upper()

    # ── Write operations must be rejected ───────────────────────────────

    @pytest.mark.parametrize("bad_sql,label", [
        ("DROP TABLE users", "DROP TABLE"),
        ("DELETE FROM orders WHERE 1=1", "DELETE"),
        ("INSERT INTO users VALUES (1,'pwned')", "INSERT"),
        ("UPDATE users SET name='x' WHERE 1=1", "UPDATE"),
        ("CREATE TABLE evil (id INT)", "CREATE TABLE"),
        ("TRUNCATE TABLE orders", "TRUNCATE"),
        ("ALTER TABLE users ADD COLUMN x TEXT", "ALTER TABLE"),
    ])
    def test_write_operations_rejected(self, bad_sql, label):
        with pytest.raises(QueryValidationError):
            validate_and_sanitize_sql(bad_sql)

    def test_error_message_does_not_leak_sql(self):
        """The error message must never contain the invalid SQL."""
        bad_sql = "DROP TABLE secret_table_name_xyz"
        with pytest.raises(QueryValidationError) as exc_info:
            validate_and_sanitize_sql(bad_sql)
        assert "secret_table_name_xyz" not in str(exc_info.value)

    # ── Dialect handling ─────────────────────────────────────────────────

    def test_mysql_dialect_accepted(self):
        sql = validate_and_sanitize_sql("SELECT * FROM orders", dialect="mysql")
        assert sql

    def test_sqlite_dialect_accepted(self):
        sql = validate_and_sanitize_sql("SELECT id FROM users", dialect="sqlite")
        assert sql

    # ── Edge cases ───────────────────────────────────────────────────────

    def test_empty_string_raises(self):
        with pytest.raises(QueryValidationError):
            validate_and_sanitize_sql("")

    def test_multiple_statements_rejected(self):
        with pytest.raises(QueryValidationError):
            validate_and_sanitize_sql("SELECT 1; DROP TABLE users")


# --- Tie-out comparisons: UNION read queries are allowed, writes never are ---

def test_union_all_select_is_allowed_and_limited():
    from api.utils.sql_validator import validate_and_sanitize_sql

    out = validate_and_sanitize_sql(
        "SELECT 'revenue' AS metric, SUM(credit) AS v FROM tb "
        "UNION ALL SELECT 'opex', SUM(debit) FROM tb",
        "sqlite",
    )
    assert "UNION ALL" in out and "LIMIT 500" in out


def test_cte_with_union_is_allowed():
    from api.utils.sql_validator import validate_and_sanitize_sql

    out = validate_and_sanitize_sql(
        "WITH a AS (SELECT 1 AS x) SELECT x FROM a UNION ALL SELECT 2", "postgresql"
    )
    assert out.upper().startswith("WITH")





@pytest.mark.parametrize(
    "sql",
    [
        "WITH d AS (DELETE FROM t RETURNING *) SELECT * FROM d",
        "WITH u AS (UPDATE t SET a = 1 RETURNING *) SELECT * FROM u",
        "INSERT INTO t SELECT 1 UNION SELECT 2",
    ],
)
def test_write_nodes_anywhere_are_rejected(sql):
    from api.utils.sql_validator import QueryValidationError, validate_and_sanitize_sql

    with pytest.raises(QueryValidationError):
        validate_and_sanitize_sql(sql, "postgresql")


@pytest.mark.parametrize(
    "sql,dialect",
    [
        ("SELECT * INTO evil FROM t", "postgresql"),
        ("SELECT * INTO evil FROM t", "sqlite"),
        ("SELECT a INTO evil FROM t UNION ALL SELECT b FROM u", "postgresql"),
        ("SELECT * FROM t INTO OUTFILE '/tmp/x'", "mysql"),
        ("SELECT * INTO OUTFILE '/tmp/x' FROM t", "mysql"),
    ],
)
def test_select_into_is_rejected(sql, dialect):
    """SELECT ... INTO creates a table (Postgres) or writes a file (MySQL)."""
    from api.utils.sql_validator import QueryValidationError, validate_and_sanitize_sql

    with pytest.raises(QueryValidationError):
        validate_and_sanitize_sql(sql, dialect)


@pytest.mark.parametrize(
    "reply,expected_start",
    [
        ("SELECT 1 LIMIT 5", "SELECT 1"),
        ("```sql\nSELECT a FROM t\n```", "SELECT a FROM t"),
        ("Here is the query:\n```sql\nSELECT a FROM t;\n```\nIt sums revenue.", "SELECT a FROM t"),
        ("To compare the books:\nWITH x AS (SELECT 1 AS v) SELECT v FROM x;\nThis returns v.", "WITH x AS"),
        ("SELECT 'rev' AS m, 1 AS v UNION ALL SELECT 'opex', 2;\n\nExplanation: ...", "SELECT 'rev'"),
    ],
)
def test_extract_sql_strips_prose_and_fences(reply, expected_start):
    from api.utils.sql_validator import extract_sql, validate_and_sanitize_sql

    sql = extract_sql(reply)
    assert sql.startswith(expected_start)
    validate_and_sanitize_sql(sql, "sqlite")  # parses and passes SELECT-only


def test_extract_sql_never_makes_prose_or_writes_valid():
    from api.utils.sql_validator import QueryValidationError, extract_sql, validate_and_sanitize_sql

    for reply in ("I cannot answer this from the schema.", "DELETE FROM t; SELECT 1"):
        with pytest.raises(QueryValidationError):
            validate_and_sanitize_sql(extract_sql(reply), "sqlite")


@pytest.mark.parametrize("dialect", ["mysql", "postgresql", "sqlite", None])
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1 /*! ; DROP TABLE t */",
        "SELECT * FROM t /*!50000 INTO OUTFILE '/tmp/x' */",
        "SELECT 1 /*M! ; DELETE FROM t */",
    ],
)
def test_mysql_executable_comments_rejected(sql, dialect):
    """MySQL runs /*! ... */ bodies; sqlglot sees a comment. Never accept."""
    from api.utils.sql_validator import QueryValidationError, validate_and_sanitize_sql

    with pytest.raises(QueryValidationError):
        validate_and_sanitize_sql(sql, dialect)


@pytest.mark.parametrize("dialect", ["mysql", "postgresql", "azure_sql"])
def test_non_sqlite_executes_regenerated_sql_not_raw_text(dialect):
    """Comments / trailing text in the raw reply never reach a network DB."""
    from api.utils.sql_validator import validate_and_sanitize_sql

    out = validate_and_sanitize_sql("SELECT a FROM t -- note: ; DROP TABLE t\n", dialect)
    assert "DROP" not in out.upper()
    assert "--" not in out


def test_sqlite_keeps_cte_column_list_verbatim():
    from api.utils.sql_validator import validate_and_sanitize_sql

    out = validate_and_sanitize_sql(
        "WITH doc(metric, amount) AS (VALUES ('rev', 1)) SELECT metric, amount FROM doc",
        "sqlite",
    )
    assert "doc(metric, amount)" in out and out.rstrip().endswith("LIMIT 500")
