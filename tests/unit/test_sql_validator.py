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
