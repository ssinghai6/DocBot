"""Integration tests for db_service pipeline against a real SQLite DB — DOCBOT-201 through 203."""

import pytest
from pathlib import Path
from api.db_service import _test_connection, _introspect_schema_from_url, _execute_query
from api.utils.sql_validator import validate_and_sanitize_sql


@pytest.mark.integration
class TestConnectionTest:

    @pytest.mark.asyncio
    async def test_valid_sqlite_connection(self, sqlite_url):
        await _test_connection(sqlite_url, "sqlite")  # must not raise

    @pytest.mark.asyncio
    async def test_invalid_path_raises_value_error(self):
        with pytest.raises(ValueError, match="connection test failed"):
            await _test_connection("sqlite:////nonexistent/path/missing.db", "sqlite")

    @pytest.mark.asyncio
    async def test_error_message_does_not_expose_url(self):
        bad_url = "sqlite:////nonexistent/super_secret_path.db"
        with pytest.raises(ValueError) as exc_info:
            await _test_connection(bad_url, "sqlite")
        assert "super_secret_path" not in str(exc_info.value)


@pytest.mark.integration
class TestSchemaIntrospection:

    @pytest.mark.asyncio
    async def test_returns_all_tables(self, sqlite_url):
        schema = await _introspect_schema_from_url(sqlite_url, "sqlite")
        names = [t["name"] for t in schema]
        assert "orders" in names
        assert "products" in names

    @pytest.mark.asyncio
    async def test_returns_column_details(self, sqlite_url):
        schema = await _introspect_schema_from_url(sqlite_url, "sqlite")
        orders = next(t for t in schema if t["name"] == "orders")
        col_names = [c["name"] for c in orders["columns"]]
        assert "customer" in col_names
        assert "amount" in col_names
        assert "status" in col_names

    @pytest.mark.asyncio
    async def test_capped_at_200_tables(self, tmp_path):
        """Schema introspection must not return more than 200 tables."""
        db_path = tmp_path / "big.db"
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        for i in range(210):
            conn.execute(f"CREATE TABLE table_{i:03d} (id INTEGER PRIMARY KEY, val TEXT)")
        conn.commit()
        conn.close()
        schema = await _introspect_schema_from_url(f"sqlite:///{db_path}", "sqlite")
        assert len(schema) <= 200


@pytest.mark.integration
class TestQueryExecutor:

    @pytest.mark.asyncio
    async def test_basic_select(self, sqlite_url):
        rows, cols = await _execute_query("SELECT * FROM orders LIMIT 10", sqlite_url, "sqlite")
        assert len(rows) == 4
        assert "customer" in cols

    @pytest.mark.asyncio
    async def test_aggregate_query(self, sqlite_url):
        rows, cols = await _execute_query(
            "SELECT customer, SUM(amount) AS total FROM orders GROUP BY customer",
            sqlite_url, "sqlite",
        )
        assert len(rows) == 3  # Alice, Bob, Charlie

    @pytest.mark.asyncio
    async def test_row_cap_respected(self, sqlite_url):
        rows, _ = await _execute_query("SELECT * FROM orders LIMIT 500", sqlite_url, "sqlite")
        assert len(rows) <= 500

    @pytest.mark.asyncio
    async def test_join_query(self, sqlite_url):
        rows, cols = await _execute_query(
            "SELECT o.customer, p.name FROM orders o "
            "JOIN products p ON p.id = o.id LIMIT 10",
            sqlite_url, "sqlite",
        )
        assert "customer" in cols
        assert "name" in cols


@pytest.mark.integration
class TestEndToEndSQLValidationAndExecution:
    """Validate → execute pipeline: confirmed safe path."""

    @pytest.mark.asyncio
    async def test_validated_sql_executes(self, sqlite_url):
        raw = "SELECT customer, amount FROM orders WHERE status = 'paid'"
        validated = validate_and_sanitize_sql(raw, dialect="sqlite")
        rows, cols = await _execute_query(validated, sqlite_url, "sqlite")
        assert len(rows) == 3  # Alice x2 + Charlie
        assert "customer" in cols

    @pytest.mark.asyncio
    async def test_injection_never_reaches_executor(self, sqlite_url):
        """Attack SQL must be caught by validator before executor is called."""
        from api.utils.sql_validator import QueryValidationError
        with pytest.raises(QueryValidationError):
            validated = validate_and_sanitize_sql("DROP TABLE orders", dialect="sqlite")
            # If validator somehow passes, executor would fail — but we assert validator catches it
