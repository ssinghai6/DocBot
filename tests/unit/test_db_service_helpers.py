"""Unit tests for db_service helper functions — DOCBOT-201 through 205."""

import pytest
from api.db_service import (
    DBConnectionRequest,
    DBChatRequest,
    _build_connection_url,
    _build_explanation,
    SUPPORTED_DIALECTS,
)
from pydantic import ValidationError


@pytest.mark.unit
class TestDBConnectionRequest:

    def test_valid_postgresql_request(self):
        req = DBConnectionRequest(
            session_id="sess-1", dialect="postgresql",
            host="8.8.8.8", port=5432,
            dbname="mydb", user="user", password="pass",
        )
        assert req.dialect == "postgresql"

    def test_valid_mysql_request(self):
        req = DBConnectionRequest(
            session_id="sess-1", dialect="mysql",
            host="8.8.8.8", port=3306,
            dbname="db", user="u", password="p",
        )
        assert req.dialect == "mysql"

    def test_valid_sqlite_request(self):
        req = DBConnectionRequest(
            session_id="s", dialect="sqlite",
            host="8.8.8.8", port=0,
            dbname="/tmp/test.db", user="", password="",
        )
        assert req.dialect == "sqlite"

    def test_dialect_is_lowercased(self):
        req = DBConnectionRequest(
            session_id="s", dialect="PostgreSQL",
            host="8.8.8.8", port=5432,
            dbname="d", user="u", password="p",
        )
        assert req.dialect == "postgresql"

    @pytest.mark.parametrize("dialect", ["oracle", "mssql", "cassandra", "mongo"])
    def test_unsupported_dialect_rejected(self, dialect):
        with pytest.raises(ValidationError) as exc_info:
            DBConnectionRequest(
                session_id="s", dialect=dialect,
                host="8.8.8.8", port=1234,
                dbname="d", user="u", password="p",
            )
        assert "Unsupported dialect" in str(exc_info.value)

    @pytest.mark.parametrize("private_host", [
        "192.168.1.1",
        "10.0.0.1",
        "172.16.5.5",
        "127.0.0.1",
        "169.254.1.1",
    ])
    def test_private_ip_rejected_by_ssrf_validator(self, private_host):
        with pytest.raises(ValidationError):
            DBConnectionRequest(
                session_id="s", dialect="postgresql",
                host=private_host, port=5432,
                dbname="d", user="u", password="p",
            )


@pytest.mark.unit
class TestDBChatRequest:

    def test_defaults(self):
        req = DBChatRequest(
            connection_id="conn-1",
            question="How many orders?",
            session_id="sess-1",
        )
        assert req.persona == "Generalist"

    def test_custom_persona(self):
        req = DBChatRequest(
            connection_id="c", question="q",
            persona="Finance Expert", session_id="s",
        )
        assert req.persona == "Finance Expert"


@pytest.mark.unit
class TestBuildConnectionURL:

    def test_postgresql_url(self):
        url = _build_connection_url("postgresql", "myhost.com", 5432, "mydb", "user", "pass")
        assert url == "postgresql+psycopg2://user:pass@myhost.com:5432/mydb?sslmode=prefer"

    def test_mysql_uses_pymysql_driver(self):
        url = _build_connection_url("mysql", "myhost.com", 3306, "mydb", "user", "pass")
        assert "pymysql" in url
        assert url == "mysql+pymysql://user:pass@myhost.com:3306/mydb"

    def test_sqlite_url(self):
        url = _build_connection_url("sqlite", "", 0, "/tmp/test.db", "", "")
        assert url == "sqlite:////tmp/test.db"


@pytest.mark.unit
class TestBuildExplanation:

    def _schema(self, *names):
        return [{"name": n, "columns": []} for n in names]

    def test_includes_table_names(self):
        expl = _build_explanation("SELECT * FROM orders LIMIT 10", self._schema("orders"))
        assert "orders" in expl

    def test_detects_group_by(self):
        expl = _build_explanation(
            "SELECT month, SUM(amount) FROM orders GROUP BY month",
            self._schema("orders"),
        )
        assert "groups results" in expl

    def test_detects_join(self):
        expl = _build_explanation(
            "SELECT o.id FROM orders o JOIN products p ON o.pid = p.id",
            self._schema("orders", "products"),
        )
        assert "joins multiple tables" in expl

    def test_detects_where(self):
        expl = _build_explanation(
            "SELECT * FROM orders WHERE status = 'paid' LIMIT 10",
            self._schema("orders"),
        )
        assert "filters rows" in expl

    def test_detects_aggregates(self):
        expl = _build_explanation(
            "SELECT COUNT(*) FROM orders",
            self._schema("orders"),
        )
        assert "computes aggregates" in expl

    def test_detects_order_by(self):
        expl = _build_explanation(
            "SELECT * FROM orders ORDER BY created_at DESC LIMIT 10",
            self._schema("orders"),
        )
        assert "orders by a column" in expl

    def test_empty_schema_fallback(self):
        expl = _build_explanation("SELECT 1", [])
        assert "database" in expl
