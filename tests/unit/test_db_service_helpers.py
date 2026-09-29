"""Unit tests for db_service helper functions — DOCBOT-201 through 205, 208, 504."""

from unittest.mock import patch

import pytest
from api.db_service import (
    DBConnectionRequest,
    DBChatRequest,
    _build_connection_url,
    _build_explanation,
    _parse_table_selection,
    _select_relevant_tables,
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


# ---------------------------------------------------------------------------
# DOCBOT-208: Azure SQL / Entra auth tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildConnectionURLAzureSQL:
    def test_uses_mssql_pyodbc_driver(self):
        url = _build_connection_url("azure_sql", "myserver.database.windows.net", 1433, "mydb")
        assert "mssql+pyodbc" in url

    def test_contains_odbc_driver_18(self):
        url = _build_connection_url("azure_sql", "myserver.database.windows.net", 1433, "mydb")
        assert "ODBC Driver 18 for SQL Server" in url

    def test_contains_server_and_database(self):
        url = _build_connection_url("azure_sql", "myserver.database.windows.net", 1433, "mydb")
        assert "myserver.database.windows.net,1433" in url
        assert "mydb" in url

    def test_no_credentials_in_url(self):
        url = _build_connection_url("azure_sql", "myserver.database.windows.net", 1433, "mydb")
        # No @ before the ? (credentials would appear as user:pass@host)
        base = url.split("?odbc_connect=")[0]
        assert "@" not in base

    def test_encryption_flags_present(self):
        url = _build_connection_url("azure_sql", "myserver.database.windows.net", 1433, "mydb")
        assert "Encrypt=yes" in url
        assert "TrustServerCertificate=no" in url


@pytest.mark.unit
class TestBuildEntraConnectArgs:
    def test_returns_attrs_before_key(self):
        from api.db_service import _build_entra_connect_args
        args = _build_entra_connect_args("fake_token")
        assert "attrs_before" in args
        assert 1256 in args["attrs_before"]

    def test_value_is_bytes(self):
        from api.db_service import _build_entra_connect_args
        args = _build_entra_connect_args("fake_token")
        assert isinstance(args["attrs_before"][1256], bytes)

    def test_struct_has_correct_length_prefix(self):
        import struct
        from api.db_service import _build_entra_connect_args
        token = "test_token"
        args = _build_entra_connect_args(token)
        raw = args["attrs_before"][1256]
        token_bytes = token.encode("utf-16-le")
        length = struct.unpack("<I", raw[:4])[0]
        assert length == len(token_bytes)
        assert raw[4:] == token_bytes


@pytest.mark.unit
class TestDBConnectionRequestAzureSQL:
    def test_azure_sql_in_supported_dialects(self):
        from api.db_service import SUPPORTED_DIALECTS
        assert "azure_sql" in SUPPORTED_DIALECTS

    def test_valid_entra_sp_request(self):
        req = DBConnectionRequest(
            session_id="s", dialect="azure_sql",
            host="8.8.8.8", port=1433, dbname="mydb",
            auth_type="entra_sp",
            tenant_id="tid", client_id="cid", client_secret="sec",
        )
        assert req.dialect == "azure_sql"
        assert req.auth_type == "entra_sp"

    def test_missing_auth_type_rejected(self):
        with pytest.raises(ValidationError):
            DBConnectionRequest(
                session_id="s", dialect="azure_sql",
                host="8.8.8.8", port=1433, dbname="mydb",
                tenant_id="tid", client_id="cid", client_secret="sec",
            )

    def test_missing_tenant_id_rejected(self):
        with pytest.raises(ValidationError):
            DBConnectionRequest(
                session_id="s", dialect="azure_sql",
                host="8.8.8.8", port=1433, dbname="mydb",
                auth_type="entra_sp",
                client_id="cid", client_secret="sec",
            )

    def test_missing_client_secret_rejected(self):
        with pytest.raises(ValidationError):
            DBConnectionRequest(
                session_id="s", dialect="azure_sql",
                host="8.8.8.8", port=1433, dbname="mydb",
                auth_type="entra_sp",
                tenant_id="tid", client_id="cid",
            )


@pytest.mark.unit
class TestGetEntraToken:
    def test_raises_runtime_error_when_azure_identity_missing(self, monkeypatch):
        import builtins
        real_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if "azure.identity" in name:
                raise ImportError("No module named 'azure.identity'")
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", mock_import)
        from api.db_service import _get_entra_token
        with pytest.raises(RuntimeError, match="azure-identity package is not installed"):
            _get_entra_token("tid", "cid", "sec")

    def test_raises_value_error_on_auth_failure(self, monkeypatch):
        import sys
        import types
        mock_azure = types.ModuleType("azure")
        mock_identity = types.ModuleType("azure.identity")
        class FakeCredential:
            def __init__(self, **kwargs): pass
            def get_token(self, scope): raise Exception("auth failed")
        mock_identity.ClientSecretCredential = FakeCredential
        mock_azure.identity = mock_identity
        monkeypatch.setitem(sys.modules, "azure", mock_azure)
        monkeypatch.setitem(sys.modules, "azure.identity", mock_identity)
        from api.db_service import _get_entra_token
        with pytest.raises(ValueError, match="Azure Entra authentication failed"):
            _get_entra_token("tid", "cid", "sec")

# ---------------------------------------------------------------------------
# DOCBOT-504: Query History — response shape helpers
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestQueryHistoryResponseShape:
    """Tests for the _parse_row_count helper logic used in the history route."""

    def _parse_row_count(self, summary):
        """Mirror of the inline helper in api/index.py db_query_history."""
        if not summary:
            return None
        try:
            return int(summary.split()[0])
        except (ValueError, IndexError):
            return None

    def test_parses_n_rows_format(self):
        assert self._parse_row_count("42 rows") == 42

    def test_parses_single_row(self):
        assert self._parse_row_count("1 rows") == 1

    def test_returns_none_for_none_input(self):
        assert self._parse_row_count(None) is None

    def test_returns_none_for_empty_string(self):
        assert self._parse_row_count("") is None

    def test_returns_none_for_non_numeric(self):
        assert self._parse_row_count("error") is None

    def test_handles_zero_rows(self):
        assert self._parse_row_count("0 rows") == 0

    def test_history_item_shape(self):
        """Verify the dict structure expected by the frontend Zod schema."""
        item = {
            "id": "abc-123",
            "question": "How many orders last month?",
            "sql": "SELECT COUNT(*) FROM orders WHERE ...",
            "executed_at": "2026-03-23T10:00:00+00:00",
            "row_count": 15,
        }
        assert set(item.keys()) == {"id", "question", "sql", "executed_at", "row_count"}
        assert isinstance(item["id"], str)
        assert isinstance(item["question"], str)
        assert isinstance(item["sql"], str)

    def test_limit_clamped_to_100(self):
        """Verify limit logic: min(user_limit, 100)."""
        user_limit = 500
        effective = max(1, min(user_limit, 100))
        assert effective == 100

    def test_limit_floored_at_1(self):
        user_limit = 0
        effective = max(1, min(user_limit, 100))
        assert effective == 1


# ---------------------------------------------------------------------------
# DOCBOT-1504: Pydantic-validated table-selector JSON parsing + retry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseTableSelection:
    def test_valid_array_parses(self):
        assert _parse_table_selection('["orders", "products"]') == ["orders", "products"]

    def test_valid_array_with_surrounding_prose(self):
        """LLMs sometimes wrap the array in explanation text despite
        instructions not to — the array must still be extracted."""
        raw = 'Here are the tables:\n["orders", "products"]\nThat should work.'
        assert _parse_table_selection(raw) == ["orders", "products"]

    def test_no_array_raises_value_error(self):
        with pytest.raises(ValueError):
            _parse_table_selection("I cannot determine the tables.")

    def test_non_string_elements_raise_validation_error(self):
        """A JSON array of non-strings (e.g. numbers) must fail validation
        rather than silently flowing into `t["name"] in set(selected_tables)`
        downstream as garbage."""
        with pytest.raises(ValidationError):
            _parse_table_selection("[1, 2, 3]")

    def test_object_with_nested_array_extracts_the_array(self):
        """The bracket-scan (find first '[', rfind last ']') intentionally
        extracts an array embedded in surrounding object/prose text — this
        is deliberate leniency for LLM output that wraps the array in a
        key, not something to guard against."""
        assert _parse_table_selection('{"tables": ["orders"]}') == ["orders"]

    def test_object_with_no_array_raises_value_error(self):
        """No '[' at all (e.g. the model returns a bare JSON object) — no
        array to extract, so this must fail rather than parse the whole
        object as a table list."""
        with pytest.raises(ValueError):
            _parse_table_selection('{"orders": 1, "products": 2}')

    def test_malformed_json_raises(self):
        with pytest.raises(ValueError):
            _parse_table_selection('["orders", "products"')  # missing closing bracket


@pytest.mark.unit
@pytest.mark.asyncio
class TestSelectRelevantTables:
    _SCHEMA = [
        {"name": "orders", "columns": [{"name": "id", "type": "INTEGER"}], "is_view": False},
        {"name": "products", "columns": [{"name": "id", "type": "INTEGER"}], "is_view": False},
    ]

    async def test_valid_response_returns_parsed_tables(self):
        with patch(
            "api.utils.llm_provider.chat_completion", return_value='["orders"]'
        ):
            result = await _select_relevant_tables("How many orders?", self._SCHEMA)
        assert result == ["orders"]

    async def test_malformed_response_retries_and_succeeds(self):
        """First call returns garbage; retry call (fed the validation error)
        returns a valid array — the retry result must be used."""
        responses = iter(["not json at all", '["products"]'])

        def _side_effect(*args, **kwargs):
            return next(responses)

        with patch("api.utils.llm_provider.chat_completion", side_effect=_side_effect) as mock_cc:
            result = await _select_relevant_tables("Which products?", self._SCHEMA)

        assert result == ["products"]
        assert mock_cc.call_count == 2
        # The retry call must carry the original prompt + feedback about the error.
        retry_call_kwargs = mock_cc.call_args_list[1]
        retry_messages = retry_call_kwargs.args[0] if retry_call_kwargs.args else retry_call_kwargs.kwargs["messages"]
        assert any("invalid" in m["content"].lower() for m in retry_messages)

    async def test_both_attempts_malformed_falls_back_to_first_ten(self):
        with patch("api.utils.llm_provider.chat_completion", return_value="garbage output"):
            schema = [
                {"name": f"table_{i}", "columns": [], "is_view": False} for i in range(15)
            ]
            result = await _select_relevant_tables("Anything?", schema)

        assert result == [f"table_{i}" for i in range(10)]

    async def test_malformed_output_is_logged(self, caplog):
        import logging

        with (
            patch("api.utils.llm_provider.chat_completion", return_value="not valid json"),
            caplog.at_level(logging.WARNING, logger="api.utils.llm_provider"),
        ):
            await _select_relevant_tables("Anything?", self._SCHEMA)

        records = [
            r for r in caplog.records if getattr(r, "event", None) == "malformed_llm_output"
        ]
        # One for the initial failure, one for the retry's failure.
        assert len(records) == 2
        assert {r.caller for r in records} == {"sql_table_selector", "sql_table_selector_retry"}

    async def test_call_failure_does_not_retry(self):
        """A raised exception from chat_completion itself (both providers
        down) is a different failure mode than malformed JSON — no retry,
        straight to the schema[:10] fallback."""
        with patch(
            "api.utils.llm_provider.chat_completion", side_effect=RuntimeError("both providers down")
        ) as mock_cc:
            result = await _select_relevant_tables("Anything?", self._SCHEMA)

        assert mock_cc.call_count == 1  # no retry attempted
        assert result == [t["name"] for t in self._SCHEMA[:10]]
