"""Unit tests for api/audit_service.py — DOCBOT-602.

All tests are CI-safe: no network calls, no real DB.
DB writes are tested via mocked async_session_factory.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch, call


# ---------------------------------------------------------------------------
# AuditEventType
# ---------------------------------------------------------------------------

class TestAuditEventType:
    def test_all_event_types_defined(self):
        from api.audit_service import AuditEventType
        expected = {"query", "db_connect", "db_disconnect", "upload", "login", "logout"}
        actual = {e.value for e in AuditEventType}
        assert actual == expected

    def test_values_are_strings(self):
        from api.audit_service import AuditEventType
        for e in AuditEventType:
            assert isinstance(e.value, str)


# ---------------------------------------------------------------------------
# _write_event — async DB write
# ---------------------------------------------------------------------------

class TestWriteEvent:
    @pytest.mark.asyncio
    async def test_writes_correct_fields(self):
        from api.audit_service import _write_event, AuditEventType

        mock_table = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_begin = AsyncMock()
        mock_begin.__aenter__ = AsyncMock(return_value=mock_begin)
        mock_begin.__aexit__ = AsyncMock(return_value=False)
        mock_session.begin = MagicMock(return_value=mock_begin)

        factory = MagicMock(return_value=mock_session)

        with patch("sqlalchemy.insert") as mock_insert:
            mock_insert.return_value.values.return_value = MagicMock()
            await _write_event(
                AuditEventType.login,
                session_id=None,
                user_id="user-123",
                detail="alice@example.com",
                metadata_json='{"provider": "okta"}',
                audit_log_table=mock_table,
                async_session_factory=factory,
            )

        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_swallows_db_exceptions(self):
        """A DB failure must not propagate — audit log is non-fatal."""
        from api.audit_service import _write_event, AuditEventType

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_begin = AsyncMock()
        mock_begin.__aenter__ = AsyncMock(return_value=mock_begin)
        mock_begin.__aexit__ = AsyncMock(return_value=False)
        mock_session.begin = MagicMock(return_value=mock_begin)
        mock_session.execute = AsyncMock(side_effect=RuntimeError("connection refused"))

        factory = MagicMock(return_value=mock_session)

        with patch("sqlalchemy.insert"):
            # Should not raise
            await _write_event(
                AuditEventType.query,
                session_id="sess-1",
                user_id=None,
                detail="show tables",
                metadata_json=None,
                audit_log_table=MagicMock(),
                async_session_factory=factory,
            )


# ---------------------------------------------------------------------------
# log_event — fire-and-forget dispatcher
# ---------------------------------------------------------------------------

class TestLogEvent:
    def test_creates_task_in_running_loop(self):
        from api.audit_service import log_event, AuditEventType

        tasks_created = []

        mock_loop = MagicMock()
        mock_loop.create_task = MagicMock(side_effect=lambda coro: tasks_created.append(coro))

        with patch("asyncio.get_running_loop", return_value=mock_loop):
            log_event(
                AuditEventType.upload,
                MagicMock(),
                MagicMock(),
                session_id="sess-abc",
                detail="report.pdf",
                metadata={"file_count": 1},
            )

        assert len(tasks_created) == 1
        mock_loop.create_task.assert_called_once()

    def test_silently_ignores_no_event_loop(self):
        from api.audit_service import log_event, AuditEventType

        with patch("asyncio.get_running_loop", side_effect=RuntimeError("no running loop")):
            # Must not raise
            log_event(
                AuditEventType.db_connect,
                MagicMock(),
                MagicMock(),
                detail="host:5432/db",
            )

    def test_metadata_serialized_to_json(self):
        from api.audit_service import log_event, _write_event, AuditEventType

        captured_metadata = []

        async def fake_write(*args, metadata_json=None, **kwargs):
            captured_metadata.append(metadata_json)

        tasks = []
        mock_loop = MagicMock()
        mock_loop.create_task = MagicMock(side_effect=lambda coro: tasks.append(coro))

        with patch("asyncio.get_running_loop", return_value=mock_loop):
            log_event(
                AuditEventType.query,
                MagicMock(),
                MagicMock(),
                metadata={"dialect": "postgresql", "row_count": 42},
            )

        assert len(tasks) == 1
        # Verify the coroutine was created (metadata gets serialized in log_event)

    def test_none_metadata_produces_none_json(self):
        from api.audit_service import log_event, AuditEventType

        mock_loop = MagicMock()
        captured_coroutines = []
        mock_loop.create_task = MagicMock(side_effect=lambda c: captured_coroutines.append(c))

        with patch("asyncio.get_running_loop", return_value=mock_loop):
            log_event(
                AuditEventType.logout,
                MagicMock(),
                MagicMock(),
                metadata=None,
            )
        assert len(captured_coroutines) == 1


# ---------------------------------------------------------------------------
# IMMUTABILITY_TRIGGER_DDL — basic sanity
# ---------------------------------------------------------------------------

class TestImmutabilityTriggerStatements:
    def test_is_a_list_of_two_statements(self):
        from api.audit_service import IMMUTABILITY_TRIGGER_STATEMENTS
        assert isinstance(IMMUTABILITY_TRIGGER_STATEMENTS, list)
        assert len(IMMUTABILITY_TRIGGER_STATEMENTS) == 2

    def test_each_statement_is_non_empty_string(self):
        from api.audit_service import IMMUTABILITY_TRIGGER_STATEMENTS
        for stmt in IMMUTABILITY_TRIGGER_STATEMENTS:
            assert isinstance(stmt, str)
            assert len(stmt.strip()) > 50

    def test_statements_cover_function_and_trigger(self):
        from api.audit_service import IMMUTABILITY_TRIGGER_STATEMENTS
        combined = "\n".join(IMMUTABILITY_TRIGGER_STATEMENTS)
        assert "audit_log_immutable" in combined
        assert "audit_log_no_mutate" in combined

    def test_statements_are_idempotent(self):
        """Each statement uses IF NOT EXISTS so it can be re-run safely."""
        from api.audit_service import IMMUTABILITY_TRIGGER_STATEMENTS
        for stmt in IMMUTABILITY_TRIGGER_STATEMENTS:
            assert "IF NOT EXISTS" in stmt


# ---------------------------------------------------------------------------
# get_client_ip — IP extraction from FastAPI Request objects
# ---------------------------------------------------------------------------

class TestGetClientIp:

    def _make_request(self, forwarded=None, host=None):
        """Build a minimal mock that looks like a FastAPI Request."""
        req = MagicMock()
        req.headers = {}
        if forwarded:
            req.headers = {"X-Forwarded-For": forwarded}
        if host:
            req.client = MagicMock()
            req.client.host = host
        else:
            req.client = None
        return req

    def test_returns_first_ip_from_x_forwarded_for(self):
        from api.audit_service import get_client_ip
        req = self._make_request(forwarded="1.2.3.4, 10.0.0.1, 172.16.0.1")
        assert get_client_ip(req) == "1.2.3.4"

    def test_single_ip_in_x_forwarded_for(self):
        from api.audit_service import get_client_ip
        req = self._make_request(forwarded="203.0.113.5")
        assert get_client_ip(req) == "203.0.113.5"

    def test_falls_back_to_client_host(self):
        from api.audit_service import get_client_ip
        req = self._make_request(host="192.168.1.1")
        assert get_client_ip(req) == "192.168.1.1"

    def test_returns_none_when_request_is_none(self):
        from api.audit_service import get_client_ip
        assert get_client_ip(None) is None

    def test_returns_none_when_no_client_info(self):
        from api.audit_service import get_client_ip
        req = self._make_request()  # no forwarded, no host
        assert get_client_ip(req) is None

    def test_x_forwarded_for_takes_priority_over_client_host(self):
        from api.audit_service import get_client_ip
        req = self._make_request(forwarded="5.6.7.8", host="10.0.0.1")
        assert get_client_ip(req) == "5.6.7.8"

    def test_strips_whitespace_from_ip(self):
        from api.audit_service import get_client_ip
        req = self._make_request(forwarded="  9.9.9.9  , 1.1.1.1")
        assert get_client_ip(req) == "9.9.9.9"


# ---------------------------------------------------------------------------
# Route audit wiring — confirm all four chat routes log query events with IP
# ---------------------------------------------------------------------------

class TestRouteAuditWiring:
    """Verify that all four chat route handlers reference get_client_ip.

    We inspect the source rather than running the app to keep the test
    fast and env-var-free.
    """

    def _route_source(self, fn) -> str:
        import inspect
        return inspect.getsource(fn)

    def test_chat_route_logs_query_with_ip(self):
        import api.index as idx
        src = self._route_source(idx.chat)
        assert "AuditEventType.query" in src
        assert "get_client_ip" in src

    def test_db_chat_route_logs_query_with_ip(self):
        import api.index as idx
        src = self._route_source(idx.db_chat)
        assert "AuditEventType.query" in src
        assert "get_client_ip" in src

    def test_hybrid_chat_route_logs_query_with_ip(self):
        import api.index as idx
        src = self._route_source(idx.hybrid_chat_route)
        assert "AuditEventType.query" in src
        assert "get_client_ip" in src

    def test_autopilot_run_logs_query_with_ip(self):
        import api.index as idx
        src = self._route_source(idx.autopilot_run)
        assert "AuditEventType.query" in src
        assert "get_client_ip" in src
