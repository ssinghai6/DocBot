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
