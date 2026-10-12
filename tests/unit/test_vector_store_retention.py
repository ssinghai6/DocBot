"""Unit tests for vector store retention/deletion lifecycle — DOCBOT-1525."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import Column, DateTime, MetaData, String, Table

from api.utils.vector_store import cleanup_expired_vector_stores


@pytest.mark.unit
def _sessions_table() -> Table:
    metadata = MetaData()
    return Table(
        "sessions", metadata,
        Column("session_id", String, primary_key=True),
        Column("updated_at", DateTime(timezone=True)),
    )


class _FakeAsyncSession:
    """Minimal async context manager mimicking AsyncSession.execute()."""

    def __init__(self, rows):
        self._rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def execute(self, *_args, **_kwargs):
        result = MagicMock()
        result.fetchall.return_value = self._rows
        return result


def _factory_for(rows):
    """Returns a callable usable as `async_session_factory()`."""
    def _factory():
        return _FakeAsyncSession(rows)
    return _factory


def _row(session_id: str, updated_at: datetime):
    return SimpleNamespace(session_id=session_id, updated_at=updated_at)


@pytest.mark.unit
class TestCleanupExpiredVectorStores:

    @pytest.mark.asyncio
    async def test_identifies_expired_vs_active_and_deletes_only_expired(self):
        now = datetime.now(timezone.utc)
        expired_row = _row("expired-sess", now - timedelta(days=45))
        active_row = _row("active-sess", now - timedelta(days=2))

        with patch(
            "api.utils.vector_store.list_stored_sessions",
            return_value=["expired-sess", "active-sess"],
        ), patch("api.utils.vector_store.delete_store") as mock_delete:
            removed = await cleanup_expired_vector_stores(
                sessions_table=_sessions_table(),
                async_session_factory=_factory_for([expired_row, active_row]),
                retention_days=30,
            )

        assert removed == 1
        mock_delete.assert_called_once_with("expired-sess")

    @pytest.mark.asyncio
    async def test_nothing_deleted_when_nothing_expired(self):
        now = datetime.now(timezone.utc)
        rows = [
            _row("sess-a", now - timedelta(days=1)),
            _row("sess-b", now - timedelta(days=5)),
        ]

        with patch(
            "api.utils.vector_store.list_stored_sessions",
            return_value=["sess-a", "sess-b"],
        ), patch("api.utils.vector_store.delete_store") as mock_delete:
            removed = await cleanup_expired_vector_stores(
                sessions_table=_sessions_table(),
                async_session_factory=_factory_for(rows),
                retention_days=30,
            )

        assert removed == 0
        mock_delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_stored_sessions_short_circuits_without_db_call(self):
        factory = MagicMock(side_effect=AssertionError("should not query DB"))

        with patch("api.utils.vector_store.list_stored_sessions", return_value=[]):
            removed = await cleanup_expired_vector_stores(
                sessions_table=_sessions_table(),
                async_session_factory=factory,
                retention_days=30,
            )

        assert removed == 0
        factory.assert_not_called()

    @pytest.mark.asyncio
    async def test_failure_deleting_one_store_does_not_stop_the_rest(self):
        now = datetime.now(timezone.utc)
        rows = [
            _row("bad-sess", now - timedelta(days=40)),
            _row("good-sess", now - timedelta(days=40)),
        ]

        def _delete_side_effect(session_id):
            if session_id == "bad-sess":
                raise RuntimeError("disk error")
            return None

        with patch(
            "api.utils.vector_store.list_stored_sessions",
            return_value=["bad-sess", "good-sess"],
        ), patch(
            "api.utils.vector_store.delete_store", side_effect=_delete_side_effect
        ) as mock_delete:
            removed = await cleanup_expired_vector_stores(
                sessions_table=_sessions_table(),
                async_session_factory=_factory_for(rows),
                retention_days=30,
            )

        # Both deletions were attempted despite the first one raising.
        assert mock_delete.call_count == 2
        # Only the successful deletion counts toward the returned total.
        assert removed == 1

    @pytest.mark.asyncio
    async def test_orphaned_store_with_no_session_row_is_cleaned_up(self):
        """A store on disk whose session row was deleted directly should
        still be cleaned up rather than leaked forever."""
        with patch(
            "api.utils.vector_store.list_stored_sessions",
            return_value=["orphan-sess"],
        ), patch("api.utils.vector_store.delete_store") as mock_delete:
            removed = await cleanup_expired_vector_stores(
                sessions_table=_sessions_table(),
                async_session_factory=_factory_for([]),  # no matching session row
                retention_days=30,
            )

        assert removed == 1
        mock_delete.assert_called_once_with("orphan-sess")

    @pytest.mark.asyncio
    async def test_retention_days_zero_disables_cleanup(self):
        factory = MagicMock(side_effect=AssertionError("should not query DB"))

        with patch(
            "api.utils.vector_store.list_stored_sessions",
            return_value=["sess-a"],
        ):
            removed = await cleanup_expired_vector_stores(
                sessions_table=_sessions_table(),
                async_session_factory=factory,
                retention_days=0,
            )

        assert removed == 0
        factory.assert_not_called()
