"""Tests for api/connectors/sync_jobs.py — DOCBOT-704.

Verifies background sync scheduling, incremental sync logic,
rate-limit backoff, and auth failure handling.
"""

from __future__ import annotations

import asyncio
import types
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from apscheduler.schedulers.background import BackgroundScheduler

from api.connectors.base import ConnectorAuthError, ConnectorRateLimitError
from api.connectors.sync_jobs import (
    _job_id,
    deregister_sync_schedules,
    register_all_active_connectors,
    register_sync_schedules,
    sync_connector_incremental,
    wire_sync_scheduler,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class FakeConnector:
    connector_type = "amazon"

    def __init__(self):
        self.fetch_orders = AsyncMock(return_value=[
            {"marketplace_order_id": "ord-1", "status": "paid", "total_amount": 100.0},
        ])
        self.fetch_financials = AsyncMock(return_value=[
            {"period_start": "2024-01-01", "period_end": "2024-01-31", "revenue": 500.0,
             "refunds": 10.0, "fees": 20.0, "net_proceeds": 470.0},
        ])


@pytest.fixture
def scheduler():
    """Use BackgroundScheduler for tests — doesn't require a running event loop."""
    s = BackgroundScheduler()
    s.start()
    yield s
    if s.running:
        s.shutdown(wait=False)


@pytest.fixture
def app_state():
    state = types.SimpleNamespace()
    state.connectors = {}
    return state


@pytest.fixture
def wired_scheduler(scheduler, app_state):
    wire_sync_scheduler(scheduler, app_state)
    yield {"scheduler": scheduler, "app_state": app_state}
    wire_sync_scheduler(None, None)


# ---------------------------------------------------------------------------
# Schedule Registration
# ---------------------------------------------------------------------------

class TestRegisterSyncSchedules:
    def test_registers_orders_and_financials_jobs(self, wired_scheduler):
        register_sync_schedules("conn-1")
        sched = wired_scheduler["scheduler"]
        assert sched.get_job(_job_id("conn-1", "orders")) is not None
        assert sched.get_job(_job_id("conn-1", "financials")) is not None

    def test_idempotent_registration(self, wired_scheduler):
        register_sync_schedules("conn-2")
        register_sync_schedules("conn-2")
        sched = wired_scheduler["scheduler"]
        assert sched.get_job(_job_id("conn-2", "orders")) is not None

    def test_deregister_removes_jobs(self, wired_scheduler):
        register_sync_schedules("conn-3")
        deregister_sync_schedules("conn-3")
        sched = wired_scheduler["scheduler"]
        assert sched.get_job(_job_id("conn-3", "orders")) is None
        assert sched.get_job(_job_id("conn-3", "financials")) is None

    def test_deregister_nonexistent_is_noop(self, wired_scheduler):
        deregister_sync_schedules("does-not-exist")  # should not raise


class TestRegisterAllActive:
    def test_registers_for_all_connectors_in_memory(self, wired_scheduler):
        state = wired_scheduler["app_state"]
        state.connectors = {"c1": FakeConnector(), "c2": FakeConnector()}
        count = register_all_active_connectors()
        assert count == 2
        sched = wired_scheduler["scheduler"]
        assert sched.get_job(_job_id("c1", "orders")) is not None
        assert sched.get_job(_job_id("c2", "financials")) is not None

    def test_returns_zero_when_no_connectors(self, wired_scheduler):
        wired_scheduler["app_state"].connectors = {}
        assert register_all_active_connectors() == 0


# ---------------------------------------------------------------------------
# Incremental Sync
# ---------------------------------------------------------------------------

class TestSyncConnectorIncremental:
    @pytest.mark.asyncio
    @patch("api.connector_store.update_sync_status", new_callable=AsyncMock)
    @patch("api.connector_store.update_last_sync", new_callable=AsyncMock)
    @patch("api.connector_store.get_sync_cursor", new_callable=AsyncMock, return_value=None)
    @patch("api.commerce_service.persist_orders", new_callable=AsyncMock, return_value=1)
    async def test_orders_sync_success(
        self, mock_persist, mock_cursor, mock_update, mock_status, wired_scheduler
    ):
        connector = FakeConnector()
        wired_scheduler["app_state"].connectors = {"c1": connector}

        result = await sync_connector_incremental("c1", "orders")

        assert result is not None
        assert result["sync_type"] == "orders"
        assert result["fetched"] == 1
        assert result["written"] == 1
        connector.fetch_orders.assert_called_once()
        mock_persist.assert_called_once()
        mock_status.assert_any_call("c1", "syncing")
        mock_update.assert_called_once()

    @pytest.mark.asyncio
    @patch("api.connector_store.update_sync_status", new_callable=AsyncMock)
    @patch("api.connector_store.update_last_sync", new_callable=AsyncMock)
    @patch("api.connector_store.get_sync_cursor", new_callable=AsyncMock, return_value="2024-01-15T00:00:00Z")
    @patch("api.commerce_service.persist_financials", new_callable=AsyncMock, return_value=1)
    async def test_financials_sync_with_cursor(
        self, mock_persist, mock_cursor, mock_update, mock_status, wired_scheduler
    ):
        connector = FakeConnector()
        wired_scheduler["app_state"].connectors = {"c2": connector}

        result = await sync_connector_incremental("c2", "financials")

        assert result is not None
        assert result["sync_type"] == "financials"
        connector.fetch_financials.assert_called_once()
        call_args = connector.fetch_financials.call_args
        assert call_args[0][0] == "2024-01-15T00:00:00Z"

    @pytest.mark.asyncio
    async def test_missing_connector_returns_none(self, wired_scheduler):
        wired_scheduler["app_state"].connectors = {}
        result = await sync_connector_incremental("nonexistent", "orders")
        assert result is None

    @pytest.mark.asyncio
    @patch("api.connector_store.update_sync_status", new_callable=AsyncMock)
    @patch("api.connector_store.get_sync_cursor", new_callable=AsyncMock, return_value=None)
    @patch("api.connectors.sync_jobs._schedule_backoff_retry")
    async def test_rate_limit_triggers_backoff(
        self, mock_backoff, mock_cursor, mock_status, wired_scheduler
    ):
        connector = FakeConnector()
        connector.fetch_orders = AsyncMock(
            side_effect=ConnectorRateLimitError(retry_after=5.0)
        )
        wired_scheduler["app_state"].connectors = {"c3": connector}

        result = await sync_connector_incremental("c3", "orders")

        assert result is None
        mock_backoff.assert_called_once_with("c3", "orders", 5.0)
        mock_status.assert_any_call("c3", "idle")

    @pytest.mark.asyncio
    @patch("api.connector_store.update_sync_status", new_callable=AsyncMock)
    @patch("api.connector_store.get_sync_cursor", new_callable=AsyncMock, return_value=None)
    @patch("api.connectors.sync_jobs._deregister_sync_schedules")
    async def test_auth_error_disables_schedules(
        self, mock_dereg, mock_cursor, mock_status, wired_scheduler
    ):
        connector = FakeConnector()
        connector.fetch_orders = AsyncMock(
            side_effect=ConnectorAuthError("Token expired")
        )
        wired_scheduler["app_state"].connectors = {"c4": connector}

        result = await sync_connector_incremental("c4", "orders")

        assert result is None
        mock_status.assert_any_call("c4", "auth_failed")
        mock_dereg.assert_called_once_with("c4")

    @pytest.mark.asyncio
    @patch("api.connector_store.update_sync_status", new_callable=AsyncMock)
    @patch("api.connector_store.get_sync_cursor", new_callable=AsyncMock, return_value=None)
    async def test_generic_error_marks_error_status(
        self, mock_cursor, mock_status, wired_scheduler
    ):
        connector = FakeConnector()
        connector.fetch_orders = AsyncMock(side_effect=RuntimeError("Network down"))
        wired_scheduler["app_state"].connectors = {"c5": connector}

        result = await sync_connector_incremental("c5", "orders")

        assert result is None
        mock_status.assert_any_call("c5", "error")


# ---------------------------------------------------------------------------
# Job ID Format
# ---------------------------------------------------------------------------

class TestJobId:
    def test_format(self):
        assert _job_id("abc-123", "orders") == "sync_orders_abc-123"
        assert _job_id("abc-123", "financials") == "sync_financials_abc-123"


# ---------------------------------------------------------------------------
# Exception Classes
# ---------------------------------------------------------------------------

class TestExceptionClasses:
    def test_connector_auth_error(self):
        exc = ConnectorAuthError("bad token")
        assert str(exc) == "bad token"

    def test_connector_rate_limit_error(self):
        exc = ConnectorRateLimitError(retry_after=10.0)
        assert exc.retry_after == 10.0

    def test_connector_rate_limit_default(self):
        exc = ConnectorRateLimitError()
        assert exc.retry_after == 2.0
