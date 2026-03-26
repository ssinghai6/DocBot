"""Tests for commerce_service — DOCBOT-702.

Covers table registration, persistence helpers, query helpers with
mandatory connection_id filtering (RLS), and sync orchestration.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import MetaData, create_engine, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from api.commerce_service import (
    _parse_dt,
    register_commerce_tables,
    wire_commerce,
    persist_orders,
    persist_financials,
    query_orders,
    query_financials,
    get_order_count,
    sync_connector_data,
    CommerceSyncRequest,
    CommerceQueryParams,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def metadata():
    return MetaData()


@pytest.fixture
def commerce_tables(metadata):
    orders, financials = register_commerce_tables(metadata)
    return orders, financials


# ---------------------------------------------------------------------------
# Table registration tests
# ---------------------------------------------------------------------------

class TestTableRegistration:
    def test_register_creates_two_tables(self, commerce_tables):
        orders, financials = commerce_tables
        assert orders.name == "commerce_orders"
        assert financials.name == "commerce_financials"

    def test_orders_table_columns(self, commerce_tables):
        orders, _ = commerce_tables
        col_names = {c.name for c in orders.columns}
        expected = {
            "id", "connection_id", "connector_type", "marketplace_order_id",
            "status", "total_amount", "currency", "order_date",
            "customer_id", "raw_json", "created_at", "updated_at",
        }
        assert expected.issubset(col_names)

    def test_financials_table_columns(self, commerce_tables):
        _, financials = commerce_tables
        col_names = {c.name for c in financials.columns}
        expected = {
            "id", "connection_id", "connector_type",
            "period_start", "period_end",
            "revenue", "refunds", "fees", "net_proceeds",
            "currency", "raw_json", "created_at",
        }
        assert expected.issubset(col_names)

    def test_orders_primary_key(self, commerce_tables):
        orders, _ = commerce_tables
        pk_cols = [c.name for c in orders.primary_key.columns]
        assert pk_cols == ["id"]

    def test_financials_primary_key(self, commerce_tables):
        _, financials = commerce_tables
        pk_cols = [c.name for c in financials.primary_key.columns]
        assert pk_cols == ["id"]

    def test_connection_id_indexed(self, commerce_tables):
        orders, financials = commerce_tables
        for table in (orders, financials):
            conn_col = table.c.connection_id
            assert conn_col.index or any(
                conn_col in idx.columns for idx in table.indexes
            ), f"connection_id not indexed on {table.name}"

    def test_orders_has_unique_constraint(self, commerce_tables):
        orders, _ = commerce_tables
        unique_indexes = [idx for idx in orders.indexes if idx.unique]
        assert len(unique_indexes) >= 1
        idx = unique_indexes[0]
        idx_col_names = {c.name for c in idx.columns}
        assert "connection_id" in idx_col_names
        assert "marketplace_order_id" in idx_col_names


# ---------------------------------------------------------------------------
# _parse_dt tests
# ---------------------------------------------------------------------------

class TestParseDt:
    def test_parse_iso_string(self):
        result = _parse_dt("2024-01-15T10:30:00+00:00")
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1

    def test_parse_date_only_string(self):
        result = _parse_dt("2024-06-15")
        assert isinstance(result, datetime)

    def test_parse_z_suffix(self):
        result = _parse_dt("2024-01-15T10:30:00Z")
        assert isinstance(result, datetime)

    def test_parse_none(self):
        assert _parse_dt(None) is None

    def test_parse_datetime_passthrough(self):
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert _parse_dt(dt) is dt

    def test_parse_invalid_returns_none(self):
        assert _parse_dt("not-a-date") is None


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------

class TestPydanticModels:
    def test_sync_request(self):
        req = CommerceSyncRequest(start_date="2024-01-01", end_date="2024-01-31")
        assert req.start_date == "2024-01-01"

    def test_query_params_defaults(self):
        params = CommerceQueryParams()
        assert params.limit == 50
        assert params.offset == 0
        assert params.status is None


# ---------------------------------------------------------------------------
# Persistence tests (mocked DB)
# ---------------------------------------------------------------------------

class TestPersistOrders:
    @pytest.mark.asyncio
    async def test_empty_orders_returns_zero(self):
        result = await persist_orders("conn-1", "amazon", [])
        assert result == 0

    @pytest.mark.asyncio
    async def test_returns_zero_when_not_wired(self):
        # _orders_table is None by default unless wired
        import api.commerce_service as cs
        old_table = cs._orders_table
        cs._orders_table = None
        try:
            result = await persist_orders("conn-1", "amazon", [{"marketplace_order_id": "123"}])
            assert result == 0
        finally:
            cs._orders_table = old_table

    @pytest.mark.asyncio
    async def test_skips_orders_without_marketplace_id(self):
        import api.commerce_service as cs
        mock_table = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_begin = AsyncMock()
        mock_begin.__aenter__ = AsyncMock(return_value=mock_begin)
        mock_begin.__aexit__ = AsyncMock(return_value=False)
        mock_session.begin = MagicMock(return_value=mock_begin)

        mock_factory = MagicMock(return_value=mock_session)

        old_table, old_factory = cs._orders_table, cs._async_session_factory
        cs._orders_table = mock_table
        cs._async_session_factory = mock_factory
        try:
            result = await persist_orders("conn-1", "amazon", [{"status": "shipped"}])
            assert result == 0
        finally:
            cs._orders_table = old_table
            cs._async_session_factory = old_factory


class TestPersistFinancials:
    @pytest.mark.asyncio
    async def test_empty_financials_returns_zero(self):
        result = await persist_financials("conn-1", "amazon", [])
        assert result == 0


# ---------------------------------------------------------------------------
# Query tests (mocked DB)
# ---------------------------------------------------------------------------

class TestQueryOrders:
    @pytest.mark.asyncio
    async def test_returns_empty_when_not_wired(self):
        import api.commerce_service as cs
        old_table = cs._orders_table
        cs._orders_table = None
        try:
            result = await query_orders("conn-1")
            assert result == []
        finally:
            cs._orders_table = old_table

    @pytest.mark.asyncio
    async def test_connection_id_is_mandatory(self):
        """query_orders requires connection_id — cannot query all connections."""
        import inspect
        sig = inspect.signature(query_orders)
        params = list(sig.parameters.keys())
        assert params[0] == "connection_id"
        # No default value — it's required
        assert sig.parameters["connection_id"].default is inspect.Parameter.empty


class TestQueryFinancials:
    @pytest.mark.asyncio
    async def test_returns_empty_when_not_wired(self):
        import api.commerce_service as cs
        old_table = cs._financials_table
        cs._financials_table = None
        try:
            result = await query_financials("conn-1")
            assert result == []
        finally:
            cs._financials_table = old_table


class TestGetOrderCount:
    @pytest.mark.asyncio
    async def test_returns_zero_when_not_wired(self):
        import api.commerce_service as cs
        old_table = cs._orders_table
        cs._orders_table = None
        try:
            result = await get_order_count("conn-1")
            assert result == 0
        finally:
            cs._orders_table = old_table


# ---------------------------------------------------------------------------
# Sync orchestrator tests
# ---------------------------------------------------------------------------

class TestSyncConnectorData:
    @pytest.mark.asyncio
    async def test_sync_calls_fetch_and_persist(self):
        mock_connector = AsyncMock()
        mock_connector.connector_type = "amazon"
        mock_connector.fetch_orders.return_value = [
            {"marketplace_order_id": "AMZ-001", "status": "Shipped", "total_amount": 99.99},
        ]
        mock_connector.fetch_financials.return_value = [
            {"period_start": "2024-01-01", "period_end": "2024-01-31", "revenue": 1000.0,
             "refunds": 50.0, "fees": 100.0, "net_proceeds": 850.0},
        ]

        with patch("api.commerce_service.persist_orders", new_callable=AsyncMock, return_value=1) as mock_po, \
             patch("api.commerce_service.persist_financials", new_callable=AsyncMock, return_value=1) as mock_pf:
            result = await sync_connector_data("conn-1", mock_connector, "2024-01-01", "2024-01-31")

        assert result["connector_id"] == "conn-1"
        assert result["connector_type"] == "amazon"
        assert result["orders_fetched"] == 1
        assert result["orders_persisted"] == 1
        assert result["financials_fetched"] == 1
        assert result["financials_persisted"] == 1
        mock_po.assert_awaited_once()
        mock_pf.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sync_with_empty_data(self):
        mock_connector = AsyncMock()
        mock_connector.connector_type = "amazon"
        mock_connector.fetch_orders.return_value = []
        mock_connector.fetch_financials.return_value = []

        with patch("api.commerce_service.persist_orders", new_callable=AsyncMock, return_value=0), \
             patch("api.commerce_service.persist_financials", new_callable=AsyncMock, return_value=0):
            result = await sync_connector_data("conn-1", mock_connector, "2024-01-01", "2024-01-31")

        assert result["orders_fetched"] == 0
        assert result["financials_fetched"] == 0


# ---------------------------------------------------------------------------
# wire_commerce tests
# ---------------------------------------------------------------------------

class TestWireCommerce:
    def test_wire_sets_module_globals(self):
        import api.commerce_service as cs
        mock_orders = MagicMock()
        mock_fin = MagicMock()
        mock_factory = MagicMock()

        old = (cs._orders_table, cs._financials_table, cs._async_session_factory)
        try:
            wire_commerce(mock_orders, mock_fin, mock_factory)
            assert cs._orders_table is mock_orders
            assert cs._financials_table is mock_fin
            assert cs._async_session_factory is mock_factory
        finally:
            cs._orders_table, cs._financials_table, cs._async_session_factory = old


# ---------------------------------------------------------------------------
# RLS boundary enforcement test
# ---------------------------------------------------------------------------

class TestRLSEnforcement:
    """Verify that all query functions require connection_id and never
    expose a 'query all connections' path."""

    def test_query_orders_requires_connection_id(self):
        import inspect
        sig = inspect.signature(query_orders)
        assert "connection_id" in sig.parameters
        assert sig.parameters["connection_id"].default is inspect.Parameter.empty

    def test_query_financials_requires_connection_id(self):
        import inspect
        sig = inspect.signature(query_financials)
        assert "connection_id" in sig.parameters
        assert sig.parameters["connection_id"].default is inspect.Parameter.empty

    def test_get_order_count_requires_connection_id(self):
        import inspect
        sig = inspect.signature(get_order_count)
        assert "connection_id" in sig.parameters
        assert sig.parameters["connection_id"].default is inspect.Parameter.empty

    def test_persist_orders_requires_connection_id(self):
        import inspect
        sig = inspect.signature(persist_orders)
        assert "connection_id" in sig.parameters
        assert sig.parameters["connection_id"].default is inspect.Parameter.empty

    def test_persist_financials_requires_connection_id(self):
        import inspect
        sig = inspect.signature(persist_financials)
        assert "connection_id" in sig.parameters
        assert sig.parameters["connection_id"].default is inspect.Parameter.empty
