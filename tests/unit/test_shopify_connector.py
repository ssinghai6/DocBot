"""Unit tests for ShopifyConnector — all httpx calls are mocked."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.connectors.base import (
    ConnectorAuthError,
    ConnectorCredentials,
    ConnectorRateLimitError,
)
from api.connectors.shopify_connector import (
    ShopifyConnector,
    normalize_shopify_status,
    verify_shopify_hmac,
    _MAX_PAGES,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_creds(**overrides: str) -> ConnectorCredentials:
    defaults = {
        "shop_domain": "testshop.myshopify.com",
        "access_token": "shpat_test_token_123",
        "webhook_secret": "whsec_test_secret",
    }
    defaults.update(overrides)
    return ConnectorCredentials(connector_type="shopify", credentials=defaults)


def _connector(**cred_overrides: str) -> ShopifyConnector:
    return ShopifyConnector(_make_creds(**cred_overrides))


def _shopify_response(
    payload: dict[str, Any],
    status: int = 200,
    link_header: str = "",
) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = payload
    mock.text = json.dumps(payload)
    mock.headers = MagicMock()
    mock.headers.get = lambda key, default="": {
        "link": link_header,
        "Link": link_header,
        "Retry-After": "2",
    }.get(key, default)
    return mock


def _make_order(
    order_id: int = 12345,
    total_price: str = "99.99",
    currency: str = "USD",
    financial_status: str = "paid",
    fulfillment_status: str | None = None,
    email: str = "buyer@example.com",
    refunds: list[dict[str, Any]] | None = None,
    total_tax: str = "8.00",
    total_discounts: str = "5.00",
) -> dict[str, Any]:
    order: dict[str, Any] = {
        "id": order_id,
        "total_price": total_price,
        "currency": currency,
        "financial_status": financial_status,
        "fulfillment_status": fulfillment_status,
        "created_at": "2024-01-15T10:00:00-05:00",
        "customer": {"email": email, "id": 111},
        "total_tax": total_tax,
        "total_discounts": total_discounts,
        "refunds": refunds or [],
    }
    return order


# ── Status normalization tests ───────────────────────────────────────────────


class TestStatusNormalization:
    def test_paid_no_fulfillment(self) -> None:
        assert normalize_shopify_status("paid", None) == "confirmed"

    def test_paid_fulfilled(self) -> None:
        assert normalize_shopify_status("paid", "fulfilled") == "delivered"

    def test_paid_partial_fulfillment(self) -> None:
        assert normalize_shopify_status("paid", "partial") == "shipped"

    def test_authorized_pending(self) -> None:
        assert normalize_shopify_status("authorized", None) == "pending"

    def test_refunded(self) -> None:
        assert normalize_shopify_status("refunded", None) == "refunded"

    def test_voided(self) -> None:
        assert normalize_shopify_status("voided", None) == "cancelled"

    def test_restocked_fulfillment(self) -> None:
        assert normalize_shopify_status("paid", "restocked") == "refunded"

    def test_unfulfilled(self) -> None:
        assert normalize_shopify_status("paid", "unfulfilled") == "confirmed"

    def test_unknown_status_passthrough(self) -> None:
        assert normalize_shopify_status("weird_status", None) == "weird_status"


# ── HMAC verification tests ─────────────────────────────────────────────────


class TestHmacVerification:
    def test_valid_hmac(self) -> None:
        body = b'{"topic":"orders/create"}'
        secret = "my_webhook_secret"
        computed = hmac.new(secret.encode(), body, hashlib.sha256).digest()
        header_hmac = base64.b64encode(computed).decode()

        assert verify_shopify_hmac(body, secret, header_hmac) is True

    def test_invalid_hmac(self) -> None:
        body = b'{"topic":"orders/create"}'
        secret = "my_webhook_secret"
        assert verify_shopify_hmac(body, secret, "invalid_hmac_value") is False

    def test_tampered_body(self) -> None:
        body = b'{"topic":"orders/create"}'
        secret = "my_webhook_secret"
        computed = hmac.new(secret.encode(), body, hashlib.sha256).digest()
        header_hmac = base64.b64encode(computed).decode()

        tampered = b'{"topic":"orders/delete"}'
        assert verify_shopify_hmac(tampered, secret, header_hmac) is False


# ── Connector property tests ─────────────────────────────────────────────────


class TestConnectorProperties:
    def test_connector_type(self) -> None:
        connector = _connector()
        assert connector.connector_type == "shopify"

    def test_base_url(self) -> None:
        connector = _connector()
        assert connector._base_url() == "https://testshop.myshopify.com/admin/api/2024-01"

    def test_custom_api_version(self) -> None:
        connector = _connector(api_version="2024-07")
        assert "2024-07" in connector._base_url()

    def test_missing_shop_domain_raises(self) -> None:
        creds = ConnectorCredentials(
            connector_type="shopify",
            credentials={"access_token": "tok"},
        )
        connector = ShopifyConnector(creds)
        with pytest.raises(ValueError, match="Missing required credential 'shop_domain'"):
            connector._base_url()

    def test_missing_access_token_raises(self) -> None:
        creds = ConnectorCredentials(
            connector_type="shopify",
            credentials={"shop_domain": "test.myshopify.com"},
        )
        connector = ShopifyConnector(creds)
        with pytest.raises(ValueError, match="Missing required credential 'access_token'"):
            connector._headers()


# ── test_connection tests ────────────────────────────────────────────────────


class TestTestConnection:
    def test_returns_true_on_200(self) -> None:
        connector = _connector()

        with patch.object(
            connector,
            "_shopify_get",
            AsyncMock(return_value=({"shop": {"name": "Test Shop"}}, MagicMock())),
        ):
            result = asyncio.run(connector.test_connection())
        assert result is True

    def test_returns_false_on_auth_error(self) -> None:
        connector = _connector()

        with patch.object(
            connector,
            "_shopify_get",
            AsyncMock(side_effect=ConnectorAuthError("401")),
        ):
            result = asyncio.run(connector.test_connection())
        assert result is False

    def test_returns_false_on_exception(self) -> None:
        connector = _connector()

        with patch.object(
            connector,
            "_shopify_get",
            AsyncMock(side_effect=RuntimeError("network")),
        ):
            result = asyncio.run(connector.test_connection())
        assert result is False


# ── HTTP helper tests ────────────────────────────────────────────────────────


class TestShopifyGet:
    def test_401_raises_auth_error(self) -> None:
        connector = _connector()
        resp_401 = MagicMock()
        resp_401.status_code = 401
        resp_401.text = "Unauthorized"

        with patch("httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=resp_401)

            async def run() -> None:
                with pytest.raises(ConnectorAuthError, match="Shopify auth failed"):
                    await connector._shopify_get("/shop.json")

            asyncio.run(run())

    def test_429_raises_rate_limit_error(self) -> None:
        connector = _connector()
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.text = "Too Many Requests"
        mock_headers = MagicMock()
        mock_headers.get = lambda key, default="": {"Retry-After": "3"}.get(key, default)
        resp_429.headers = mock_headers

        with patch("httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=resp_429)

            async def run() -> None:
                with pytest.raises(ConnectorRateLimitError):
                    await connector._shopify_get("/orders.json")

            asyncio.run(run())

    def test_500_raises_runtime_error(self) -> None:
        connector = _connector()
        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_500.text = "Internal Server Error"

        with patch("httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=resp_500)

            async def run() -> None:
                with pytest.raises(RuntimeError, match="Shopify API error 500"):
                    await connector._shopify_get("/orders.json")

            asyncio.run(run())


# ── Pagination tests ─────────────────────────────────────────────────────────


class TestPagination:
    def test_extract_next_page_url(self) -> None:
        connector = _connector()
        headers = MagicMock()
        headers.get = lambda key, default="": (
            '<https://testshop.myshopify.com/admin/api/2024-01/orders.json?page_info=abc123>; rel="next"'
            if key == "link" else default
        )
        url = connector._extract_next_page_url(headers)
        assert url == "https://testshop.myshopify.com/admin/api/2024-01/orders.json?page_info=abc123"

    def test_no_next_page(self) -> None:
        connector = _connector()
        headers = MagicMock()
        headers.get = lambda key, default="": (
            '<https://example.com>; rel="previous"'
            if key == "link" else default
        )
        assert connector._extract_next_page_url(headers) is None

    def test_empty_link_header(self) -> None:
        connector = _connector()
        headers = MagicMock()
        headers.get = lambda key, default="": default
        assert connector._extract_next_page_url(headers) is None


# ── fetch_orders tests ───────────────────────────────────────────────────────


class TestFetchOrders:
    def test_maps_fields_correctly(self) -> None:
        connector = _connector()
        order = _make_order(
            order_id=99999,
            total_price="149.50",
            currency="CAD",
            financial_status="paid",
            fulfillment_status="fulfilled",
        )

        headers_mock = MagicMock()
        headers_mock.get = lambda key, default="": default

        with patch.object(
            connector,
            "_shopify_get",
            AsyncMock(return_value=({"orders": [order]}, headers_mock)),
        ):
            orders = asyncio.run(connector.fetch_orders("2024-01-01", "2024-01-31"))

        assert len(orders) == 1
        o = orders[0]
        assert o["marketplace_order_id"] == "99999"
        assert o["status"] == "delivered"  # paid + fulfilled -> delivered
        assert o["total_amount"] == 149.50
        assert o["currency"] == "CAD"
        assert o["order_date"] == "2024-01-15T10:00:00-05:00"
        assert o["customer_id"] == "buyer@example.com"
        assert isinstance(o["raw_json"], dict)

    def test_empty_orders(self) -> None:
        connector = _connector()
        headers_mock = MagicMock()
        headers_mock.get = lambda key, default="": default

        with patch.object(
            connector,
            "_shopify_get",
            AsyncMock(return_value=({"orders": []}, headers_mock)),
        ):
            orders = asyncio.run(connector.fetch_orders("2024-01-01", "2024-01-31"))
        assert orders == []

    def test_pagination_follows_link_header(self) -> None:
        connector = _connector()

        order1 = _make_order(order_id=1)
        order2 = _make_order(order_id=2)

        page1_headers = MagicMock()
        page1_headers.get = lambda key, default="": (
            '<https://testshop.myshopify.com/admin/api/2024-01/orders.json?page_info=next123>; rel="next"'
            if key == "link" else default
        )

        page2_headers = MagicMock()
        page2_headers.get = lambda key, default="": default  # no next

        call_count = 0

        async def mock_get(path, params=None):
            nonlocal call_count
            call_count += 1
            return ({"orders": [order1]}, page1_headers)

        async def mock_get_paginated(url):
            nonlocal call_count
            call_count += 1
            return ({"orders": [order2]}, page2_headers)

        with patch.object(connector, "_shopify_get", side_effect=mock_get), \
             patch.object(connector, "_shopify_get_paginated", side_effect=mock_get_paginated):
            orders = asyncio.run(connector.fetch_orders("2024-01-01", "2024-01-31"))

        assert len(orders) == 2
        assert orders[0]["marketplace_order_id"] == "1"
        assert orders[1]["marketplace_order_id"] == "2"

    def test_pagination_stops_at_max_pages(self) -> None:
        connector = _connector()
        order = _make_order(order_id=1)

        always_next_headers = MagicMock()
        always_next_headers.get = lambda key, default="": (
            '<https://test.myshopify.com/orders.json?page_info=always>; rel="next"'
            if key == "link" else default
        )

        async def mock_get(path, params=None):
            return ({"orders": [order]}, always_next_headers)

        async def mock_get_paginated(url):
            return ({"orders": [order]}, always_next_headers)

        with patch.object(connector, "_shopify_get", side_effect=mock_get), \
             patch.object(connector, "_shopify_get_paginated", side_effect=mock_get_paginated):
            orders = asyncio.run(connector.fetch_orders("2024-01-01", "2024-01-31"))

        assert len(orders) == _MAX_PAGES  # 1 initial + 4 paginated = 5

    def test_no_customer_field(self) -> None:
        connector = _connector()
        order = _make_order()
        order["customer"] = None

        headers_mock = MagicMock()
        headers_mock.get = lambda key, default="": default

        with patch.object(
            connector,
            "_shopify_get",
            AsyncMock(return_value=({"orders": [order]}, headers_mock)),
        ):
            orders = asyncio.run(connector.fetch_orders("2024-01-01", "2024-01-31"))

        assert orders[0]["customer_id"] is None


# ── fetch_financials tests ───────────────────────────────────────────────────


class TestFetchFinancials:
    def test_aggregates_revenue_and_refunds(self) -> None:
        connector = _connector()

        refund_txn = {
            "transactions": [{"kind": "refund", "amount": "25.00"}],
        }
        orders = [
            _make_order(order_id=1, total_price="100.00", total_tax="8.00", total_discounts="5.00", refunds=[refund_txn]),
            _make_order(order_id=2, total_price="200.00", total_tax="16.00", total_discounts="10.00"),
        ]

        headers_mock = MagicMock()
        headers_mock.get = lambda key, default="": default

        with patch.object(
            connector,
            "_shopify_get",
            AsyncMock(return_value=({"orders": orders}, headers_mock)),
        ):
            result = asyncio.run(connector.fetch_financials("2024-01-01", "2024-01-31"))

        assert len(result) == 1
        fin = result[0]
        assert fin["revenue"] == 300.00
        assert fin["refunds"] == 25.00
        assert fin["fees"] == 0.0
        assert fin["net_proceeds"] == 275.00
        assert fin["currency"] == "USD"
        assert fin["period_start"] == "2024-01-01"
        assert fin["period_end"] == "2024-01-31"
        assert fin["raw_json"]["order_count"] == 2
        assert fin["raw_json"]["total_tax"] == 24.00
        assert fin["raw_json"]["total_discounts"] == 15.00

    def test_empty_orders_returns_zero_financials(self) -> None:
        connector = _connector()
        headers_mock = MagicMock()
        headers_mock.get = lambda key, default="": default

        with patch.object(
            connector,
            "_shopify_get",
            AsyncMock(return_value=({"orders": []}, headers_mock)),
        ):
            result = asyncio.run(connector.fetch_financials("2024-01-01", "2024-01-31"))

        assert len(result) == 1
        assert result[0]["revenue"] == 0.0
        assert result[0]["refunds"] == 0.0
        assert result[0]["net_proceeds"] == 0.0


# ── Registry tests ───────────────────────────────────────────────────────────


class TestShopifyRegistry:
    def test_shopify_registered(self) -> None:
        from api.connectors.registry import get_connector_class

        cls = get_connector_class("shopify")
        assert cls is ShopifyConnector

    def test_connector_type_property(self) -> None:
        connector = _connector()
        assert connector.connector_type == "shopify"

    def test_in_list_types(self) -> None:
        from api.connectors.registry import list_connector_types

        types = list_connector_types()
        assert "shopify" in types
        assert "amazon" in types
