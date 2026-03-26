"""Unit tests for AmazonConnector — all httpx calls are mocked."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.connectors.amazon_connector import AmazonConnector, _TOKEN_TTL_SECONDS
from api.connectors.base import ConnectorCredentials


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_creds(**overrides: str) -> ConnectorCredentials:
    defaults = {
        "client_id": "amzn1.application-oa2-client.test",
        "client_secret": "secret-test",
        "refresh_token": "Atzr|test-refresh-token",
        "marketplace_id": "ATVPDKIKX0DER",
    }
    defaults.update(overrides)
    return ConnectorCredentials(connector_type="amazon", credentials=defaults)


def _connector() -> AmazonConnector:
    return AmazonConnector(_make_creds())


def _lwa_response(token: str = "access-token-1") -> MagicMock:
    """Mock a successful LWA token response."""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {"access_token": token, "expires_in": 3600}
    return mock


def _spapi_response(payload: dict[str, Any], status: int = 200) -> MagicMock:
    """Mock a successful SP-API response."""
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = payload
    mock.text = json.dumps(payload)
    mock.headers = {}
    return mock


# ── Token refresh tests ───────────────────────────────────────────────────────

class TestTokenRefresh:
    def test_refresh_caches_token(self) -> None:
        """Token fetched once; second call reuses cached value without HTTP."""
        connector = _connector()

        with patch("httpx.AsyncClient") as mock_client_cls:
            instance = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.post = AsyncMock(return_value=_lwa_response("tok-1"))
            instance.get = AsyncMock(
                return_value=_spapi_response({"payload": {"ParticipationList": []}})
            )

            async def run() -> None:
                token1 = await connector._refresh_access_token()
                assert token1 == "tok-1"
                assert connector._token_is_valid()
                # Second call should return cached token without HTTP call
                token2 = await connector._get_token()
                assert token2 == "tok-1"
                # Only one POST should have been made
                assert instance.post.call_count == 1

            asyncio.run(run())

    def test_expired_token_triggers_refresh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If token_fetched_at is old enough, _get_token forces a refresh."""
        connector = _connector()
        connector._access_token = "old-token"
        # Simulate token being fetched well beyond TTL
        monkeypatch.setattr(
            "api.connectors.amazon_connector.time.monotonic",
            lambda: connector._token_fetched_at + _TOKEN_TTL_SECONDS + 1,
        )

        with patch("httpx.AsyncClient") as mock_client_cls:
            instance = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.post = AsyncMock(return_value=_lwa_response("new-token"))

            async def run() -> None:
                token = await connector._get_token()
                assert token == "new-token"

            asyncio.run(run())

    def test_refresh_raises_on_non_200(self) -> None:
        """LWA non-200 response raises RuntimeError."""
        connector = _connector()
        bad_response = MagicMock()
        bad_response.status_code = 400
        bad_response.text = "invalid_client"

        with patch("httpx.AsyncClient") as mock_client_cls:
            instance = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.post = AsyncMock(return_value=bad_response)

            async def run() -> None:
                with pytest.raises(RuntimeError, match="LWA token refresh failed 400"):
                    await connector._refresh_access_token()

            asyncio.run(run())


# ── test_connection tests ─────────────────────────────────────────────────────

class TestTestConnection:
    def _patch_client(self, get_response: MagicMock) -> tuple[MagicMock, AsyncMock]:
        """Return a context manager patcher for httpx.AsyncClient."""
        mock_client_cls = MagicMock()
        instance = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        instance.post = AsyncMock(return_value=_lwa_response())
        instance.get = AsyncMock(return_value=get_response)
        return mock_client_cls, instance

    def test_returns_true_on_200(self) -> None:
        connector = _connector()
        ok_response = _spapi_response({"payload": {"ParticipationList": []}})
        mock_cls, _ = self._patch_client(ok_response)

        with patch("httpx.AsyncClient", mock_cls):
            result = asyncio.run(connector.test_connection())
        assert result is True

    def test_returns_false_on_403(self) -> None:
        connector = _connector()
        bad_response = MagicMock()
        bad_response.status_code = 403
        bad_response.text = "Forbidden"
        bad_response.headers = {}

        mock_cls, _ = self._patch_client(bad_response)

        # First GET returns 403 (auth error); after 401 retry it also returns 403
        mock_instance = AsyncMock()
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_instance.post = AsyncMock(return_value=_lwa_response())
        mock_instance.get = AsyncMock(return_value=bad_response)

        with patch("httpx.AsyncClient", mock_cls):
            result = asyncio.run(connector.test_connection())
        assert result is False

    def test_returns_false_on_exception(self) -> None:
        connector = _connector()

        with patch.object(
            connector,
            "_spapi_get",
            AsyncMock(side_effect=RuntimeError("network error")),
        ):
            result = asyncio.run(connector.test_connection())
        assert result is False


# ── fetch_orders tests ────────────────────────────────────────────────────────

class TestFetchOrders:
    def _single_page_payload(
        self,
        order_id: str = "111-1234567-1234567",
        status: str = "Shipped",
        amount: str = "99.99",
        currency: str = "USD",
        next_token: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "payload": {
                "Orders": [
                    {
                        "AmazonOrderId": order_id,
                        "OrderStatus": status,
                        "OrderTotal": {"Amount": amount, "CurrencyCode": currency},
                        "PurchaseDate": "2024-01-15T10:00:00Z",
                        "BuyerInfo": {"BuyerEmail": "buyer@example.com"},
                    }
                ],
            }
        }
        if next_token:
            payload["payload"]["NextToken"] = next_token
        return payload

    def test_maps_fields_correctly(self) -> None:
        connector = _connector()
        api_payload = self._single_page_payload()

        with patch.object(
            connector, "_get_token", AsyncMock(return_value="tok")
        ), patch.object(
            connector, "_spapi_get", AsyncMock(return_value=api_payload)
        ):
            orders = asyncio.run(
                connector.fetch_orders("2024-01-01", "2024-01-31")
            )

        assert len(orders) == 1
        order = orders[0]
        assert order["marketplace_order_id"] == "111-1234567-1234567"
        assert order["status"] == "Shipped"
        assert order["total_amount"] == 99.99
        assert order["currency"] == "USD"
        assert order["order_date"] == "2024-01-15T10:00:00Z"
        assert order["customer_id"] == "buyer@example.com"
        assert isinstance(order["raw_json"], dict)

    def test_empty_orders_list(self) -> None:
        connector = _connector()
        api_payload = {"payload": {"Orders": []}}

        with patch.object(
            connector, "_spapi_get", AsyncMock(return_value=api_payload)
        ):
            orders = asyncio.run(
                connector.fetch_orders("2024-01-01", "2024-01-31")
            )
        assert orders == []

    def test_pagination_follows_next_token(self) -> None:
        """Two pages: first page has NextToken, second page has none."""
        connector = _connector()

        page1 = self._single_page_payload(order_id="order-1", next_token="PAGE2TOKEN")
        page2 = self._single_page_payload(order_id="order-2")  # no NextToken

        call_count = 0

        async def fake_spapi_get(path: str, params: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return page1
            return page2

        with patch.object(connector, "_spapi_get", side_effect=fake_spapi_get):
            orders = asyncio.run(
                connector.fetch_orders("2024-01-01", "2024-01-31")
            )

        assert call_count == 2
        assert len(orders) == 2
        assert orders[0]["marketplace_order_id"] == "order-1"
        assert orders[1]["marketplace_order_id"] == "order-2"

    def test_pagination_stops_at_max_pages(self) -> None:
        """Connector never fetches more than _MAX_PAGES pages."""
        from api.connectors.amazon_connector import _MAX_PAGES

        connector = _connector()
        # Every page always returns a NextToken to simulate infinite pagination
        always_next = self._single_page_payload(order_id="x", next_token="ALWAYS")
        call_count = 0

        async def fake_spapi_get(path: str, params: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return always_next

        with patch.object(connector, "_spapi_get", side_effect=fake_spapi_get):
            orders = asyncio.run(
                connector.fetch_orders("2024-01-01", "2024-01-31")
            )

        assert call_count == _MAX_PAGES
        assert len(orders) == _MAX_PAGES


# ── fetch_financials tests ───────────────────────────────────────────────────

class TestFetchFinancials:
    def test_aggregates_revenue_refunds_fees(self) -> None:
        """Verify revenue, refunds, and fees are extracted and aggregated correctly."""
        connector = _connector()

        financial_payload = {
            "payload": {
                "FinancialEvents": {
                    "ShipmentEventList": [
                        {
                            "ShipmentItemList": [
                                {
                                    "ItemChargeList": [
                                        {
                                            "ChargeType": "Principal",
                                            "ChargeAmount": {"CurrencyCode": "USD", "Amount": "100.00"},
                                        },
                                        {
                                            "ChargeType": "Tax",
                                            "ChargeAmount": {"CurrencyCode": "USD", "Amount": "8.00"},
                                        },
                                    ],
                                    "ItemFeeList": [
                                        {
                                            "FeeType": "Commission",
                                            "FeeAmount": {"CurrencyCode": "USD", "Amount": "-15.00"},
                                        },
                                    ],
                                }
                            ]
                        }
                    ],
                    "RefundEventList": [
                        {
                            "ShipmentItemAdjustmentList": [
                                {
                                    "ItemChargeAdjustmentList": [
                                        {
                                            "ChargeType": "Principal",
                                            "ChargeAmount": {"CurrencyCode": "USD", "Amount": "-25.00"},
                                        }
                                    ]
                                }
                            ]
                        }
                    ],
                }
            }
        }

        with patch.object(
            connector, "_spapi_get", AsyncMock(return_value=financial_payload)
        ):
            result = asyncio.run(
                connector.fetch_financials("2024-01-01", "2024-01-31")
            )

        assert len(result) == 1
        fin = result[0]
        assert fin["revenue"] == 100.00
        assert fin["refunds"] == 25.00
        assert fin["fees"] == 15.00
        assert fin["net_proceeds"] == 60.00
        assert fin["currency"] == "USD"
        assert fin["period_start"] == "2024-01-01"
        assert fin["period_end"] == "2024-01-31"

    def test_empty_financial_events(self) -> None:
        """Empty financial events returns zero aggregates."""
        connector = _connector()
        empty_payload = {"payload": {"FinancialEvents": {}}}

        with patch.object(
            connector, "_spapi_get", AsyncMock(return_value=empty_payload)
        ):
            result = asyncio.run(
                connector.fetch_financials("2024-01-01", "2024-01-31")
            )

        assert len(result) == 1
        assert result[0]["revenue"] == 0.0
        assert result[0]["refunds"] == 0.0
        assert result[0]["fees"] == 0.0
        assert result[0]["net_proceeds"] == 0.0


# ── 429 retry tests ───────────────────────────────────────────────────────────

class TestRateLimitRetry:
    def test_429_waits_and_retries(self) -> None:
        """On 429, connector sleeps Retry-After seconds then retries once."""
        connector = _connector()
        connector._access_token = "valid-tok"
        connector._token_fetched_at = 9_999_999.0  # far future — avoids refresh

        throttled = MagicMock()
        throttled.status_code = 429
        throttled.headers = {"Retry-After": "1"}
        throttled.text = "throttled"

        ok = _spapi_response({"payload": {}})

        call_count = 0

        async def fake_client_get(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return throttled
            return ok

        with patch("httpx.AsyncClient") as mock_cls, \
             patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            instance = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = fake_client_get

            result = asyncio.run(connector._spapi_get("/some/path"))

        # Should have slept for Retry-After=1
        mock_sleep.assert_called_once_with(1.0)
        assert result == {"payload": {}}

    def test_non_200_non_retryable_raises(self) -> None:
        """4xx errors that are not 401/429 raise RuntimeError immediately."""
        connector = _connector()
        connector._access_token = "valid-tok"
        connector._token_fetched_at = 9_999_999.0

        server_error = MagicMock()
        server_error.status_code = 500
        server_error.text = "Internal Server Error"
        server_error.headers = {}

        with patch("httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=server_error)

            async def run() -> None:
                with pytest.raises(RuntimeError, match="SP-API error 500"):
                    await connector._spapi_get("/some/path")

            asyncio.run(run())


# ── Credential validation tests ──────────────────────────────────────────────

class TestCredentialValidation:
    def test_missing_client_id_raises(self) -> None:
        creds = ConnectorCredentials(
            connector_type="amazon",
            credentials={"client_secret": "s", "refresh_token": "r", "marketplace_id": "m"},
        )
        connector = AmazonConnector(creds)

        async def run() -> None:
            with pytest.raises(ValueError, match="Missing required credential 'client_id'"):
                await connector._refresh_access_token()

        asyncio.run(run())

    def test_missing_marketplace_id_raises(self) -> None:
        creds = ConnectorCredentials(
            connector_type="amazon",
            credentials={"client_id": "c", "client_secret": "s", "refresh_token": "r"},
        )
        connector = AmazonConnector(creds)

        async def run() -> None:
            with pytest.raises(ValueError, match="Missing required credential 'marketplace_id'"):
                await connector.fetch_orders("2024-01-01", "2024-01-31")

        asyncio.run(run())


# ── Registry tests ────────────────────────────────────────────────────────────

class TestRegistry:
    def test_amazon_registered(self) -> None:
        from api.connectors.registry import get_connector_class

        cls = get_connector_class("amazon")
        assert cls is AmazonConnector

    def test_connector_type_property(self) -> None:
        connector = _connector()
        assert connector.connector_type == "amazon"

    def test_list_connector_types(self) -> None:
        from api.connectors.registry import list_connector_types

        types = list_connector_types()
        assert "amazon" in types

    def test_unknown_connector_raises(self) -> None:
        from api.connectors.registry import get_connector_class

        with pytest.raises(KeyError, match="No connector registered for type 'nonexistent'"):
            get_connector_class("nonexistent")


# ── Rate limiter tests ───────────────────────────────────────────────────────

class TestRateLimiter:
    def test_invalid_rate_raises(self) -> None:
        from api.connectors.rate_limiter import RateLimiter

        with pytest.raises(ValueError, match="calls_per_second must be > 0"):
            RateLimiter(calls_per_second=0)

        with pytest.raises(ValueError, match="calls_per_second must be > 0"):
            RateLimiter(calls_per_second=-1)

    def test_limiter_allows_first_call_immediately(self) -> None:
        from api.connectors.rate_limiter import RateLimiter

        limiter = RateLimiter(calls_per_second=10.0)

        async def run() -> None:
            async with limiter:
                pass  # Should not block

        asyncio.run(run())
