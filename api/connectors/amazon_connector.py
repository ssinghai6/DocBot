"""Amazon Selling Partner API (SP-API) connector.

Handles OAuth token refresh, rate limiting (0.5 req/sec on Orders API),
and maps SP-API responses to the unified commerce schema shape.

Required credentials dict keys:
  - client_id: LWA client ID
  - client_secret: LWA client secret
  - refresh_token: LWA refresh token
  - marketplace_id: e.g. "ATVPDKIKX0DER" for US
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from api.connectors.base import BaseConnector, ConnectorCredentials
from api.connectors.rate_limiter import RateLimiter
from api.connectors.registry import register

# SP-API base URL for North America region
_SPAPI_BASE = "https://sellingpartnerapi-na.amazon.com"
# LWA token endpoint
_LWA_TOKEN_URL = "https://api.amazon.com/auth/o2/token"
# Token TTL from Amazon docs is 3600 s; refresh 5 min early → 55 min
_TOKEN_TTL_SECONDS = 55 * 60
# Maximum pages to paginate (100 orders/page × 5 = 500 orders max)
_MAX_PAGES = 5
_PAGE_SIZE = 100


@register("amazon")
class AmazonConnector(BaseConnector):
    """Amazon SP-API connector for orders and financial events."""

    def __init__(self, creds: ConnectorCredentials) -> None:
        super().__init__(creds)
        self._access_token: str | None = None
        self._token_fetched_at: float = 0.0
        self._rate_limiter = RateLimiter(calls_per_second=0.5)

    # ── Internal helpers ────────────────────────────────────────────────────

    @property
    def connector_type(self) -> str:
        return "amazon"

    def _token_is_valid(self) -> bool:
        """Return True if the cached access token is still within its TTL."""
        return (
            self._access_token is not None
            and (time.monotonic() - self._token_fetched_at) < _TOKEN_TTL_SECONDS
        )

    async def _refresh_access_token(self) -> str:
        """Fetch a fresh LWA access token and cache it for 55 minutes."""
        client_id = self._creds.require("client_id")
        client_secret = self._creds.require("client_secret")
        refresh_token = self._creds.require("refresh_token")

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                _LWA_TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15.0,
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"LWA token refresh failed {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        token = data.get("access_token")
        if not token:
            raise RuntimeError("LWA response missing access_token field")

        self._access_token = token
        self._token_fetched_at = time.monotonic()
        return token

    async def _get_token(self) -> str:
        """Return a valid access token, refreshing if needed."""
        if self._token_is_valid() and self._access_token is not None:
            return self._access_token
        return await self._refresh_access_token()

    async def _spapi_get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        retry_on_401: bool = True,
    ) -> dict[str, Any]:
        """Perform an authenticated GET against the SP-API with retry logic.

        Handles:
        - 401: refresh token once, retry
        - 429: honour Retry-After header (default 2 s), retry once
        - 4xx/5xx: raise RuntimeError
        """
        token = await self._get_token()

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{_SPAPI_BASE}{path}",
                params=params,
                headers={
                    "x-amz-access-token": token,
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 401 and retry_on_401:
            # Force a token refresh and retry exactly once
            self._access_token = None
            return await self._spapi_get(path, params, retry_on_401=False)

        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", "2"))
            await asyncio.sleep(wait)
            # Retry once after back-off
            return await self._spapi_get(path, params, retry_on_401=False)

        raise RuntimeError(f"SP-API error {resp.status_code}: {resp.text}")

    # ── Public interface ────────────────────────────────────────────────────

    async def test_connection(self) -> bool:
        """Return True if SP-API marketplace participations returns 200."""
        try:
            await self._spapi_get("/sellers/v1/marketplaceParticipations")
            return True
        except Exception:
            return False

    async def fetch_orders(self, start_date: str, end_date: str) -> list[dict[str, Any]]:
        """Fetch up to 500 orders (5 pages × 100) from the Orders API.

        Maps each SP-API order to the unified commerce schema:
        {
            marketplace_order_id, status, total_amount, currency,
            order_date, customer_id, raw_json
        }
        """
        marketplace_id = self._creds.require("marketplace_id")
        orders: list[dict[str, Any]] = []
        next_token: str | None = None
        pages_fetched = 0

        while pages_fetched < _MAX_PAGES:
            async with self._rate_limiter:
                params: dict[str, Any] = {
                    "MarketplaceIds": marketplace_id,
                    "MaxResultsPerPage": _PAGE_SIZE,
                }
                if next_token:
                    params["NextToken"] = next_token
                else:
                    params["CreatedAfter"] = start_date
                    params["CreatedBefore"] = end_date

                data = await self._spapi_get("/orders/v0/orders", params)

            payload = data.get("payload", {})
            raw_orders = payload.get("Orders", [])

            for order in raw_orders:
                order_total = order.get("OrderTotal", {})
                orders.append(
                    {
                        "marketplace_order_id": order.get("AmazonOrderId"),
                        "status": order.get("OrderStatus"),
                        "total_amount": float(order_total.get("Amount", 0) or 0),
                        "currency": order_total.get("CurrencyCode"),
                        "order_date": order.get("PurchaseDate"),
                        "customer_id": order.get("BuyerInfo", {}).get(
                            "BuyerEmail"
                        ),
                        "raw_json": order,
                    }
                )

            pages_fetched += 1
            next_token = payload.get("NextToken")
            if not next_token:
                break

        return orders

    async def fetch_financials(self, start_date: str, end_date: str) -> list[dict[str, Any]]:
        """Fetch financial events from the Finances API.

        Maps events to the unified commerce schema:
        {
            period_start, period_end, revenue, refunds, fees,
            net_proceeds, currency, raw_json
        }

        Aggregation logic:
        - revenue   = sum of ItemPrice amounts in ShipmentEventList
        - refunds   = sum of ItemPrice amounts in RefundEventList
        - fees      = sum of FeeComponent amounts across all events
        - net_proceeds = revenue - refunds - fees
        """
        async with self._rate_limiter:
            data = await self._spapi_get(
                "/finances/v0/financialEvents",
                params={
                    "PostedAfter": start_date,
                    "PostedBefore": end_date,
                },
            )

        payload = data.get("payload", {}).get("FinancialEvents", {})

        # Determine dominant currency from first available amount field
        currency = "USD"

        def _safe_amount(amount_obj: dict[str, Any] | Any) -> float:
            """Safely extract a numeric amount and update currency from a CurrencyAmount dict."""
            nonlocal currency
            if not isinstance(amount_obj, dict):
                return 0.0
            currency = amount_obj.get("CurrencyCode", currency)
            try:
                return float(amount_obj.get("Amount", 0) or 0)
            except (TypeError, ValueError):
                return 0.0

        revenue = 0.0
        for shipment_event in payload.get("ShipmentEventList", []):
            for order_item in shipment_event.get("ShipmentItemList", []):
                for item_charge in order_item.get("ItemChargeList", []):
                    if item_charge.get("ChargeType") == "Principal":
                        revenue += _safe_amount(item_charge.get("ChargeAmount", {}))

        refunds = 0.0
        for refund_event in payload.get("RefundEventList", []):
            for order_item in refund_event.get("ShipmentItemAdjustmentList", []):
                for item_charge in order_item.get("ItemChargeAdjustmentList", []):
                    if item_charge.get("ChargeType") == "Principal":
                        refunds += abs(_safe_amount(item_charge.get("ChargeAmount", {})))

        fees = 0.0
        for shipment_event in payload.get("ShipmentEventList", []):
            for order_item in shipment_event.get("ShipmentItemList", []):
                for fee_component in order_item.get("ItemFeeList", []):
                    fees += abs(_safe_amount(fee_component.get("FeeAmount", {})))

        net_proceeds = revenue - refunds - fees

        return [
            {
                "period_start": start_date,
                "period_end": end_date,
                "revenue": round(revenue, 2),
                "refunds": round(refunds, 2),
                "fees": round(fees, 2),
                "net_proceeds": round(net_proceeds, 2),
                "currency": currency,
                "raw_json": payload,
            }
        ]
