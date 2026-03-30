"""Shopify Admin API connector — DOCBOT-705.

Connects to a Shopify store via the Admin REST API using a custom app
access token.  Fetches orders with cursor-based pagination (``page_info``
in ``Link`` header) and aggregates financial data from order transactions.

Rate limit: Shopify allows 2 req/s for REST Admin API (bucket leak rate).

Required credentials dict keys:
  - shop_domain: e.g. "mystore.myshopify.com"
  - access_token: Admin API access token (from custom app or OAuth)

Optional credentials:
  - webhook_secret: HMAC secret for verifying Shopify webhook payloads
  - api_version: Shopify API version, defaults to "2024-01"
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import re
from typing import Any

import httpx

from api.connectors.base import (
    BaseConnector,
    ConnectorAuthError,
    ConnectorCredentials,
    ConnectorRateLimitError,
)
from api.connectors.rate_limiter import RateLimiter
from api.connectors.registry import register

logger = logging.getLogger(__name__)

# Shopify REST Admin API defaults
_DEFAULT_API_VERSION = "2024-01"
_MAX_PAGES = 5
_PAGE_SIZE = 250  # Shopify max per page for orders

# Shopify REST rate limit: 2 requests/second (bucket leak)
_RATE_LIMIT_RPS = 2.0

# Link header pagination regex
_NEXT_PAGE_RE = re.compile(r'<([^>]+)>;\s*rel="next"')

# Status normalization: Shopify financial_status -> unified status
_STATUS_MAP = {
    "authorized": "pending",
    "pending": "pending",
    "paid": "confirmed",
    "partially_paid": "confirmed",
    "partially_refunded": "confirmed",
    "refunded": "refunded",
    "voided": "cancelled",
}

# Shopify fulfillment_status -> unified status (overrides financial if present)
_FULFILLMENT_STATUS_MAP = {
    "fulfilled": "delivered",
    "partial": "shipped",
    "unfulfilled": "confirmed",
    "restocked": "refunded",
    None: None,  # use financial_status
}


def normalize_shopify_status(financial_status: str, fulfillment_status: str | None) -> str:
    """Map Shopify's dual-status model to a single unified status string."""
    # Fulfillment status takes priority if meaningful
    if fulfillment_status and fulfillment_status in _FULFILLMENT_STATUS_MAP:
        mapped = _FULFILLMENT_STATUS_MAP[fulfillment_status]
        if mapped:
            return mapped
    # Fall back to financial status
    return _STATUS_MAP.get(financial_status, financial_status)


def verify_shopify_hmac(body: bytes, secret: str, header_hmac: str) -> bool:
    """Verify Shopify webhook HMAC-SHA256 signature."""
    computed = hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).digest()
    import base64
    expected = base64.b64encode(computed).decode("utf-8")
    return hmac.compare_digest(expected, header_hmac)


@register("shopify")
class ShopifyConnector(BaseConnector):
    """Shopify Admin REST API connector for orders and financial events."""

    def __init__(self, creds: ConnectorCredentials) -> None:
        super().__init__(creds)
        self._rate_limiter = RateLimiter(calls_per_second=_RATE_LIMIT_RPS)
        self._api_version = creds.credentials.get("api_version", _DEFAULT_API_VERSION)

    @property
    def connector_type(self) -> str:
        return "shopify"

    def _base_url(self) -> str:
        shop = self._creds.require("shop_domain")
        return f"https://{shop}/admin/api/{self._api_version}"

    def _headers(self) -> dict[str, str]:
        token = self._creds.require("access_token")
        return {
            "X-Shopify-Access-Token": token,
            "Content-Type": "application/json",
        }

    # ── HTTP helper ─────────────────────────────────────────────────────────

    async def _shopify_get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], httpx.Headers]:
        """Authenticated GET with rate limiting, 401/429 handling.

        Returns (json_body, response_headers) so caller can read Link header.
        """
        async with self._rate_limiter:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self._base_url()}{path}",
                    params=params,
                    headers=self._headers(),
                    timeout=30.0,
                )

        if resp.status_code == 200:
            return resp.json(), resp.headers

        if resp.status_code == 401:
            raise ConnectorAuthError(
                f"Shopify auth failed: {resp.status_code} {resp.text}"
            )

        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "2"))
            raise ConnectorRateLimitError(
                retry_after=retry_after,
                message=f"Shopify rate limited, retry after {retry_after}s",
            )

        raise RuntimeError(f"Shopify API error {resp.status_code}: {resp.text}")

    def _extract_next_page_url(self, headers: httpx.Headers) -> str | None:
        """Parse cursor-based pagination from Shopify Link header."""
        link = headers.get("link", "")
        match = _NEXT_PAGE_RE.search(link)
        return match.group(1) if match else None

    async def _shopify_get_paginated(
        self,
        url: str,
    ) -> tuple[dict[str, Any], httpx.Headers]:
        """GET a full URL (for pagination follow-up) with rate limiting."""
        async with self._rate_limiter:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    url,
                    headers=self._headers(),
                    timeout=30.0,
                )

        if resp.status_code == 200:
            return resp.json(), resp.headers

        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "2"))
            raise ConnectorRateLimitError(retry_after=retry_after)

        raise RuntimeError(f"Shopify API error {resp.status_code}: {resp.text}")

    # ── Public interface ────────────────────────────────────────────────────

    async def test_connection(self) -> bool:
        """Verify credentials by calling GET /shop.json."""
        try:
            await self._shopify_get("/shop.json")
            return True
        except Exception:
            return False

    async def fetch_orders(
        self, start_date: str, end_date: str
    ) -> list[dict[str, Any]]:
        """Fetch orders created in [start_date, end_date] with cursor pagination.

        Maps each Shopify order to unified commerce schema:
        {
            marketplace_order_id, status, total_amount, currency,
            order_date, customer_id, raw_json
        }
        """
        orders: list[dict[str, Any]] = []
        pages_fetched = 0

        data, headers = await self._shopify_get(
            "/orders.json",
            params={
                "created_at_min": start_date,
                "created_at_max": end_date,
                "limit": _PAGE_SIZE,
                "status": "any",
            },
        )

        raw_orders = data.get("orders", [])
        orders.extend(self._normalize_orders(raw_orders))
        pages_fetched += 1

        # Cursor-based pagination via Link header
        next_url = self._extract_next_page_url(headers)
        while next_url and pages_fetched < _MAX_PAGES:
            data, headers = await self._shopify_get_paginated(next_url)
            raw_orders = data.get("orders", [])
            orders.extend(self._normalize_orders(raw_orders))
            pages_fetched += 1
            next_url = self._extract_next_page_url(headers)

        logger.info(
            "Shopify fetch_orders: %d orders across %d pages",
            len(orders), pages_fetched,
        )
        return orders

    def _normalize_orders(self, raw_orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Map raw Shopify orders to unified schema."""
        normalized = []
        for order in raw_orders:
            customer = order.get("customer") or {}
            normalized.append({
                "marketplace_order_id": str(order.get("id", "")),
                "status": normalize_shopify_status(
                    order.get("financial_status", ""),
                    order.get("fulfillment_status"),
                ),
                "total_amount": float(order.get("total_price", 0) or 0),
                "currency": order.get("currency", "USD"),
                "order_date": order.get("created_at"),
                "customer_id": customer.get("email"),
                "raw_json": order,
            })
        return normalized

    async def fetch_financials(
        self, start_date: str, end_date: str
    ) -> list[dict[str, Any]]:
        """Aggregate financial data from orders in the date range.

        Shopify doesn't have a dedicated financials API like Amazon,
        so we compute revenue/refunds from the orders themselves.

        Maps to unified commerce schema:
        {
            period_start, period_end, revenue, refunds, fees,
            net_proceeds, currency, raw_json
        }
        """
        # Fetch all orders in the period (reuse the same pagination logic)
        data, headers = await self._shopify_get(
            "/orders.json",
            params={
                "created_at_min": start_date,
                "created_at_max": end_date,
                "limit": _PAGE_SIZE,
                "status": "any",
                "fields": "id,total_price,total_tax,financial_status,currency,refunds,total_discounts",
            },
        )

        all_orders = data.get("orders", [])
        pages = 1

        next_url = self._extract_next_page_url(headers)
        while next_url and pages < _MAX_PAGES:
            data, headers = await self._shopify_get_paginated(next_url)
            all_orders.extend(data.get("orders", []))
            pages += 1
            next_url = self._extract_next_page_url(headers)

        # Aggregate
        currency = "USD"
        revenue = 0.0
        refunds = 0.0
        taxes = 0.0
        discounts = 0.0

        for order in all_orders:
            currency = order.get("currency", currency)
            revenue += float(order.get("total_price", 0) or 0)
            taxes += float(order.get("total_tax", 0) or 0)
            discounts += float(order.get("total_discounts", 0) or 0)

            # Sum refund amounts
            for refund in order.get("refunds", []):
                for txn in refund.get("transactions", []):
                    if txn.get("kind") == "refund":
                        refunds += abs(float(txn.get("amount", 0) or 0))

        net_proceeds = revenue - refunds

        return [
            {
                "period_start": start_date,
                "period_end": end_date,
                "revenue": round(revenue, 2),
                "refunds": round(refunds, 2),
                "fees": 0.0,  # Shopify doesn't expose platform fees via REST API
                "net_proceeds": round(net_proceeds, 2),
                "currency": currency,
                "raw_json": {
                    "order_count": len(all_orders),
                    "total_tax": round(taxes, 2),
                    "total_discounts": round(discounts, 2),
                },
            }
        ]
