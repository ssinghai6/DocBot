"""Marketplace connector + commerce routes — DOCBOT-1602 (SEC-03).

Every ``/api/connectors*`` and ``/api/commerce/*`` route requires a logged-in
analyst+ in EVERY mode (including open/demo mode): connectors hold
third-party store credentials and drive outbound HTTP, so anonymous access is
never allowed. All data is scoped to the caller's organization:

* ``marketplace_connections.org_id`` is stamped from the caller on register;
* lookups by ``connector_id`` check org ownership first and answer 404 for a
  connector in another org (existence is not disclosed);
* commerce rows are keyed by ``connection_id``, so the same ownership check
  gates ``/api/commerce/{connector_id}/*``.

Error responses never echo raw exception text; details are logged
server-side by exception type only (exception messages from connector HTTP
calls can carry upstream response bodies).

The Shopify webhook (``POST /api/marketplace/webhook/shopify``) is
deliberately public: Shopify calls it server-to-server and it is
authenticated by HMAC-SHA256 over the raw body.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from api.rbac_service import UserRole, get_user_org_id, require_role

logger = logging.getLogger(__name__)

router = APIRouter()

_MAX_CREDENTIAL_FIELDS = 20
_MAX_CREDENTIAL_VALUE_LEN = 4096


# ---------------------------------------------------------------------------
# Auth / scoping dependencies
# ---------------------------------------------------------------------------

_require_analyst_always = require_role(UserRole.analyst, enforce_in_open_mode=True)


async def require_connector_org(user: Any = Depends(_require_analyst_always)) -> str:
    """Analyst+ session in every mode; returns the caller's org_id.

    401 anonymous, 403 below analyst, 403 when the account has no org
    (fail closed rather than an unscoped view).
    """
    org_id = get_user_org_id(user)
    if not org_id:
        raise HTTPException(status_code=403, detail="Your account is not attached to an organization.")
    return str(org_id)


def _connector_cache(request: Request) -> dict[str, Any]:
    state = request.app.state
    if not hasattr(state, "connectors"):
        state.connectors = {}
    return state.connectors


async def _owned_connector(request: Request, connector_id: str, org_id: str) -> Any:
    """Return the live connector instance if it belongs to ``org_id``, else 404."""
    from api.connector_store import connector_in_org

    if not await connector_in_org(connector_id, org_id):
        raise HTTPException(status_code=404, detail="Connector not found.")
    connector = _connector_cache(request).get(connector_id)
    if connector is None:
        raise HTTPException(status_code=404, detail="Connector not found.")
    return connector


async def _require_owned_connector_id(connector_id: str, org_id: str) -> None:
    from api.connector_store import connector_in_org

    if not await connector_in_org(connector_id, org_id):
        raise HTTPException(status_code=404, detail="Connector not found.")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ConnectorRegisterRequest(BaseModel):
    """Register a new commerce connector with encrypted credentials."""

    connector_type: str = Field(..., min_length=1, max_length=64)  # e.g. "amazon", "shopify"
    credentials: Dict[str, str]

    @model_validator(mode="after")
    def _validate_credentials(self) -> "ConnectorRegisterRequest":
        if len(self.credentials) > _MAX_CREDENTIAL_FIELDS:
            raise ValueError("Too many credential fields.")
        if any(len(v) > _MAX_CREDENTIAL_VALUE_LEN for v in self.credentials.values()):
            raise ValueError("Credential value too long.")
        self.connector_type = self.connector_type.strip().lower()
        if self.connector_type == "shopify":
            from api.connectors.shopify_connector import validate_api_version, validate_shop_domain

            self.credentials["shop_domain"] = validate_shop_domain(self.credentials.get("shop_domain"))
            if "api_version" in self.credentials:
                self.credentials["api_version"] = validate_api_version(self.credentials["api_version"])
        return self


class ConnectorDateRangeRequest(BaseModel):
    """Date range for fetching orders or financials."""

    start_date: str = Field(..., max_length=40)  # ISO 8601, e.g. "2024-01-01"
    end_date: str = Field(..., max_length=40)


class ConnectorRegisterResponse(BaseModel):
    connector_id: str
    connector_type: str
    status: str


class ConnectorListResponse(BaseModel):
    connectors: list[dict[str, Any]]


class ConnectorTypesResponse(BaseModel):
    types: list[str]


class ConnectorDisconnectResponse(BaseModel):
    status: str
    connector_id: str


class ConnectorOrdersResponse(BaseModel):
    connector_id: str
    order_count: int
    orders: list[dict[str, Any]]


class ConnectorFinancialsResponse(BaseModel):
    connector_id: str
    financials: list[dict[str, Any]]


class WebhookAck(BaseModel):
    status: str
    topic: str


# ---------------------------------------------------------------------------
# Connector routes
# ---------------------------------------------------------------------------


@router.post("/api/connectors/register", response_model=ConnectorRegisterResponse)
async def register_connector(
    body: ConnectorRegisterRequest,
    request: Request,
    org_id: str = Depends(require_connector_org),
) -> ConnectorRegisterResponse:
    """Register and test a commerce connector in the caller's org."""
    from api.connector_store import save_connector
    from api.connectors.base import ConnectorCredentials
    from api.connectors.registry import get_connector_class
    from api.connectors.sync_jobs import register_sync_schedules

    try:
        cls = get_connector_class(body.connector_type)
    except KeyError:
        raise HTTPException(status_code=400, detail="Unsupported connector type.")

    connector = cls(ConnectorCredentials(connector_type=body.connector_type, credentials=body.credentials))

    try:
        ok = await connector.test_connection()
    except Exception as exc:  # connector HTTP stacks raise heterogeneous errors
        logger.warning("Connector test_connection failed (type=%s): %s", body.connector_type, type(exc).__name__)
        raise HTTPException(status_code=502, detail="Connection test failed. Check your credentials.")
    if not ok:
        raise HTTPException(status_code=502, detail="Connection test failed. Check your credentials.")

    connector_id = str(uuid.uuid4())
    _connector_cache(request)[connector_id] = connector
    await save_connector(connector_id, body.connector_type, body.credentials, org_id=org_id)
    register_sync_schedules(connector_id)

    logger.info("Registered %s connector %s", body.connector_type, connector_id)
    return ConnectorRegisterResponse(
        connector_id=connector_id, connector_type=body.connector_type, status="connected",
    )


@router.get("/api/connectors", response_model=ConnectorListResponse)
async def list_connectors(org_id: str = Depends(require_connector_org)) -> ConnectorListResponse:
    """List the caller org's active connectors (metadata only)."""
    from api.connector_store import list_active_connectors

    return ConnectorListResponse(connectors=await list_active_connectors(org_id=org_id))


@router.get("/api/connectors/types", response_model=ConnectorTypesResponse)
async def list_connector_types(_org_id: str = Depends(require_connector_org)) -> ConnectorTypesResponse:
    """Return all registered connector type identifiers."""
    from api.connectors.registry import list_connector_types as _list_types

    return ConnectorTypesResponse(types=list(_list_types()))


@router.delete("/api/connectors/{connector_id}", response_model=ConnectorDisconnectResponse)
async def disconnect_connector(
    connector_id: str,
    request: Request,
    org_id: str = Depends(require_connector_org),
) -> ConnectorDisconnectResponse:
    """Soft-delete one of the caller org's connectors."""
    from api.connector_store import delete_connector
    from api.connectors.sync_jobs import deregister_sync_schedules

    if not await delete_connector(connector_id, org_id=org_id):
        raise HTTPException(status_code=404, detail="Connector not found.")
    _connector_cache(request).pop(connector_id, None)
    deregister_sync_schedules(connector_id)
    logger.info("Disconnected connector %s", connector_id)
    return ConnectorDisconnectResponse(status="disconnected", connector_id=connector_id)


@router.post("/api/connectors/{connector_id}/orders", response_model=ConnectorOrdersResponse)
async def fetch_connector_orders(
    connector_id: str,
    body: ConnectorDateRangeRequest,
    request: Request,
    org_id: str = Depends(require_connector_org),
) -> ConnectorOrdersResponse:
    """Fetch live orders from one of the caller org's connectors."""
    connector = await _owned_connector(request, connector_id, org_id)
    try:
        orders = await connector.fetch_orders(body.start_date, body.end_date)
    except Exception as exc:  # connector HTTP stacks raise heterogeneous errors
        logger.error("fetch_orders failed for %s: %s", connector_id, type(exc).__name__)
        raise HTTPException(status_code=502, detail="Failed to fetch orders from the marketplace.")
    return ConnectorOrdersResponse(connector_id=connector_id, order_count=len(orders), orders=orders)


@router.post("/api/connectors/{connector_id}/financials", response_model=ConnectorFinancialsResponse)
async def fetch_connector_financials(
    connector_id: str,
    body: ConnectorDateRangeRequest,
    request: Request,
    org_id: str = Depends(require_connector_org),
) -> ConnectorFinancialsResponse:
    """Fetch live financial events from one of the caller org's connectors."""
    connector = await _owned_connector(request, connector_id, org_id)
    try:
        financials = await connector.fetch_financials(body.start_date, body.end_date)
    except Exception as exc:  # connector HTTP stacks raise heterogeneous errors
        logger.error("fetch_financials failed for %s: %s", connector_id, type(exc).__name__)
        raise HTTPException(status_code=502, detail="Failed to fetch financials from the marketplace.")
    return ConnectorFinancialsResponse(connector_id=connector_id, financials=financials)


@router.post("/api/connectors/{connector_id}/sync")
async def sync_connector(
    connector_id: str,
    body: ConnectorDateRangeRequest,
    request: Request,
    org_id: str = Depends(require_connector_org),
) -> dict[str, Any]:
    """Fetch orders + financials and persist them to the commerce tables."""
    from api.commerce_service import sync_connector_data
    from api.connector_store import update_last_sync

    connector = await _owned_connector(request, connector_id, org_id)
    try:
        summary = await sync_connector_data(connector_id, connector, body.start_date, body.end_date)
    except Exception as exc:  # connector HTTP stacks raise heterogeneous errors
        logger.error("sync failed for %s: %s", connector_id, type(exc).__name__)
        raise HTTPException(status_code=502, detail="Sync failed. Please try again later.")
    await update_last_sync(connector_id)
    return summary


# ---------------------------------------------------------------------------
# Commerce query routes (persisted data)
# ---------------------------------------------------------------------------


@router.get("/api/commerce/{connector_id}/orders", response_model=ConnectorOrdersResponse)
async def get_commerce_orders(
    connector_id: str,
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
    org_id: str = Depends(require_connector_org),
) -> ConnectorOrdersResponse:
    """Query persisted orders for one of the caller org's connectors."""
    from api.commerce_service import query_orders

    await _require_owned_connector_id(connector_id, org_id)
    orders = await query_orders(connector_id, limit=limit, offset=offset, status=status)
    return ConnectorOrdersResponse(connector_id=connector_id, order_count=len(orders), orders=orders)


@router.get("/api/commerce/{connector_id}/financials", response_model=ConnectorFinancialsResponse)
async def get_commerce_financials(
    connector_id: str,
    limit: int = 50,
    offset: int = 0,
    org_id: str = Depends(require_connector_org),
) -> ConnectorFinancialsResponse:
    """Query persisted financials for one of the caller org's connectors."""
    from api.commerce_service import query_financials

    await _require_owned_connector_id(connector_id, org_id)
    financials = await query_financials(connector_id, limit=limit, offset=offset)
    return ConnectorFinancialsResponse(connector_id=connector_id, financials=financials)


# ---------------------------------------------------------------------------
# Shopify webhook — public, HMAC-authenticated (DOCBOT-705)
# ---------------------------------------------------------------------------


def _find_shopify_connector(connectors: dict[str, Any], shop_domain: str) -> Optional[tuple[str, Any]]:
    from api.connectors.shopify_connector import InvalidShopDomainError, validate_shop_domain

    for cid, conn in connectors.items():
        if getattr(conn, "connector_type", None) != "shopify":
            continue
        try:
            cred_domain = validate_shop_domain(conn._creds.credentials.get("shop_domain", ""))
        except InvalidShopDomainError:
            continue
        if cred_domain == shop_domain:
            return cid, conn
    return None


@router.post("/api/marketplace/webhook/shopify", response_model=WebhookAck)
async def shopify_webhook(request: Request) -> WebhookAck:
    """Handle a Shopify webhook (HMAC-SHA256 verified) and trigger an incremental sync."""
    from api.commerce_service import sync_connector_data
    from api.connectors.shopify_connector import (
        InvalidShopDomainError,
        validate_shop_domain,
        verify_shopify_hmac,
    )

    body = await request.body()
    hmac_header = request.headers.get("X-Shopify-Hmac-SHA256", "")
    raw_domain = request.headers.get("X-Shopify-Shop-Domain", "")
    topic = request.headers.get("X-Shopify-Topic", "")[:128]

    if not hmac_header or not raw_domain:
        raise HTTPException(status_code=401, detail="Missing Shopify webhook headers.")
    try:
        shop_domain = validate_shop_domain(raw_domain)
    except InvalidShopDomainError:
        raise HTTPException(status_code=401, detail="Invalid Shopify webhook.")

    match = _find_shopify_connector(_connector_cache(request), shop_domain)
    if match is None:
        raise HTTPException(status_code=404, detail="No connector for this shop.")
    connector_id, connector = match

    webhook_secret = connector._creds.credentials.get("webhook_secret", "")
    if not webhook_secret or not verify_shopify_hmac(body, webhook_secret, hmac_header):
        raise HTTPException(status_code=401, detail="Invalid Shopify webhook.")

    logger.info("Shopify webhook received: topic=%s connector=%s", topic, connector_id)

    now = datetime.now(timezone.utc)
    try:
        summary = await sync_connector_data(
            connector_id, connector, (now - timedelta(hours=24)).isoformat(), now.isoformat(),
        )
        logger.info("Shopify webhook sync complete for %s: %s", connector_id, summary)
    except Exception as exc:  # sync failure must not make Shopify retry-storm the webhook
        logger.error("Shopify webhook sync failed for %s: %s", connector_id, type(exc).__name__)

    return WebhookAck(status="ok", topic=topic)
