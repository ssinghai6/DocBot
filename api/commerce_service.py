"""Commerce data persistence — DOCBOT-702.

Unified commerce schema with multi-tenant isolation via mandatory
connection_id filtering. Tables are defined here and registered with
the shared SQLAlchemy metadata from index.py at import time.

Tables:
    commerce_orders      — normalized orders from any marketplace connector
    commerce_financials  — normalized financial events (revenue, refunds, fees)

Every read/write function requires a connection_id parameter — this is
the RLS boundary.  There is no way to query across connections.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Table,
    Text,
    func,
    select,
    text,
)
from sqlalchemy import insert as sa_insert

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Table definitions — attached to shared metadata at import time
# ---------------------------------------------------------------------------

def register_commerce_tables(metadata):
    """Define commerce tables on the shared metadata object.

    Called once during app startup (imported by index.py).
    Returns the table objects for use in service functions.
    """
    commerce_orders = Table(
        "commerce_orders", metadata,
        Column("id", String, primary_key=True),
        Column("connection_id", String, nullable=False, index=True),
        Column("connector_type", String, nullable=False),
        Column("marketplace_order_id", String, nullable=False),
        Column("status", String, nullable=False),
        Column("total_amount", Float, nullable=False),
        Column("currency", String, server_default="USD"),
        Column("order_date", DateTime(timezone=True), index=True),
        Column("customer_id", String),
        Column("raw_json", Text),
        Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
        Column("updated_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
        # Composite unique: one order per marketplace per connection
        Index("uq_commerce_orders_conn_mkt",
              "connection_id", "marketplace_order_id", unique=True),
    )

    commerce_financials = Table(
        "commerce_financials", metadata,
        Column("id", String, primary_key=True),
        Column("connection_id", String, nullable=False, index=True),
        Column("connector_type", String, nullable=False),
        Column("period_start", DateTime(timezone=True), nullable=False, index=True),
        Column("period_end", DateTime(timezone=True), nullable=False),
        Column("revenue", Float, nullable=False),
        Column("refunds", Float, nullable=False),
        Column("fees", Float, nullable=False),
        Column("net_proceeds", Float, nullable=False),
        Column("currency", String, server_default="USD"),
        Column("raw_json", Text),
        Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    )

    return commerce_orders, commerce_financials


# ---------------------------------------------------------------------------
# Module-level table references — set by wire_commerce()
# ---------------------------------------------------------------------------

_orders_table: Optional[Table] = None
_financials_table: Optional[Table] = None
_async_session_factory: Any = None


def wire_commerce(orders_table, financials_table, async_session_factory):
    """Inject table + session references.  Called once from index.py lifespan."""
    global _orders_table, _financials_table, _async_session_factory
    _orders_table = orders_table
    _financials_table = financials_table
    _async_session_factory = async_session_factory


# ---------------------------------------------------------------------------
# Pydantic models for API routes
# ---------------------------------------------------------------------------

class CommerceSyncRequest(BaseModel):
    start_date: str  # ISO 8601
    end_date: str


class CommerceQueryParams(BaseModel):
    limit: int = 50
    offset: int = 0
    status: Optional[str] = None


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _parse_dt(val: Any) -> Optional[datetime]:
    """Best-effort parse of a date/datetime string."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


async def persist_orders(
    connection_id: str,
    connector_type: str,
    orders: list[dict[str, Any]],
) -> int:
    """Upsert normalized orders.  Returns count of rows written."""
    if not orders or _orders_table is None or _async_session_factory is None:
        return 0

    rows_written = 0
    async with _async_session_factory() as session:
        async with session.begin():
            for order in orders:
                row_id = str(uuid.uuid4())
                mkt_id = str(order.get("marketplace_order_id", ""))
                if not mkt_id:
                    continue

                # Upsert: try insert, on conflict update
                insert_stmt = sa_insert(_orders_table).values(
                    id=row_id,
                    connection_id=connection_id,
                    connector_type=connector_type,
                    marketplace_order_id=mkt_id,
                    status=str(order.get("status", "unknown")),
                    total_amount=float(order.get("total_amount", 0)),
                    currency=str(order.get("currency", "USD")),
                    order_date=_parse_dt(order.get("order_date")),
                    customer_id=str(order.get("customer_id", "")) or None,
                    raw_json=json.dumps(order.get("raw_json")) if order.get("raw_json") else None,
                    updated_at=datetime.now(timezone.utc),
                )
                # PostgreSQL ON CONFLICT upsert
                from sqlalchemy.dialects.postgresql import insert as pg_insert
                pg_stmt = pg_insert(_orders_table).values(
                    id=row_id,
                    connection_id=connection_id,
                    connector_type=connector_type,
                    marketplace_order_id=mkt_id,
                    status=str(order.get("status", "unknown")),
                    total_amount=float(order.get("total_amount", 0)),
                    currency=str(order.get("currency", "USD")),
                    order_date=_parse_dt(order.get("order_date")),
                    customer_id=str(order.get("customer_id", "")) or None,
                    raw_json=json.dumps(order.get("raw_json")) if order.get("raw_json") else None,
                    updated_at=datetime.now(timezone.utc),
                ).on_conflict_do_update(
                    index_elements=["connection_id", "marketplace_order_id"],
                    set_={
                        "status": str(order.get("status", "unknown")),
                        "total_amount": float(order.get("total_amount", 0)),
                        "currency": str(order.get("currency", "USD")),
                        "order_date": _parse_dt(order.get("order_date")),
                        "customer_id": str(order.get("customer_id", "")) or None,
                        "raw_json": json.dumps(order.get("raw_json")) if order.get("raw_json") else None,
                        "updated_at": datetime.now(timezone.utc),
                    },
                )
                try:
                    await session.execute(pg_stmt)
                    rows_written += 1
                except Exception:
                    # Fallback for non-PostgreSQL (tests with SQLite)
                    try:
                        await session.execute(insert_stmt)
                        rows_written += 1
                    except Exception as exc:
                        logger.warning("persist_orders row skip: %s", exc)

    logger.info("persist_orders: wrote %d/%d for connection %s", rows_written, len(orders), connection_id)
    return rows_written


async def persist_financials(
    connection_id: str,
    connector_type: str,
    financials: list[dict[str, Any]],
) -> int:
    """Insert financial period records.  Returns count of rows written."""
    if not financials or _financials_table is None or _async_session_factory is None:
        return 0

    rows_written = 0
    async with _async_session_factory() as session:
        async with session.begin():
            for fin in financials:
                row_id = str(uuid.uuid4())
                try:
                    await session.execute(
                        sa_insert(_financials_table).values(
                            id=row_id,
                            connection_id=connection_id,
                            connector_type=connector_type,
                            period_start=_parse_dt(fin.get("period_start")),
                            period_end=_parse_dt(fin.get("period_end")),
                            revenue=float(fin.get("revenue", 0)),
                            refunds=float(fin.get("refunds", 0)),
                            fees=float(fin.get("fees", 0)),
                            net_proceeds=float(fin.get("net_proceeds", 0)),
                            currency=str(fin.get("currency", "USD")),
                            raw_json=json.dumps(fin.get("raw_json")) if fin.get("raw_json") else None,
                        )
                    )
                    rows_written += 1
                except Exception as exc:
                    logger.warning("persist_financials row skip: %s", exc)

    logger.info("persist_financials: wrote %d/%d for connection %s", rows_written, len(financials), connection_id)
    return rows_written


# ---------------------------------------------------------------------------
# Query helpers — mandatory connection_id (RLS boundary)
# ---------------------------------------------------------------------------

async def query_orders(
    connection_id: str,
    *,
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Fetch orders for a specific connection.  connection_id is mandatory (RLS)."""
    if _orders_table is None or _async_session_factory is None:
        return []

    stmt = (
        select(_orders_table)
        .where(_orders_table.c.connection_id == connection_id)
        .order_by(_orders_table.c.order_date.desc())
        .limit(limit)
        .offset(offset)
    )
    if status:
        stmt = stmt.where(_orders_table.c.status == status)

    async with _async_session_factory() as session:
        result = await session.execute(stmt)
        rows = result.mappings().all()

    return [
        {
            "id": r["id"],
            "marketplace_order_id": r["marketplace_order_id"],
            "status": r["status"],
            "total_amount": r["total_amount"],
            "currency": r["currency"],
            "order_date": r["order_date"].isoformat() if r["order_date"] else None,
            "customer_id": r["customer_id"],
            "connector_type": r["connector_type"],
        }
        for r in rows
    ]


async def query_financials(
    connection_id: str,
    *,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Fetch financial records for a specific connection.  connection_id is mandatory (RLS)."""
    if _financials_table is None or _async_session_factory is None:
        return []

    stmt = (
        select(_financials_table)
        .where(_financials_table.c.connection_id == connection_id)
        .order_by(_financials_table.c.period_start.desc())
        .limit(limit)
        .offset(offset)
    )

    async with _async_session_factory() as session:
        result = await session.execute(stmt)
        rows = result.mappings().all()

    return [
        {
            "id": r["id"],
            "period_start": r["period_start"].isoformat() if r["period_start"] else None,
            "period_end": r["period_end"].isoformat() if r["period_end"] else None,
            "revenue": r["revenue"],
            "refunds": r["refunds"],
            "fees": r["fees"],
            "net_proceeds": r["net_proceeds"],
            "currency": r["currency"],
            "connector_type": r["connector_type"],
        }
        for r in rows
    ]


async def get_order_count(connection_id: str) -> int:
    """Return total order count for a connection."""
    if _orders_table is None or _async_session_factory is None:
        return 0

    stmt = (
        select(func.count())
        .select_from(_orders_table)
        .where(_orders_table.c.connection_id == connection_id)
    )
    async with _async_session_factory() as session:
        result = await session.execute(stmt)
        return result.scalar() or 0


# ---------------------------------------------------------------------------
# Sync orchestrator — fetch from connector + persist
# ---------------------------------------------------------------------------

async def sync_connector_data(
    connector_id: str,
    connector: Any,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """Fetch orders + financials from connector and persist to DB.

    Returns summary dict with counts.
    """
    connector_type = connector.connector_type

    orders = await connector.fetch_orders(start_date, end_date)
    financials = await connector.fetch_financials(start_date, end_date)

    orders_written = await persist_orders(connector_id, connector_type, orders)
    financials_written = await persist_financials(connector_id, connector_type, financials)

    return {
        "connector_id": connector_id,
        "connector_type": connector_type,
        "orders_fetched": len(orders),
        "orders_persisted": orders_written,
        "financials_fetched": len(financials),
        "financials_persisted": financials_written,
        "period": {"start": start_date, "end": end_date},
    }
