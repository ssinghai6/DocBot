"""Pro subscription lifecycle + dunning — DOCBOT-1810 (EPIC-18).

Pro = ``STRIPE_PRO_PRICE_ID`` (a recurring Stripe price, $79/mo), 25k credits
per paid invoice. State lives in ``org_subscriptions`` (one row per org) and is
driven only by verified Stripe webhooks (``api/routes/billing_routes.py``):

* ``customer.subscription.created|updated|deleted`` -> status from Stripe.
* ``invoice.paid`` -> ``active`` + grant :data:`PRO_MONTHLY_CREDITS` once per
  invoice (ledger ``ref`` = invoice id, ``UNIQUE(kind, ref)`` = idempotent).
* ``invoice.payment_failed`` -> ``past_due``.

Dunning: ``past_due`` keeps Pro access while Stripe retries the card (Smart
Retries). When retries are exhausted Stripe cancels the subscription
(dashboard setting) -> ``customer.subscription.deleted`` -> ``canceled`` ->
the org is back on the free plan. Credits already granted are kept.

Events can arrive out of order; a row only moves to the state of an event
whose ``created`` is >= the last one applied.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from sqlalchemy import BigInteger, Column, DateTime, String, Table, select

from api import credit_service

logger = logging.getLogger(__name__)

PRO_MONTHLY_CREDITS = 25_000
SUBSCRIPTION_EVENTS = frozenset({
    "customer.subscription.created",
    "customer.subscription.updated",
    "customer.subscription.deleted",
    "invoice.paid",
    "invoice.payment_failed",
})
# Only the first invoice and renewals carry the monthly grant (not prorations).
GRANT_BILLING_REASONS = frozenset({"subscription_create", "subscription_cycle"})

SubscriptionState = Literal["active", "past_due", "canceled"]
# Stripe status -> ours. Anything else (incomplete, unpaid, paused, ...) has no Pro access.
_STATE: dict[str, SubscriptionState] = {"active": "active", "trialing": "active", "past_due": "past_due"}

_table: Optional[Table] = None
_session_factory: Any = None


def pro_price_id() -> str:
    return os.getenv("STRIPE_PRO_PRICE_ID", "").strip()


def register_subscription_table(metadata: Any) -> Table:
    return Table(
        "org_subscriptions",
        metadata,
        Column("org_id", String, primary_key=True),
        Column("stripe_subscription_id", String, index=True),
        Column("stripe_customer_id", String),
        Column("status", String, nullable=False),
        Column("current_period_end", DateTime(timezone=True)),
        Column("last_event_created", BigInteger, nullable=False),  # Stripe event.created (unix s)
        Column("updated_at", DateTime(timezone=True), nullable=False),
    )


def wire_subscription_store(table: Table, async_session_factory: Any) -> None:
    global _table, _session_factory
    _table = table
    _session_factory = async_session_factory


def is_wired() -> bool:
    return _table is not None and _session_factory is not None


def stripe_state(status: Optional[str]) -> SubscriptionState:
    return _STATE.get(status or "", "canceled")


def plan_for(status: Optional[str]) -> Literal["pro", "free"]:
    """``past_due`` keeps Pro during Stripe's retry window."""
    return "pro" if status in ("active", "past_due") else "free"


@dataclass(frozen=True)
class SubscriptionEvent:
    subscription_id: str
    org_id: Optional[str]
    user_id: Optional[str]
    customer_id: Optional[str]
    state: SubscriptionState
    current_period_end: Optional[datetime]
    created: int
    grant_invoice_id: Optional[str] = None
    first_invoice: bool = False


def _ts(value: Any) -> Optional[datetime]:
    return datetime.fromtimestamp(value, tz=timezone.utc) if isinstance(value, int) else None


def _invoice_subscription(inv: dict) -> tuple[str, dict]:
    """Subscription id + its metadata across Stripe API versions (pre/post 2025-03 ``parent``)."""
    details = ((inv.get("parent") or {}).get("subscription_details")) or inv.get("subscription_details") or {}
    sub = details.get("subscription") or inv.get("subscription") or ""
    if isinstance(sub, dict):
        sub = sub.get("id") or ""
    return str(sub), details.get("metadata") or {}


def parse_event(event: dict) -> Optional[SubscriptionEvent]:
    """The state change a verified event implies, or None (not a Pro subscription event)."""
    etype = event.get("type")
    if etype not in SUBSCRIPTION_EVENTS:
        return None
    obj = (event.get("data") or {}).get("object") or {}
    created = event.get("created") if isinstance(event.get("created"), int) else 0
    state: SubscriptionState
    if etype.startswith("customer.subscription."):
        items = ((obj.get("items") or {}).get("data")) or [{}]
        meta = obj.get("metadata") or {}
        sub_id = obj.get("id") or ""
        state = "canceled" if etype.endswith(".deleted") else stripe_state(obj.get("status"))
        period_end = _ts(obj.get("current_period_end")) or _ts(items[0].get("current_period_end"))
        grant = None
        first = False
    else:
        sub_id, meta = _invoice_subscription(obj)
        state = "active" if etype == "invoice.paid" else "past_due"
        period_end = None
        first = obj.get("billing_reason") == "subscription_create"
        grant = (
            obj.get("id")
            if etype == "invoice.paid" and obj.get("billing_reason") in GRANT_BILLING_REASONS
            else None
        )
    if not sub_id:
        return None  # one-off invoice, not a subscription
    customer = obj.get("customer")
    return SubscriptionEvent(
        subscription_id=sub_id,
        org_id=meta.get("org_id") or None,
        user_id=meta.get("user_id") or None,
        customer_id=customer if isinstance(customer, str) else None,
        state=state,
        current_period_end=period_end,
        created=created,
        grant_invoice_id=grant,
        first_invoice=first,
    )


async def _org_for_subscription(session: Any, subscription_id: str) -> Optional[str]:
    assert _table is not None
    row = (
        await session.execute(select(_table.c.org_id).where(_table.c.stripe_subscription_id == subscription_id))
    ).first()
    return row.org_id if row else None


async def apply_event(ev: SubscriptionEvent) -> Optional[str]:
    """Persist ``ev`` (newest event wins) and grant invoice credits. Returns the org id or None.

    Raises ``credit_service.UnknownOrg`` / ``SQLAlchemyError`` like ``grant_purchase``.
    """
    assert _table is not None
    t = _table
    now = datetime.now(timezone.utc)
    async with _session_factory() as session:
        async with session.begin():
            org_id = ev.org_id or await _org_for_subscription(session, ev.subscription_id)
            if not org_id:
                logger.warning("subscription_service: no org for subscription event")
                return None
            row = (
                await session.execute(select(t).where(t.c.org_id == org_id).with_for_update())
            ).first()
            values = {
                "stripe_subscription_id": ev.subscription_id,
                "status": ev.state,
                "last_event_created": ev.created,
                "updated_at": now,
            }
            if ev.customer_id:
                values["stripe_customer_id"] = ev.customer_id
            if ev.current_period_end:
                values["current_period_end"] = ev.current_period_end
            if row is None:
                await session.execute(t.insert().values(org_id=org_id, **values))
            elif ev.created >= row.last_event_created:
                await session.execute(t.update().where(t.c.org_id == org_id).values(**values))
    if ev.grant_invoice_id:
        result = await credit_service.grant_purchase(
            org_id, PRO_MONTHLY_CREDITS, ev.grant_invoice_id, "Pro monthly credits"
        )
        if ev.first_invoice and not result.duplicate:
            from api import funnel_service

            funnel_service.record_paid(ev.user_id, org_id, {"plan": "pro", "credits": PRO_MONTHLY_CREDITS})
    logger.info("subscription_service: org=%s state=%s", org_id, ev.state)
    return org_id


async def get_subscription(org_id: str) -> Optional[dict[str, Any]]:
    if not is_wired():
        return None
    assert _table is not None
    async with _session_factory() as session:
        row = (await session.execute(select(_table).where(_table.c.org_id == org_id))).first()
    return dict(row._mapping) if row else None
