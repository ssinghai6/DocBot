"""Server-side activation funnel events — DOCBOT-1706.

One row per (user, event) in ``funnel_events``: the first time a user signs
up, verifies, uploads, gets an answer, gets a tie-out, hits the paywall,
starts checkout or pays. Anonymous callers are keyed by ``anon_owner`` (the
anon cookie digest) instead. Two unique indexes enforce the dedupe; inserts use
``ON CONFLICT DO NOTHING`` and an in-process seen-set skips repeat writes.

``record_funnel_event`` is fire-and-forget: it schedules the insert on the
running loop and never raises, so a funnel write can never fail a request.
``record_checkout_started`` / ``record_paid`` are the hooks the Stripe ticket
calls.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from pydantic import BaseModel
from sqlalchemy import JSON, Column, DateTime, Index, String, Table, func, select
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

# Funnel order — the admin report computes step conversion in this order.
FUNNEL_EVENTS: tuple[str, ...] = (
    "signup",
    "email_verified",
    "first_upload",
    "first_answer",
    "first_tieout",
    "paywall_shown",
    "checkout_started",
    "paid",
)

# Idempotent; covers DBs where funnel_events predates its indexes.
FUNNEL_EVENTS_DDL: list[str] = [
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_funnel_user_event ON funnel_events (user_id, event)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_funnel_anon_event ON funnel_events (anon_owner, event)",
]

_SEEN_MAX = 50_000


def register_funnel_events_table(metadata: Any) -> Table:
    return Table(
        "funnel_events",
        metadata,
        Column("id", String, primary_key=True),
        Column("user_id", String),
        Column("org_id", String),
        Column("anon_owner", String),
        Column("event", String, nullable=False),
        Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False, index=True),
        Column("props", JSON),
        # NULLs are distinct in both indexes, so user rows and anon rows never collide.
        Index("uq_funnel_user_event", "user_id", "event", unique=True),
        Index("uq_funnel_anon_event", "anon_owner", "event", unique=True),
    )


_table: Optional[Table] = None
_session_factory: Any = None
_dialect: str = "postgresql"
_seen: set[tuple[str, str]] = set()
_tasks: set[asyncio.Task] = set()


def wire_funnel_store(table: Table, async_session_factory: Any, dialect: str) -> None:
    global _table, _session_factory, _dialect
    _table, _session_factory, _dialect = table, async_session_factory, dialect
    _seen.clear()


def _insert() -> Any:
    if _dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as dialect_insert
    else:
        from sqlalchemy.dialects.postgresql import insert as dialect_insert
    return dialect_insert(_table)


async def _persist(row: dict, seen_key: tuple[str, str]) -> None:
    try:
        async with _session_factory() as session, session.begin():
            await session.execute(_insert().values(**row).on_conflict_do_nothing())
    except (SQLAlchemyError, OSError) as exc:
        _seen.discard(seen_key)  # allow a later retry
        logger.warning("funnel_service: insert %s failed (%s)", row["event"], type(exc).__name__)


def record_funnel_event(
    event: str,
    *,
    user_id: Optional[str] = None,
    org_id: Optional[str] = None,
    anon_owner: Optional[str] = None,
    props: Optional[dict] = None,
) -> None:
    """Record ``event`` once per user (or anon owner). Fire-and-forget, never raises."""
    try:
        if event not in FUNNEL_EVENTS:
            logger.warning("funnel_service: unknown event %r", event)
            return
        if user_id:
            anon_owner = None
        key = user_id or anon_owner
        if not key or _table is None or _session_factory is None:
            return
        seen_key = (key, event)
        if seen_key in _seen:
            return
        loop = asyncio.get_running_loop()
        if len(_seen) >= _SEEN_MAX:
            _seen.clear()
        _seen.add(seen_key)
        row = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "org_id": org_id,
            "anon_owner": anon_owner,
            "event": event,
            "created_at": datetime.now(timezone.utc),
            "props": props or None,
        }
        task = loop.create_task(_persist(row, seen_key))
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
    except RuntimeError:
        logger.debug("funnel_service: no running loop; dropping %s", event)


def record_for_scope(event: str, scope: Any, props: Optional[dict] = None) -> None:
    """``record_funnel_event`` keyed by a request's ``CallerScope``."""
    record_funnel_event(
        event,
        user_id=getattr(scope, "user_id", None),
        org_id=getattr(scope, "org_id", None),
        anon_owner=getattr(scope, "anon_token", None),
        props=props,
    )


def record_checkout_started(user_id: Optional[str], org_id: Optional[str], props: Optional[dict] = None) -> None:
    """Hook for the Stripe checkout-session route."""
    record_funnel_event("checkout_started", user_id=user_id, org_id=org_id, props=props)


def record_paid(user_id: Optional[str], org_id: Optional[str], props: Optional[dict] = None) -> None:
    """Hook for the Stripe payment-succeeded webhook."""
    record_funnel_event("paid", user_id=user_id, org_id=org_id, props=props)


# ---------------------------------------------------------------------------
# Admin report
# ---------------------------------------------------------------------------


class FunnelStep(BaseModel):
    event: str
    count: int
    # % of the previous step's count; None for the first step or a zero denominator.
    conversion_pct: Optional[float] = None


class FunnelReport(BaseModel):
    days: int
    since: datetime
    steps: list[FunnelStep]


async def get_funnel_report(days: int) -> FunnelReport:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    counts: dict[str, int] = {}
    if _table is not None and _session_factory is not None:
        async with _session_factory() as session:
            rows = await session.execute(
                select(_table.c.event, func.count())
                .where(_table.c.created_at >= since)
                .group_by(_table.c.event)
            )
            counts = {event: int(n) for event, n in rows.all()}
    steps: list[FunnelStep] = []
    prev: Optional[int] = None
    for event in FUNNEL_EVENTS:
        n = counts.get(event, 0)
        pct = round(100.0 * n / prev, 1) if prev else None
        steps.append(FunnelStep(event=event, count=n, conversion_pct=pct))
        prev = n
    return FunnelReport(days=days, since=since, steps=steps)
