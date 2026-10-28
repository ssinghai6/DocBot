"""Credit balance, reservations and ledger — DOCBOT-1802 (EPIC-18).

Every LLM-spending request runs inside a *billing run*:

1. **Reserve** — before any provider call, one atomic statement moves an
   up-front estimate from available to reserved::

       UPDATE org_balances
          SET reserved_credits = reserved_credits + :est
        WHERE org_id = :org
          AND balance_credits - reserved_credits >= :est
          AND NOT frozen
       RETURNING ...

   Postgres re-evaluates the WHERE clause against the latest committed row
   after taking the row lock (READ COMMITTED), and SQLite serializes writers
   on the database lock, so concurrent reservations can never oversubscribe
   a balance. No row back -> HTTP 402 ``insufficient_credits`` (or
   ``account_frozen``) when ``BILLING_ENFORCE`` is on.

2. **Settle** — in a ``finally`` (stream end, client disconnect or
   exception), the actual raw provider cost from ``RequestMeter.snapshot()``
   is converted to retail credits (``credits_for_cost`` x ``BILLING_MARKUP``),
   the reservation is released, the balance is debited and one ``usage``
   ledger row is written with ``ref = run_id``.

3. **Reap** — reservations whose settle never ran (process crash, generator
   never started) are released after ``BILLING_RESERVATION_TTL_MINUTES`` by
   :func:`reap_stale_reservations` (startup + APScheduler interval job).

Lazy monthly free grant: the first billed request of a calendar month (UTC)
inserts ``free_grant`` / ``grant:<org>:<YYYY-MM>``; ``UNIQUE(kind, ref)``
on the append-only ledger makes a double grant impossible even across
replicas.

Shadow mode (default): with ``BILLING_ENFORCE`` unset/false the full
reserve/settle/ledger cycle runs but a failed reservation only logs
``would_block`` — the request is never refused. Billing-store errors fail
open (logged) in both modes so a ledger outage cannot take chat down.

Exempt (no reservation, no ledger): anonymous callers (no user/org — gated by
the request_guard budgets / DOCBOT-1711 demo quota instead) and requests that
only touch server-generated demo resources.

Balances never go negative: a settle that costs more than the remaining
balance debits what is left and logs the forgiven overrun (bounded by the
per-run estimate + the free-tier per-run ceiling). The ledger ``delta``
is always the amount actually applied, so ``SUM(delta) == balance``.

Units: credits are integers, 1 credit = $0.001 retail (``api/billing_models``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from types import TracebackType
from typing import Any, AsyncIterator, Optional

from fastapi import HTTPException, Request
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Index,
    String,
    Table,
    UniqueConstraint,
    and_,
    case,
    false,
    select,
    update,
)
from sqlalchemy.exc import SQLAlchemyError

from api.billing_models import (
    LedgerEntryKind,
    QuotaErrorCode,
    QuotaErrorPayload,
    credits_for_cost,
)
from api.utils.caller_scope import CallerScope
from api.funnel_service import record_funnel_event
from api.utils.billing_context import (
    BillingContextSnapshot,
    Principal,
    RequestMeter,
    Role,
    capture_billing_context,
    reset_meter,
    reset_principal,
    set_meter,
    set_principal,
    use_billing_context,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (env, with safe defaults)
# ---------------------------------------------------------------------------

# Direct (raw provider) cost must be <= 30% of the retail price, so the markup
# can never drop below 1 / 0.30.
MIN_MARKUP = Decimal(10) / Decimal(3)
DEFAULT_MARKUP = Decimal("3.5")  # raw cost = 28.6% of price
DEFAULT_FREE_MONTHLY_CREDITS = 500  # $0.50 retail -> <= $0.143 raw cost per org/month
DEFAULT_FREE_RUN_CEILING_CREDITS = 150  # $0.15 retail max debit per free-tier run
DEFAULT_RESERVATION_TTL_MINUTES = 15
REAPER_INTERVAL_MINUTES = 5

# Up-front reservation per route, in credits. Override with
# BILLING_EST_CREDITS_<ROUTE> (e.g. BILLING_EST_CREDITS_AUTOPILOT=200).
DEFAULT_ESTIMATES: dict[str, int] = {
    "chat": 20,
    "db_chat": 30,
    "hybrid": 40,
    "sandbox": 50,
    "autopilot": 150,
}

_TRUTHY = {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        logger.warning("credit_service: %s=%r is not an integer; using %d", name, raw, default)
        return default
    return max(value, minimum)


def is_enforced() -> bool:
    """``BILLING_ENFORCE`` — refuse (402) on insufficient credits. Default off (shadow)."""
    return os.getenv("BILLING_ENFORCE", "").strip().lower() in _TRUTHY


def markup() -> Decimal:
    raw = os.getenv("BILLING_MARKUP")
    if raw is None or raw.strip() == "":
        return DEFAULT_MARKUP
    try:
        value = Decimal(raw.strip())
    except InvalidOperation:
        logger.warning("credit_service: BILLING_MARKUP=%r invalid; using %s", raw, DEFAULT_MARKUP)
        return DEFAULT_MARKUP
    if not value.is_finite() or value < MIN_MARKUP:
        logger.warning(
            "credit_service: BILLING_MARKUP=%r below the 30%% cost ceiling; clamped to %s",
            raw,
            MIN_MARKUP,
        )
        return MIN_MARKUP
    return value


def free_monthly_credits() -> int:
    return _env_int("BILLING_FREE_MONTHLY_CREDITS", DEFAULT_FREE_MONTHLY_CREDITS)


def free_run_ceiling_credits() -> int:
    return _env_int("BILLING_FREE_RUN_CEILING_CREDITS", DEFAULT_FREE_RUN_CEILING_CREDITS, minimum=1)


def reservation_ttl_minutes() -> int:
    return _env_int("BILLING_RESERVATION_TTL_MINUTES", DEFAULT_RESERVATION_TTL_MINUTES, minimum=1)


def estimate_for(route: str) -> int:
    default = DEFAULT_ESTIMATES.get(route, DEFAULT_ESTIMATES["chat"])
    return _env_int(f"BILLING_EST_CREDITS_{route.upper()}", default, minimum=1)


def cost_to_credits(cost_micro_usd: int) -> int:
    """Raw provider cost (micro-USD) -> retail credits, rounded up."""
    if cost_micro_usd <= 0:
        return 0
    return credits_for_cost(cost_micro_usd, markup())


def grant_ref(org_id: str, period: str) -> str:
    return f"grant:{org_id}:{period}"


def current_period(now: Optional[datetime] = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

LEDGER_KINDS = tuple(k.value for k in LedgerEntryKind)
_KIND_CHECK = "kind IN (" + ", ".join(f"'{k}'" for k in LEDGER_KINDS) + ")"


@dataclass(frozen=True)
class CreditTables:
    org_balances: Table
    credit_ledger: Table
    credit_reservations: Table


def register_credit_tables(metadata: Any) -> CreditTables:
    """Define the three billing tables on the shared metadata (``create_all`` is idempotent)."""
    org_balances = Table(
        "org_balances",
        metadata,
        Column("org_id", String, primary_key=True),
        Column("balance_credits", BigInteger, nullable=False, server_default="0"),
        Column("reserved_credits", BigInteger, nullable=False, server_default="0"),
        Column("updated_at", DateTime(timezone=True), nullable=False),
        Column("frozen", Boolean, nullable=False, server_default=false()),
    )
    credit_ledger = Table(
        "credit_ledger",
        metadata,
        Column("id", String, primary_key=True),  # UUID
        Column("org_id", String, nullable=False),
        Column("kind", String, nullable=False),
        Column("ref", String, nullable=False),
        Column("delta_credits", BigInteger, nullable=False),
        Column("balance_after_credits", BigInteger),
        Column("description", String),
        Column("created_at", DateTime(timezone=True), nullable=False),
        UniqueConstraint("kind", "ref", name="uq_credit_ledger_kind_ref"),
        CheckConstraint(_KIND_CHECK, name="ck_credit_ledger_kind"),
        Index("ix_credit_ledger_org_created", "org_id", "created_at"),
    )
    credit_reservations = Table(
        "credit_reservations",
        metadata,
        Column("id", String, primary_key=True),  # billing run_id
        Column("org_id", String, nullable=False),
        Column("route", String, nullable=False),
        Column("credits", BigInteger, nullable=False),
        Column("status", String, nullable=False),  # open | settled | released
        Column("actual_credits", BigInteger),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("closed_at", DateTime(timezone=True)),
        Index("ix_credit_reservations_status_created", "status", "created_at"),
    )
    return CreditTables(org_balances, credit_ledger, credit_reservations)


# Postgres-only: make credit_ledger append-only (mirrors audit_log, DOCBOT-602).
# Each entry is one statement — asyncpg rejects multi-statement strings.
LEDGER_IMMUTABILITY_STATEMENTS: list[str] = [
    """
    DO $outer$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_proc p
            JOIN pg_namespace n ON n.oid = p.pronamespace
            WHERE p.proname = 'credit_ledger_immutable'
              AND n.nspname = 'public'
        ) THEN
            EXECUTE $body$
                CREATE FUNCTION public.credit_ledger_immutable()
                RETURNS TRIGGER LANGUAGE plpgsql AS $$
                BEGIN
                    RAISE EXCEPTION 'credit_ledger rows are immutable';
                END;
                $$
            $body$;
        END IF;
    END
    $outer$
    """,
    """
    DO $outer$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_trigger t
            JOIN pg_class c ON c.oid = t.tgrelid
            WHERE t.tgname = 'credit_ledger_no_mutate'
              AND c.relname = 'credit_ledger'
        ) THEN
            EXECUTE $body$
                CREATE TRIGGER credit_ledger_no_mutate
                BEFORE UPDATE OR DELETE ON credit_ledger
                FOR EACH ROW EXECUTE FUNCTION public.credit_ledger_immutable()
            $body$;
        END IF;
    END
    $outer$
    """,
]


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

_tables: Optional[CreditTables] = None
_session_factory: Any = None
_dialect: str = "postgresql"
# org_id -> YYYY-MM already granted (per-process shortcut; the UNIQUE
# constraint is the real guard).
_granted_period: dict[str, str] = {}


def wire_credit_store(tables: CreditTables, async_session_factory: Any, dialect: str) -> None:
    """Inject tables + session factory. Called once from the index.py lifespan."""
    global _tables, _session_factory, _dialect
    _tables = tables
    _session_factory = async_session_factory
    _dialect = dialect
    _granted_period.clear()


def is_wired() -> bool:
    return _tables is not None and _session_factory is not None


def _insert(table: Table) -> Any:
    if _dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as dialect_insert
    else:
        from sqlalchemy.dialects.postgresql import insert as dialect_insert
    return dialect_insert(table)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Core operations (each its own transaction)
# ---------------------------------------------------------------------------


class InsufficientCredits(Exception):
    def __init__(self, required: int, available: int, frozen: bool) -> None:
        super().__init__("insufficient credits")
        self.required = required
        self.available = available
        self.frozen = frozen


@dataclass(frozen=True)
class ReserveResult:
    reserved_credits: int
    free_tier: bool


async def _ensure_row_and_grant(session: Any, org_id: str, now: datetime) -> str:
    """Upsert the balance row and apply the lazy monthly grant. Write-first.

    Returns the period; the caller records it in ``_granted_period`` only
    after the transaction commits.
    """
    assert _tables is not None
    ob = _tables.org_balances
    await session.execute(
        _insert(ob)
        .values(org_id=org_id, balance_credits=0, reserved_credits=0, updated_at=now, frozen=False)
        .on_conflict_do_nothing(index_elements=["org_id"])
    )
    period = current_period(now)
    if _granted_period.get(org_id) == period:
        return period
    amount = free_monthly_credits()
    inserted = await _credit_once(
        session, org_id, LedgerEntryKind.free_grant, grant_ref(org_id, period),
        amount, f"Monthly free credits {period}", now,
    )
    if inserted and amount > 0:
        logger.info("credit_service: monthly grant org=%s period=%s credits=%d", org_id, period, amount)
    return period


async def _credit_once(
    session: Any,
    org_id: str,
    kind: LedgerEntryKind,
    ref: str,
    amount: int,
    description: str,
    now: datetime,
) -> bool:
    """Write one positive ledger row and add it to the balance, at most once per (kind, ref).

    Caller holds the transaction and has already upserted the balance row.
    Returns False when ``(kind, ref)`` already exists (nothing applied).
    """
    assert _tables is not None
    ob, ledger = _tables.org_balances, _tables.credit_ledger
    current = (
        await session.execute(
            select(ob.c.balance_credits).where(ob.c.org_id == org_id).with_for_update()
        )
    ).scalar_one()
    inserted = (
        await session.execute(
            _insert(ledger)
            .values(
                id=str(uuid.uuid4()),
                org_id=org_id,
                kind=kind.value,
                ref=ref,
                delta_credits=amount,
                balance_after_credits=int(current) + amount,
                description=description,
                created_at=now,
            )
            .on_conflict_do_nothing(index_elements=["kind", "ref"])
            .returning(ledger.c.id)
        )
    ).first()
    if inserted is not None and amount > 0:
        await session.execute(
            update(ob)
            .where(ob.c.org_id == org_id)
            .values(balance_credits=ob.c.balance_credits + amount, updated_at=now)
        )
    return inserted is not None


@dataclass(frozen=True)
class GrantResult:
    duplicate: bool
    balance_credits: int


async def grant_purchase(org_id: str, credits: int, ref: str, description: str) -> GrantResult:
    """Credit a purchase (DOCBOT-1803 manual grant; DOCBOT-1804 webhook later).

    One transaction: balance row upsert + monthly grant + ``purchase`` ledger
    row + balance update. ``UNIQUE(kind, ref)`` makes a replayed ``ref`` a
    no-op (``duplicate=True``), so the operator can safely retry.
    """
    if not is_wired():
        raise RuntimeError("credit store not wired")
    if credits <= 0:
        raise ValueError("credits must be positive")
    assert _tables is not None
    ob = _tables.org_balances
    now = _now()
    async with _session_factory() as session:
        async with session.begin():
            period = await _ensure_row_and_grant(session, org_id, now)
            inserted = await _credit_once(
                session, org_id, LedgerEntryKind.purchase, ref, credits, description, now
            )
            balance = (
                await session.execute(select(ob.c.balance_credits).where(ob.c.org_id == org_id))
            ).scalar_one()
    _granted_period[org_id] = period
    logger.info(
        "credit_service: purchase org=%s credits=%d duplicate=%s", org_id, credits, not inserted
    )
    return GrantResult(duplicate=not inserted, balance_credits=int(balance))


async def _is_free_tier(session: Any, org_id: str) -> bool:
    assert _tables is not None
    ledger = _tables.credit_ledger
    row = (
        await session.execute(
            select(ledger.c.id)
            .where(ledger.c.org_id == org_id, ledger.c.kind == LedgerEntryKind.purchase.value)
            .limit(1)
        )
    ).first()
    return row is None


async def is_paid_org(org_id: str) -> bool:
    """True when ``org_id`` has ever purchased credits (DOCBOT-1615). False if unwired."""
    if not is_wired():
        return False
    async with _session_factory() as session:
        return not await _is_free_tier(session, org_id)


async def reserve(org_id: str, estimate: int, run_id: str, route: str) -> ReserveResult:
    """Atomically reserve ``estimate`` credits (capped for free tier).

    Raises :class:`InsufficientCredits` when the org cannot cover it.
    """
    if not is_wired():
        raise RuntimeError("credit store not wired")
    assert _tables is not None
    ob, res = _tables.org_balances, _tables.credit_reservations
    now = _now()
    failure: Optional[InsufficientCredits] = None
    async with _session_factory() as session:
        async with session.begin():
            period = await _ensure_row_and_grant(session, org_id, now)
            free_tier = await _is_free_tier(session, org_id)
            est = min(estimate, free_run_ceiling_credits()) if free_tier else estimate
            est = max(est, 1)
            row = (
                await session.execute(
                    update(ob)
                    .where(
                        ob.c.org_id == org_id,
                        (ob.c.balance_credits - ob.c.reserved_credits) >= est,
                        ob.c.frozen == false(),
                    )
                    .values(reserved_credits=ob.c.reserved_credits + est, updated_at=now)
                    .returning(ob.c.balance_credits, ob.c.reserved_credits)
                )
            ).first()
            if row is None:
                current = (
                    await session.execute(
                        select(ob.c.balance_credits, ob.c.reserved_credits, ob.c.frozen).where(
                            ob.c.org_id == org_id
                        )
                    )
                ).first()
                available = max(current.balance_credits - current.reserved_credits, 0) if current else 0
                frozen = bool(current.frozen) if current else False
                # Raised after the block so the balance row + monthly grant commit.
                failure = InsufficientCredits(required=est, available=available, frozen=frozen)
            else:
                await session.execute(
                    res.insert().values(
                        id=run_id,
                        org_id=org_id,
                        route=route,
                        credits=est,
                        status="open",
                        created_at=now,
                    )
                )
    _granted_period[org_id] = period
    if failure is not None:
        raise failure
    return ReserveResult(reserved_credits=est, free_tier=free_tier)


@dataclass(frozen=True)
class SettleResult:
    actual_credits: int
    applied_credits: int
    released_credits: int


async def settle(
    org_id: str,
    run_id: str,
    cost_micro_usd: int,
    *,
    free_tier: bool,
) -> SettleResult:
    """Release the run's reservation and debit the actual cost. Idempotent per run_id."""
    if not is_wired():
        raise RuntimeError("credit store not wired")
    assert _tables is not None
    ob, ledger, res = _tables.org_balances, _tables.credit_ledger, _tables.credit_reservations
    actual = cost_to_credits(cost_micro_usd)
    if free_tier:
        actual = min(actual, free_run_ceiling_credits())
    now = _now()
    async with _session_factory() as session:
        async with session.begin():
            # Write first (takes the SQLite write lock; open->settled only once).
            released_row = (
                await session.execute(
                    update(res)
                    .where(res.c.id == run_id, res.c.status == "open")
                    .values(status="settled", actual_credits=actual, closed_at=now)
                    .returning(res.c.credits)
                )
            ).first()
            released = int(released_row.credits) if released_row else 0
            bal = (
                await session.execute(
                    select(ob.c.balance_credits).where(ob.c.org_id == org_id).with_for_update()
                )
            ).first()
            if bal is None:
                return SettleResult(actual, 0, released)
            balance = int(bal.balance_credits)
            applied = min(actual, max(balance, 0))
            if actual > 0:
                inserted = (
                    await session.execute(
                        _insert(ledger)
                        .values(
                            id=str(uuid.uuid4()),
                            org_id=org_id,
                            kind=LedgerEntryKind.usage.value,
                            ref=run_id,
                            delta_credits=-applied,
                            balance_after_credits=balance - applied,
                            description=(
                                f"actual={actual}" if applied == actual
                                else f"actual={actual} applied={applied} (overrun forgiven)"
                            ),
                            created_at=now,
                        )
                        .on_conflict_do_nothing(index_elements=["kind", "ref"])
                        .returning(ledger.c.id)
                    )
                ).first()
                if inserted is None:  # duplicate settle — never debit twice
                    applied = 0
            await session.execute(
                update(ob)
                .where(ob.c.org_id == org_id)
                .values(
                    balance_credits=ob.c.balance_credits - applied,
                    reserved_credits=case(
                        (ob.c.reserved_credits >= released, ob.c.reserved_credits - released),
                        else_=0,
                    ),
                    updated_at=now,
                )
            )
    if applied < actual:
        logger.warning(
            "credit_service: overrun forgiven org=%s run=%s actual=%d applied=%d",
            org_id, run_id, actual, applied,
        )
    return SettleResult(actual, applied, released)


async def release_reservation(run_id: str) -> int:
    """Release an open reservation without debiting (e.g. request refused later)."""
    if not is_wired():
        return 0
    assert _tables is not None
    ob, res = _tables.org_balances, _tables.credit_reservations
    now = _now()
    async with _session_factory() as session:
        async with session.begin():
            row = (
                await session.execute(
                    update(res)
                    .where(res.c.id == run_id, res.c.status == "open")
                    .values(status="released", closed_at=now)
                    .returning(res.c.org_id, res.c.credits)
                )
            ).first()
            if row is None:
                return 0
            await _release_reserved(session, row.org_id, int(row.credits), now)
            return int(row.credits)


async def _release_reserved(session: Any, org_id: str, credits: int, now: datetime) -> None:
    assert _tables is not None
    ob = _tables.org_balances
    await session.execute(
        update(ob)
        .where(ob.c.org_id == org_id)
        .values(
            reserved_credits=case(
                (ob.c.reserved_credits >= credits, ob.c.reserved_credits - credits),
                else_=0,
            ),
            updated_at=now,
        )
    )


async def reap_stale_reservations(
    max_age_minutes: Optional[int] = None, *, now: Optional[datetime] = None
) -> int:
    """Release open reservations older than ``max_age_minutes``. Returns the count."""
    if not is_wired():
        return 0
    assert _tables is not None
    res = _tables.credit_reservations
    age = max_age_minutes if max_age_minutes is not None else reservation_ttl_minutes()
    current = now or _now()
    cutoff = current - timedelta(minutes=age)
    async with _session_factory() as session:
        async with session.begin():
            rows = (
                await session.execute(
                    update(res)
                    .where(and_(res.c.status == "open", res.c.created_at < cutoff))
                    .values(status="released", closed_at=current)
                    .returning(res.c.org_id, res.c.credits)
                )
            ).all()
            per_org: dict[str, int] = {}
            for r in rows:
                per_org[r.org_id] = per_org.get(r.org_id, 0) + int(r.credits)
            for org_id, credits in per_org.items():
                await _release_reserved(session, org_id, credits, current)
    if rows:
        logger.warning("credit_service: reaped %d stale reservation(s)", len(rows))
    return len(rows)


async def run_reaper_job() -> None:
    """APScheduler / startup entry point. Never raises."""
    try:
        await reap_stale_reservations()
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("credit_service: reaper failed: %s", type(exc).__name__)


def register_reaper_job(scheduler: Any) -> None:
    """Add the periodic reaper to the shared DOCBOT-704 APScheduler instance."""
    scheduler.add_job(
        run_reaper_job,
        "interval",
        minutes=REAPER_INTERVAL_MINUTES,
        id="credit_reservation_reaper",
        replace_existing=True,
        max_instances=1,
    )


# ---------------------------------------------------------------------------
# Request integration
# ---------------------------------------------------------------------------


def is_demo_request(
    session_id: Optional[str],
    connection_id: Optional[str],
    *,
    session_exists: bool,
) -> bool:
    """True when the request only touches server-generated demo resources.

    Demo ids are ``demo-<uuid>`` minted by ``demo_service``. A ``demo-`` session
    id that does not exist server-side does not count (otherwise any caller
    could prefix an id to skip billing); connection ownership is verified by
    the route's ``_require_connection`` before billing begins.
    """
    sid = session_id or ""
    cid = connection_id or ""
    if not sid and not cid:
        return False
    session_ok = (not sid) or (sid.startswith("demo-") and session_exists)
    connection_ok = (not cid) or cid.startswith("demo-")
    return session_ok and connection_ok


def billing_org_for(user_id: Optional[str], org_id: Optional[str]) -> Optional[str]:
    """The billing account: the org, or a personal account for an org-less user."""
    if org_id:
        return org_id
    if user_id:
        return f"user:{user_id}"
    return None


@dataclass
class BillingRun:
    """One request's reservation. Settle exactly once via :meth:`stream` or ``async with``."""

    route: str
    run_id: str
    org_id: Optional[str]
    principal: Optional[Principal]
    meter: RequestMeter = field(default_factory=RequestMeter)
    reserved_credits: int = 0
    free_tier: bool = True
    exempt: bool = False
    settled: bool = False
    anon_owner: Optional[str] = None  # DOCBOT-1706 funnel key for anonymous callers
    _tokens: list[Any] = field(default_factory=list, repr=False)

    async def settle(self) -> Optional[SettleResult]:
        """Settle from the meter snapshot. Idempotent, never raises."""
        if self.settled:
            return None
        self.settled = True
        if self.exempt or self.org_id is None or not is_wired():
            return None
        snap = self.meter.snapshot()
        try:
            result = await settle(
                self.org_id, self.run_id, snap.cost_micro_usd, free_tier=self.free_tier
            )
        except (SQLAlchemyError, OSError) as exc:
            logger.error(
                "credit_service: settle failed org=%s run=%s: %s",
                self.org_id, self.run_id, type(exc).__name__,
            )
            return None
        logger.info(
            "credit_service: settled route=%s org=%s run=%s cost_micro_usd=%d credits=%d reserved=%d mode=%s",
            self.route, self.org_id, self.run_id, snap.cost_micro_usd,
            result.applied_credits, result.released_credits,
            "enforce" if is_enforced() else "shadow",
        )
        return result

    async def _settle_shielded(self) -> None:
        # Shielded: if the request task is cancelled (client gone) the settle
        # still runs to completion on its own task.
        await asyncio.shield(asyncio.ensure_future(self.settle()))

    # -- streaming ---------------------------------------------------------

    def _snapshot(self, base: BillingContextSnapshot) -> BillingContextSnapshot:
        return BillingContextSnapshot(
            principal=base.principal or self.principal,
            key_ctx=base.key_ctx,
            meter=self.meter,
        )

    def stream(self, agen: AsyncIterator[Any]) -> AsyncIterator[Any]:
        """Wrap an SSE generator: bind billing context, settle in ``finally``.

        Replaces ``usage_service.bind_billing_stream`` at the callsite — the
        snapshot is captured here, on the request path.
        """
        snapshot = self._snapshot(capture_billing_context())

        async def _billed() -> AsyncIterator[Any]:
            failed = False
            try:
                with use_billing_context(snapshot):
                    async for chunk in agen:
                        failed = failed or (isinstance(chunk, str) and '"type": "error"' in chunk)
                        yield chunk
                if not failed:
                    self._record_first_answer()
            finally:
                await self._settle_shielded()

        return _billed()

    def _record_first_answer(self) -> None:
        user_id = self.principal.user_id if self.principal else None
        record_funnel_event(
            "first_answer", user_id=user_id, org_id=self.org_id if user_id else None,
            anon_owner=self.anon_owner, props={"route": self.route},
        )

    # -- non-streaming -----------------------------------------------------

    async def __aenter__(self) -> "BillingRun":
        self._tokens = [set_meter(self.meter)]
        if self.principal is not None:
            self._tokens.append(set_principal(self.principal))
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        try:
            await self._settle_shielded()
        finally:
            tokens, self._tokens = self._tokens, []
            if len(tokens) > 1:
                reset_principal(tokens[1])
            if tokens:
                reset_meter(tokens[0])


def _quota_error(exc: InsufficientCredits) -> HTTPException:
    if exc.frozen:
        payload = QuotaErrorPayload(
            error=QuotaErrorCode.account_frozen,
            message="This account is frozen. Contact support to restore access.",
        )
    else:
        payload = QuotaErrorPayload(
            error=QuotaErrorCode.insufficient_credits,
            message="You're out of credits for this request. Add credits to continue.",
            required_credits=exc.required,
            available_credits=exc.available,
            action_path="/settings/billing",
        )
    return HTTPException(status_code=402, detail=payload.model_dump(mode="json"))


async def start_billing_run(
    *,
    route: str,
    user_id: Optional[str],
    org_id: Optional[str],
    role: Role = "viewer",
    demo: bool = False,
) -> BillingRun:
    """Reserve credits for one run. Raises HTTP 402 only in enforce mode."""
    billing_org = billing_org_for(user_id, org_id)
    principal: Optional[Principal] = None
    if user_id is not None:
        principal = Principal(
            user_id=user_id,
            org_id=billing_org,
            role=role,
            auth_method="session",
        )
    run = BillingRun(
        route=route,
        run_id=f"bill-{uuid.uuid4()}",
        org_id=billing_org,
        principal=principal,
        exempt=demo or billing_org is None,
    )
    if run.exempt or not is_wired():
        return run
    estimate = estimate_for(route)
    try:
        result = await reserve(billing_org, estimate, run.run_id, route)
        run.reserved_credits = result.reserved_credits
        run.free_tier = result.free_tier
    except InsufficientCredits as exc:
        if is_enforced():
            run.settled = True  # nothing reserved, nothing to settle
            logger.info(
                "credit_service: 402 route=%s org=%s required=%d available=%d frozen=%s",
                route, billing_org, exc.required, exc.available, exc.frozen,
            )
            record_funnel_event(
                "paywall_shown", user_id=user_id, org_id=org_id, props={"route": route},
            )
            raise _quota_error(exc) from None
        logger.info(
            "credit_service: shadow would_block route=%s org=%s required=%d available=%d frozen=%s",
            route, billing_org, exc.required, exc.available, exc.frozen,
        )
    except (SQLAlchemyError, OSError) as exc:
        # Fail open: a billing-store outage must not take chat down.
        logger.error(
            "credit_service: reserve failed route=%s org=%s: %s — failing open",
            route, billing_org, type(exc).__name__,
        )
    return run


async def begin_billing(
    request: Request,
    route: str,
    *,
    session_id: Optional[str] = None,
    connection_id: Optional[str] = None,
    session_exists: bool = False,
    scope: Optional[CallerScope] = None,
) -> BillingRun:
    """Route-facing entry point: resolve the caller, detect demo, reserve.

    Call it last on the request path, immediately before returning the
    ``StreamingResponse`` (``run.stream(...)``) or entering ``async with run``,
    so no reservation is taken for a request that then fails validation.

    ``scope``: pass the route's existing ``CallerScope`` dependency when the
    route already resolved one — the caller is then NOT re-resolved from the
    session cookie (one auth lookup per request, not two). Omitted, it falls
    back to ``resolve_caller_scope(request)``.
    """
    if scope is None:
        from api.utils.caller_scope import resolve_caller_scope

        scope = await resolve_caller_scope(request)
    demo = is_demo_request(session_id, connection_id, session_exists=session_exists)
    run = await start_billing_run(
        route=route,
        user_id=scope.user_id,
        org_id=scope.org_id,
        role="admin" if scope.is_admin else "viewer",
        demo=demo,
    )
    run.anon_owner = scope.anon_token
    return run


__all__ = [
    "BillingRun",
    "CreditTables",
    "DEFAULT_ESTIMATES",
    "InsufficientCredits",
    "LEDGER_IMMUTABILITY_STATEMENTS",
    "MIN_MARKUP",
    "begin_billing",
    "billing_org_for",
    "cost_to_credits",
    "estimate_for",
    "free_monthly_credits",
    "free_run_ceiling_credits",
    "GrantResult",
    "grant_purchase",
    "grant_ref",
    "is_demo_request",
    "is_enforced",
    "markup",
    "reap_stale_reservations",
    "register_credit_tables",
    "register_reaper_job",
    "release_reservation",
    "reserve",
    "run_reaper_job",
    "settle",
    "start_billing_run",
    "wire_credit_store",
]
