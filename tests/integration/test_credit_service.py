"""DOCBOT-1802 — credit balance, atomic reservations, settle, grant, reaper.

Runs against a real on-disk SQLite database (aiosqlite) so the atomic
``UPDATE ... WHERE balance - reserved >= :est RETURNING`` and the
``UNIQUE(kind, ref)`` grant guard are exercised by the actual engine. The same
concurrency test runs against Postgres when ``TEST_POSTGRES_URL`` is set
(``@pytest.mark.postgres``, skipped in CI).
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from sqlalchemy import MetaData, func, select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api import credit_service
from api.billing_models import LedgerEntryKind
from api.utils.billing_context import get_meter, get_principal


async def _make_store(url: str, **engine_kw):
    metadata = MetaData()
    tables = credit_service.register_credit_tables(metadata)
    engine = create_async_engine(url, **engine_kw)
    async with engine.begin() as conn:
        await conn.run_sync(metadata.drop_all)
        await conn.run_sync(metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    credit_service.wire_credit_store(tables, factory, engine.dialect.name)
    return engine, tables, factory


@pytest.fixture(autouse=True)
def _billing_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("BILLING_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BILLING_FREE_MONTHLY_CREDITS", "500")
    yield
    credit_service._tables = None
    credit_service._session_factory = None
    credit_service._granted_period.clear()


@pytest.fixture
async def store(tmp_path):
    engine, tables, factory = await _make_store(
        f"sqlite+aiosqlite:///{tmp_path / 'credits.db'}",
        connect_args={"timeout": 30},
    )
    yield tables, factory
    await engine.dispose()


async def _balance(factory, tables, org="org-1"):
    ob = tables.org_balances
    async with factory() as s:
        return (
            await s.execute(select(ob.c.balance_credits, ob.c.reserved_credits).where(ob.c.org_id == org))
        ).first()


async def _ledger(factory, tables, org="org-1"):
    lg = tables.credit_ledger
    async with factory() as s:
        return (await s.execute(select(lg).where(lg.c.org_id == org).order_by(lg.c.created_at))).all()


async def _run_concurrent_limit(tables, factory, n=50, est=10):
    """Balance = n*est - 1 -> exactly n-1 reservations may succeed."""
    os.environ["BILLING_FREE_MONTHLY_CREDITS"] = str(n * est - 1)

    async def one(i: int) -> bool:
        try:
            await credit_service.reserve("org-c", est, f"run-{i}", "chat")
            return True
        except credit_service.InsufficientCredits:
            return False

    results = await asyncio.gather(*(one(i) for i in range(n)))
    assert sum(results) == n - 1
    row = await _balance(factory, tables, "org-c")
    assert row.balance_credits == n * est - 1
    assert row.reserved_credits == (n - 1) * est
    grants = [r for r in await _ledger(factory, tables, "org-c") if r.kind == LedgerEntryKind.free_grant.value]
    assert len(grants) == 1  # 50 racing first-requests -> exactly one grant


# ---------------------------------------------------------------------------
# Atomic reserve
# ---------------------------------------------------------------------------


async def test_50_concurrent_reserves_at_limit_minus_one_sqlite(store):
    tables, factory = store
    await _run_concurrent_limit(tables, factory)


@pytest.mark.postgres
async def test_50_concurrent_reserves_at_limit_minus_one_postgres():
    url = os.getenv("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_URL not set")
    engine, tables, factory = await _make_store(url, pool_size=20, max_overflow=40)
    try:
        await _run_concurrent_limit(tables, factory)
    finally:
        await engine.dispose()


async def test_reserve_insufficient_and_frozen(store):
    tables, factory = store
    os.environ["BILLING_FREE_MONTHLY_CREDITS"] = "5"
    with pytest.raises(credit_service.InsufficientCredits) as ei:
        await credit_service.reserve("org-1", 10, "r1", "chat")
    assert ei.value.required == 10 and ei.value.available == 5 and not ei.value.frozen
    async with factory() as s, s.begin():
        await s.execute(update(tables.org_balances).values(balance_credits=1000, frozen=True))
    with pytest.raises(credit_service.InsufficientCredits) as ei:
        await credit_service.reserve("org-1", 10, "r2", "chat")
    assert ei.value.frozen


async def test_free_tier_estimate_capped_at_run_ceiling(store):
    tables, factory = store
    os.environ["BILLING_FREE_MONTHLY_CREDITS"] = "1000"
    res = await credit_service.reserve("org-1", 400, "r1", "autopilot")
    assert res.free_tier and res.reserved_credits == 150


# ---------------------------------------------------------------------------
# Settle
# ---------------------------------------------------------------------------


async def test_settle_debits_actual_and_releases_reservation(store):
    tables, factory = store
    await credit_service.reserve("org-1", 20, "run-a", "chat")
    # 2_000 micro-USD raw x 3.5 markup = 7 credits
    result = await credit_service.settle("org-1", "run-a", 2_000, free_tier=True)
    assert (result.actual_credits, result.applied_credits, result.released_credits) == (7, 7, 20)
    row = await _balance(factory, tables)
    assert (row.balance_credits, row.reserved_credits) == (493, 0)
    usage = [r for r in await _ledger(factory, tables) if r.kind == "usage"]
    assert len(usage) == 1 and usage[0].ref == "run-a" and usage[0].delta_credits == -7
    assert usage[0].balance_after_credits == 493
    # idempotent: a second settle never debits twice
    again = await credit_service.settle("org-1", "run-a", 2_000, free_tier=True)
    assert again.applied_credits == 0 and again.released_credits == 0
    assert (await _balance(factory, tables)).balance_credits == 493


async def test_settle_never_drives_balance_negative_and_ledger_reconciles(store):
    tables, factory = store
    os.environ["BILLING_FREE_MONTHLY_CREDITS"] = "30"
    await credit_service.reserve("org-1", 20, "run-a", "chat")
    # 100_000 micro-USD x 3.5 = 350 credits, free-tier ceiling 150, balance 30
    result = await credit_service.settle("org-1", "run-a", 100_000, free_tier=True)
    assert result.actual_credits == 150 and result.applied_credits == 30
    row = await _balance(factory, tables)
    assert (row.balance_credits, row.reserved_credits) == (0, 0)
    lg = tables.credit_ledger
    async with factory() as s:
        total = (await s.execute(select(func.sum(lg.c.delta_credits)).where(lg.c.org_id == "org-1"))).scalar()
    assert total == row.balance_credits


async def test_billing_run_settles_in_finally_on_exception(store):
    tables, factory = store
    run = await credit_service.start_billing_run(route="sandbox", user_id="u1", org_id="org-1")
    assert run.reserved_credits == 50
    with pytest.raises(RuntimeError):
        async with run:
            assert get_meter() is run.meter
            assert get_principal().org_id == "org-1"
            run.meter.record_llm(100, 50, "0.001")  # 1_000 micro-USD -> 4 credits
            raise RuntimeError("boom")
    assert run.settled and get_meter() is None
    row = await _balance(factory, tables)
    assert (row.balance_credits, row.reserved_credits) == (496, 0)


async def test_stream_binds_meter_and_settles_when_generator_raises(store):
    tables, factory = store
    run = await credit_service.start_billing_run(route="chat", user_id="u1", org_id="org-1")
    seen = []

    async def gen():
        seen.append(get_meter())
        get_meter().record_llm(10, 10, "0.002")  # 2_000 -> 7 credits
        yield "a"
        raise ValueError("mid-stream failure")

    out = []
    with pytest.raises(ValueError):
        async for chunk in run.stream(gen()):
            out.append(chunk)
    assert out == ["a"] and seen == [run.meter] and run.settled
    row = await _balance(factory, tables)
    assert (row.balance_credits, row.reserved_credits) == (493, 0)


async def test_stream_settles_on_client_disconnect(store):
    tables, factory = store
    run = await credit_service.start_billing_run(route="chat", user_id="u1", org_id="org-1")

    async def gen():
        for i in range(10):
            yield i

    agen = run.stream(gen())
    assert await agen.__anext__() == 0
    await agen.aclose()  # what Starlette does when the client goes away
    assert run.settled
    assert (await _balance(factory, tables)).reserved_credits == 0


# ---------------------------------------------------------------------------
# Lazy monthly grant
# ---------------------------------------------------------------------------


async def test_double_grant_prevented_and_next_month_grants_again(store, monkeypatch):
    tables, factory = store
    await credit_service.reserve("org-1", 1, "r1", "chat")
    credit_service._granted_period.clear()  # force the DB path, not the cache
    await credit_service.reserve("org-1", 1, "r2", "chat")
    grants = [r for r in await _ledger(factory, tables) if r.kind == "free_grant"]
    assert len(grants) == 1
    assert grants[0].ref == credit_service.grant_ref("org-1", credit_service.current_period())
    assert (await _balance(factory, tables)).balance_credits == 500

    future = datetime.now(timezone.utc) + timedelta(days=40)
    monkeypatch.setattr(credit_service, "_now", lambda: future)
    await credit_service.reserve("org-1", 1, "r3", "chat")
    grants = [r for r in await _ledger(factory, tables) if r.kind == "free_grant"]
    assert len(grants) == 2
    assert (await _balance(factory, tables)).balance_credits == 1000


# ---------------------------------------------------------------------------
# Reaper
# ---------------------------------------------------------------------------


async def test_reaper_releases_only_stale_reservations(store):
    tables, factory = store
    await credit_service.reserve("org-1", 20, "old", "chat")
    await credit_service.reserve("org-1", 30, "new", "chat")
    async with factory() as s, s.begin():
        await s.execute(
            update(tables.credit_reservations)
            .where(tables.credit_reservations.c.id == "old")
            .values(created_at=datetime.now(timezone.utc) - timedelta(minutes=60))
        )
    assert await credit_service.reap_stale_reservations(15) == 1
    assert (await _balance(factory, tables)).reserved_credits == 30
    # A late settle of the reaped run debits usage but does not double-release.
    late = await credit_service.settle("org-1", "old", 1_000, free_tier=True)
    assert late.released_credits == 0
    row = await _balance(factory, tables)
    assert (row.balance_credits, row.reserved_credits) == (496, 30)
    assert await credit_service.reap_stale_reservations(15) == 0


async def test_reaper_job_is_noop_when_unwired():
    await credit_service.run_reaper_job()  # must not raise


def test_register_reaper_job_adds_interval_job():
    calls = []

    class FakeScheduler:
        def add_job(self, fn, trigger, **kw):
            calls.append((fn, trigger, kw))

    credit_service.register_reaper_job(FakeScheduler())
    assert calls[0][0] is credit_service.run_reaper_job
    assert calls[0][1] == "interval" and calls[0][2]["id"] == "credit_reservation_reaper"


# ---------------------------------------------------------------------------
# Shadow vs enforce, exemptions, fail-open
# ---------------------------------------------------------------------------


async def test_shadow_mode_never_402_but_still_settles(store):
    tables, factory = store
    os.environ["BILLING_FREE_MONTHLY_CREDITS"] = "3"
    run = await credit_service.start_billing_run(route="chat", user_id="u1", org_id="org-1")
    assert run.reserved_credits == 0 and not run.exempt
    async with run:
        run.meter.record_llm(1, 1, "0.0002")  # 200 micro -> 1 credit
    row = await _balance(factory, tables)
    assert (row.balance_credits, row.reserved_credits) == (2, 0)


async def test_enforce_mode_raises_402_payload(store, monkeypatch):
    tables, factory = store
    monkeypatch.setenv("BILLING_ENFORCE", "true")
    os.environ["BILLING_FREE_MONTHLY_CREDITS"] = "3"
    with pytest.raises(HTTPException) as ei:
        await credit_service.start_billing_run(route="chat", user_id="u1", org_id="org-1")
    assert ei.value.status_code == 402
    assert ei.value.detail["error"] == "insufficient_credits"
    assert ei.value.detail["required_credits"] == 20
    assert ei.value.detail["available_credits"] == 3
    async with factory() as s, s.begin():
        await s.execute(update(tables.org_balances).values(balance_credits=999, frozen=True))
    with pytest.raises(HTTPException) as ei:
        await credit_service.start_billing_run(route="chat", user_id="u1", org_id="org-1")
    assert ei.value.detail["error"] == "account_frozen"


async def test_enforce_mode_allows_when_funded(store, monkeypatch):
    monkeypatch.setenv("BILLING_ENFORCE", "1")
    run = await credit_service.start_billing_run(route="hybrid", user_id="u1", org_id="org-1")
    assert run.reserved_credits == 40
    await run.settle()


async def test_demo_and_anonymous_are_exempt(store, monkeypatch):
    tables, factory = store
    monkeypatch.setenv("BILLING_ENFORCE", "true")
    os.environ["BILLING_FREE_MONTHLY_CREDITS"] = "0"
    demo = await credit_service.start_billing_run(route="chat", user_id="u1", org_id="org-1", demo=True)
    anon = await credit_service.start_billing_run(route="chat", user_id=None, org_id=None)
    assert demo.exempt and anon.exempt
    async with demo:
        demo.meter.record_llm(1, 1, "1")
    assert await _balance(factory, tables) is None  # nothing touched


def test_is_demo_request_rules():
    f = credit_service.is_demo_request
    assert f("demo-abc", None, session_exists=True)
    assert f("demo-abc", "demo-def", session_exists=True)
    assert f(None, "demo-def", session_exists=False)
    assert not f("demo-abc", None, session_exists=False)  # spoofed id
    assert not f("demo-abc", "conn-real", session_exists=True)  # real DB
    assert not f("sess-1", None, session_exists=True)
    assert not f(None, None, session_exists=True)


def test_billing_org_falls_back_to_personal_account():
    assert credit_service.billing_org_for("u1", "org-1") == "org-1"
    assert credit_service.billing_org_for("u1", None) == "user:u1"
    assert credit_service.billing_org_for(None, None) is None


async def test_reserve_store_error_fails_open(store, monkeypatch):
    async def broken(*a, **k):
        raise OperationalError("SELECT 1", {}, Exception("db down"))

    monkeypatch.setattr(credit_service, "reserve", broken)
    monkeypatch.setenv("BILLING_ENFORCE", "true")
    run = await credit_service.start_billing_run(route="chat", user_id="u1", org_id="org-1")
    assert run.reserved_credits == 0 and not run.exempt


async def test_unwired_store_is_noop():
    run = await credit_service.start_billing_run(route="chat", user_id="u1", org_id="org-1")
    async with run:
        pass
    assert run.reserved_credits == 0


# ---------------------------------------------------------------------------
# Pricing weights
# ---------------------------------------------------------------------------


def test_markup_clamped_so_cost_is_at_most_30_percent(monkeypatch):
    monkeypatch.setenv("BILLING_MARKUP", "2")
    assert credit_service.markup() == credit_service.MIN_MARKUP
    monkeypatch.setenv("BILLING_MARKUP", "nope")
    assert credit_service.markup() == credit_service.DEFAULT_MARKUP
    monkeypatch.delenv("BILLING_MARKUP")
    m = credit_service.markup()
    assert 1 / m <= 0.30
    # default free tier: raw cost of a full month stays small
    assert credit_service.free_monthly_credits() * 1000 / m <= 150_000  # <= $0.15 raw


def test_estimates_from_env(monkeypatch):
    assert credit_service.estimate_for("autopilot") == 150
    monkeypatch.setenv("BILLING_EST_CREDITS_AUTOPILOT", "220")
    assert credit_service.estimate_for("autopilot") == 220
    monkeypatch.setenv("BILLING_EST_CREDITS_CHAT", "-5")
    assert credit_service.estimate_for("chat") == 1


def test_cost_to_credits_rounds_up():
    assert credit_service.cost_to_credits(0) == 0
    assert credit_service.cost_to_credits(1) == 1
    assert credit_service.cost_to_credits(1_000) == 4  # 3.5 -> 4


# ---------------------------------------------------------------------------
# DOCBOT-1803: manual purchase grant (Payment Link flow)
# ---------------------------------------------------------------------------


async def test_grant_purchase_idempotent_on_reference(store):
    tables, factory = store
    first = await credit_service.grant_purchase("org-1", 10_000, "pi_123", "Manual grant")
    again = await credit_service.grant_purchase("org-1", 10_000, "pi_123", "Manual grant")
    assert first.duplicate is False and again.duplicate is True
    # monthly free grant (500) + one purchase, never two
    assert first.balance_credits == again.balance_credits == 10_500
    bal = await _balance(factory, tables)
    assert bal.balance_credits == 10_500
    rows = await _ledger(factory, tables)
    purchases = [r for r in rows if r.kind == LedgerEntryKind.purchase.value]
    assert len(purchases) == 1 and purchases[0].ref == "pi_123"
    assert purchases[0].balance_after_credits == 10_500
    assert sum(r.delta_credits for r in rows) == bal.balance_credits


async def test_grant_purchase_ends_free_tier(store):
    tables, factory = store
    await credit_service.grant_purchase("org-1", 1_000, "pi_free", "Manual grant")
    result = await credit_service.reserve("org-1", 400, "run-x", "/api/chat")
    assert result.free_tier is False
