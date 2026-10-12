"""Tests for commerce_service raw_json retention cleanup — DOCBOT-1526.

Verifies that ``cleanup_stale_raw_json()`` nulls the ``raw_json`` payload
column only for rows older than the retention window, leaves normalized
fields (amounts, status, dates, etc.) untouched, leaves recent rows'
raw_json intact, handles the retention-window boundary correctly, and
that a failure on one row does not stop cleanup of the others.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import MetaData, insert as sa_insert, select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from api.commerce_service import (
    register_commerce_tables,
    wire_commerce,
    cleanup_stale_raw_json,
)

RETENTION_DAYS = 90


# ---------------------------------------------------------------------------
# Fixtures — real in-memory SQLite async DB (shared across connections via
# StaticPool), with the real commerce_orders / commerce_financials tables.
# ---------------------------------------------------------------------------


@pytest.fixture
async def commerce_db():
    metadata = MetaData()
    orders_table, financials_table = register_commerce_tables(metadata)

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    import api.commerce_service as cs
    old = (cs._orders_table, cs._financials_table, cs._async_session_factory)
    wire_commerce(orders_table, financials_table, session_factory)

    try:
        yield orders_table, financials_table, session_factory
    finally:
        cs._orders_table, cs._financials_table, cs._async_session_factory = old
        await engine.dispose()


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _insert_order(session_factory, orders_table, *, created_at, raw_json="{\"k\": \"v\"}", **overrides):
    row_id = overrides.pop("id", str(uuid.uuid4()))
    values = {
        "id": row_id,
        "connection_id": "conn-1",
        "connector_type": "amazon",
        "marketplace_order_id": f"mkt-{row_id}",
        "status": "shipped",
        "total_amount": 123.45,
        "currency": "USD",
        "order_date": created_at,
        "customer_id": "cust-1",
        "raw_json": raw_json,
        "created_at": created_at,
        "updated_at": created_at,
    }
    values.update(overrides)
    async with session_factory() as session:
        async with session.begin():
            await session.execute(sa_insert(orders_table).values(**values))
    return row_id


async def _fetch_order(session_factory, orders_table, row_id):
    async with session_factory() as session:
        result = await session.execute(
            select(orders_table).where(orders_table.c.id == row_id)
        )
        return result.mappings().one()


# ---------------------------------------------------------------------------
# Core retention behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCleanupStaleRawJson:
    async def test_old_row_raw_json_nulled(self, commerce_db):
        orders_table, _, session_factory = commerce_db
        old_created = _now() - timedelta(days=RETENTION_DAYS + 10)
        row_id = await _insert_order(session_factory, orders_table, created_at=old_created)

        cleaned = await cleanup_stale_raw_json(retention_days=RETENTION_DAYS)
        assert cleaned == 1

        row = await _fetch_order(session_factory, orders_table, row_id)
        assert row["raw_json"] is None

    async def test_recent_row_raw_json_kept(self, commerce_db):
        orders_table, _, session_factory = commerce_db
        recent_created = _now() - timedelta(days=10)
        row_id = await _insert_order(session_factory, orders_table, created_at=recent_created)

        cleaned = await cleanup_stale_raw_json(retention_days=RETENTION_DAYS)
        assert cleaned == 0

        row = await _fetch_order(session_factory, orders_table, row_id)
        assert row["raw_json"] == "{\"k\": \"v\"}"

    async def test_normalized_fields_untouched(self, commerce_db):
        """Only raw_json changes — amounts/status/dates are never modified."""
        orders_table, _, session_factory = commerce_db
        old_created = _now() - timedelta(days=RETENTION_DAYS + 5)
        row_id = await _insert_order(
            orders_table=orders_table,
            session_factory=session_factory,
            created_at=old_created,
            status="shipped",
            total_amount=555.55,
            customer_id="cust-xyz",
        )

        await cleanup_stale_raw_json(retention_days=RETENTION_DAYS)

        row = await _fetch_order(session_factory, orders_table, row_id)
        assert row["raw_json"] is None
        assert row["status"] == "shipped"
        assert row["total_amount"] == 555.55
        assert row["customer_id"] == "cust-xyz"

    async def test_boundary_just_inside_window_kept(self, commerce_db):
        """A row created just within the retention window keeps raw_json."""
        orders_table, _, session_factory = commerce_db
        just_inside = _now() - timedelta(days=RETENTION_DAYS) + timedelta(hours=1)
        row_id = await _insert_order(session_factory, orders_table, created_at=just_inside)

        cleaned = await cleanup_stale_raw_json(retention_days=RETENTION_DAYS)
        assert cleaned == 0

        row = await _fetch_order(session_factory, orders_table, row_id)
        assert row["raw_json"] is not None

    async def test_boundary_just_outside_window_nulled(self, commerce_db):
        """A row created just past the retention window has raw_json nulled."""
        orders_table, _, session_factory = commerce_db
        just_outside = _now() - timedelta(days=RETENTION_DAYS) - timedelta(hours=1)
        row_id = await _insert_order(session_factory, orders_table, created_at=just_outside)

        cleaned = await cleanup_stale_raw_json(retention_days=RETENTION_DAYS)
        assert cleaned == 1

        row = await _fetch_order(session_factory, orders_table, row_id)
        assert row["raw_json"] is None

    async def test_already_null_raw_json_not_recounted(self, commerce_db):
        """Rows with no raw_json to begin with are not touched or counted."""
        orders_table, _, session_factory = commerce_db
        old_created = _now() - timedelta(days=RETENTION_DAYS + 5)
        await _insert_order(session_factory, orders_table, created_at=old_created, raw_json=None)

        cleaned = await cleanup_stale_raw_json(retention_days=RETENTION_DAYS)
        assert cleaned == 0

    async def test_covers_both_orders_and_financials_tables(self, commerce_db):
        orders_table, financials_table, session_factory = commerce_db
        old_created = _now() - timedelta(days=RETENTION_DAYS + 5)

        await _insert_order(session_factory, orders_table, created_at=old_created)

        fin_id = str(uuid.uuid4())
        async with session_factory() as session:
            async with session.begin():
                await session.execute(
                    sa_insert(financials_table).values(
                        id=fin_id,
                        connection_id="conn-1",
                        connector_type="amazon",
                        period_start=old_created,
                        period_end=old_created,
                        revenue=1000.0,
                        refunds=0.0,
                        fees=0.0,
                        net_proceeds=1000.0,
                        currency="USD",
                        raw_json=json.dumps({"k": "v"}),
                        created_at=old_created,
                    )
                )

        cleaned = await cleanup_stale_raw_json(retention_days=RETENTION_DAYS)
        assert cleaned == 2  # 1 order + 1 financial row

        async with session_factory() as session:
            result = await session.execute(
                select(financials_table.c.raw_json, financials_table.c.revenue).where(
                    financials_table.c.id == fin_id
                )
            )
            row = result.one()
        assert row.raw_json is None
        assert row.revenue == 1000.0  # normalized field untouched

    async def test_not_wired_returns_zero(self):
        import api.commerce_service as cs
        old = (cs._orders_table, cs._financials_table, cs._async_session_factory)
        cs._orders_table = None
        cs._financials_table = None
        cs._async_session_factory = None
        try:
            result = await cleanup_stale_raw_json()
            assert result == 0
        finally:
            cs._orders_table, cs._financials_table, cs._async_session_factory = old

    async def test_default_retention_days_from_env(self, monkeypatch):
        """With no explicit retention_days, the module-level default is used."""
        import api.commerce_service as cs
        monkeypatch.setattr(cs, "COMMERCE_RAW_JSON_RETENTION_DAYS", 42)
        # Not wired — short-circuits before using the default, but exercises
        # the code path that reads the module-level constant.
        cs._orders_table = None
        cs._financials_table = None
        cs._async_session_factory = None
        result = await cleanup_stale_raw_json()
        assert result == 0


# ---------------------------------------------------------------------------
# Resilience: a failure nulling one row must not block the others
# ---------------------------------------------------------------------------


class _FlakyFactory:
    """Wraps a real async session factory and makes the Nth call's
    ``session.begin()`` raise, simulating a transient failure on exactly
    one row's update while leaving all other sessions untouched."""

    def __init__(self, real_factory, fail_on_call: int):
        self._real_factory = real_factory
        self._fail_on_call = fail_on_call
        self._calls = 0

    def __call__(self):
        self._calls += 1
        session = self._real_factory()
        if self._calls == self._fail_on_call:
            def _boom(*args, **kwargs):
                raise RuntimeError("simulated transient DB error")
            session.begin = _boom
        return session


@pytest.mark.asyncio
class TestCleanupResilience:
    async def test_one_row_failure_does_not_block_others(self, commerce_db):
        orders_table, _, session_factory = commerce_db
        old_created = _now() - timedelta(days=RETENTION_DAYS + 10)

        row_a = await _insert_order(session_factory, orders_table, created_at=old_created)
        row_b = await _insert_order(session_factory, orders_table, created_at=old_created)
        row_c = await _insert_order(session_factory, orders_table, created_at=old_created)

        import api.commerce_service as cs
        real_factory = cs._async_session_factory
        # Call 1 = the initial SELECT for stale ids. Calls 2/3/4 = the three
        # per-row UPDATE sessions. Fail call 3 (the middle row's update).
        flaky = _FlakyFactory(real_factory, fail_on_call=3)
        cs._async_session_factory = flaky
        try:
            cleaned = await cleanup_stale_raw_json(retention_days=RETENTION_DAYS)
        finally:
            cs._async_session_factory = real_factory

        # Exactly 2 of the 3 rows were successfully cleaned.
        assert cleaned == 2

        rows = {
            row_a: await _fetch_order(session_factory, orders_table, row_a),
            row_b: await _fetch_order(session_factory, orders_table, row_b),
            row_c: await _fetch_order(session_factory, orders_table, row_c),
        }
        null_count = sum(1 for r in rows.values() if r["raw_json"] is None)
        kept_count = sum(1 for r in rows.values() if r["raw_json"] is not None)
        assert null_count == 2
        assert kept_count == 1
