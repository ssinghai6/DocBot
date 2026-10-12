"""Unit tests for api/org_service.py — DOCBOT-1524.

Covers the three properties the ticket exists to deliver:

1. Backfill creates exactly one default org and assigns it to every
   pre-existing row (across users, sessions, db_connections,
   marketplace_connections) that had org_id IS NULL. audit_log is
   intentionally excluded from the UPDATE-based backfill — see
   api/org_service.py's _BACKFILL_TABLES docstring: DOCBOT-602's
   append-only immutability trigger rejects any UPDATE against
   audit_log, so pre-existing audit rows keep org_id = NULL by design
   rather than violate that compliance boundary.
2. Backfill is idempotent — running it twice never creates a second
   default org and never errors.
3. (Most important) Admin user-list / audit-log queries scoped by
   org_id correctly isolate one organization's rows from another's —
   an admin from org A cannot see org B's users or audit rows.

Uses an in-memory SQLite async engine (mirrors tests/unit/test_trace_service.py
and tests/unit/test_ontology_service.py's approach) so no live PostgreSQL is
required. No network calls.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    MetaData,
    String,
    Table,
    Text,
    func,
    select,
)
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api.org_service import (
    DEFAULT_ORG_NAME,
    backfill_default_organization,
    register_organizations_table,
)
from api.rbac_service import get_user_org_id


# ---------------------------------------------------------------------------
# Minimal table fixtures — mirror the real column shape closely enough for
# org_id backfill/scoping tests without pulling in the full production
# metadata (which needs DATABASE_URL-bound engine setup out of scope here).
# org_service.backfill_default_organization() only ever references these
# tables by the fixed name strings "users", "sessions", "db_connections",
# "marketplace_connections" — exactly what it does against the real schema
# in api/index.py. audit_log is included here too (for the cross-org
# isolation tests below) but is never UPDATEd by the backfill itself — see
# api/org_service.py's _BACKFILL_TABLES docstring.
# ---------------------------------------------------------------------------


def _build_schema() -> tuple[MetaData, dict[str, Table]]:
    metadata = MetaData()
    organizations = register_organizations_table(metadata)

    users = Table(
        "users", metadata,
        Column("id", String, primary_key=True),
        Column("email", String, nullable=False),
        Column("role", String, server_default="analyst", nullable=False),
        Column("org_id", String, index=True),
    )
    sessions = Table(
        "sessions", metadata,
        Column("session_id", String, primary_key=True),
        Column("user_id", String),
        Column("org_id", String, index=True),
    )
    db_connections = Table(
        "db_connections", metadata,
        Column("id", String, primary_key=True),
        Column("session_id", String, nullable=False),
        Column("org_id", String, index=True),
    )
    marketplace_connections = Table(
        "marketplace_connections", metadata,
        Column("id", String, primary_key=True),
        Column("connector_type", String, nullable=False),
        Column("org_id", String, index=True),
    )
    audit_log = Table(
        "audit_log", metadata,
        Column("id", String, primary_key=True),
        Column("event_type", String, nullable=False),
        Column("user_id", String),
        Column("org_id", String, index=True),
        # DOCBOT-1528: occurred_at, for the audit-log date-range filter test below.
        Column("occurred_at", DateTime(timezone=True)),
    )

    return metadata, {
        "organizations": organizations,
        "users": users,
        "sessions": sessions,
        "db_connections": db_connections,
        "marketplace_connections": marketplace_connections,
        "audit_log": audit_log,
    }


@pytest.fixture
async def db():
    """In-memory SQLite async engine + the minimal org-scoped tables."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    metadata, tables = _build_schema()

    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    yield engine, session_factory, tables
    await engine.dispose()


async def _seed_preexisting_rows(session_factory, tables) -> None:
    """Insert rows with org_id left NULL, simulating a pre-DOCBOT-1524 DB."""
    async with session_factory() as session:
        async with session.begin():
            await session.execute(
                tables["users"].insert().values(id="u1", email="a@co.com", role="admin")
            )
            await session.execute(
                tables["sessions"].insert().values(session_id="s1", user_id="u1")
            )
            await session.execute(
                tables["db_connections"].insert().values(id="c1", session_id="s1")
            )
            await session.execute(
                tables["marketplace_connections"].insert().values(
                    id="m1", connector_type="shopify"
                )
            )
            await session.execute(
                tables["audit_log"].insert().values(id="a1", event_type="login", user_id="u1")
            )


# ---------------------------------------------------------------------------
# 1. Backfill creates exactly one default org and assigns it to all rows
# ---------------------------------------------------------------------------


class TestBackfillCreatesDefaultOrg:
    @pytest.mark.asyncio
    async def test_creates_exactly_one_default_org(self, db):
        engine, session_factory, tables = db
        await _seed_preexisting_rows(session_factory, tables)

        async with engine.begin() as conn:
            org_id = await backfill_default_organization(conn)

        async with session_factory() as session:
            result = await session.execute(select(tables["organizations"]))
            orgs = result.fetchall()

        assert len(orgs) == 1
        assert orgs[0].id == org_id
        assert orgs[0].name == DEFAULT_ORG_NAME

    @pytest.mark.asyncio
    async def test_backfills_org_id_on_every_scoped_table(self, db):
        engine, session_factory, tables = db
        await _seed_preexisting_rows(session_factory, tables)

        async with engine.begin() as conn:
            org_id = await backfill_default_organization(conn)

        async with session_factory() as session:
            for name in ("users", "sessions", "db_connections", "marketplace_connections"):
                result = await session.execute(select(tables[name]))
                rows = result.fetchall()
                assert len(rows) == 1, f"expected exactly 1 seeded row in {name}"
                assert rows[0].org_id == org_id, f"{name} row was not backfilled to the default org"

    @pytest.mark.asyncio
    async def test_audit_log_is_not_backfilled_by_update(self, db):
        """audit_log is append-only (DOCBOT-602 immutability trigger blocks
        UPDATE/DELETE at the DB level) — the backfill must never attempt to
        UPDATE it, so a pre-existing audit row keeps org_id = NULL."""
        engine, session_factory, tables = db
        await _seed_preexisting_rows(session_factory, tables)

        async with engine.begin() as conn:
            await backfill_default_organization(conn)

        async with session_factory() as session:
            result = await session.execute(select(tables["audit_log"]))
            rows = result.fetchall()
        assert len(rows) == 1
        assert rows[0].org_id is None

    @pytest.mark.asyncio
    async def test_returns_default_org_id_even_with_no_rows(self, db):
        """Backfill must create the default org even on an otherwise-empty DB."""
        engine, session_factory, tables = db

        async with engine.begin() as conn:
            org_id = await backfill_default_organization(conn)

        assert org_id
        async with session_factory() as session:
            result = await session.execute(select(tables["organizations"]))
            orgs = result.fetchall()
        assert len(orgs) == 1
        assert orgs[0].id == org_id


# ---------------------------------------------------------------------------
# 2. Idempotency — running backfill twice never duplicates or errors
# ---------------------------------------------------------------------------


class TestBackfillIdempotent:
    @pytest.mark.asyncio
    async def test_running_twice_does_not_create_second_org(self, db):
        engine, session_factory, tables = db
        await _seed_preexisting_rows(session_factory, tables)

        async with engine.begin() as conn:
            org_id_1 = await backfill_default_organization(conn)
        async with engine.begin() as conn:
            org_id_2 = await backfill_default_organization(conn)

        assert org_id_1 == org_id_2

        async with session_factory() as session:
            result = await session.execute(select(tables["organizations"]))
            orgs = result.fetchall()
        assert len(orgs) == 1

    @pytest.mark.asyncio
    async def test_running_twice_does_not_error_or_change_backfilled_rows(self, db):
        engine, session_factory, tables = db
        await _seed_preexisting_rows(session_factory, tables)

        async with engine.begin() as conn:
            await backfill_default_organization(conn)
        # Second run must be a pure no-op — no exception, no row changes.
        async with engine.begin() as conn:
            org_id = await backfill_default_organization(conn)

        async with session_factory() as session:
            result = await session.execute(select(tables["users"]))
            users = result.fetchall()
        assert len(users) == 1
        assert users[0].org_id == org_id

    @pytest.mark.asyncio
    async def test_new_rows_added_between_runs_get_backfilled_on_next_run(self, db):
        """A row created after the first backfill (org_id NULL) is swept up
        by the next startup's backfill — this is how org_id reaches rows
        created by flows this ticket does not touch (e.g. new JIT-provisioned
        users) until a dedicated org-assignment flow ships."""
        engine, session_factory, tables = db
        await _seed_preexisting_rows(session_factory, tables)

        async with engine.begin() as conn:
            org_id = await backfill_default_organization(conn)

        # Simulate a new user signing up after the first backfill ran.
        async with session_factory() as session:
            async with session.begin():
                await session.execute(
                    tables["users"].insert().values(id="u2", email="b@co.com", role="analyst")
                )

        async with engine.begin() as conn:
            org_id_2 = await backfill_default_organization(conn)

        assert org_id == org_id_2
        async with session_factory() as session:
            result = await session.execute(
                select(tables["users"]).where(tables["users"].c.id == "u2")
            )
            new_user = result.fetchone()
        assert new_user.org_id == org_id


# ---------------------------------------------------------------------------
# 3. Cross-org isolation — the actual security property this ticket exists
#    to deliver. An admin from org A must never see org B's users or audit
#    rows in the query result.
# ---------------------------------------------------------------------------


class _FakeUser:
    """Stand-in for the Row require_role() returns — just needs .org_id."""

    def __init__(self, org_id):
        self.org_id = org_id


class TestCrossOrgIsolation:
    @pytest.mark.asyncio
    async def test_admin_cannot_see_other_orgs_users(self, db):
        engine, session_factory, tables = db
        org_a, org_b = str(uuid.uuid4()), str(uuid.uuid4())

        async with session_factory() as session:
            async with session.begin():
                await session.execute(tables["organizations"].insert().values(id=org_a, name="Org A"))
                await session.execute(tables["organizations"].insert().values(id=org_b, name="Org B"))
                await session.execute(
                    tables["users"].insert().values(
                        id="admin_a", email="admin@a.com", role="admin", org_id=org_a
                    )
                )
                await session.execute(
                    tables["users"].insert().values(
                        id="viewer_a", email="viewer@a.com", role="viewer", org_id=org_a
                    )
                )
                await session.execute(
                    tables["users"].insert().values(
                        id="admin_b", email="admin@b.com", role="admin", org_id=org_b
                    )
                )

        admin_a = _FakeUser(org_id=org_a)

        # Exact scoping pattern used in api/index.py's GET /admin/users route.
        stmt = select(tables["users"]).order_by(tables["users"].c.id)
        org_id = get_user_org_id(admin_a)
        if org_id is not None:
            stmt = stmt.where(tables["users"].c.org_id == org_id)

        async with session_factory() as session:
            result = await session.execute(stmt)
            rows = result.fetchall()

        seen_ids = {row.id for row in rows}
        assert seen_ids == {"admin_a", "viewer_a"}
        assert "admin_b" not in seen_ids, "admin from org A saw a user from org B — isolation broken"

    @pytest.mark.asyncio
    async def test_admin_cannot_see_other_orgs_audit_log(self, db):
        engine, session_factory, tables = db
        org_a, org_b = str(uuid.uuid4()), str(uuid.uuid4())

        async with session_factory() as session:
            async with session.begin():
                await session.execute(tables["organizations"].insert().values(id=org_a, name="Org A"))
                await session.execute(tables["organizations"].insert().values(id=org_b, name="Org B"))
                await session.execute(
                    tables["audit_log"].insert().values(
                        id="ev1", event_type="login", user_id="admin_a", org_id=org_a
                    )
                )
                await session.execute(
                    tables["audit_log"].insert().values(
                        id="ev2", event_type="query", user_id="viewer_a", org_id=org_a
                    )
                )
                await session.execute(
                    tables["audit_log"].insert().values(
                        id="ev3", event_type="login", user_id="admin_b", org_id=org_b
                    )
                )

        admin_a = _FakeUser(org_id=org_a)

        # Exact scoping pattern used in api/index.py's GET /admin/audit-log route.
        stmt = select(tables["audit_log"]).order_by(tables["audit_log"].c.id)
        org_id = get_user_org_id(admin_a)
        if org_id is not None:
            stmt = stmt.where(tables["audit_log"].c.org_id == org_id)

        async with session_factory() as session:
            result = await session.execute(stmt)
            rows = result.fetchall()

        seen_ids = {row.id for row in rows}
        assert seen_ids == {"ev1", "ev2"}
        assert "ev3" not in seen_ids, "admin from org A saw an audit row from org B — isolation broken"

    @pytest.mark.asyncio
    async def test_open_demo_mode_stays_unscoped(self, db):
        """get_user_org_id(None) — AUTH_REQUIRED off — must not filter at all,
        preserving pre-ticket open/demo-mode behavior."""
        engine, session_factory, tables = db
        org_a, org_b = str(uuid.uuid4()), str(uuid.uuid4())

        async with session_factory() as session:
            async with session.begin():
                await session.execute(
                    tables["users"].insert().values(id="u_a", email="a@a.com", role="admin", org_id=org_a)
                )
                await session.execute(
                    tables["users"].insert().values(id="u_b", email="b@b.com", role="admin", org_id=org_b)
                )

        stmt = select(tables["users"])
        org_id = get_user_org_id(None)
        if org_id is not None:
            stmt = stmt.where(tables["users"].c.org_id == org_id)

        async with session_factory() as session:
            result = await session.execute(stmt)
            rows = result.fetchall()

        assert {row.id for row in rows} == {"u_a", "u_b"}

    @pytest.mark.asyncio
    async def test_date_range_filter_composes_with_org_scoping(self, db):
        """DOCBOT-1528: GET /admin/audit-log's new start_date/end_date params
        must compose with (not bypass) DOCBOT-1524's org_id scoping — an
        admin's date-range query must still never surface another org's
        rows, and the date bounds must correctly include/exclude rows at
        the boundary (inclusive start, inclusive end via a < next-day bound,
        matching api/index.py's get_audit_log implementation)."""
        from datetime import datetime, timedelta, timezone

        engine, session_factory, tables = db
        org_a, org_b = str(uuid.uuid4()), str(uuid.uuid4())

        day1 = datetime(2026, 1, 2, tzinfo=timezone.utc)
        day5 = datetime(2026, 1, 6, tzinfo=timezone.utc)

        async with session_factory() as session:
            async with session.begin():
                await session.execute(tables["organizations"].insert().values(id=org_a, name="Org A"))
                await session.execute(tables["organizations"].insert().values(id=org_b, name="Org B"))
                await session.execute(
                    tables["audit_log"].insert().values(
                        id="in_range_a", event_type="login", user_id="admin_a",
                        org_id=org_a, occurred_at=day1,
                    )
                )
                await session.execute(
                    tables["audit_log"].insert().values(
                        id="out_of_range_a", event_type="login", user_id="admin_a",
                        org_id=org_a, occurred_at=day5,
                    )
                )
                await session.execute(
                    tables["audit_log"].insert().values(
                        id="in_range_b", event_type="login", user_id="admin_b",
                        org_id=org_b, occurred_at=day1,
                    )
                )

        admin_a = _FakeUser(org_id=org_a)

        # Mirrors api/index.py's get_audit_log: org scoping, then start_date
        # (inclusive, >=) and end_date (inclusive, via < end + 1 day).
        start_date, end_date = "2026-01-01", "2026-01-03"
        stmt = select(tables["audit_log"]).order_by(tables["audit_log"].c.id)
        org_id = get_user_org_id(admin_a)
        if org_id is not None:
            stmt = stmt.where(tables["audit_log"].c.org_id == org_id)
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        stmt = stmt.where(tables["audit_log"].c.occurred_at >= start_dt)
        end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc) + timedelta(days=1)
        stmt = stmt.where(tables["audit_log"].c.occurred_at < end_dt)

        async with session_factory() as session:
            result = await session.execute(stmt)
            rows = result.fetchall()

        seen_ids = {row.id for row in rows}
        assert seen_ids == {"in_range_a"}, (
            "date range + org scoping must isolate org A's in-range row only — "
            f"got {seen_ids}"
        )


# ---------------------------------------------------------------------------
# get_user_org_id — None-safe accessor
# ---------------------------------------------------------------------------


class TestGetUserOrgId:
    def test_none_user_returns_none(self):
        assert get_user_org_id(None) is None

    def test_user_with_org_id_returns_it(self):
        assert get_user_org_id(_FakeUser(org_id="org-123")) == "org-123"

    def test_user_without_org_id_attr_returns_none(self):
        class _NoOrgField:
            pass

        assert get_user_org_id(_NoOrgField()) is None
