"""DOCBOT-1702 — personal org per signup, legacy-only backfill, remediation.

In-memory SQLite (aiosqlite); no network, no live PostgreSQL.
"""

from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

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

from api.auth_service import jit_provision_user
from api.org_service import (
    DEFAULT_ORG_NAME,
    PERSONAL_ORG_NAME_SUFFIX,
    backfill_default_organization,
    get_user_org,
    personal_org_name,
    register_organizations_table,
    remediate_personal_orgs,
)

T0 = datetime(2026, 10, 12, 11, 0, tzinfo=timezone.utc)


def _schema() -> tuple[MetaData, dict[str, Table]]:
    md = MetaData()
    orgs = register_organizations_table(md)
    users = Table(
        "users", md,
        Column("id", String, primary_key=True),
        Column("email", String, nullable=False, unique=True),
        Column("name", String, nullable=False),
        Column("provider", String, nullable=False),
        Column("provider_id", String),
        Column("password_hash", Text),
        Column("role", String, server_default="analyst", nullable=False),
        Column("last_login_at", DateTime(timezone=True)),
        Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
        Column("org_id", String),
    )
    sessions = Table(
        "sessions", md,
        Column("session_id", String, primary_key=True),
        Column("user_id", String),
        Column("org_id", String),
    )
    db_connections = Table(
        "db_connections", md,
        Column("id", String, primary_key=True),
        Column("session_id", String, nullable=False),
        Column("credentials_blob", Text, nullable=False, server_default="x"),
        Column("user_id", String),
        Column("org_id", String),
    )
    marketplace = Table(
        "marketplace_connections", md,
        Column("id", String, primary_key=True),
        Column("connector_type", String, nullable=False),
        Column("is_active", Boolean, server_default="1"),
        Column("org_id", String),
    )
    return md, {
        "organizations": orgs, "users": users, "sessions": sessions,
        "db_connections": db_connections, "marketplace_connections": marketplace,
    }


@pytest.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    md, tables = _schema()
    async with engine.begin() as conn:
        await conn.run_sync(md.create_all)
    yield engine, async_sessionmaker(engine, expire_on_commit=False), tables
    await engine.dispose()


async def _rows(factory, table):
    async with factory() as s:
        return (await s.execute(select(table))).fetchall()


async def _user(factory, tables, uid):
    async with factory() as s:
        return (await s.execute(select(tables["users"]).where(tables["users"].c.id == uid))).fetchone()


def _attrs(email, name="Alice", provider="email"):
    return {"email": email, "name": name, "provider": provider, "password_hash": "h"}


# ---------------------------------------------------------------------------
# Signup → personal org
# ---------------------------------------------------------------------------


class TestPersonalOrgOnSignup:
    async def test_two_signups_get_two_distinct_orgs_each_admin(self, db):
        _, factory, tables = db
        a = await jit_provision_user(_attrs("a@x.com", "Alice"), tables["users"], factory)
        b = await jit_provision_user(_attrs("b@y.com", "Bob", "github"), tables["users"], factory)
        ua, ub = await _user(factory, tables, a), await _user(factory, tables, b)
        assert ua.org_id and ub.org_id and ua.org_id != ub.org_id
        assert ua.role == "admin" and ub.role == "admin"
        orgs = {o.id: o.name for o in await _rows(factory, tables["organizations"])}
        assert set(orgs) == {ua.org_id, ub.org_id}
        assert all(n.endswith(PERSONAL_ORG_NAME_SUFFIX) for n in orgs.values())
        assert DEFAULT_ORG_NAME not in orgs.values()

    @pytest.mark.parametrize("provider", ["saml", "okta", "azure_ad", "github", "google", "email"])
    async def test_every_provider_path_gets_an_org(self, db, provider):
        _, factory, tables = db
        uid = await jit_provision_user(
            {"email": f"{provider}@x.com", "name": "N", "provider": provider, "provider_id": "1"},
            tables["users"], factory,
        )
        assert (await _user(factory, tables, uid)).org_id is not None

    async def test_returning_user_keeps_org_and_role(self, db):
        _, factory, tables = db
        uid = await jit_provision_user(_attrs("a@x.com"), tables["users"], factory)
        async with factory() as s:
            async with s.begin():
                await s.execute(tables["users"].update().values(role="viewer"))
        org_before = (await _user(factory, tables, uid)).org_id
        again = await jit_provision_user(_attrs("a@x.com", "Alice 2"), tables["users"], factory)
        u = await _user(factory, tables, uid)
        assert again == uid and u.org_id == org_before and u.role == "viewer"
        assert len(await _rows(factory, tables["organizations"])) == 1

    async def test_failed_user_insert_rolls_back_org(self, db):
        """Atomicity: if the user INSERT fails, the org INSERT is rolled back."""
        _, factory, tables = db
        bad = _attrs("a@x.com")
        bad["name"] = None  # violates users.name NOT NULL → user insert fails
        with pytest.raises(Exception):
            await jit_provision_user(bad, tables["users"], factory)
        assert await _rows(factory, tables["organizations"]) == []
        assert await _rows(factory, tables["users"]) == []

    def test_personal_org_name(self):
        assert personal_org_name("Ann") == "Ann's workspace"
        assert personal_org_name("  ") == "Personal workspace"
        assert personal_org_name(None) == "Personal workspace"
        assert personal_org_name(DEFAULT_ORG_NAME) != DEFAULT_ORG_NAME


# ---------------------------------------------------------------------------
# Backfill: legacy-only, idempotent
# ---------------------------------------------------------------------------


class TestBackfillLegacyOnly:
    async def test_backfill_does_not_move_personal_org_users(self, db):
        engine, factory, tables = db
        async with engine.begin() as conn:
            default = await backfill_default_organization(conn)
        uid = await jit_provision_user(_attrs("new@x.com"), tables["users"], factory)
        personal = (await _user(factory, tables, uid)).org_id
        async with factory() as s:
            async with s.begin():
                await s.execute(tables["users"].insert().values(
                    id="legacy", email="old@x.com", name="Old", provider="email"))
                # personal user's session written with NULL org_id (route didn't set it)
                await s.execute(tables["sessions"].insert().values(session_id="s_new", user_id=uid))
                await s.execute(tables["db_connections"].insert().values(
                    id="c_new", session_id="s_new", user_id=uid))
                await s.execute(tables["sessions"].insert().values(session_id="s_old", user_id="legacy"))
                await s.execute(tables["sessions"].insert().values(session_id="s_guest"))

        for _ in range(2):  # idempotent
            async with engine.begin() as conn:
                assert await backfill_default_organization(conn) == default

        assert (await _user(factory, tables, uid)).org_id == personal
        assert (await _user(factory, tables, "legacy")).org_id == default
        sess = {r.session_id: r.org_id for r in await _rows(factory, tables["sessions"])}
        assert sess == {"s_new": personal, "s_old": default, "s_guest": default}
        conns = {r.id: r.org_id for r in await _rows(factory, tables["db_connections"])}
        assert conns == {"c_new": personal}
        names = [o.name for o in await _rows(factory, tables["organizations"])]
        assert names.count(DEFAULT_ORG_NAME) == 1

    async def test_default_org_found_by_name_not_age(self, db):
        """A personal org created before the default org must not be reused as default."""
        engine, factory, tables = db
        uid = await jit_provision_user(_attrs("first@x.com"), tables["users"], factory)
        personal = (await _user(factory, tables, uid)).org_id
        async with engine.begin() as conn:
            default = await backfill_default_organization(conn)
        assert default != personal


# ---------------------------------------------------------------------------
# Remediation
# ---------------------------------------------------------------------------


async def _seed_post_1524_world(factory, tables) -> str:
    """Default org at T0; legacy users before T0; post-1524 signups after T0."""
    async with factory() as s:
        async with s.begin():
            await s.execute(tables["organizations"].insert().values(
                id="org_default", name=DEFAULT_ORG_NAME, created_at=T0))
            await s.execute(tables["organizations"].insert().values(
                id="org_other", name="Bob's workspace", created_at=T0 + timedelta(days=3)))
            u = tables["users"]
            rows = [
                ("legacy", T0 - timedelta(days=30), "org_default", "admin"),
                ("legacy2", T0 - timedelta(days=1), "org_default", "analyst"),
                ("late1", T0 + timedelta(hours=1), "org_default", "analyst"),
                ("late2", T0 + timedelta(days=2), None, "analyst"),
                ("late_admin", T0 + timedelta(days=1), "org_default", "admin"),
                ("already", T0 + timedelta(days=3), "org_other", "admin"),
            ]
            for uid, created, org, role in rows:
                await s.execute(u.insert().values(
                    id=uid, email=f"{uid}@x.com", name=uid, provider="email",
                    created_at=created, org_id=org, role=role))
            await s.execute(tables["sessions"].insert().values(
                session_id="s_late1", user_id="late1", org_id="org_default"))
            await s.execute(tables["sessions"].insert().values(
                session_id="s_legacy", user_id="legacy", org_id="org_default"))
            await s.execute(tables["db_connections"].insert().values(
                id="c_late1", session_id="s_late1", user_id="late1", org_id="org_default",
                credentials_blob="ENV1:org_default:token"))
            await s.execute(tables["marketplace_connections"].insert().values(
                id="m1", connector_type="shopify", org_id="org_default"))
    return "org_default"


class TestRemediation:
    async def test_dry_run_makes_no_changes(self, db):
        engine, factory, tables = db
        await _seed_post_1524_world(factory, tables)
        before = {t: sorted(map(tuple, await _rows(factory, tables[t]))) for t in tables}
        async with engine.begin() as conn:
            report = await remediate_personal_orgs(conn, apply=False)
        after = {t: sorted(map(tuple, await _rows(factory, tables[t]))) for t in tables}
        assert before == after
        assert not report.applied
        assert sorted(m.user_id for m in report.moves) == ["late1", "late2"]
        assert report.skipped_admins == ["late_admin"]
        late1 = next(m for m in report.moves if m.user_id == "late1")
        assert (late1.sessions, late1.db_connections, late1.connections_on_default_dek) == (1, 1, 1)
        assert report.unattributable_marketplace_connections == 1

    async def test_apply_moves_only_post_1524_signups(self, db):
        engine, factory, tables = db
        await _seed_post_1524_world(factory, tables)
        async with engine.begin() as conn:
            report = await remediate_personal_orgs(conn, apply=True)
        assert report.applied
        users = {r.id: r for r in await _rows(factory, tables["users"])}
        for legacy in ("legacy", "legacy2", "late_admin"):
            assert users[legacy].org_id == "org_default"
        assert users["already"].org_id == "org_other"
        new1, new2 = users["late1"].org_id, users["late2"].org_id
        assert new1 and new2 and len({new1, new2, "org_default", "org_other"}) == 4
        assert users["late1"].role == "admin" and users["late2"].role == "admin"
        sess = {r.session_id: r.org_id for r in await _rows(factory, tables["sessions"])}
        assert sess == {"s_late1": new1, "s_legacy": "org_default"}
        conns = {r.id: r.org_id for r in await _rows(factory, tables["db_connections"])}
        assert conns == {"c_late1": new1}
        # summary contains ids only — no emails
        text_out = "\n".join(report.summary_lines())
        assert "late1" in text_out and "@x.com" not in text_out

        # idempotent: second run selects nobody
        async with engine.begin() as conn:
            again = await remediate_personal_orgs(conn, apply=True)
        assert again.moves == []

    async def test_options_include_admins_exclude_keep_roles_since(self, db):
        engine, factory, tables = db
        await _seed_post_1524_world(factory, tables)
        async with engine.begin() as conn:
            report = await remediate_personal_orgs(
                conn, apply=True, include_admins=True, exclude_user_ids=["late2"],
                promote_to_admin=False, since=T0 + timedelta(minutes=30),
            )
        assert sorted(m.user_id for m in report.moves) == ["late1", "late_admin"]
        assert report.skipped_excluded == ["late2"]
        users = {r.id: r for r in await _rows(factory, tables["users"])}
        assert users["late1"].role == "analyst"
        assert users["late2"].org_id is None

    async def test_no_default_org_is_noop(self, db):
        engine, _, _ = db
        async with engine.begin() as conn:
            report = await remediate_personal_orgs(conn, apply=True)
        assert report.default_org_id is None and report.moves == []

    async def test_script_dry_run_rolls_back_and_apply_commits(self, db):
        engine, factory, tables = db
        await _seed_post_1524_world(factory, tables)
        path = Path(__file__).resolve().parents[2] / "scripts" / "remediate_personal_orgs.py"
        spec = importlib.util.spec_from_file_location("remediate_personal_orgs", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        args = mod.build_parser().parse_args([])
        assert args.apply is False  # dry run is the default
        report = await mod.run(engine, args)
        assert len(report.moves) == 2
        assert (await _user(factory, tables, "late1")).org_id == "org_default"

        report = await mod.run(engine, mod.build_parser().parse_args(["--apply"]))
        assert report.applied
        assert (await _user(factory, tables, "late1")).org_id != "org_default"
        assert isinstance(args, argparse.Namespace)


# ---------------------------------------------------------------------------
# §4 gate #2 — org B cannot see org A's rows through org-scoped helpers
# ---------------------------------------------------------------------------


class TestTwoOrgIsolation:
    async def test_org_b_cannot_see_org_a_rows(self, db):
        engine, factory, tables = db
        a = await jit_provision_user(_attrs("a@x.com", "A"), tables["users"], factory)
        b = await jit_provision_user(_attrs("b@x.com", "B"), tables["users"], factory)
        org_a = (await _user(factory, tables, a)).org_id
        org_b = (await _user(factory, tables, b)).org_id
        async with factory() as s:
            async with s.begin():
                await s.execute(tables["sessions"].insert().values(session_id="sa", user_id=a))
        async with engine.begin() as conn:
            await backfill_default_organization(conn)

        async with factory() as s:
            # Admin-users listing scoped like /admin/users
            seen = (await s.execute(
                select(tables["users"].c.id).where(tables["users"].c.org_id == org_b)
            )).scalars().all()
            assert seen == [b]
            # Org-B-scoped session lookup for org A's id → not found (→ 404)
            hit = (await s.execute(
                select(tables["sessions"]).where(
                    tables["sessions"].c.session_id == "sa",
                    tables["sessions"].c.org_id == org_b,
                )
            )).fetchone()
            assert hit is None
            # get_user_org helper: target in org A is not in org B → route must 404
            assert await get_user_org(s, a) == org_a != org_b
            assert await get_user_org(s, "missing") is None
