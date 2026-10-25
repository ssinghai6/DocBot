"""DOCBOT-1610: two-org isolation for every admin route (§4 security gate #2).

Since DOCBOT-1702 every signup is admin of its own personal org, so passing
require_admin() only proves "admin of SOME org". Each admin route must scope
to the caller's org: org B's admin gets 404 (or an empty/filtered listing)
on org A's resources, and platform-wide /admin/metrics is 403 to any admin
not listed in PLATFORM_ADMIN_USER_IDS.

Runs the real route handlers against an in-memory SQLite DB (no network, no
Postgres). Session resolution is stubbed: the cookie value is the user id.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import api.api_key_service as api_key_service
import api.index as idx
from api import rbac_service
from api.utils import request_guard

ORG_A, ORG_B = "org-a", "org-b"

USERS = {
    "a_admin": SimpleNamespace(id="a_admin", email="admin@a.com", role="admin", org_id=ORG_A),
    "a_viewer": SimpleNamespace(id="a_viewer", email="viewer@a.com", role="viewer", org_id=ORG_A),
    "b_admin": SimpleNamespace(id="b_admin", email="admin@b.com", role="admin", org_id=ORG_B),
    "orphan_admin": SimpleNamespace(id="orphan_admin", email="orphan@x.com", role="admin", org_id=None),
}


@pytest.fixture
async def db(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    tables = [idx.users_table, idx.audit_log_table, idx.api_keys_table, idx.organizations_table]
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: idx.metadata.create_all(c, tables=tables))
    factory = async_sessionmaker(engine, expire_on_commit=False)

    now = datetime.now(timezone.utc)
    async with factory() as s:
        async with s.begin():
            await s.execute(insert(idx.organizations_table), [
                {"id": ORG_A, "name": "A", "created_at": now},
                {"id": ORG_B, "name": "B", "created_at": now},
            ])
            await s.execute(insert(idx.users_table), [
                {"id": u.id, "email": u.email, "name": u.id, "provider": "email",
                 "role": u.role, "org_id": u.org_id, "created_at": now}
                for u in USERS.values()
            ])
            await s.execute(insert(idx.audit_log_table), [
                {"id": "ev_a", "event_type": "login", "user_id": "a_admin", "org_id": ORG_A, "occurred_at": now},
                {"id": "ev_b", "event_type": "login", "user_id": "b_admin", "org_id": ORG_B, "occurred_at": now},
                {"id": "ev_null", "event_type": "login", "user_id": "x", "org_id": None, "occurred_at": now},
            ])

    monkeypatch.setattr(idx, "async_session_factory", factory)
    monkeypatch.setattr(rbac_service, "_users_table", idx.users_table)
    monkeypatch.setattr(rbac_service, "_user_sessions_table", idx.user_sessions_table)
    monkeypatch.setattr(rbac_service, "_async_session_factory", factory)
    monkeypatch.setattr(api_key_service, "_api_keys_table", idx.api_keys_table)
    monkeypatch.setattr(api_key_service, "_async_session_factory", factory)
    monkeypatch.setattr(api_key_service, "_audit_log_table", idx.audit_log_table)
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    monkeypatch.delenv("PLATFORM_ADMIN_USER_IDS", raising=False)
    monkeypatch.setattr(request_guard, "RATE_LIMIT_PER_MIN", 0)
    request_guard.reset_state()

    async def _lookup(token, *_args):
        return USERS.get(token)

    with patch("api.auth_service.get_user_from_session", new=AsyncMock(side_effect=_lookup)):
        yield factory
    await engine.dispose()


def _client(as_user: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=idx.app),
        base_url="http://test",
        cookies={"docbot_session": as_user},
    )


async def _role_of(factory, user_id):
    async with factory() as s:
        return (await s.execute(select(idx.users_table.c.role).where(idx.users_table.c.id == user_id))).scalar_one()


async def _seed_key(factory, org_id):
    async with factory() as s:
        async with s.begin():
            await s.execute(insert(idx.api_keys_table).values(
                id=f"key_{org_id}", org_id=org_id, key_hash=f"h_{org_id}", key_prefix="dk_live_x",
                name="k", scopes='["hybrid:read"]', created_at=datetime.now(timezone.utc),
            ))


# --- /admin/users ----------------------------------------------------------

async def test_users_list_scoped_to_caller_org(db):
    async with _client("b_admin") as c:
        resp = await c.get("/admin/users")
    assert resp.status_code == 200
    assert {u["id"] for u in resp.json()["users"]} == {"b_admin"}


async def test_admin_without_org_fails_closed(db):
    async with _client("orphan_admin") as c:
        for path in ("/admin/users", "/admin/audit-log", "/admin/api-keys"):
            assert (await c.get(path)).status_code == 403, path


# --- /admin/audit-log ------------------------------------------------------

async def test_audit_log_scoped_to_caller_org(db):
    async with _client("b_admin") as c:
        resp = await c.get("/admin/audit-log")
        csv_resp = await c.get("/admin/audit-log", params={"format": "csv"})
    assert {e["id"] for e in resp.json()["events"]} == {"ev_b"}
    assert "ev_a" not in csv_resp.text and "ev_null" not in csv_resp.text


async def test_audit_log_user_filter_cannot_cross_org(db):
    async with _client("b_admin") as c:
        resp = await c.get("/admin/audit-log", params={"user_id": "a_admin"})
    assert resp.json()["events"] == []


# --- PATCH /admin/users/{id}/role -------------------------------------------

async def test_other_org_admin_cannot_change_role(db):
    async with _client("b_admin") as c:
        resp = await c.patch("/admin/users/a_viewer/role", json={"role": "admin"})
    assert resp.status_code == 404
    assert await _role_of(db, "a_viewer") == "viewer"


async def test_other_org_admin_cannot_demote_admin(db):
    async with _client("b_admin") as c:
        resp = await c.patch("/admin/users/a_admin/role", json={"role": "viewer"})
    assert resp.status_code == 404
    assert await _role_of(db, "a_admin") == "admin"


async def test_same_org_admin_can_change_role(db):
    async with _client("a_admin") as c:
        resp = await c.patch("/admin/users/a_viewer/role", json={"role": "analyst"})
    assert resp.status_code == 200
    assert await _role_of(db, "a_viewer") == "analyst"


async def test_last_admin_cannot_demote_self(db):
    async with _client("b_admin") as c:
        resp = await c.patch("/admin/users/b_admin/role", json={"role": "viewer"})
    assert resp.status_code == 409
    assert await _role_of(db, "b_admin") == "admin"


async def test_admin_can_step_down_when_another_admin_exists(db):
    async with _client("a_admin") as c:
        assert (await c.patch("/admin/users/a_viewer/role", json={"role": "admin"})).status_code == 200
        resp = await c.patch("/admin/users/a_admin/role", json={"role": "analyst"})
    assert resp.status_code == 200
    assert await _role_of(db, "a_admin") == "analyst"


async def test_invalid_role_rejected(db):
    async with _client("a_admin") as c:
        resp = await c.patch("/admin/users/a_viewer/role", json={"role": "superuser"})
    assert resp.status_code == 400


# --- /admin/api-keys -------------------------------------------------------

async def test_api_key_list_scoped(db):
    await _seed_key(db, ORG_A)
    async with _client("b_admin") as c:
        resp = await c.get("/admin/api-keys")
    assert resp.status_code == 200
    assert resp.json()["org_id"] == ORG_B
    assert resp.json()["keys"] == []


async def test_api_key_revoke_cross_org_404(db):
    await _seed_key(db, ORG_A)
    async with _client("b_admin") as c:
        resp = await c.delete(f"/admin/api-keys/key_{ORG_A}")
    assert resp.status_code == 404
    async with db() as s:
        revoked = (await s.execute(
            select(idx.api_keys_table.c.revoked_at).where(idx.api_keys_table.c.id == f"key_{ORG_A}")
        )).scalar_one()
    assert revoked is None


async def test_api_key_create_lands_in_caller_org(db):
    async with _client("b_admin") as c:
        resp = await c.post("/admin/api-keys", json={"name": "ci", "scopes": ["hybrid:read"]})
    assert resp.status_code == 200
    assert resp.json()["key"]["org_id"] == ORG_B


# --- /admin/metrics (platform-wide) ----------------------------------------

async def test_metrics_forbidden_to_org_admins_by_default(db):
    with patch("api.metrics_service.get_platform_metrics", new=AsyncMock(return_value={"ok": 1})) as m:
        async with _client("b_admin") as c:
            resp = await c.get("/admin/metrics")
    assert resp.status_code == 403
    m.assert_not_called()


async def test_metrics_forbidden_to_unlisted_admin(db, monkeypatch):
    monkeypatch.setenv("PLATFORM_ADMIN_USER_IDS", "a_admin")
    with patch("api.metrics_service.get_platform_metrics", new=AsyncMock(return_value={"ok": 1})):
        async with _client("b_admin") as c:
            resp = await c.get("/admin/metrics")
    assert resp.status_code == 403


async def test_metrics_allowed_for_listed_platform_admin(db, monkeypatch):
    monkeypatch.setenv("PLATFORM_ADMIN_USER_IDS", " a_admin , other ")
    with patch("api.metrics_service.get_platform_metrics", new=AsyncMock(return_value={"ok": 1})):
        async with _client("a_admin") as c:
            resp = await c.get("/admin/metrics")
    assert resp.status_code == 200
    assert resp.json() == {"ok": 1}


async def test_platform_allowlist_still_requires_admin_role(db, monkeypatch):
    monkeypatch.setenv("PLATFORM_ADMIN_USER_IDS", "a_viewer")
    async with _client("a_viewer") as c:
        resp = await c.get("/admin/metrics")
    assert resp.status_code == 403
