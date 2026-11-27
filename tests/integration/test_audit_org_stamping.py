"""DOCBOT-4016: browser audit events carry the actor's user_id + org_id.

/admin/audit-log filters on ``audit_log.org_id``; events written without it
were invisible to org admins. Runs the real login/register/logout handlers
against in-memory SQLite (no network) and reads back via /admin/audit-log.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import api.index as idx
from api import email_verification_service as evs
from api import oauth_service, rbac_service
from api.auth_service import create_user_session
from api.oauth_service import hash_password
from api.utils import request_guard

ORG = "org-a"
PASSWORD = "Sup3r-secret-pass!"


@pytest.fixture
async def factory(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    tables = [idx.organizations_table, idx.users_table, idx.user_sessions_table, idx.audit_log_table]
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: idx.metadata.create_all(c, tables=tables))
    fac = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(timezone.utc)
    async with fac() as s:
        async with s.begin():
            await s.execute(insert(idx.organizations_table).values(id=ORG, name="A", created_at=now))
            await s.execute(insert(idx.users_table), [
                {"id": "admin", "email": "admin@a.com", "name": "Admin", "provider": "email",
                 "role": "admin", "org_id": ORG, "email_verified": True,
                 "password_hash": None, "created_at": now},
                {"id": "alice", "email": "alice@a.com", "name": "Alice", "provider": "email",
                 "role": "analyst", "org_id": ORG, "email_verified": True,
                 "password_hash": hash_password(PASSWORD), "created_at": now},
            ])

    monkeypatch.setattr(idx, "async_session_factory", fac)
    monkeypatch.setattr(rbac_service, "_users_table", idx.users_table)
    monkeypatch.setattr(rbac_service, "_user_sessions_table", idx.user_sessions_table)
    monkeypatch.setattr(rbac_service, "_async_session_factory", fac)
    monkeypatch.setattr(oauth_service, "is_password_breached", AsyncMock(return_value=False))
    monkeypatch.setattr(evs, "is_email_verification_enabled", lambda: False)
    monkeypatch.setattr(request_guard, "RATE_LIMIT_PER_MIN", 0)
    monkeypatch.setattr(request_guard, "is_demo_blocked", lambda *_a, **_k: False)
    monkeypatch.setattr(request_guard, "check_ip_rate_limit", AsyncMock(return_value=None))
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    request_guard.reset_state()
    yield fac
    request_guard.reset_state()
    await engine.dispose()


def _client(token: str | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=idx.app), base_url="http://test",
        cookies={"docbot_session": token} if token else None,
    )


async def _drain() -> None:
    """Let log_event's fire-and-forget writes finish."""
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending:
        await asyncio.wait(pending, timeout=2)


async def _rows(fac, event_type):
    async with fac() as s:
        return (await s.execute(
            select(idx.audit_log_table).where(idx.audit_log_table.c.event_type == event_type)
        )).fetchall()


async def _admin_view(fac) -> list[dict]:
    token = await create_user_session("admin", idx.user_sessions_table, fac)
    async with _client(token) as c:
        resp = await c.get("/admin/audit-log")
    assert resp.status_code == 200
    return resp.json()["events"]


async def test_login_and_logout_stamped_and_visible_to_org_admin(factory):
    async with _client() as c:
        resp = await c.post("/api/auth/login", json={"email": "alice@a.com", "password": PASSWORD})
        assert resp.status_code == 200, resp.text
        token = resp.cookies["docbot_session"]
    await _drain()
    async with _client(token) as c:
        assert (await c.post("/api/auth/logout")).status_code == 200
    await _drain()

    for event_type in ("login", "logout"):
        rows = await _rows(factory, event_type)
        assert [(r.user_id, r.org_id) for r in rows] == [("alice", ORG)], event_type

    seen = {(e["event_type"], e["user_id"]) for e in await _admin_view(factory)}
    assert {("login", "alice"), ("logout", "alice")} <= seen


async def test_register_event_carries_new_users_org(factory):
    async with _client() as c:
        resp = await c.post(
            "/api/auth/register",
            json={"email": "new@b.com", "password": PASSWORD, "accepted_terms": True},
        )
    assert resp.status_code == 200
    await _drain()
    async with factory() as s:
        user = (await s.execute(
            select(idx.users_table).where(idx.users_table.c.email == "new@b.com")
        )).fetchone()
    rows = await _rows(factory, "login")
    assert user.org_id
    assert [(r.user_id, r.org_id) for r in rows] == [(user.id, user.org_id)]
