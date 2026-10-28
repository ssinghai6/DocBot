"""DOCBOT-1706 — server-side funnel events + /admin/funnel report.

Real SQLite (aiosqlite) so the unique-index dedupe and ON CONFLICT DO NOTHING
are exercised by the actual engine.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from sqlalchemy import MetaData, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api import credit_service, funnel_service
from api.utils.billing_context import Principal

os.environ.setdefault("DB_ENCRYPTION_KEY", Fernet.generate_key().decode())
os.environ.setdefault("DATABASE_URL", "postgresql://fake:fake@localhost:5432/testdb")


@pytest.fixture
async def store(tmp_path):
    metadata = MetaData()
    table = funnel_service.register_funnel_events_table(metadata)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'funnel.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    funnel_service.wire_funnel_store(table, factory, "sqlite")
    yield table, factory
    funnel_service._table = None
    funnel_service._session_factory = None
    await engine.dispose()


async def _drain() -> None:
    await asyncio.gather(*list(funnel_service._tasks))


async def _rows(table, factory):
    async with factory() as s:
        return (await s.execute(select(table.c.user_id, table.c.anon_owner, table.c.event))).all()


async def test_recorded_once_per_user_per_event(store):
    table, factory = store
    for _ in range(3):
        funnel_service.record_funnel_event("first_answer", user_id="u1", org_id="o1")
        await _drain()
        funnel_service._seen.clear()  # force the DB unique index to do the dedupe
    funnel_service.record_funnel_event("first_upload", user_id="u1")
    funnel_service.record_funnel_event("first_answer", user_id="u2")
    funnel_service.record_funnel_event("first_answer", anon_owner="anon-1")
    funnel_service.record_funnel_event("first_answer", anon_owner="anon-1")
    await _drain()
    rows = sorted(await _rows(table, factory), key=lambda r: (r.event, r.user_id or "", r.anon_owner or ""))
    assert [(r.user_id, r.anon_owner, r.event) for r in rows] == [
        (None, "anon-1", "first_answer"),
        ("u1", None, "first_answer"),
        ("u2", None, "first_answer"),
        ("u1", None, "first_upload"),
    ]


async def test_unknown_event_and_keyless_are_ignored(store):
    table, factory = store
    funnel_service.record_funnel_event("not_a_step", user_id="u1")
    funnel_service.record_funnel_event("signup")
    await _drain()
    assert await _rows(table, factory) == []


def test_never_raises_without_a_running_loop(store):
    funnel_service.record_funnel_event("signup", user_id="u-sync")  # no loop → dropped silently


async def test_db_error_never_raises(store):
    broken = MagicMock(side_effect=OSError("db down"))
    funnel_service._session_factory = broken
    funnel_service.record_funnel_event("signup", user_id="u-err")
    await _drain()
    assert ("u-err", "signup") not in funnel_service._seen  # retry allowed later


async def test_report_counts_and_conversion(store):
    for i in range(4):
        funnel_service.record_funnel_event("signup", user_id=f"u{i}")
    for i in range(2):
        funnel_service.record_funnel_event("email_verified", user_id=f"u{i}")
    funnel_service.record_funnel_event("first_upload", user_id="u0")
    await _drain()
    report = await funnel_service.get_funnel_report(30)
    steps = {s.event: s for s in report.steps}
    assert [s.event for s in report.steps] == list(funnel_service.FUNNEL_EVENTS)
    assert steps["signup"].count == 4 and steps["signup"].conversion_pct is None
    assert steps["email_verified"].conversion_pct == 50.0
    assert steps["first_upload"].conversion_pct == 50.0
    assert steps["first_answer"].count == 0 and steps["first_answer"].conversion_pct == 0.0
    assert steps["first_tieout"].conversion_pct is None  # zero denominator


async def test_billing_stream_records_first_answer_only_on_success(monkeypatch):
    calls = []
    monkeypatch.setattr(credit_service, "record_funnel_event", lambda e, **kw: calls.append((e, kw)))

    async def ok():
        yield 'data: {"type": "token", "content": "hi"}\n\n'

    async def err():
        yield 'data: {"type": "error", "detail": "x"}\n\n'

    for gen in (ok, err):
        run = credit_service.BillingRun(
            route="chat", run_id="r", org_id="o1", exempt=True,
            principal=Principal(user_id="u1", org_id="o1", role="viewer", auth_method="session"),
        )
        async for _ in run.stream(gen()):
            pass
    assert [c[0] for c in calls] == ["first_answer"]
    assert calls[0][1]["user_id"] == "u1"


async def test_402_records_paywall_shown(monkeypatch):
    calls = []
    monkeypatch.setattr(credit_service, "record_funnel_event", lambda e, **kw: calls.append((e, kw)))
    monkeypatch.setattr(credit_service, "is_wired", lambda: True)
    monkeypatch.setattr(credit_service, "is_enforced", lambda: True)
    monkeypatch.setattr(
        credit_service, "reserve",
        AsyncMock(side_effect=credit_service.InsufficientCredits(10, 0, False)),
    )
    with pytest.raises(Exception) as exc_info:
        await credit_service.start_billing_run(route="chat", user_id="u1", org_id="o1")
    assert getattr(exc_info.value, "status_code", None) == 402
    assert calls == [("paywall_shown", {"user_id": "u1", "org_id": "o1", "props": {"route": "chat"}})]


def test_admin_funnel_is_platform_admin_only(monkeypatch):
    import api.index as idx

    client = TestClient(idx.app, raise_server_exceptions=False)
    assert client.get("/admin/funnel").status_code in (401, 403)

    report = funnel_service.FunnelReport(days=7, since="2026-01-01T00:00:00Z", steps=[])
    monkeypatch.setattr(funnel_service, "get_funnel_report", AsyncMock(return_value=report))
    idx.app.dependency_overrides[idx._rbac_platform_admin.dependency] = lambda: object()
    try:
        resp = client.get("/admin/funnel?days=7")
        assert resp.status_code == 200
        assert resp.json()["days"] == 7
        assert client.get("/admin/funnel?days=0").status_code == 422
    finally:
        idx.app.dependency_overrides.clear()
