"""DOCBOT-1615 — platform daily spend kill switch (api/utils/spend_guard.py).

Real SQLite (aiosqlite) for the cost aggregate + credit ledger; a tiny FastAPI
app exercises the dependency end to end.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import MetaData, insert
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api import credit_service
from api.llm_trace_service import register_llm_calls_table
from api.usage_service import register_usage_events_table
from api.utils import caller_scope, request_guard, spend_guard
from api.utils.caller_scope import CallerScope


@pytest.fixture
async def store(tmp_path, monkeypatch):
    metadata = MetaData()
    llm = register_llm_calls_table(metadata)
    ue = register_usage_events_table(metadata)
    credit_tables = credit_service.register_credit_tables(metadata)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'spend.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    spend_guard.wire_spend_guard(llm, ue, factory)
    credit_service.wire_credit_store(credit_tables, factory, "sqlite")
    monkeypatch.setenv("PLATFORM_DAILY_USD_CAP", "1")
    yield llm, ue, credit_tables, factory
    spend_guard._session_factory = None
    credit_service._tables = None
    credit_service._session_factory = None
    await engine.dispose()


async def _spend(factory, llm, ue, *, llm_usd=0.0, sandbox_micro=0, byok_usd=0.0, when=None):
    when = when or datetime.now(timezone.utc)
    async with factory() as s, s.begin():
        for usd, ks in ((llm_usd, "platform"), (byok_usd, "byok")):
            if usd:
                await s.execute(insert(llm).values(
                    id=str(uuid.uuid4()), run_id="r", provider="groq", model="m", success=True,
                    estimated_cost_usd=usd, key_source=ks, created_at=when,
                ))
        if sandbox_micro:
            await s.execute(insert(ue).values(
                id=str(uuid.uuid4()), kind="sandbox", cost_micro_usd=sandbox_micro, created_at=when,
            ))


async def _make_paid(factory, tables, org_id):
    async with factory() as s, s.begin():
        await s.execute(insert(tables.credit_ledger).values(
            id=str(uuid.uuid4()), org_id=org_id, kind="purchase", ref="ch_1",
            delta_credits=1000, created_at=datetime.now(timezone.utc),
        ))


def _client(monkeypatch, scope: CallerScope) -> TestClient:
    monkeypatch.setattr(caller_scope, "resolve_caller_scope", AsyncMock(return_value=scope))
    app = FastAPI()

    @app.post("/api/chat", dependencies=[Depends(spend_guard.check_platform_capacity)])
    async def chat():
        return {"ok": True}

    @app.post("/api/demo/init", dependencies=[Depends(spend_guard.check_platform_capacity)])
    async def demo():
        return {"ok": True}

    return TestClient(app)


ANON = CallerScope(user_id=None, org_id=None, anon_token="a" * 64)
FREE = CallerScope(user_id="u-free", org_id="org-free", anon_token=None)
PAID = CallerScope(user_id="u-paid", org_id="org-paid", anon_token=None)


async def test_spend_aggregates_llm_and_sandbox_excluding_byok_and_yesterday(store):
    llm, ue, _, factory = store
    await _spend(factory, llm, ue, llm_usd=0.25, sandbox_micro=500_000, byok_usd=9.0)
    await _spend(factory, llm, ue, llm_usd=50.0, when=datetime.now(timezone.utc) - timedelta(days=2))
    assert await spend_guard.todays_spend_usd() == pytest.approx(0.75)


async def test_under_cap_everyone_passes(store, monkeypatch):
    llm, ue, _, factory = store
    await _spend(factory, llm, ue, llm_usd=0.5)
    assert _client(monkeypatch, ANON).post("/api/chat").status_code == 200


async def test_over_cap_blocks_anon_free_and_demo_but_not_paid(store, monkeypatch):
    llm, ue, tables, factory = store
    await _spend(factory, llm, ue, llm_usd=2.0)
    await _make_paid(factory, tables, "org-paid")

    for scope in (ANON, FREE):
        resp = _client(monkeypatch, scope).post("/api/chat")
        assert resp.status_code == 503
        assert resp.json()["detail"] == spend_guard.CAPACITY_ERROR
    assert _client(monkeypatch, PAID).post("/api/chat").status_code == 200
    # demo routes are free tier even for a paid org
    assert _client(monkeypatch, PAID).post("/api/demo/init").status_code == 503


async def test_owner_bypasses(store, monkeypatch):
    llm, ue, _, factory = store
    await _spend(factory, llm, ue, llm_usd=2.0)
    monkeypatch.setattr(request_guard, "is_owner", lambda _r: True)
    assert _client(monkeypatch, ANON).post("/api/chat").status_code == 200


async def test_zero_cap_disables(store, monkeypatch):
    llm, ue, _, factory = store
    await _spend(factory, llm, ue, llm_usd=2.0)
    monkeypatch.setenv("PLATFORM_DAILY_USD_CAP", "0")
    assert _client(monkeypatch, ANON).post("/api/chat").status_code == 200


async def test_alert_fires_once_per_day(store, monkeypatch):
    llm, ue, _, factory = store
    await _spend(factory, llm, ue, llm_usd=2.0)
    sentry = MagicMock()
    sentry.is_initialized.return_value = True
    monkeypatch.setitem(__import__("sys").modules, "sentry_sdk", sentry)
    client = _client(monkeypatch, ANON)
    for _ in range(3):
        assert client.post("/api/chat").status_code == 503
    assert sentry.capture_message.call_count == 1


async def test_fail_open_on_db_error(store, monkeypatch):
    monkeypatch.setattr(
        spend_guard, "_query_spend", AsyncMock(side_effect=OperationalError("x", {}, Exception()))
    )
    assert await spend_guard.todays_spend_usd() is None
    assert _client(monkeypatch, ANON).post("/api/chat").status_code == 200


async def test_result_is_cached(store, monkeypatch):
    llm, ue, _, factory = store
    q = AsyncMock(return_value=2.0)
    monkeypatch.setattr(spend_guard, "_query_spend", q)
    assert await spend_guard.todays_spend_usd() == 2.0
    assert await spend_guard.todays_spend_usd() == 2.0
    assert q.await_count == 1
    # expired cache → re-query
    day, val, at = spend_guard._cache
    spend_guard._cache = (day, val, at - spend_guard._CACHE_TTL_SEC - 1)
    await spend_guard.todays_spend_usd()
    assert q.await_count == 2
