"""DOCBOT-2330 — /v1 LLM routes are billed to the API key's org, spend-capped and IP-limited.

No network: the API-key dependency is overridden, the hybrid/autopilot
pipelines are fake generators, and the credit ledger / spend tables live in a
temp SQLite file. The app lifespan is not run.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from sqlalchemy import MetaData, insert, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import api.index as idx
import api.routes.v1_routes as v1_routes
from api import credit_service
from api.llm_trace_service import register_llm_calls_table
from api.usage_service import register_usage_events_table
from api.utils import request_guard, spend_guard
from api.utils.billing_context import get_meter

ORG = "org-key"
HEADERS = {"Authorization": "Bearer dk_test"}
BODY = {"question": "q", "session_id": "s-1"}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("BILLING_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BILLING_FREE_MONTHLY_CREDITS", "500")
    monkeypatch.setenv("PLATFORM_DAILY_USD_CAP", "0")
    monkeypatch.setattr(request_guard, "RATE_LIMIT_PER_MIN", 0)
    monkeypatch.setattr(request_guard, "OWNER_KEY", "")
    request_guard.reset_state()
    yield
    request_guard.reset_state()
    credit_service._tables = None
    credit_service._session_factory = None
    credit_service._granted_period.clear()
    spend_guard._session_factory = None


@pytest.fixture
async def store(tmp_path):
    metadata = MetaData()
    tables = credit_service.register_credit_tables(metadata)
    llm = register_llm_calls_table(metadata)
    ue = register_usage_events_table(metadata)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'v1.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    credit_service.wire_credit_store(tables, factory, "sqlite")
    spend_guard.wire_spend_guard(llm, ue, factory)
    yield SimpleNamespace(tables=tables, llm=llm, factory=factory)
    await engine.dispose()


def _fake_key(request: Request):
    request.state.org_id = ORG  # what require_api_key does on success
    return SimpleNamespace(id="key-1", org_id=ORG)


async def _fake_hybrid(**_kw):
    get_meter().record_llm(100, 50, "0.002")  # 2_000 micro-USD -> 7 credits
    yield f"data: {json.dumps({'type': 'token', 'content': 'hi'})}\n\n"
    yield f"data: {json.dumps({'type': 'done', 'citations': []})}\n\n"


async def _fake_autopilot(**_kw):
    get_meter().record_llm(100, 50, "0.002")
    yield f"data: {json.dumps({'type': 'answer', 'content': 'ok'})}\n\n"


@pytest.fixture
def client(monkeypatch):
    overrides = {v1_routes._v1_hybrid_read.dependency: _fake_key, v1_routes._v1_autopilot_run.dependency: _fake_key}
    idx.app.dependency_overrides.update(overrides)
    monkeypatch.setattr(v1_routes, "_require_v1_resources", AsyncMock())
    monkeypatch.setitem(v1_routes._d, "charge_budget", AsyncMock())
    sources = SimpleNamespace(has_docs=True, has_db=False, has_csv=False)
    with patch("api.audit_service.log_event", MagicMock()), \
         patch("api.hybrid_service.hybrid_chat", _fake_hybrid), \
         patch("api.autopilot_service.run_autopilot", _fake_autopilot), \
         patch("api.utils.autopilot_sources.resolve_autopilot_sources", AsyncMock(return_value=sources)):
        yield TestClient(idx.app, raise_server_exceptions=False)
    for dep in overrides:
        idx.app.dependency_overrides.pop(dep, None)


async def _balance(store):
    ob = store.tables.org_balances
    async with store.factory() as s:
        return (await s.execute(select(ob.c.balance_credits, ob.c.reserved_credits)
                                .where(ob.c.org_id == ORG))).first()


@pytest.mark.parametrize("path", ["/v1/hybrid/chat", "/v1/autopilot/run"])
async def test_v1_call_settles_against_key_org(store, client, path):
    resp = client.post(path, json=BODY, headers=HEADERS)
    assert resp.status_code == 200, resp.text
    row = await _balance(store)
    assert (row.balance_credits, row.reserved_credits) == (493, 0)
    lg = store.tables.credit_ledger
    async with store.factory() as s:
        usage = (await s.execute(select(lg).where(lg.c.org_id == ORG, lg.c.kind == "usage"))).all()
    assert len(usage) == 1 and usage[0].delta_credits == -7


@pytest.mark.parametrize("path", ["/v1/hybrid/chat", "/v1/autopilot/run"])
async def test_enforced_empty_balance_is_402(store, client, monkeypatch, path):
    monkeypatch.setenv("BILLING_ENFORCE", "true")
    monkeypatch.setenv("BILLING_FREE_MONTHLY_CREDITS", "0")
    resp = client.post(path, json=BODY, headers=HEADERS)
    assert resp.status_code == 402, resp.text
    # DOCBOT-3901: /v1 errors use the envelope; the QuotaErrorPayload code survives.
    assert resp.json()["error"]["code"] == "insufficient_credits"


async def test_demo_prefix_without_server_session_is_billed(store, client):
    resp = client.post("/v1/hybrid/chat", json={**BODY, "session_id": "demo-fake"}, headers=HEADERS)
    assert resp.status_code == 200
    assert (await _balance(store)).balance_credits == 493


async def _over_cap(store, monkeypatch):
    monkeypatch.setenv("PLATFORM_DAILY_USD_CAP", "1")
    spend_guard._cache = None
    async with store.factory() as s, s.begin():
        await s.execute(insert(store.llm).values(
            id=str(uuid.uuid4()), run_id="r", provider="groq", model="m", success=True,
            estimated_cost_usd=5.0, key_source="platform", created_at=datetime.now(timezone.utc),
        ))


@pytest.mark.parametrize("path", ["/v1/hybrid/chat", "/v1/autopilot/run"])
async def test_over_platform_cap_is_503_for_free_key_org(store, client, monkeypatch, path):
    await _over_cap(store, monkeypatch)
    resp = client.post(path, json=BODY, headers=HEADERS)
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "platform_capacity"


async def test_over_platform_cap_paid_key_org_passes(store, client, monkeypatch):
    await _over_cap(store, monkeypatch)
    async with store.factory() as s, s.begin():
        await s.execute(insert(store.tables.credit_ledger).values(
            id=str(uuid.uuid4()), org_id=ORG, kind="purchase", ref="ch_1",
            delta_credits=1000, created_at=datetime.now(timezone.utc),
        ))
    assert client.post("/v1/hybrid/chat", json=BODY, headers=HEADERS).status_code == 200


async def test_ip_limiter_applies_to_v1(store, client, monkeypatch):
    monkeypatch.setattr(request_guard, "RATE_LIMIT_PER_MIN", 2)
    codes = [client.post("/v1/hybrid/chat", json=BODY, headers=HEADERS).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
