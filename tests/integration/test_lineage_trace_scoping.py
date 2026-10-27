"""DOCBOT-1620 (SEC-16): lineage + trace feedback are owner/org scoped.

``GET /api/lineage/{run_id}``, ``POST /api/traces/{trace_id}/feedback`` and
``GET /v1/lineage/{run_id}`` only serve rows whose session or connection the
caller can access; anything else (foreign, anonymous-other, unbound, missing)
is 404.

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
from api import lineage_service, rbac_service, trace_service
from api.api_key_service import APIKeyRecord
from api.utils import request_guard
from api.utils.caller_scope import anon_token_digest
from api.utils.lineage import LineageCollector

ORG_A, ORG_B = "org-a", "org-b"
COOKIE_A, COOKIE_B = "a" * 43, "b" * 43

USERS = {
    "alice": SimpleNamespace(id="alice", email="a@a.com", role="viewer", org_id=ORG_A),
    "bob": SimpleNamespace(id="bob", email="b@b.com", role="viewer", org_id=ORG_B),
}

# run_id -> (session_id, connection_id)
RUNS = {
    "run-a": ("sess-a", None),
    "run-b": ("sess-b", None),
    "run-conn-a": ("anonymous", "conn-a"),    # DB-only turn: unbound label + owned connection
    "run-unbound": ("anonymous", None),       # no owner at all
    "run-anon-a": ("sess-anon-a", None),      # anonymous browser session (default-org backfilled)
}


@pytest.fixture
async def db(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    tables = [
        idx.users_table, idx.audit_log_table, idx.sessions_table, idx.db_connections_table,
        idx.answer_lineage_table, idx.agent_traces_table,
    ]
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: idx.metadata.create_all(c, tables=tables))
    factory = async_sessionmaker(engine, expire_on_commit=False)

    now = datetime.now(timezone.utc)
    async with factory() as s:
        async with s.begin():
            await s.execute(insert(idx.users_table), [
                {"id": u.id, "email": u.email, "name": u.id, "provider": "email",
                 "role": u.role, "org_id": u.org_id, "created_at": now}
                for u in USERS.values()
            ])
            await s.execute(insert(idx.sessions_table), [
                {"session_id": "sess-a", "org_id": ORG_A, "user_id": "alice", "anon_owner": None},
                {"session_id": "sess-b", "org_id": ORG_B, "user_id": "bob", "anon_owner": None},
                {"session_id": "sess-anon-a", "org_id": ORG_A, "user_id": None,
                 "anon_owner": anon_token_digest(COOKIE_A)},
            ])
            await s.execute(insert(idx.db_connections_table), [{
                "id": "conn-a", "session_id": "sess-a", "dialect": "postgresql",
                "host": "db.example.com", "port": 5432, "dbname": "prod",
                "credentials_blob": "x", "created_at": now, "user_id": "alice", "org_id": ORG_A,
            }])
            await s.execute(insert(idx.answer_lineage_table), [
                {"run_id": run_id, "session_id": sid, "connection_id": cid, "mode": "docs",
                 "data_json": LineageCollector(run_id, "docs", question="q").build().model_dump_json()}
                for run_id, (sid, cid) in RUNS.items()
            ])
            await s.execute(insert(idx.agent_traces_table), [
                {"id": f"trace-{run_id}", "run_id": run_id, "session_id": sid,
                 "connection_id": cid, "pipeline": "hybrid", "question": "q"}
                for run_id, (sid, cid) in RUNS.items()
            ])

    monkeypatch.setattr(idx, "async_session_factory", factory)
    monkeypatch.setattr(lineage_service, "_lineage_table", idx.answer_lineage_table)
    monkeypatch.setattr(lineage_service, "_async_session_factory", factory)
    monkeypatch.setattr(trace_service, "_agent_traces_table", idx.agent_traces_table)
    monkeypatch.setattr(trace_service, "_async_session_factory", factory)
    monkeypatch.setattr(rbac_service, "_users_table", idx.users_table)
    monkeypatch.setattr(rbac_service, "_user_sessions_table", idx.user_sessions_table)
    monkeypatch.setattr(rbac_service, "_async_session_factory", factory)
    monkeypatch.setattr(api_key_service, "_async_session_factory", factory)
    monkeypatch.setattr(api_key_service, "_audit_log_table", idx.audit_log_table)
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    monkeypatch.setattr(request_guard, "RATE_LIMIT_PER_MIN", 0)
    request_guard.reset_state()

    async def _lookup(token, *_args):
        return USERS.get(token)

    with patch("api.auth_service.get_user_from_session", new=AsyncMock(side_effect=_lookup)), \
         patch("api.llm_trace_service.get_calls_by_run", new=AsyncMock(return_value=[])):
        yield factory
    await engine.dispose()


def _client(*, user: str | None = None, anon: str | None = None) -> httpx.AsyncClient:
    cookies = {}
    if user:
        cookies["docbot_session"] = user
    if anon:
        cookies["docbot_anon"] = anon
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=idx.app), base_url="http://test", cookies=cookies,
    )


@pytest.mark.parametrize("caller,run_id,expected", [
    ({"user": "alice"}, "run-a", 200),
    ({"user": "alice"}, "run-conn-a", 200),
    ({"user": "alice"}, "run-b", 404),          # other org
    ({"user": "alice"}, "run-anon-a", 404),     # anonymous row, even though org-stamped
    ({"user": "alice"}, "run-unbound", 404),
    ({"user": "alice"}, "missing", 404),
    ({"user": "bob"}, "run-conn-a", 404),
    ({"anon": COOKIE_A}, "run-anon-a", 200),    # own anon session
    ({"anon": COOKIE_B}, "run-anon-a", 404),    # someone else's anon session
    ({"anon": COOKIE_A}, "run-a", 404),
    ({"anon": COOKIE_A}, "run-unbound", 404),
])
async def test_lineage_scoped(db, caller, run_id, expected):
    async with _client(**caller) as c:
        resp = await c.get(f"/api/lineage/{run_id}")
    assert resp.status_code == expected
    if expected == 200:
        assert resp.json()["run_id"] == run_id


async def _feedback_of(factory, trace_id):
    async with factory() as s:
        return (await s.execute(
            select(idx.agent_traces_table.c.user_feedback).where(idx.agent_traces_table.c.id == trace_id)
        )).scalar_one()


@pytest.mark.parametrize("caller,run_id,expected", [
    ({"user": "alice"}, "run-a", 200),
    ({"user": "alice"}, "run-conn-a", 200),
    ({"user": "alice"}, "run-b", 404),
    ({"user": "bob"}, "run-conn-a", 404),
    ({"anon": COOKIE_A}, "run-anon-a", 200),
    ({"anon": COOKIE_B}, "run-anon-a", 404),
    ({"user": "alice"}, "run-unbound", 404),
])
async def test_trace_feedback_scoped(db, caller, run_id, expected):
    async with _client(**caller) as c:
        resp = await c.post(f"/api/traces/trace-{run_id}/feedback", json={"feedback": "up"})
    assert resp.status_code == expected
    assert await _feedback_of(db, f"trace-{run_id}") == ("up" if expected == 200 else None)


def _key_for(org_id: str) -> APIKeyRecord:
    return APIKeyRecord(
        id=f"key_{org_id}", org_id=org_id, key_prefix="dk_live_x", name="k", scopes=["lineage:read"],
    )


@pytest.mark.parametrize("run_id,expected", [
    ("run-a", 200),
    ("run-conn-a", 200),
    ("run-b", 404),          # org B's run
    ("run-anon-a", 404),     # anonymous session backfilled into org A
    ("run-unbound", 404),
])
async def test_v1_lineage_scoped_to_key_org(db, run_id, expected):
    with patch.object(api_key_service, "validate_api_key", new=AsyncMock(return_value=_key_for(ORG_A))):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=idx.app), base_url="http://test",
            headers={"Authorization": "Bearer dk_live_test"},
        ) as c:
            resp = await c.get(f"/v1/lineage/{run_id}")
    assert resp.status_code == expected
