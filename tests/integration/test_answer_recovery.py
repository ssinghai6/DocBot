"""DOCBOT-2636: recover a hybrid answer whose browser stream dropped.

Owner check runs the real ``run_row_accessible`` / ``ids_accessible`` queries
against in-memory SQLite. A run the caller cannot access must look exactly
like a missing one (None -> 404), whether it is finished or still running.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

if not os.environ.get("DB_ENCRYPTION_KEY"):
    os.environ["DB_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql://fake:fake@localhost:5432/testdb"

import api.index as idx  # noqa: E402
from api import answer_recovery_service as svc  # noqa: E402
from api.answer_models import RecoveredAnswer  # noqa: E402
from api.routes import answer_routes  # noqa: E402
from api.utils.caller_scope import CallerScope  # noqa: E402

ALICE = CallerScope(user_id="alice", org_id="org-a", anon_token=None)
BOB = CallerScope(user_id="bob", org_id="org-b", anon_token=None)
ANSWER = "## Executive Summary\nLab Services differs by $2.2M.\n" + "x" * 5000  # longer than the old 4000 cap


@pytest.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    tables = [idx.sessions_table, idx.db_connections_table, idx.agent_traces_table]
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: idx.metadata.create_all(c, tables=tables))
    factory = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(timezone.utc)
    async with factory() as s:
        async with s.begin():
            await s.execute(insert(idx.sessions_table), [
                {"session_id": "sess-a", "org_id": "org-a", "user_id": "alice"},
            ])
            await s.execute(insert(idx.db_connections_table), [
                {"id": "conn-a", "session_id": "sess-a", "user_id": "alice", "org_id": "org-a",
                 "dialect": "sqlite", "host": "", "port": 0, "dbname": "x", "credentials_blob": "x",
                 "created_at": now},
            ])
            await s.execute(insert(idx.agent_traces_table), [
                {"id": "trace-1", "run_id": "run-done", "session_id": "sess-a", "connection_id": "conn-a",
                 "pipeline": "hybrid", "question": "q", "final_answer": ANSWER,
                 "retrieved_refs_json": json.dumps([{"source": "Tidewell-Draft-10K-FY2025.pdf", "page": 9, "text": "NOTE 12"}])},
            ])
    yield factory
    svc._running.clear()
    await engine.dispose()


async def _recover(run_id: str, scope: CallerScope, factory) -> RecoveredAnswer | None:
    with patch.object(svc, "get_lineage", new_callable=AsyncMock, return_value=None):
        return await svc.recover_answer(
            run_id, scope, idx.agent_traces_table, idx.sessions_table, idx.db_connections_table, factory
        )


@pytest.mark.asyncio
async def test_owner_gets_full_answer_and_citations(db):
    got = await _recover("run-done", ALICE, db)
    assert got is not None and got.status == "ready"
    assert got.answer == ANSWER
    assert got.trace_id == "trace-1"
    assert got.citations[0].source == "Tidewell-Draft-10K-FY2025.pdf" and got.citations[0].page == 9


@pytest.mark.asyncio
async def test_other_caller_cannot_see_finished_run(db):
    assert await _recover("run-done", BOB, db) is None


@pytest.mark.asyncio
async def test_running_turn_is_pending_for_owner_only(db):
    svc.mark_running("run-live", "sess-a", "conn-a")
    got = await _recover("run-live", ALICE, db)
    assert got is not None and got.status == "pending" and got.answer is None
    assert await _recover("run-live", BOB, db) is None
    svc.mark_done("run-live")
    assert await _recover("run-live", ALICE, db) is None  # ended without a persisted answer


@pytest.mark.asyncio
async def test_unknown_run_is_none(db):
    assert await _recover("nope", ALICE, db) is None


# ── route: HTTP mapping ──────────────────────────────────────────────────────


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    monkeypatch.setattr(answer_routes, "_state", {"traces": None, "sessions": None, "connections": None, "factory": None})
    app = FastAPI()
    app.include_router(answer_routes.router)
    c = TestClient(app)
    c.cookies.set("docbot_session", "tok")
    return c


def _user():
    return patch("api.auth_service.get_user_from_session", new_callable=AsyncMock,
                 return_value=SimpleNamespace(id="alice", org_id="org-a", role="viewer"))


@pytest.mark.parametrize("result,code,status", [
    (RecoveredAnswer(run_id="r", status="ready", answer="A"), 200, "ready"),
    (RecoveredAnswer(run_id="r", status="pending"), 200, "pending"),
    (None, 404, None),
])
def test_route_maps_service_result(client, result, code, status):
    with _user(), patch.object(answer_routes, "recover_answer", new_callable=AsyncMock, return_value=result):
        resp = client.get("/api/hybrid/answer/r?session_id=sess-a&connection_id=conn-a")
    assert resp.status_code == code
    if status:
        assert resp.json()["status"] == status


def test_route_anonymous_rejected_when_auth_required(client, monkeypatch):
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    client.cookies.clear()
    assert client.get("/api/hybrid/answer/r").status_code == 401


# ── detach: the turn finishes even when the client stops reading ─────────────


@pytest.mark.asyncio
async def test_detached_stream_runs_to_end_after_client_drop():
    finished = asyncio.Event()

    async def turn():
        yield "data: metadata\n\n"
        await asyncio.sleep(0.01)
        yield "data: token\n\n"
        finished.set()  # where hybrid_chat persists the answer

    relay = svc.detach(turn())
    assert await relay.__anext__() == "data: metadata\n\n"
    await relay.aclose()  # client disconnected
    await asyncio.wait_for(finished.wait(), timeout=1)


@pytest.mark.asyncio
async def test_detached_stream_relays_everything_and_errors():
    async def ok():
        for i in range(3):
            yield str(i)

    assert [x async for x in svc.detach(ok())] == ["0", "1", "2"]

    async def boom():
        yield "a"
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        [x async for x in svc.detach(boom())]
