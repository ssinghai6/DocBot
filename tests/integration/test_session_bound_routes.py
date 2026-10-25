"""DOCBOT-1611 (SEC-02): session-bound routes reject foreign session ids.

Every route that takes a ``session_id`` must answer 404 (same as missing) when
the caller does not own it, and must do so BEFORE any budget / vector-store /
LLM work. Covered pairs: anonymous owner cookie A vs B, and logged-in user in
org X vs org Y.

Runs the real route handlers against an in-memory SQLite DB (no network, no
Postgres). Session resolution is stubbed: the session cookie value is the user
id. ``charge_budget`` (the first expensive step in the LLM routes) is replaced
with a sentinel that raises 418, so "guard passed" == 418 and "guard refused"
== 404 without ever running a pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import HTTPException
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import api.index as idx
from api import rbac_service, session_service
from api.utils import request_guard
from api.utils.caller_scope import CallerScope, anon_token_digest

ORG_X, ORG_Y = "org-x", "org-y"
USERS = {
    "xavier": SimpleNamespace(id="xavier", email="x@x.com", role="analyst", org_id=ORG_X),
    "yolanda": SimpleNamespace(id="yolanda", email="y@y.com", role="analyst", org_id=ORG_Y),
}
ANON_A = "a" * 43
ANON_B = "b" * 43

SESS_X, SESS_Y = "sess-org-x", "sess-org-y"
SESS_ANON_A, SESS_ANON_B = "sess-anon-a", "sess-anon-b"
CONN_X, CONN_ANON_A = "conn-x", "conn-anon-a"
ART_X, ART_Y = "art-x", "art-y"

# (caller kwargs, own session, foreign session)
PAIRS = [
    pytest.param({"user": "xavier"}, SESS_X, SESS_Y, id="org-x-vs-org-y"),
    pytest.param({"user": "yolanda"}, SESS_Y, SESS_X, id="org-y-vs-org-x"),
    pytest.param({"anon": ANON_A}, SESS_ANON_A, SESS_ANON_B, id="anon-a-vs-anon-b"),
    pytest.param({"anon": ANON_B}, SESS_ANON_B, SESS_ANON_A, id="anon-b-vs-anon-a"),
]


@pytest.fixture
async def db(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    tables = [
        idx.users_table, idx.audit_log_table, idx.sessions_table,
        idx.session_artifacts_table, idx.db_connections_table, idx.schema_cache_table,
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
                {"session_id": sid, "org_id": org, "user_id": uid, "anon_owner": anon,
                 "persona": "Generalist"}
                for sid, org, uid, anon in [
                    (SESS_X, ORG_X, "xavier", None),
                    (SESS_Y, ORG_Y, "yolanda", None),
                    (SESS_ANON_A, None, None, anon_token_digest(ANON_A)),
                    (SESS_ANON_B, None, None, anon_token_digest(ANON_B)),
                ]
            ])
            await s.execute(insert(idx.session_artifacts_table), [
                {"id": aid, "session_id": sid, "turn_id": 1, "artifact_type": "sql_result",
                 "name": name, "data_json": "[]", "created_at": now}
                for aid, sid, name in [(ART_X, SESS_X, "x secret"), (ART_Y, SESS_Y, "y secret")]
            ])
            base = {"dialect": "postgresql", "host": "db.example.com", "port": 5432,
                    "dbname": "prod", "credentials_blob": "x", "created_at": now}
            await s.execute(insert(idx.db_connections_table), [
                {**base, "id": CONN_X, "session_id": SESS_X, "user_id": "xavier", "org_id": ORG_X},
            ])

    monkeypatch.setattr(idx, "async_session_factory", factory)
    monkeypatch.setattr(idx, "engine", engine)
    monkeypatch.setattr(rbac_service, "_users_table", idx.users_table)
    monkeypatch.setattr(rbac_service, "_user_sessions_table", idx.user_sessions_table)
    monkeypatch.setattr(rbac_service, "_async_session_factory", factory)
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    monkeypatch.setattr(request_guard, "RATE_LIMIT_PER_MIN", 0)
    request_guard.reset_state()
    monkeypatch.setattr(idx, "VECTOR_STORES", {})

    async def _lookup(token, *_args):
        return USERS.get(token)

    with patch("api.auth_service.get_user_from_session", new=AsyncMock(side_effect=_lookup)):
        yield factory
    await engine.dispose()


@pytest.fixture
def budget(monkeypatch):
    """charge_budget sentinel: reaching it means the ownership guard passed."""
    mock = AsyncMock(side_effect=HTTPException(status_code=418, detail="guard passed"))
    monkeypatch.setattr(idx, "charge_budget", mock)
    return mock


def _client(user: str | None = None, anon: str | None = None) -> httpx.AsyncClient:
    cookies = {}
    if user:
        cookies["docbot_session"] = user
    if anon:
        cookies["docbot_anon"] = anon
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=idx.app), base_url="http://test", cookies=cookies
    )


# --- /api/chat ---------------------------------------------------------------

@pytest.mark.parametrize("caller,own,foreign", PAIRS)
async def test_chat_foreign_session_404_before_budget(db, budget, caller, own, foreign):
    async with _client(**caller) as c:
        resp = await c.post("/api/chat", json={"session_id": foreign, "message": "hi"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Session not found"
    budget.assert_not_awaited()


@pytest.mark.parametrize("caller,own,foreign", PAIRS)
async def test_chat_own_session_passes_guard(db, budget, caller, own, foreign):
    async with _client(**caller) as c:
        resp = await c.post("/api/chat", json={"session_id": own, "message": "hi"})
    assert resp.status_code == 418


async def test_chat_missing_session_same_404(db, budget):
    async with _client(user="xavier") as c:
        resp = await c.post("/api/chat", json={"session_id": "nope", "message": "hi"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Session not found"


# --- /api/session/{id}/update-persona ----------------------------------------

async def _persona(factory, sid: str) -> str:
    async with factory() as s:
        row = (await s.execute(
            select(idx.sessions_table.c.persona).where(idx.sessions_table.c.session_id == sid)
        )).fetchone()
    return row.persona


@pytest.mark.parametrize("caller,own,foreign", PAIRS)
async def test_update_persona_scoped(db, caller, own, foreign):
    async with _client(**caller) as c:
        bad = await c.post(f"/api/session/{foreign}/update-persona", params={"persona": "Finance Expert"})
        good = await c.post(f"/api/session/{own}/update-persona", params={"persona": "Finance Expert"})
    assert bad.status_code == 404
    assert await _persona(db, foreign) == "Generalist"
    assert good.status_code == 200
    assert await _persona(db, own) == "Finance Expert"


# --- /api/artifacts/* --------------------------------------------------------

@pytest.mark.parametrize("caller,own,foreign", PAIRS)
async def test_artifact_list_scoped(db, caller, own, foreign):
    async with _client(**caller) as c:
        bad = await c.get(f"/api/artifacts/{foreign}")
        good = await c.get(f"/api/artifacts/{own}")
    assert bad.status_code == 404
    assert "secret" not in bad.text
    assert good.status_code == 200


async def test_artifact_item_scoped(db):
    async with _client(user="xavier") as c:
        mine = await c.get(f"/api/artifacts/item/{ART_X}")
        theirs = await c.get(f"/api/artifacts/item/{ART_Y}")
        missing = await c.get("/api/artifacts/item/nope")
    assert mine.status_code == 200
    assert mine.json()["name"] == "x secret"
    assert theirs.status_code == 404
    assert theirs.json() == missing.json()


async def test_artifact_item_anon_cannot_read_org_artifact(db):
    async with _client(anon=ANON_A) as c:
        resp = await c.get(f"/api/artifacts/item/{ART_X}")
    assert resp.status_code == 404


# --- /api/hybrid/chat --------------------------------------------------------

@pytest.mark.parametrize("caller,own,foreign", PAIRS)
async def test_hybrid_foreign_session_404_before_budget(db, budget, caller, own, foreign):
    async with _client(**caller) as c:
        bad = await c.post("/api/hybrid/chat", json={"question": "q", "session_id": foreign})
        good = await c.post("/api/hybrid/chat", json={"question": "q", "session_id": own})
    assert bad.status_code == 404
    assert good.status_code == 418
    assert budget.await_count == 1


# --- /api/autopilot/run ------------------------------------------------------

@pytest.mark.parametrize("caller,own,foreign", PAIRS)
async def test_autopilot_foreign_session_404_before_budget(db, budget, caller, own, foreign):
    async with _client(**caller) as c:
        bad_docs = await c.post("/api/autopilot/run",
                                json={"question": "q", "session_id": foreign, "has_docs": True})
        bad_nodocs = await c.post("/api/autopilot/run",
                                  json={"question": "q", "session_id": foreign, "has_docs": False})
        good = await c.post("/api/autopilot/run",
                            json={"question": "q", "session_id": own, "has_docs": True})
    assert bad_docs.status_code == 404
    assert bad_nodocs.status_code == 404
    assert good.status_code == 418
    assert budget.await_count == 1


async def test_autopilot_unbound_placeholder_allowed_without_docs(db, budget):
    """CSV/DB-only turns send session_id="anonymous" (no sessions row)."""
    async with _client(anon=ANON_A) as c:
        ok = await c.post("/api/autopilot/run",
                          json={"question": "q", "session_id": "anonymous", "has_docs": False})
        docs = await c.post("/api/autopilot/run",
                            json={"question": "q", "session_id": "anonymous", "has_docs": True})
    assert ok.status_code == 418
    assert docs.status_code == 404


async def test_autopilot_unbound_id_with_loaded_store_refused(db, budget, monkeypatch):
    monkeypatch.setattr(idx, "VECTOR_STORES", {"orphan-store": object()})
    async with _client(anon=ANON_A) as c:
        resp = await c.post("/api/autopilot/run",
                            json={"question": "q", "session_id": "orphan-store", "has_docs": False})
    assert resp.status_code == 404


# --- /api/db/chat (artifacts are saved under session_id) ----------------------

async def test_db_chat_cannot_target_foreign_session(db):
    with patch.object(idx, "run_sql_pipeline") as pipeline:
        async with _client(user="xavier") as c:
            resp = await c.post("/api/db/chat", json={
                "connection_id": CONN_X, "question": "q", "session_id": SESS_Y,
            })
    assert resp.status_code == 404
    pipeline.assert_not_called()


async def test_db_chat_unbound_placeholder_allowed(db):
    async def _empty(**_kwargs):
        if False:
            yield ""

    with (
        patch.object(idx, "run_sql_pipeline", side_effect=_empty),
        patch("api.audit_service.log_event"),
    ):
        async with _client(user="xavier") as c:
            resp = await c.post("/api/db/chat", json={
                "connection_id": CONN_X, "question": "q", "session_id": "anonymous",
            })
    assert resp.status_code == 200


# --- /api/edgar/cache never discloses the first ingester's session id ---------

async def test_edgar_cache_listing_strips_session_id(db):
    cached = [{"accession_number": "0001", "ticker": "AAPL", "session_id": SESS_Y}]
    with patch("api.connector_store.list_cached_filings", new=AsyncMock(return_value=cached)):
        async with _client(user="xavier") as c:
            resp = await c.get("/api/edgar/cache")
    assert resp.status_code == 200
    assert SESS_Y not in resp.text
    assert resp.json()["filings"][0]["accession_number"] == "0001"


# --- session_service.is_session_bindable --------------------------------------

async def test_is_session_bindable(db):
    x = CallerScope(user_id="xavier", org_id=ORG_X, anon_token=None)
    args = (idx.sessions_table, db)
    assert await session_service.is_session_bindable(SESS_X, x, *args)
    assert await session_service.is_session_bindable("anonymous", x, *args)
    assert not await session_service.is_session_bindable(SESS_Y, x, *args)
    assert not await session_service.is_session_bindable(SESS_ANON_A, x, *args)
