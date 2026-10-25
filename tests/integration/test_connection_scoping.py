"""DOCBOT-1603 (SEC-06): connection / schema-cache / query-history scoping.

User A must not be able to use, list, read the schema of, read the history
of, refresh, or disconnect user B's DB connection. `/v1/*` callers may only
reference sessions/connections in their API key's org; foreign ids → 404.

Runs the real route handlers against an in-memory SQLite DB (no network, no
Postgres). Session resolution is stubbed: the cookie value is the user id.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
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
from api.api_key_service import APIKeyRecord
from api.db_service import connection_access_clause
from api.utils import request_guard
from api.utils.caller_scope import CallerScope, scope_filter

ORG_A, ORG_B = "org-a", "org-b"

USERS = {
    "alice": SimpleNamespace(id="alice", email="a@a.com", role="analyst", org_id=ORG_A),
    "bob": SimpleNamespace(id="bob", email="b@b.com", role="analyst", org_id=ORG_B),
    "a_admin": SimpleNamespace(id="a_admin", email="adm@a.com", role="admin", org_id=ORG_A),
}

CONN_A, CONN_B, CONN_ANON = "conn-alice", "conn-bob", "conn-anon"
SESS_A, SESS_B = "sess-a", "sess-b"
SCHEMA_A = [{"name": "alice_secret_table", "columns": [{"name": "x", "type": "INT"}]}]
SCHEMA_B = [{"name": "bob_secret_table", "columns": [{"name": "y", "type": "INT"}]}]


@pytest.fixture
async def db(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    tables = [
        idx.users_table, idx.audit_log_table, idx.sessions_table,
        idx.db_connections_table, idx.schema_cache_table, idx.query_history_table,
    ]
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: idx.metadata.create_all(c, tables=tables))
    factory = async_sessionmaker(engine, expire_on_commit=False)

    now = datetime.now(timezone.utc)
    later = now + timedelta(hours=12)
    async with factory() as s:
        async with s.begin():
            await s.execute(insert(idx.users_table), [
                {"id": u.id, "email": u.email, "name": u.id, "provider": "email",
                 "role": u.role, "org_id": u.org_id, "created_at": now}
                for u in USERS.values()
            ])
            await s.execute(insert(idx.sessions_table), [
                {"session_id": SESS_A, "org_id": ORG_A, "user_id": "alice"},
                {"session_id": SESS_B, "org_id": ORG_B, "user_id": "bob"},
            ])
            base = {"dialect": "postgresql", "host": "db.example.com", "port": 5432,
                    "dbname": "prod", "credentials_blob": "x", "created_at": now}
            await s.execute(insert(idx.db_connections_table), [
                {**base, "id": CONN_A, "session_id": SESS_A, "user_id": "alice", "org_id": ORG_A},
                {**base, "id": CONN_B, "session_id": SESS_B, "user_id": "bob", "org_id": ORG_B},
                {**base, "id": CONN_ANON, "session_id": "sess-anon", "user_id": None, "org_id": None},
            ])
            await s.execute(insert(idx.schema_cache_table), [
                {"connection_id": CONN_A, "schema_json": json.dumps(SCHEMA_A), "expires_at": later},
                {"connection_id": CONN_B, "schema_json": json.dumps(SCHEMA_B), "expires_at": later},
            ])
            await s.execute(insert(idx.query_history_table), [
                {"id": "qa", "connection_id": CONN_A, "nl_question": "alice q",
                 "sql_query": "SELECT 1", "result_summary": "1 rows", "created_at": now},
                {"id": "qb", "connection_id": CONN_B, "nl_question": "bob q",
                 "sql_query": "SELECT 2", "result_summary": "2 rows", "created_at": now},
            ])

    monkeypatch.setattr(idx, "async_session_factory", factory)
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

    with patch("api.auth_service.get_user_from_session", new=AsyncMock(side_effect=_lookup)):
        yield factory
    await engine.dispose()


def _client(as_user: str | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=idx.app),
        base_url="http://test",
        cookies={"docbot_session": as_user} if as_user else {},
    )


async def _conn_exists(factory, conn_id: str) -> bool:
    async with factory() as s:
        row = (await s.execute(
            select(idx.db_connections_table.c.id).where(idx.db_connections_table.c.id == conn_id)
        )).fetchone()
    return row is not None


# --- /api/db/sessions -------------------------------------------------------

async def test_sessions_lists_only_callers_connections(db):
    async with _client("alice") as c:
        resp = await c.get("/api/db/sessions")
    assert resp.status_code == 200
    assert {r["connection_id"] for r in resp.json()["connections"]} == {CONN_A}
    assert "credentials_blob" not in resp.text


async def test_sessions_anonymous_gets_empty_list(db):
    async with _client() as c:
        resp = await c.get("/api/db/sessions")
    assert resp.json()["connections"] == []


async def test_sessions_org_admin_sees_org_only(db):
    async with _client("a_admin") as c:
        resp = await c.get("/api/db/sessions")
    assert {r["connection_id"] for r in resp.json()["connections"]} == {CONN_A}


# --- schema / history / refresh / disconnect ---------------------------------

async def test_schema_own_ok_foreign_404(db):
    async with _client("alice") as c:
        own = await c.get(f"/api/db/schema/{CONN_A}")
        foreign = await c.get(f"/api/db/schema/{CONN_B}")
    assert own.status_code == 200
    assert own.json()["tables"] == SCHEMA_A
    assert foreign.status_code == 404
    assert "bob_secret_table" not in foreign.text


async def test_anonymous_cannot_read_owned_schema(db):
    async with _client() as c:
        assert (await c.get(f"/api/db/schema/{CONN_A}")).status_code == 404


async def test_history_own_ok_foreign_404(db):
    async with _client("alice") as c:
        own = await c.get(f"/api/db/history/{CONN_A}")
        foreign = await c.get(f"/api/db/history/{CONN_B}")
    assert [h["question"] for h in own.json()["history"]] == ["alice q"]
    assert foreign.status_code == 404
    assert "bob q" not in foreign.text


async def test_refresh_schema_foreign_404_and_cache_untouched(db):
    async with _client("alice") as c:
        resp = await c.post(f"/api/db/refresh-schema/{CONN_B}")
    assert resp.status_code == 404
    async with db() as s:
        row = (await s.execute(
            select(idx.schema_cache_table).where(idx.schema_cache_table.c.connection_id == CONN_B)
        )).fetchone()
    assert row is not None


async def test_disconnect_foreign_404_and_row_kept(db):
    async with _client("alice") as c:
        resp = await c.request(
            "DELETE", f"/api/db/disconnect/{CONN_B}", json={"session_id": SESS_B}
        )
    assert resp.status_code == 404
    assert await _conn_exists(db, CONN_B)


async def test_disconnect_own_ok(db):
    async with _client("alice") as c:
        resp = await c.request(
            "DELETE", f"/api/db/disconnect/{CONN_A}", json={"session_id": SESS_A}
        )
    assert resp.status_code == 200
    assert not await _conn_exists(db, CONN_A)


# --- use: /api/db/chat, /api/hybrid/chat, /api/autopilot/run -----------------

async def _fake_pipeline(**_kwargs):
    yield 'data: {"type": "done"}\n\n'


async def test_db_chat_foreign_connection_404(db):
    with patch.object(idx, "run_sql_pipeline", side_effect=_fake_pipeline) as pipe:
        async with _client("alice") as c:
            foreign = await c.post("/api/db/chat", json={
                "connection_id": CONN_B, "question": "q", "session_id": SESS_A,
            })
            own = await c.post("/api/db/chat", json={
                "connection_id": CONN_A, "question": "q", "session_id": SESS_A,
            })
    assert foreign.status_code == 404
    assert own.status_code == 200
    assert pipe.call_count == 1
    assert pipe.call_args.kwargs["scope"].user_id == "alice"


async def test_hybrid_and_autopilot_foreign_connection_404(db):
    async with _client("alice") as c:
        hybrid = await c.post("/api/hybrid/chat", json={
            "question": "q", "session_id": SESS_A, "connection_id": CONN_B,
        })
        autopilot = await c.post("/api/autopilot/run", json={
            "question": "q", "session_id": SESS_A, "connection_id": CONN_B,
        })
    assert hybrid.status_code == 404
    assert autopilot.status_code == 404


async def test_anonymous_connection_still_usable_by_id(db):
    """Unowned (anonymous/demo) connections keep working — no regression."""
    with patch.object(idx, "run_sql_pipeline", side_effect=_fake_pipeline):
        async with _client() as c:
            resp = await c.post("/api/db/chat", json={
                "connection_id": CONN_ANON, "question": "q", "session_id": "sess-anon",
            })
    assert resp.status_code == 200


# --- /v1/* ------------------------------------------------------------------

def _key_for(org_id: str) -> APIKeyRecord:
    return APIKeyRecord(
        id=f"key_{org_id}", org_id=org_id, key_prefix="dk_live_x", name="k",
        scopes=["hybrid:read", "autopilot:run"],
    )


@pytest.fixture
def v1_key_org_a():
    with patch.object(api_key_service, "validate_api_key", new=AsyncMock(return_value=_key_for(ORG_A))):
        yield


def _v1_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=idx.app),
        base_url="http://test",
        headers={"Authorization": "Bearer dk_live_test"},
    )


@pytest.mark.parametrize("path", ["/v1/hybrid/chat", "/v1/autopilot/run"])
@pytest.mark.parametrize("body", [
    {"session_id": SESS_B},                              # foreign session
    {"session_id": SESS_A, "connection_id": CONN_B},     # foreign connection
    {"session_id": SESS_A, "connection_id": CONN_ANON},  # unowned connection
    {"session_id": "does-not-exist"},                    # unknown session
])
async def test_v1_foreign_ids_404(db, v1_key_org_a, path, body):
    async with _v1_client() as c:
        resp = await c.post(path, json={"question": "q", **body})
    assert resp.status_code == 404


async def test_v1_own_ids_pass_scope_check(db, v1_key_org_a):
    async def _fake_hybrid(**_kwargs):
        yield 'data: {"type": "done", "citations": []}\n\n'

    with patch("api.hybrid_service.hybrid_chat", side_effect=_fake_hybrid):
        async with _v1_client() as c:
            resp = await c.post("/v1/hybrid/chat", json={
                "question": "q", "session_id": SESS_A, "connection_id": CONN_A,
            })
    assert resp.status_code == 200


# --- predicate unit checks ----------------------------------------------------

def test_scope_filter_anonymous_only_unowned():
    clause = str(scope_filter(idx.db_connections_table, CallerScope(user_id=None, org_id=None, anon_token=None)))
    assert "user_id IS NULL" in clause and "org_id IS NULL" in clause


def test_api_key_scope_is_strict_org():
    clause = str(connection_access_clause(idx.db_connections_table, CallerScope(user_id=None, org_id=ORG_A, anon_token=None)))
    assert "IS NULL" not in clause
    assert "org_id" in clause
