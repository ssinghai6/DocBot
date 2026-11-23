"""DOCBOT-1635: anonymous DB connections are owned by the ``docbot_anon`` cookie.

After DOCBOT-1603 a connection with no ``user_id`` was reachable by anyone who
knew its id. Now anonymous connections carry ``anon_owner`` (sha256 of the
anon cookie) and only that cookie's holder can use them. Legacy rows (no
``user_id``, no ``anon_owner``) are hidden from everyone, org admins included
(DOCBOT-2317), and org-scoped API keys (``/v1``) never match anonymous rows even after the
startup default-org backfill stamped an ``org_id`` on them.

Runs the real route handlers against an in-memory SQLite DB (no network).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import api.api_key_service as api_key_service
import api.index as idx
import api.routes.v1_routes as v1_routes
from api import rbac_service
from api.api_key_service import APIKeyRecord
from api.utils import request_guard
from api.utils.caller_scope import ANON_OWNER_COLUMN_MIGRATIONS, anon_token_digest

DEFAULT_ORG = "org-default"
COOKIE_A = "A" * 43
COOKIE_B = "B" * 43
OWNER_A = anon_token_digest(COOKIE_A)
OWNER_B = anon_token_digest(COOKIE_B)

USERS = {
    "admin": SimpleNamespace(id="admin", email="adm@d.com", role="admin", org_id=DEFAULT_ORG),
    "member": SimpleNamespace(id="member", email="m@d.com", role="analyst", org_id=DEFAULT_ORG),
}

CONN_A, CONN_B, CONN_LEGACY = "conn-anon-a", "conn-anon-b", "conn-legacy"
SESS_A, SESS_B, SESS_MEMBER = "sess-anon-a", "sess-anon-b", "sess-member"
SCHEMA_A = [{"name": "anon_a_table", "columns": [{"name": "x", "type": "INT"}]}]
SCHEMA_B = [{"name": "anon_b_table", "columns": [{"name": "y", "type": "INT"}]}]
SCHEMA_LEGACY = [{"name": "legacy_table", "columns": [{"name": "z", "type": "INT"}]}]


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
            # Anonymous sessions were swept into the default org by the
            # DOCBOT-1524 startup backfill (org_id set, user_id NULL).
            await s.execute(insert(idx.sessions_table), [
                {"session_id": SESS_A, "org_id": DEFAULT_ORG, "user_id": None, "anon_owner": OWNER_A},
                {"session_id": SESS_B, "org_id": DEFAULT_ORG, "user_id": None, "anon_owner": OWNER_B},
                {"session_id": SESS_MEMBER, "org_id": DEFAULT_ORG, "user_id": "member", "anon_owner": None},
            ])
            base = {"dialect": "postgresql", "host": "db.example.com", "port": 5432,
                    "dbname": "prod", "credentials_blob": "x", "created_at": now,
                    "user_id": None, "org_id": DEFAULT_ORG}
            await s.execute(insert(idx.db_connections_table), [
                {**base, "id": CONN_A, "session_id": SESS_A, "anon_owner": OWNER_A},
                {**base, "id": CONN_B, "session_id": SESS_B, "anon_owner": OWNER_B},
                {**base, "id": CONN_LEGACY, "session_id": "sess-legacy", "anon_owner": None},
            ])
            await s.execute(insert(idx.schema_cache_table), [
                {"connection_id": CONN_A, "schema_json": json.dumps(SCHEMA_A), "expires_at": later},
                {"connection_id": CONN_B, "schema_json": json.dumps(SCHEMA_B), "expires_at": later},
                {"connection_id": CONN_LEGACY, "schema_json": json.dumps(SCHEMA_LEGACY),
                 "expires_at": later},
            ])
            await s.execute(insert(idx.query_history_table), [
                {"id": "qa", "connection_id": CONN_A, "nl_question": "anon a q",
                 "sql_query": "SELECT 1", "result_summary": "1 rows", "created_at": now},
                {"id": "qb", "connection_id": CONN_B, "nl_question": "anon b q",
                 "sql_query": "SELECT 2", "result_summary": "2 rows", "created_at": now},
            ])

    monkeypatch.setattr(idx, "async_session_factory", factory)
    monkeypatch.setitem(v1_routes._d, "async_session_factory", factory)  # DOCBOT-3901: /v1 lives in v1_routes
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


def _client(*, anon: str | None = None, user: str | None = None) -> httpx.AsyncClient:
    cookies: dict[str, str] = {}
    if anon:
        cookies["docbot_anon"] = anon
    if user:
        cookies["docbot_session"] = user
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=idx.app), base_url="http://test", cookies=cookies,
    )


async def _conn_row(factory, conn_id: str):
    async with factory() as s:
        return (await s.execute(
            select(idx.db_connections_table).where(idx.db_connections_table.c.id == conn_id)
        )).fetchone()


async def _fake_pipeline(**_kwargs):
    yield 'data: {"type": "done"}\n\n'


# --- anon-A vs anon-B ---------------------------------------------------------

async def test_schema_anon_owner_only(db):
    async with _client(anon=COOKIE_A) as c:
        own = await c.get(f"/api/db/schema/{CONN_A}")
        foreign = await c.get(f"/api/db/schema/{CONN_B}")
    assert own.status_code == 200
    assert own.json()["tables"] == SCHEMA_A
    assert foreign.status_code == 404
    assert "anon_b_table" not in foreign.text


async def test_schema_no_cookie_404(db):
    async with _client() as c:
        assert (await c.get(f"/api/db/schema/{CONN_A}")).status_code == 404


async def test_history_anon_owner_only(db):
    async with _client(anon=COOKIE_B) as c:
        own = await c.get(f"/api/db/history/{CONN_B}")
        foreign = await c.get(f"/api/db/history/{CONN_A}")
    assert [h["question"] for h in own.json()["history"]] == ["anon b q"]
    assert foreign.status_code == 404
    assert "anon a q" not in foreign.text


async def test_db_chat_anon_owner_only(db):
    with patch.object(idx, "run_sql_pipeline", side_effect=_fake_pipeline) as pipe:
        async with _client(anon=COOKIE_A) as c:
            foreign = await c.post("/api/db/chat", json={
                "connection_id": CONN_B, "question": "q", "session_id": SESS_A,
            })
            own = await c.post("/api/db/chat", json={
                "connection_id": CONN_A, "question": "q", "session_id": SESS_A,
            })
    assert foreign.status_code == 404
    assert own.status_code == 200
    assert pipe.call_count == 1
    assert pipe.call_args.kwargs["scope"].anon_token == OWNER_A


async def test_hybrid_and_autopilot_foreign_anon_404(db):
    async with _client(anon=COOKIE_A) as c:
        hybrid = await c.post("/api/hybrid/chat", json={
            "question": "q", "session_id": SESS_A, "connection_id": CONN_B,
        })
        autopilot = await c.post("/api/autopilot/run", json={
            "question": "q", "session_id": SESS_A, "connection_id": CONN_B,
        })
    assert hybrid.status_code == 404
    assert autopilot.status_code == 404


async def test_disconnect_anon_owner_only(db):
    async with _client(anon=COOKIE_B) as c:
        foreign = await c.request(
            "DELETE", f"/api/db/disconnect/{CONN_A}", json={"session_id": SESS_A}
        )
        assert foreign.status_code == 404
        assert await _conn_row(db, CONN_A) is not None
        own = await c.request(
            "DELETE", f"/api/db/disconnect/{CONN_B}", json={"session_id": SESS_B}
        )
    assert own.status_code == 200
    assert await _conn_row(db, CONN_B) is None


async def test_logged_in_member_cannot_reach_anon_connection(db):
    async with _client(user="member") as c:
        assert (await c.get(f"/api/db/schema/{CONN_A}")).status_code == 404


# --- legacy rows (user_id NULL, anon_owner NULL) ---------------------------------

async def test_legacy_row_hidden_from_anon_and_members(db):
    async with _client(anon=COOKIE_A) as c:
        assert (await c.get(f"/api/db/schema/{CONN_LEGACY}")).status_code == 404
    async with _client() as c:
        assert (await c.get(f"/api/db/schema/{CONN_LEGACY}")).status_code == 404
    async with _client(user="member") as c:
        assert (await c.get(f"/api/db/schema/{CONN_LEGACY}")).status_code == 404


async def test_legacy_row_hidden_from_org_admin(db):
    """DOCBOT-2317: the backfill homed it into the default org, but admins see user rows only."""
    async with _client(user="admin") as c:
        assert (await c.get(f"/api/db/schema/{CONN_LEGACY}")).status_code == 404


async def test_default_org_admin_never_sees_anon_connections(db):
    """DOCBOT-2317: anon rows carry the backfilled default org_id; admin gets nothing."""
    async with _client(user="admin") as c:
        listed = (await c.get("/api/db/sessions")).json()["connections"]
        assert {r["connection_id"] for r in listed} == set()
        for conn_id in (CONN_A, CONN_B):
            assert (await c.get(f"/api/db/schema/{conn_id}")).status_code == 404
            assert (await c.get(f"/api/db/history/{conn_id}")).status_code == 404
    async with _client(anon=COOKIE_A) as c:
        assert (await c.get(f"/api/db/schema/{CONN_A}")).status_code == 200


# --- /v1 default-org API key -----------------------------------------------------

def _key(org_id: str) -> APIKeyRecord:
    return APIKeyRecord(
        id=f"key_{org_id}", org_id=org_id, key_prefix="dk_live_x", name="k",
        scopes=["hybrid:read", "autopilot:run"],
    )


@pytest.mark.parametrize("body", [
    {"session_id": SESS_MEMBER, "connection_id": CONN_A},       # anon connection
    {"session_id": SESS_MEMBER, "connection_id": CONN_LEGACY},  # legacy connection
    {"session_id": SESS_A},                                     # anon session
])
async def test_v1_default_org_key_never_matches_anon_rows(db, body):
    with patch.object(api_key_service, "validate_api_key", new=AsyncMock(return_value=_key(DEFAULT_ORG))):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=idx.app), base_url="http://test",
            headers={"Authorization": "Bearer dk_live_test"},
        ) as c:
            resp = await c.post("/v1/hybrid/chat", json={"question": "q", **body})
    assert resp.status_code == 404


# --- create paths stamp anon_owner -----------------------------------------------

async def test_csv_upload_stamps_anon_owner(db):
    csv_bytes = b"region,revenue\nnorth,10\nsouth,20\n"
    async with _client(anon=COOKIE_A) as c:
        resp = await c.post(
            "/api/db/upload/csv",
            files={"file": ("sales.csv", csv_bytes, "text/csv")},
            data={"session_id": SESS_A},
        )
        assert resp.status_code == 200, resp.text
        conn_id = resp.json()["connection_id"]
        assert (await c.get(f"/api/db/schema/{conn_id}")).status_code == 200
    row = await _conn_row(db, conn_id)
    assert row.anon_owner == OWNER_A
    assert row.user_id is None
    async with _client(anon=COOKIE_B) as c:
        assert (await c.get(f"/api/db/schema/{conn_id}")).status_code == 404


async def test_upload_without_cookie_primes_and_stamps(db):
    """First request without a cookie: the middleware-issued cookie owns the row."""
    csv_bytes = b"a,b\n1,2\n"
    async with _client() as c:
        resp = await c.post(
            "/api/db/upload/csv",
            files={"file": ("x.csv", csv_bytes, "text/csv")},
            data={"session_id": "s-new"},
        )
        assert resp.status_code == 200, resp.text
        issued = resp.cookies.get("docbot_anon")
        assert issued
        conn_id = resp.json()["connection_id"]
    row = await _conn_row(db, conn_id)
    assert row.anon_owner == anon_token_digest(issued)


async def test_demo_init_stamps_anon_owner(db):
    with patch("api.demo_service.create_store", new=MagicMock(return_value=MagicMock())), \
            patch.object(idx, "get_embeddings", new=MagicMock(return_value=MagicMock())):
        async with _client(anon=COOKIE_A) as c:
            resp = await c.post("/api/demo/init")
            assert resp.status_code == 200, resp.text
            body = resp.json()
            conn_id = body["connection_id"]
            assert body["schema_summary"]["table_count"] > 0
            assert (await c.get(f"/api/db/schema/{conn_id}")).status_code == 200
    row = await _conn_row(db, conn_id)
    assert row.anon_owner == OWNER_A
    assert row.user_id is None
    async with _client(anon=COOKIE_B) as c:
        assert (await c.get(f"/api/db/schema/{conn_id}")).status_code == 404


async def test_connect_stamps_anon_owner(db):
    fake = AsyncMock(return_value={"connection_id": "c-new", "schema_summary": {}})
    with patch.object(idx, "connect_database", new=fake), \
            patch("api.db_service.validate_ssrf", new=MagicMock(return_value=None)):
        async with _client(anon=COOKIE_A) as c:
            resp = await c.post("/api/db/connect", json={
                "session_id": SESS_A, "dialect": "postgresql", "host": "db.example.com",
                "port": 5432, "dbname": "prod", "user": "u", "password": "p",
            })
    assert resp.status_code == 200, resp.text
    assert fake.call_args.kwargs["anon_owner"] == OWNER_A
    assert fake.call_args.kwargs["user_id"] is None


# --- migration --------------------------------------------------------------------

def test_migration_is_idempotent_ddl():
    ddl = [d for d in ANON_OWNER_COLUMN_MIGRATIONS if "db_connections" in d]
    assert ddl == [
        "ALTER TABLE db_connections ADD COLUMN IF NOT EXISTS anon_owner TEXT",
        "CREATE INDEX IF NOT EXISTS ix_db_connections_anon_owner ON db_connections (anon_owner)",
    ]
    assert "anon_owner" in idx.db_connections_table.c
