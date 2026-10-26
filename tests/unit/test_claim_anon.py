"""DOCBOT-2316: anonymous sessions/connections are claimed on sign-in.

A visitor who runs the demo or uploads files anonymously, then signs up or
logs in, must keep that work. SQLite only, no network.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import os
from pathlib import Path
from typing import Any, Iterator, Optional

import pytest
from cryptography.fernet import Fernet

if not os.environ.get("DB_ENCRYPTION_KEY"):
    os.environ["DB_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql://fake:fake@localhost:5432/testdb"

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, insert, select  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from starlette.requests import Request  # noqa: E402

import api.index as idx  # noqa: E402
from api.oauth_service import hash_password  # noqa: E402
from api.session_service import claim_anonymous_resources, is_session_owned  # noqa: E402
from api.utils.caller_scope import ANON_COOKIE_NAME, CallerScope, anon_token_digest, scope_filter  # noqa: E402

ANON_A = "a" * 43
ANON_B = "b" * 43
DIG_A = anon_token_digest(ANON_A)
DIG_B = anon_token_digest(ANON_B)
PASSWORD = "Sup3r-secret-pass!"

# (session_id / connection_id, user_id, org_id, anon_owner)
ROWS = [
    ("s-anon-a", None, None, DIG_A),
    ("s-anon-b", None, None, DIG_B),
    ("s-bob", "bob", "org-y", None),
]
TABLES = [idx.sessions_table, idx.db_connections_table, idx.users_table,
          idx.user_sessions_table, idx.audit_log_table]


def _request(cookies: Optional[dict[str, str]] = None) -> Request:
    cookie = "; ".join(f"{k}={v}" for k, v in (cookies or {}).items())
    headers = [(b"cookie", cookie.encode())] if cookie else []
    return Request({"type": "http", "method": "POST", "path": "/api/x", "headers": headers, "state": {}})


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    path = tmp_path / "claim.db"
    sync_engine = create_engine(f"sqlite:///{path}")
    idx.metadata.create_all(sync_engine, tables=TABLES)
    with sync_engine.begin() as conn:
        conn.execute(insert(idx.users_table).values(
            id="alice", email="alice@example.com", name="Alice", provider="email",
            password_hash=hash_password(PASSWORD), email_verified=True, org_id="org-x",
        ))
        for rid, uid, org, anon in ROWS:
            conn.execute(insert(idx.sessions_table).values(
                session_id=rid, user_id=uid, org_id=org, anon_owner=anon,
            ))
            conn.execute(insert(idx.db_connections_table).values(
                id=f"c-{rid}", session_id=rid, dialect="csv", host="", port=0, dbname="d",
                credentials_blob="x", user_id=uid, org_id=org, anon_owner=anon,
            ))
    sync_engine.dispose()
    factory = async_sessionmaker(create_async_engine(f"sqlite+aiosqlite:///{path}"), expire_on_commit=False)
    monkeypatch.setattr(idx, "async_session_factory", factory)
    monkeypatch.setattr(idx.request_guard, "is_demo_blocked", lambda *_a, **_k: False)
    monkeypatch.setattr(idx.request_guard, "check_ip_rate_limit", AsyncMock(return_value=None))
    yield factory


async def _claim(factory: Any, cookies: Optional[dict[str, str]]) -> int:
    return await claim_anonymous_resources(
        _request(cookies), "alice", idx.users_table,
        [idx.sessions_table, idx.db_connections_table], factory,
    )


async def _conn_visible(factory: Any, conn_id: str, scope: CallerScope) -> bool:
    t = idx.db_connections_table
    async with factory() as s:
        row = (await s.execute(select(t.c.id).where(t.c.id == conn_id, scope_filter(t, scope)))).fetchone()
    return row is not None


ALICE = CallerScope("alice", "org-x", None, False)
ANON_A_SCOPE = CallerScope(None, None, DIG_A, False)
ANON_B_SCOPE = CallerScope(None, None, DIG_B, False)
BOB = CallerScope("bob", "org-y", None, False)


async def test_claim_moves_anon_rows_to_user(db: Any) -> None:
    assert await is_session_owned("s-anon-a", ANON_A_SCOPE, idx.sessions_table, db)
    assert await _claim(db, {ANON_COOKIE_NAME: ANON_A}) == 2

    assert await is_session_owned("s-anon-a", ALICE, idx.sessions_table, db)
    assert await _conn_visible(db, "c-s-anon-a", ALICE)
    assert not await is_session_owned("s-anon-a", ANON_A_SCOPE, idx.sessions_table, db)
    assert not await _conn_visible(db, "c-s-anon-a", ANON_A_SCOPE)
    # Other anon token and rows already owned by a user are untouched.
    assert await is_session_owned("s-anon-b", ANON_B_SCOPE, idx.sessions_table, db)
    assert await _conn_visible(db, "c-s-anon-b", ANON_B_SCOPE)
    assert await is_session_owned("s-bob", BOB, idx.sessions_table, db)
    assert not await is_session_owned("s-bob", ALICE, idx.sessions_table, db)


async def test_claim_without_cookie_is_noop(db: Any) -> None:
    assert await _claim(db, None) == 0
    assert await _claim(db, {ANON_COOKIE_NAME: "short"}) == 0
    assert await is_session_owned("s-anon-a", ANON_A_SCOPE, idx.sessions_table, db)


async def test_claim_db_error_does_not_raise(db: Any) -> None:
    from sqlalchemy.exc import OperationalError

    def broken() -> Any:
        raise OperationalError("x", {}, Exception("down"))

    assert await claim_anonymous_resources(
        _request({ANON_COOKIE_NAME: ANON_A}), "alice", idx.users_table, [idx.sessions_table], broken,
    ) == 0


def test_login_route_claims_anon_work(db: Any, tmp_path: Path) -> None:
    client = TestClient(idx.app)
    client.cookies.set(ANON_COOKIE_NAME, ANON_A)
    resp = client.post("/api/auth/login", json={"email": "alice@example.com", "password": PASSWORD})
    assert resp.status_code == 200, resp.text

    engine = create_engine(f"sqlite:///{tmp_path / 'claim.db'}")
    with engine.connect() as conn:
        t = idx.sessions_table
        row = conn.execute(select(t.c.user_id, t.c.org_id, t.c.anon_owner).where(t.c.session_id == "s-anon-a")).one()
    engine.dispose()
    assert tuple(row) == ("alice", "org-x", None)
