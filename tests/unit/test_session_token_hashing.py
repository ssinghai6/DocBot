"""DOCBOT-1621 — session tokens hashed at rest, TTL enforced, revoked on credential change.

In-memory SQLite (aiosqlite). No network.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api.auth_service import (
    create_user_session,
    delete_session,
    get_user_from_session,
    hash_legacy_session_tokens,
    hash_session_token,
)
from api.identity_service import ProviderIdentity, resolve_provider_login
from tests.unit.test_identity_linking import _schema


@pytest.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    md, tables = _schema()
    async with engine.begin() as conn:
        await conn.run_sync(md.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        async with s.begin():
            await s.execute(insert(tables["users"]).values(
                id="u1", email="u1@x.com", name="U1", provider="email",
                password_hash="hash", email_verified=False,
            ))
    yield engine, factory, tables
    await engine.dispose()


async def _tokens(factory: Any, tables: dict) -> list[str]:
    async with factory() as s:
        return [r.token for r in (await s.execute(select(tables["sessions"]))).fetchall()]


async def _lookup(factory: Any, tables: dict, token: str) -> Any:
    return await get_user_from_session(token, tables["sessions"], tables["users"], factory)


async def test_db_never_contains_raw_token(db):
    _, factory, tables = db
    token = await create_user_session("u1", tables["sessions"], factory)
    stored = await _tokens(factory, tables)
    assert stored == [hash_session_token(token)]
    assert token not in stored[0]


async def test_lookup_and_logout(db):
    _, factory, tables = db
    token = await create_user_session("u1", tables["sessions"], factory)
    assert (await _lookup(factory, tables, token)).id == "u1"
    assert await _lookup(factory, tables, "not-a-token") is None
    # The stored digest itself is not a usable cookie.
    assert await _lookup(factory, tables, hash_session_token(token)) is None
    await delete_session(token, tables["sessions"], factory)
    assert await _lookup(factory, tables, token) is None


async def test_expired_session_rejected(db):
    _, factory, tables = db
    token = await create_user_session("u1", tables["sessions"], factory)
    async with factory() as s:
        async with s.begin():
            await s.execute(update(tables["sessions"]).values(
                expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)
            ))
    assert await _lookup(factory, tables, token) is None


async def test_legacy_plaintext_rows_hashed_in_place(db):
    engine, factory, tables = db
    now = datetime.now(timezone.utc)
    async with factory() as s:
        async with s.begin():
            await s.execute(insert(tables["sessions"]).values(
                id="live", user_id="u1", token="a" * 64, expires_at=now + timedelta(hours=1),
            ))
            await s.execute(insert(tables["sessions"]).values(
                id="dead", user_id="u1", token="b" * 64, expires_at=now - timedelta(hours=1),
            ))
    async with engine.begin() as conn:
        assert await hash_legacy_session_tokens(conn, tables["sessions"]) == 1
    async with engine.begin() as conn:
        assert await hash_legacy_session_tokens(conn, tables["sessions"]) == 0  # idempotent

    assert await _tokens(factory, tables) == [hash_session_token("a" * 64)]
    # Existing users stay logged in: the old cookie still resolves.
    assert (await _lookup(factory, tables, "a" * 64)).id == "u1"


async def test_credential_change_revokes_all_sessions(db):
    """A verified OAuth claim clears the password and must kill every session."""
    _, factory, tables = db
    t1 = await create_user_session("u1", tables["sessions"], factory)
    t2 = await create_user_session("u1", tables["sessions"], factory)
    await resolve_provider_login(
        ProviderIdentity(provider="google", provider_sub="g", email="u1@x.com",
                         email_verified=True, name="G"),
        tables["users"], tables["identities"], tables["sessions"], factory,
    )
    assert await _lookup(factory, tables, t1) is None
    assert await _lookup(factory, tables, t2) is None
