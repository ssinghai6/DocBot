"""DOCBOT-2332 — canonical-email dedupe, grant-on-verify, register timing,
Turnstile client IP, boot backfill. SQLite only; Resend/Turnstile mocked."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import AsyncMock

import anyio.from_thread
import pytest
from cryptography.fernet import Fernet

if not os.environ.get("DB_ENCRYPTION_KEY"):
    os.environ["DB_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql://fake:fake@localhost:5432/testdb"

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, insert, select, text  # noqa: E402
from sqlalchemy.exc import IntegrityError  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

import api.index as idx  # noqa: E402
from api import credit_service, oauth_service  # noqa: E402
from api import email_verification_service as evs  # noqa: E402
from api.identity_service import (  # noqa: E402
    EMAIL_CANONICAL_INDEX_DDL,
    ProviderIdentity,
    backfill_email_canonical,
    resolve_provider_login,
)
from api.utils import request_guard  # noqa: E402
from api.utils.email_canonical import canonical_email  # noqa: E402

PASSWORD = "Sup3r-secret-pass!"
TABLES = [
    idx.organizations_table, idx.users_table, idx.user_sessions_table, idx.audit_log_table,
    idx.email_verification_tokens_table, idx.user_identities_table,
    idx.credit_tables.org_balances, idx.credit_tables.credit_ledger, idx.credit_tables.credit_reservations,
]


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw, canon", [
    ("A.B+promo@Gmail.com", "ab@gmail.com"),
    ("a.b.c@googlemail.com", "abc@gmail.com"),
    ("ab+1@gmail.com", "ab@gmail.com"),
    ("first.last+x@corp.com", "first.last@corp.com"),  # other domains keep dots
    ("  First.Last@Corp.COM ", "first.last@corp.com"),
    ("plain@example.org", "plain@example.org"),
])
def test_canonical_email(raw: str, canon: str) -> None:
    assert canonical_email(raw) == canon


# ---------------------------------------------------------------------------
# Route fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, Any, AsyncMock]]:
    path = tmp_path / "farm.db"
    sync_engine = create_engine(f"sqlite:///{path}")
    idx.metadata.create_all(sync_engine, tables=TABLES)
    sync_engine.dispose()
    factory = async_sessionmaker(create_async_engine(f"sqlite+aiosqlite:///{path}"), expire_on_commit=False)
    monkeypatch.setattr(idx, "async_session_factory", factory)
    monkeypatch.setattr(idx.request_guard, "is_demo_blocked", lambda *_a, **_k: False)
    monkeypatch.setattr(idx.request_guard, "check_ip_rate_limit", AsyncMock(return_value=None))
    monkeypatch.setattr(oauth_service, "is_password_breached", AsyncMock(return_value=False))
    monkeypatch.setenv("RESEND_API_KEY", "re_test")
    monkeypatch.setenv("EMAIL_FROM", "DocBot <hi@x.com>")
    monkeypatch.setenv("BILLING_FREE_MONTHLY_CREDITS", "500")
    send = AsyncMock(return_value=None)
    monkeypatch.setattr(evs, "send_verification_email", send)
    credit_service.wire_credit_store(idx.credit_tables, factory, "sqlite")
    request_guard.reset_state()
    client = TestClient(idx.app)
    # One long-lived loop: background sends must outlive the request, and the
    # test's own DB calls run on the same loop via ``client.portal.call``.
    with anyio.from_thread.start_blocking_portal() as portal:
        client.portal = portal
        yield client, factory, send
    request_guard.reset_state()
    credit_service._tables = None
    credit_service._session_factory = None
    credit_service._granted_period.clear()


def _register(client: TestClient, email: str, **headers: str) -> Any:
    return client.post(
        "/api/auth/register",
        json={"email": email, "password": PASSWORD, "accepted_terms": True},
        headers=headers,
    )


async def _users(factory: Any) -> list[Any]:
    async with factory() as s:
        return (await s.execute(select(idx.users_table))).fetchall()


def _wait_for(predicate: Any, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "background send never ran"
        time.sleep(0.01)


# ---------------------------------------------------------------------------
# Register dedupe + timing + Turnstile IP
# ---------------------------------------------------------------------------


def test_gmail_aliases_register_one_account_with_identical_responses(env) -> None:
    client, factory, send = env
    first = _register(client, "a+1@gmail.com")
    _wait_for(lambda: send.await_count == 1)
    for alias in ("a@gmail.com", "A.@gmail.com", "a+2@googlemail.com"):
        dup = _register(client, alias)
        assert dup.status_code == first.status_code == 202
        assert dup.json() == {"status": "verification_required", "email": alias.lower()}
    rows = client.portal.call(_users, factory)
    assert len(rows) == 1 and rows[0].email_canonical == "a@gmail.com"
    assert send.await_count == 1  # nothing sent for duplicates


def test_new_email_path_does_not_await_resend(env, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = env
    sent = []

    async def slow_send(email: str, raw: str) -> None:
        await asyncio.sleep(1.0)
        sent.append(email)

    monkeypatch.setattr(evs, "send_verification_email", slow_send)
    start = time.monotonic()
    resp = _register(client, "slow@example.com")
    new_elapsed = time.monotonic() - start
    assert resp.status_code == 202 and sent == []
    start = time.monotonic()
    _register(client, "slow@example.com")
    dup_elapsed = time.monotonic() - start
    assert new_elapsed < 0.8 and abs(new_elapsed - dup_elapsed) < 0.5
    _wait_for(lambda: sent == ["slow@example.com"])


def test_turnstile_receives_trusted_client_ip(env, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = env
    enforce = AsyncMock(return_value=None)
    monkeypatch.setattr(evs, "enforce_turnstile", enforce)
    monkeypatch.setattr(request_guard, "TRUSTED_PROXY_HOPS", 1)
    _register(client, "ip@example.com", **{"X-Forwarded-For": "6.6.6.6, 203.0.113.9"})
    assert enforce.await_args.args[1] == "203.0.113.9"


# ---------------------------------------------------------------------------
# Grant on verify
# ---------------------------------------------------------------------------


async def _free_grants(factory: Any) -> list[Any]:
    lg = idx.credit_tables.credit_ledger
    async with factory() as s:
        return (await s.execute(select(lg).where(lg.c.kind == "free_grant"))).fetchall()


def test_unverified_reserve_no_grant_then_verify_grants_once(env) -> None:
    client, factory, send = env
    _register(client, "v@example.com")
    _wait_for(lambda: send.await_count == 1)
    raw_token = send.await_args.args[1]
    user = client.portal.call(_users, factory)[0]
    billing_org = credit_service.billing_org_for(user.id, user.org_id)

    async def flow() -> None:
        with pytest.raises(credit_service.InsufficientCredits):
            await credit_service.reserve(billing_org, 1, "r1", "chat")
        assert await _free_grants(factory) == []
        await evs.consume_verification_token(
            raw_token, idx.email_verification_tokens_table, idx.users_table, factory,
        )
        with pytest.raises(evs.VerificationTokenInvalid):  # second click
            await evs.consume_verification_token(
                raw_token, idx.email_verification_tokens_table, idx.users_table, factory,
            )
        await evs._grant_on_verify(user.id, user.org_id)  # replayed grant
        credit_service._granted_period.clear()
        await credit_service.reserve(billing_org, 1, "r2", "chat")
        grants = await _free_grants(factory)
        assert len(grants) == 1 and grants[0].org_id == billing_org

    client.portal.call(flow)


# ---------------------------------------------------------------------------
# OAuth: an alias links instead of creating a second account
# ---------------------------------------------------------------------------


def test_verified_oauth_alias_links_to_existing_account(env) -> None:
    client, factory, send = env
    _register(client, "ab@gmail.com")
    _wait_for(lambda: send.await_count == 1)
    owner = client.portal.call(_users, factory)[0]
    ident = ProviderIdentity(
        provider="github", provider_sub="gh-1", email="a.b@gmail.com", email_verified=True, name="A",
    )
    uid = client.portal.call(
        resolve_provider_login,
        ident, idx.users_table, idx.user_identities_table, idx.user_sessions_table, factory,
    )
    assert uid == owner.id and len(client.portal.call(_users, factory)) == 1


def test_non_gmail_plus_alias_never_links_provider_login(env) -> None:
    """foo+x@corp.com may be a different mailbox: no takeover of foo@corp.com."""
    client, factory, send = env
    _register(client, "foo@corp.com")
    _wait_for(lambda: send.await_count == 1)
    owner = client.portal.call(_users, factory)[0]
    ident = ProviderIdentity(
        provider="github", provider_sub="gh-2", email="foo+x@corp.com", email_verified=True, name="X",
    )
    uid = client.portal.call(
        resolve_provider_login,
        ident, idx.users_table, idx.user_identities_table, idx.user_sessions_table, factory,
    )
    users = client.portal.call(_users, factory)
    assert uid != owner.id and len(users) == 2
    new = next(u for u in users if u.id == uid)
    assert new.email_canonical is None  # canonical stays owned by foo@corp.com


def test_canonical_is_same_mailbox_only_for_gmail() -> None:
    from api.utils.email_canonical import canonical_is_same_mailbox

    assert canonical_is_same_mailbox("A.B+x@GoogleMail.com")
    assert canonical_is_same_mailbox("ab@gmail.com")
    assert not canonical_is_same_mailbox("foo+x@corp.com")
    assert not canonical_is_same_mailbox("not-an-email")


# ---------------------------------------------------------------------------
# Boot backfill
# ---------------------------------------------------------------------------


async def test_backfill_idempotent_and_collision_safe(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'bf.db'}")
    async with engine.begin() as conn:
        # Legacy schema: column exists (ALTER ran) but no unique index yet.
        await conn.execute(text(
            "CREATE TABLE users (id TEXT PRIMARY KEY, email TEXT UNIQUE, "
            "email_canonical TEXT, created_at TEXT)"
        ))
        for uid, email, at in (
            ("u1", "ab@gmail.com", "2024-01-01"),
            ("u2", "a.b@gmail.com", "2024-02-01"),
            ("u3", "x@corp.com", "2024-03-01"),
        ):
            await conn.execute(
                text("INSERT INTO users (id, email, created_at) VALUES (:i, :e, :t)"),
                {"i": uid, "e": email, "t": at},
            )
    for _ in range(2):
        async with engine.begin() as conn:
            assert await backfill_email_canonical(conn) == 1
            await conn.execute(text(EMAIL_CANONICAL_INDEX_DDL))
    async with engine.begin() as conn:
        rows = dict((await conn.execute(text("SELECT id, email_canonical FROM users"))).fetchall())
    assert rows == {"u1": "ab@gmail.com", "u2": None, "u3": "x@corp.com"}
    await engine.dispose()


def test_users_table_insert_rejects_duplicate_canonical(tmp_path: Path) -> None:
    sync_engine = create_engine(f"sqlite:///{tmp_path / 'u.db'}")
    idx.metadata.create_all(sync_engine, tables=[idx.users_table])
    base = {"name": "n", "provider": "email"}
    with sync_engine.begin() as conn:
        conn.execute(insert(idx.users_table).values(id="1", email="ab@gmail.com", email_canonical="ab@gmail.com", **base))
    with pytest.raises(IntegrityError):
        with sync_engine.begin() as conn:
            conn.execute(insert(idx.users_table).values(id="2", email="a.b@gmail.com", email_canonical="ab@gmail.com", **base))
    sync_engine.dispose()
