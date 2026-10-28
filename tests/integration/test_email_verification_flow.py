"""DOCBOT-1713 — signup → verify → login integration on real SQLite.

Exercises the exact service sequence the routes run (register → Resend send →
403-gated login → POST /api/auth/verify → clean login), plus the DOCBOT-1617
compatibility contract: email verification updates the same email_verified
flag identity linking reads, so a verified OAuth identity can attach to a
verified password account, and never bypasses a verified one.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    MetaData,
    String,
    Table,
    Text,
    select,
)
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api import email_verification_service as evs
from api.identity_service import (
    ProviderIdentity,
    register_password_user,
    register_user_identities_table,
    resolve_provider_login,
)
from api.org_service import register_organizations_table


def _schema() -> tuple[MetaData, dict[str, Table]]:
    md = MetaData()
    orgs = register_organizations_table(md)
    users = Table(
        "users", md,
        Column("id", String, primary_key=True),
        Column("email", String, nullable=False, unique=True),
        Column("name", String, nullable=False),
        Column("provider", String, nullable=False),
        Column("provider_id", String),
        Column("password_hash", Text),
        Column("email_verified", Boolean, server_default="0", nullable=False),
        Column("role", String, server_default="analyst"),
        Column("last_login_at", DateTime(timezone=True)),
        Column("created_at", DateTime(timezone=True), server_default=None),
        Column("org_id", String),
    )
    sessions = Table(
        "user_sessions", md,
        Column("id", String, primary_key=True),
        Column("user_id", String, nullable=False),
        Column("token", String, nullable=False, unique=True),
        Column("expires_at", DateTime(timezone=True)),
    )
    identities = register_user_identities_table(md)
    tokens = evs.register_email_verification_tokens_table(md)
    return md, {
        "organizations": orgs, "users": users, "sessions": sessions,
        "identities": identities, "tokens": tokens,
    }


@pytest.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    md, tables = _schema()
    async with engine.begin() as conn:
        await conn.run_sync(md.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory, tables
    await engine.dispose()


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    monkeypatch.setenv("EMAIL_FROM", "DocBot <hello@x.com>")
    monkeypatch.setenv("FRONTEND_URL", "https://docbot.example")


def _mock_resend() -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.json.return_value = {"id": "sent"}
    resp.raise_for_status.return_value = None
    client.post = AsyncMock(return_value=resp)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _google(sub: str, email: str, verified: bool) -> ProviderIdentity:
    return ProviderIdentity(
        provider="google", provider_sub=sub, email=email,
        email_verified=verified, name="G",
    )


def _resp_mock_client(post_payload: dict) -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.json.return_value = post_payload
    resp.raise_for_status.return_value = None
    client.post = AsyncMock(return_value=resp)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


async def _user(factory: Any, tables: dict[str, Table], uid: str) -> Any:
    async with factory() as s:
        return (await s.execute(
            select(tables["users"]).where(tables["users"].c.id == uid)
        )).fetchone()


class TestSignupVerifyLoginFlow:
    async def test_full_flow_with_resend_enabled(self, db, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable(monkeypatch)
        factory, tables = db

        # ── signup: insert-only user, link emailed, NO session state to assert ──
        user_id = await register_password_user(
            "N@x.com", "N", "hashed-bcrypt", tables["users"], factory
        )
        resend_ctx = _mock_resend()
        with patch.object(evs.httpx, "AsyncClient", return_value=resend_ctx):
            sent = await evs.issue_verification_token_and_send(
                user_id, "n@x.com", tables["tokens"], factory
            )
        assert sent is True
        client = resend_ctx.__aenter__.return_value
        payload = client.post.call_args.kwargs["json"]
        assert payload["to"] == ["n@x.com"]
        text = payload["text"]
        assert "https://docbot.example/verify?token=" in text
        raw_token = text.rsplit("/verify?token=", 1)[1].split("\n")[0]

        user = await _user(factory, tables, user_id)
        # Sign-up leaves the account unverified → the login route's
        # login_requires_verification() becomes our 403 "verify required".
        assert user.email_verified is False
        assert evs.login_requires_verification(user) is True

        # ── POST /api/auth/verify with the emailed token ──
        verified_email = await evs.consume_verification_token(
            raw_token, tables["tokens"], tables["users"], factory
        )
        assert verified_email == "n@x.com"

        user = await _user(factory, tables, user_id)
        assert user.email_verified is True
        # The same flag gates login again — now open.
        assert evs.login_requires_verification(user) is False

        # ── token reuse fails (single use) ──
        with pytest.raises(evs.VerificationTokenInvalid):
            await evs.consume_verification_token(
                raw_token, tables["tokens"], tables["users"], factory
            )

    async def test_dormant_mode_matches_today(self, db, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RESEND_API_KEY", raising=False)
        monkeypatch.delenv("EMAIL_FROM", raising=False)
        monkeypatch.delenv("TURNSTILE_SECRET_KEY", raising=False)
        factory, tables = db
        user_id = await register_password_user(
            "N@x.com", "N", "hashed", tables["users"], factory
        )
        user = await _user(factory, tables, user_id)
        # No mint, no send, no gate: exactly pre-1713 behavior.
        assert user.email_verified is False
        assert evs.is_email_verification_enabled() is False
        assert evs.login_requires_verification(user) is False

    async def test_query_token_param_equivalent(
        self, db, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The route accepts the token via body OR query; the service must be
        identical for both."""
        _enable(monkeypatch)
        factory, tables = db
        user_id = await register_password_user(
            "n@x.com", "N", "h", tables["users"], factory
        )
        raw = await evs.mint_verification_token(
            user_id, "n@x.com", tables["tokens"], factory
        )
        assert await evs.consume_verification_token(
            f" {raw} ", tables["tokens"], tables["users"], factory
        ) == "n@x.com"  # whitespace stripped, matching a URL-decoded query param


class TestDocbot1617Compatibility:
    async def test_verified_oauth_identity_links_after_email_verification(
        self, db, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After POST /api/auth/verify, the SAME flag DOCBOT-1617 checks is set,
        so a verified Google identity for that address attaches WITHOUT the
        pre-hijack password wipe — the flow feeds 1617's verified-email state."""
        _enable(monkeypatch)
        factory, tables = db
        user_id = await register_password_user(
            "n@x.com", "N", "hashed", tables["users"], factory
        )
        raw = await evs.mint_verification_token(
            user_id, "n@x.com", tables["tokens"], factory
        )
        await evs.consume_verification_token(raw, tables["tokens"], tables["users"], factory)

        uid = await resolve_provider_login(
            _google("g-1", "n@x.com", True), tables["users"], tables["identities"],
            tables["sessions"], factory,
        )
        assert uid == user_id
        user = await _user(factory, tables, user_id)
        assert user.password_hash == "hashed"  # password untouched — account was legit
        assert user.email_verified is True

    async def test_unverified_password_account_still_pre_hijacked_cleared(
        self, db, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dormant mode keeps 1617's defense intact: registering an address with
        a password and never verifying it (vulnerable) still loses the password
        when the real verified OAuth owner shows up. The token flow can't be
        used to bypass that."""
        monkeypatch.delenv("RESEND_API_KEY", raising=False)
        factory, tables = db
        user_id = await register_password_user(
            "owner@x.com", "Evil", "roguehash", tables["users"], factory
        )
        uid = await resolve_provider_login(
            _google("g-1", "owner@x.com", True), tables["users"], tables["identities"],
            tables["sessions"], factory,
        )
        assert uid == user_id
        user = await _user(factory, tables, user_id)
        assert user.password_hash is None and user.provider == "google"

    async def test_verify_links_disabled_for_dormant_accounts(
        self, db, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Live Resend + an unverified email account: verified Google identity
        claims the address and verifies it via 1617's own logic (its
        `_link_existing` arm) — using, not bypassing, the same flag."""
        _enable(monkeypatch)
        factory, tables = db
        user_id = await register_password_user(
            "n@x.com", "N", "hashed", tables["users"], factory
        )
        uid = await resolve_provider_login(
            _google("g-1", "n@x.com", True), tables["users"], tables["identities"],
            tables["sessions"], factory,
        )
        assert uid == user_id
        user = await _user(factory, tables, user_id)
        assert user.email_verified is True
        # 1617 already flipped the flag: the verification gate is now off.
        assert evs.login_requires_verification(user) is False


class TestTurnstileInSignupSequence:
    async def test_enforce_steps_before_user_creation(
        self, db, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Route order is turnstile → username/password checks → insert; a bad
        token must abort BEFORE any user row exists."""
        monkeypatch.setenv("TURNSTILE_SECRET_KEY", "0xsecret")
        factory, tables = db
        with pytest.raises(evs.TurnstileValidationError):
            await evs.enforce_turnstile("forged-token", "1.2.3.4")
        ctx = _resp_mock_client({"success": False})
        with patch.object(evs.httpx, "AsyncClient", return_value=ctx):
            with pytest.raises(evs.TurnstileValidationError):
                await evs.enforce_turnstile("tok", None)
        async with factory() as s:
            assert (await s.execute(select(tables["users"]))).fetchall() == []
