"""DOCBOT-1713 — email verification token lifecycle + Turnstile gate.

In-memory SQLite (aiosqlite, real DB semantics); httpx is fully mocked for
both Resend and Cloudflare siteverify — no network, no API keys.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
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
    func,
    insert,
    select,
    update,
)
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api import email_verification_service as evs


# ---------------------------------------------------------------------------
# Schema + fixtures (mirrors tests/unit/test_identity_linking.py)
# ---------------------------------------------------------------------------


def _schema() -> tuple[MetaData, dict[str, Table]]:
    md = MetaData()
    users = Table(
        "users", md,
        Column("id", String, primary_key=True),
        Column("email", String, nullable=False, unique=True),
        Column("name", String, nullable=False),
        Column("provider", String, nullable=False),
        Column("password_hash", Text),
        Column("email_verified", Boolean, server_default="0", nullable=False),
        Column("created_at", DateTime(timezone=True), server_default=func.now()),
    )
    tokens = evs.register_email_verification_tokens_table(md)
    return md, {"users": users, "tokens": tokens}


@pytest.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    md, tables = _schema()
    async with engine.begin() as conn:
        await conn.run_sync(md.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory, tables
    await engine.dispose()


async def mk_user(factory: Any, tables: dict[str, Table], email: str = "n@x.com") -> str:
    async with factory() as s:
        async with s.begin():
            await s.execute(insert(tables["users"]).values(
                id="u-1", email=email, name="N", provider="email",
                password_hash="hash", email_verified=False,
            ))
    return "u-1"


async def one(factory: Any, table: Table, **filters: Any) -> Any:
    async with factory() as s:
        return (await s.execute(
            select(table).where(
                *[getattr(table.c, k) == v for k, v in filters.items()]
            )
        )).fetchone()


async def all_rows(factory: Any, table: Table) -> list[Any]:
    async with factory() as s:
        return (await s.execute(select(table))).fetchall()


def _resp(payload: Any) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _mock_post(payload: Any, *, fail: bool = False) -> MagicMock:
    """Async httpx.AsyncClient whose post returns `payload` (or raises)."""
    client = MagicMock()
    if fail:
        client.post = AsyncMock(side_effect=evs.httpx.ConnectError("boom"))
    else:
        resp = _resp(payload)
        if fail:
            resp.raise_for_status.side_effect = ValueError()
        client.post = AsyncMock(return_value=resp)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


# ---------------------------------------------------------------------------
# Feature toggles — dormant by default
# ---------------------------------------------------------------------------


class TestFeatureToggles:
    def test_verification_disabled_without_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RESEND_API_KEY", raising=False)
        monkeypatch.delenv("EMAIL_FROM", raising=False)
        assert evs.is_email_verification_enabled() is False

    @pytest.mark.parametrize("first,second", [
        ("key", None), (None, "docbot <a@b.com>"), ("", "docbot <a@b.com>"),
        ("key", ""), ("  ", "  "),
    ])
    def test_verification_needs_both_keys(
        self, monkeypatch: pytest.MonkeyPatch, first: Any, second: Any
    ) -> None:
        for name, value in (("RESEND_API_KEY", first), ("EMAIL_FROM", second)):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        assert evs.is_email_verification_enabled() is False

    def test_verification_enabled_with_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RESEND_API_KEY", "re_key")
        monkeypatch.setenv("EMAIL_FROM", "DocBot <hello@x.com>")
        assert evs.is_email_verification_enabled() is True

    def test_turnstile_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TURNSTILE_SECRET_KEY", raising=False)
        assert evs.is_turnstile_enabled() is False

    def test_turnstile_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TURNSTILE_SECRET_KEY", "0xsecret")
        assert evs.is_turnstile_enabled() is True

    def test_login_gate_dormant_even_for_unverified_user(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("RESEND_API_KEY", raising=False)
        assert evs.login_requires_verification(MagicMock(provider="email", email_verified=False)) is False

    def test_login_gate_live_for_unverified_password_user(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RESEND_API_KEY", "k")
        monkeypatch.setenv("EMAIL_FROM", "D <d@x.com>")
        assert evs.login_requires_verification(MagicMock(provider="email", email_verified=False)) is True

    @pytest.mark.parametrize("provider,verified", [
        ("google", False), ("github", False), ("saml", False), ("email", True),
    ])
    def test_login_gate_only_blocks_unverified_password_accounts(
        self, monkeypatch: pytest.MonkeyPatch, provider: str, verified: bool
    ) -> None:
        monkeypatch.setenv("RESEND_API_KEY", "k")
        monkeypatch.setenv("EMAIL_FROM", "D <d@x.com>")
        assert evs.login_requires_verification(
            MagicMock(provider=provider, email_verified=verified)
        ) is False


# ---------------------------------------------------------------------------
# Token mint / consume lifecycle
# ---------------------------------------------------------------------------


class TestTokenLifecycle:
    async def test_mint_stores_hash_not_raw(self, db) -> None:
        factory, tables = db
        user_id = await mk_user(factory, tables)
        raw = await evs.mint_verification_token(user_id, "n@x.com", tables["tokens"], factory)
        row = await one(factory, tables["tokens"], user_id=user_id)
        assert raw != row.token_hash
        assert row.token_hash == hashlib.sha256(raw.encode()).hexdigest()
        assert row.used_at is None and row.email == "n@x.com"

    async def test_minted_tokens_are_unique(self, db) -> None:
        factory, tables = db
        a = await evs.mint_verification_token("u", "a@x.com", tables["tokens"], factory)
        b = await evs.mint_verification_token("u", "a@x.com", tables["tokens"], factory)
        assert a != b

    async def test_mint_sets_24h_expiry(self, db) -> None:
        factory, tables = db
        await evs.mint_verification_token("u", "n@x.com", tables["tokens"], factory)
        row = await one(factory, tables["tokens"], email="n@x.com")
        delta = evs._as_aware(row.expires_at) - datetime.now(timezone.utc)
        assert timedelta(hours=23) < delta <= timedelta(hours=24)

    async def test_consume_flips_email_verified(self, db) -> None:
        factory, tables = db
        user_id = await mk_user(factory, tables)
        raw = await evs.mint_verification_token(user_id, "n@x.com", tables["tokens"], factory)
        email = await evs.consume_verification_token(
            raw, tables["tokens"], tables["users"], factory
        )
        assert email == "n@x.com"
        user = await one(factory, tables["users"], id=user_id)
        assert user.email_verified is True
        token_row = await one(factory, tables["tokens"], user_id=user_id)
        assert token_row.used_at is not None

    async def test_consume_is_single_use(self, db) -> None:
        factory, tables = db
        user_id = await mk_user(factory, tables)
        raw = await evs.mint_verification_token(user_id, "n@x.com", tables["tokens"], factory)
        await evs.consume_verification_token(raw, tables["tokens"], tables["users"], factory)
        with pytest.raises(evs.VerificationTokenInvalid):
            await evs.consume_verification_token(raw, tables["tokens"], tables["users"], factory)

    @pytest.mark.parametrize("raw", ["", "   ", "x" * 513])
    async def test_consume_rejects_malformed_input(self, db, raw: str) -> None:
        factory, tables = db
        with pytest.raises(evs.VerificationTokenInvalid):
            await evs.consume_verification_token(raw, tables["tokens"], tables["users"], factory)
        assert await all_rows(factory, tables["tokens"]) == []

    async def test_consume_rejects_unknown_token(self, db) -> None:
        factory, tables = db
        with pytest.raises(evs.VerificationTokenInvalid):
            await evs.consume_verification_token(
                "no-such-token", tables["tokens"], tables["users"], factory
            )
        assert (await one(factory, tables["users"], id="u-1")) is None

    async def test_consume_rejects_expired_token(self, db) -> None:
        factory, tables = db
        user_id = await mk_user(factory, tables)
        raw = await evs.mint_verification_token(
            user_id, "n@x.com", tables["tokens"], factory, ttl_hours=-1
        )
        with pytest.raises(evs.VerificationTokenInvalid):
            await evs.consume_verification_token(raw, tables["tokens"], tables["users"], factory)
        assert (await one(factory, tables["users"], id=user_id)).email_verified is False

    async def test_consumed_token_cannot_be_reused(self, db) -> None:
        factory, tables = db
        user_id = await mk_user(factory, tables)
        raw = await evs.mint_verification_token(user_id, "n@x.com", tables["tokens"], factory)
        await evs.consume_verification_token(raw, tables["tokens"], tables["users"], factory)
        with pytest.raises(evs.VerificationTokenInvalid):
            await evs.consume_verification_token(raw, tables["tokens"], tables["users"], factory)

    async def test_consume_rejects_email_mismatch(self, db) -> None:
        """A stale link must not verify an account whose address was re-taken."""
        factory, tables = db
        user_id = await mk_user(factory, tables)
        raw = await evs.mint_verification_token(user_id, "n@x.com", tables["tokens"], factory)
        async with factory() as s:
            async with s.begin():
                await s.execute(update(tables["users"]).values(email="moved@x.com").where(
                    tables["users"].c.id == user_id
                ))
        with pytest.raises(evs.VerificationTokenInvalid):
            await evs.consume_verification_token(raw, tables["tokens"], tables["users"], factory)
        assert (await one(factory, tables["users"], id=user_id)).email_verified is False

    async def test_second_mint_for_same_user_leaves_others_valid(self, db) -> None:
        factory, tables = db
        user_id = await mk_user(factory, tables)
        first = await evs.mint_verification_token(user_id, "n@x.com", tables["tokens"], factory)
        second = await evs.mint_verification_token(user_id, "n@x.com", tables["tokens"], factory)
        assert first != second
        email = await evs.consume_verification_token(
            first, tables["tokens"], tables["users"], factory
        )
        assert email == "n@x.com"


# ---------------------------------------------------------------------------
# Resend delivery (httpx mocked)
# ---------------------------------------------------------------------------


class TestResendDelivery:
    async def test_send_posts_correct_payload(self, db, monkeypatch: pytest.MonkeyPatch) -> None:
        factory, tables = db
        monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
        monkeypatch.setenv("EMAIL_FROM", "DocBot <hello@x.com>")
        ctx = _mock_post({"id": "abc"})
        with patch.object(evs.httpx, "AsyncClient", return_value=ctx):
            await evs.send_verification_email("n@x.com", "raw-token")
        client = ctx.__aenter__.return_value
        args, kwargs = client.post.call_args
        assert args[0] == evs.RESEND_API_URL
        assert kwargs["headers"]["Authorization"] == "Bearer re_test_key"
        payload = kwargs["json"]
        assert payload["from"] == "DocBot <hello@x.com>"
        assert payload["to"] == ["n@x.com"]
        assert "/verify?token=raw-token" in payload["text"]
        assert "/verify?token=raw-token" in payload["html"]
        assert payload["subject"] == "Verify your DocBot email"

    async def test_issue_wrapper_swallows_send_failure(
        self, db, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Resend being down must NOT 500 the signup; the log must not carry
        the raw token (it only ever names the user)."""
        factory, tables = db
        user_id = await mk_user(factory, tables)
        monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
        monkeypatch.setenv("EMAIL_FROM", "D <d@x.com>")
        ctx = _mock_post({}, fail=True)
        with patch.object(evs.httpx, "AsyncClient", return_value=ctx), caplog.at_level(
            "WARNING", logger="api.email_verification_service"
        ):
            sent = await evs.issue_verification_token_and_send(
                user_id, "n@x.com", tables["tokens"], factory
            )
        assert sent is False
        assert await one(factory, tables["tokens"], user_id=user_id) is not None
        raw_out = "\n".join(rec.getMessage() for rec in caplog.records)
        raw_out = raw_out.replace("raw-token", "")  # caplog never carries it anyway
        assert "token=" not in raw_out

    async def test_issue_wrapper_reports_unconfigured(
        self, db, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        factory, tables = db
        monkeypatch.delenv("RESEND_API_KEY", raising=False)
        monkeypatch.delenv("EMAIL_FROM", raising=False)
        user_id = await mk_user(factory, tables)
        sent = await evs.issue_verification_token_and_send(
            user_id, "n@x.com", tables["tokens"], factory
        )
        assert sent is False
        # Mint happened (self-healing), but nothing was sent.
        assert await one(factory, tables["tokens"], user_id=user_id) is not None


# ---------------------------------------------------------------------------
# Login-path re-send throttling
# ---------------------------------------------------------------------------


class TestResendIfDue:
    async def test_first_call_sends(self, db, monkeypatch: pytest.MonkeyPatch) -> None:
        factory, tables = db
        monkeypatch.setenv("RESEND_API_KEY", "k")
        monkeypatch.setenv("EMAIL_FROM", "D <d@x.com>")
        ctx = _mock_post({"id": "x"})
        with patch.object(evs.httpx, "AsyncClient", return_value=ctx):
            await evs.resend_verification_if_due(
                "u-1", "n@x.com", tables["tokens"], factory
            )
        assert ctx.__aenter__.return_value.post.call_count == 1
        assert await one(factory, tables["tokens"], user_id="u-1") is not None

    async def test_cooldown_blocks_second_send(self, db, monkeypatch: pytest.MonkeyPatch) -> None:
        factory, tables = db
        monkeypatch.setenv("RESEND_API_KEY", "k")
        monkeypatch.setenv("EMAIL_FROM", "D <d@x.com>")
        ctx = _mock_post({"id": "x"})
        with patch.object(evs.httpx, "AsyncClient", return_value=ctx):
            await evs.resend_verification_if_due("u-1", "n@x.com", tables["tokens"], factory)
            await evs.resend_verification_if_due("u-1", "n@x.com", tables["tokens"], factory)
        assert ctx.__aenter__.return_value.post.call_count == 1
        assert len(await all_rows(factory, tables["tokens"])) == 1

    @pytest.mark.parametrize("enabled", [True, False])
    async def test_dormant_toggle_never_sends(
        self, db, monkeypatch: pytest.MonkeyPatch, enabled: bool
    ) -> None:
        factory, tables = db
        if enabled:
            monkeypatch.setenv("RESEND_API_KEY", "k")
            monkeypatch.setenv("EMAIL_FROM", "D <d@x.com>")
        else:
            monkeypatch.delenv("RESEND_API_KEY", raising=False)
            monkeypatch.delenv("EMAIL_FROM", raising=False)
        ctx = _mock_post({"id": "x"})
        with patch.object(evs.httpx, "AsyncClient", return_value=ctx):
            await evs.resend_verification_if_due("u-1", "n@x.com", tables["tokens"], factory)
        assert ctx.__aenter__.return_value.post.call_count == (1 if enabled else 0)


# ---------------------------------------------------------------------------
# Turnstile bot protection (siteverify mocked)
# ---------------------------------------------------------------------------


class TestTurnstile:
    async def test_enforce_is_silent_when_dormant(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TURNSTILE_SECRET_KEY", raising=False)
        # Even a totally broken token passes in dormant mode (zero prod change).
        await evs.enforce_turnstile("stale-token", "1.2.3.4")

    async def test_missing_token_rejected_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TURNSTILE_SECRET_KEY", "0xsecret")
        with pytest.raises(evs.TurnstileValidationError, match="bot check"):
            await evs.enforce_turnstile(None, "1.2.3.4")

    async def test_siteverify_success_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TURNSTILE_SECRET_KEY", "0xsecret")
        ctx = _mock_post({"success": True})
        with patch.object(evs.httpx, "AsyncClient", return_value=ctx):
            assert await evs.verify_turnstile_token("tok", "1.2.3.4") is True
        client = ctx.__aenter__.return_value
        _, kwargs = client.post.call_args
        assert kwargs["data"] == {
            "secret": "0xsecret", "response": "tok", "remoteip": "1.2.3.4",
        }
        assert "0xsecret" not in str(client.post.call_args[0])

    async def test_siteverify_false_token_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TURNSTILE_SECRET_KEY", "0xsecret")
        ctx = _mock_post({"success": False, "error-codes": ["invalid-input-response"]})
        with patch.object(evs.httpx, "AsyncClient", return_value=ctx):
            with pytest.raises(evs.TurnstileValidationError):
                await evs.enforce_turnstile("forged", "1.2.3.4")

    async def test_siteverify_http_error_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TURNSTILE_SECRET_KEY", "0xsecret")
        client = MagicMock()
        client.post = AsyncMock(side_effect=evs.httpx.ReadTimeout("upstream"))
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch.object(evs.httpx, "AsyncClient", return_value=ctx):
            with pytest.raises(evs.TurnstileValidationError, match="temporarily unavailable"):
                await evs.enforce_turnstile("tok", None)

    async def test_siteverify_non_json_response_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TURNSTILE_SECRET_KEY", "0xsecret")
        resp = _resp({"unexpected": "shape"})
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=MagicMock(post=AsyncMock(return_value=resp)))
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch.object(evs.httpx, "AsyncClient", return_value=ctx):
            with pytest.raises(evs.TurnstileValidationError):
                await evs.verify_turnstile_token("tok", None)


# ---------------------------------------------------------------------------
# Email link construction — never leaks into logs
# ---------------------------------------------------------------------------


class TestVerificationLink:
    def test_link_points_at_verify_page(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FRONTEND_URL", "https://docbot.example")
        assert evs.verification_link("raw") == "https://docbot.example/verify?token=raw"

    def test_token_hash_never_reverses_to_raw(self) -> None:
        raw = evs.generate_token()
        assert raw not in hashlib.sha256(raw.encode()).hexdigest()
