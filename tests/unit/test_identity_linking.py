"""DOCBOT-1617 — provider identity linking, verified-email only (SEC-09).

In-memory SQLite (aiosqlite) for the identity model; httpx is fully mocked
for the OAuth exchanges. No network, no live PostgreSQL.
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
    func,
    insert,
    select,
)
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api import oauth_service
from api.identity_service import (
    EmailAlreadyRegistered,
    IdentityLinkError,
    ProviderIdentity,
    backfill_legacy_identities,
    identity_from_oauth_attrs,
    identity_from_saml_attrs,
    register_password_user,
    register_user_identities_table,
    resolve_provider_login,
)
from api.org_service import register_organizations_table


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
        Column("role", String, server_default="analyst", nullable=False),
        Column("last_login_at", DateTime(timezone=True)),
        Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
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
    return md, {"organizations": orgs, "users": users, "sessions": sessions, "identities": identities}


@pytest.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    md, tables = _schema()
    async with engine.begin() as conn:
        await conn.run_sync(md.create_all)
    yield engine, async_sessionmaker(engine, expire_on_commit=False), tables
    await engine.dispose()


async def _login(factory: Any, tables: dict[str, Table], identity: ProviderIdentity) -> str:
    return await resolve_provider_login(
        identity, tables["users"], tables["identities"], tables["sessions"], factory,
    )


async def _rows(factory: Any, table: Table) -> list[Any]:
    async with factory() as s:
        return (await s.execute(select(table))).fetchall()


async def _user(factory: Any, tables: dict[str, Table], uid: str) -> Any:
    async with factory() as s:
        users = tables["users"]
        return (await s.execute(select(users).where(users.c.id == uid))).fetchone()


async def _seed_user(
    factory: Any, tables: dict[str, Table], email: str, *,
    verified: bool, provider: str = "email", password_hash: str | None = "hash",
) -> str:
    async with factory() as s:
        async with s.begin():
            await s.execute(insert(tables["users"]).values(
                id="victim", email=email, name="Victim", provider=provider,
                password_hash=password_hash, email_verified=verified,
            ))
            await s.execute(insert(tables["sessions"]).values(
                id="s1", user_id="victim", token="victim-session-token",
            ))
    return "victim"


def _google(sub: str, email: str, verified: bool) -> ProviderIdentity:
    return ProviderIdentity(
        provider="google", provider_sub=sub, email=email, email_verified=verified, name="G",
    )


def _saml(name_id: str, email: str) -> ProviderIdentity:
    return identity_from_saml_attrs(
        {"provider": "okta", "name_id": name_id, "email": email, "name": "S"},
        idp_entity_id="https://idp.example.com",
    )


# ---------------------------------------------------------------------------
# Subject-first resolution
# ---------------------------------------------------------------------------


class TestSubjectLookup:
    async def test_same_sub_relogin_returns_same_user(self, db):
        _, factory, tables = db
        first = await _login(factory, tables, _google("g-1", "a@x.com", True))
        again = await _login(factory, tables, _google("g-1", "a@x.com", True))
        assert first == again
        assert len(await _rows(factory, tables["users"])) == 1
        assert len(await _rows(factory, tables["identities"])) == 1

    async def test_same_sub_relogin_ignores_changed_or_unverified_email(self, db):
        """Once linked, the subject is the key — a changed email doesn't matter."""
        _, factory, tables = db
        uid = await _login(factory, tables, _google("g-1", "a@x.com", True))
        again = await _login(factory, tables, _google("g-1", "renamed@x.com", False))
        assert again == uid

    async def test_new_verified_user_gets_identity_and_verified_account(self, db):
        _, factory, tables = db
        uid = await _login(factory, tables, _google("g-1", "A@X.com ", True))
        user = await _user(factory, tables, uid)
        assert user.email == "a@x.com" and user.email_verified and user.org_id
        (ident,) = await _rows(factory, tables["identities"])
        assert (ident.user_id, ident.provider, ident.provider_sub) == (uid, "google", "g-1")

    async def test_identities_unique_per_provider_and_sub(self, db):
        _, factory, tables = db
        uid = await _login(factory, tables, _google("g-1", "a@x.com", True))
        async with factory() as s:
            with pytest.raises(Exception):
                async with s.begin():
                    await s.execute(insert(tables["identities"]).values(
                        id="dup", user_id=uid, provider="google", provider_sub="g-1",
                    ))


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------


class TestGoogleLinking:
    async def test_unverified_google_email_cannot_link(self, db):
        _, factory, tables = db
        victim = await _seed_user(factory, tables, "victim@x.com", verified=True)
        with pytest.raises(IdentityLinkError) as err:
            await _login(factory, tables, _google("attacker", "victim@x.com", False))
        assert str(err.value) == "Sign-in failed."  # generic, no account oracle
        assert await _rows(factory, tables["identities"]) == []
        assert (await _user(factory, tables, victim)).password_hash == "hash"

    async def test_unverified_google_email_cannot_create_account(self, db):
        _, factory, tables = db
        with pytest.raises(IdentityLinkError):
            await _login(factory, tables, _google("g-1", "new@x.com", False))
        assert await _rows(factory, tables["users"]) == []
        assert await _rows(factory, tables["organizations"]) == []

    async def test_verified_google_links_to_verified_account(self, db):
        _, factory, tables = db
        victim = await _seed_user(factory, tables, "owner@x.com", verified=True)
        uid = await _login(factory, tables, _google("g-1", "owner@x.com", True))
        assert uid == victim
        user = await _user(factory, tables, victim)
        assert user.password_hash == "hash"  # verified account is left intact
        assert len(await _rows(factory, tables["sessions"])) == 1

    async def test_verified_google_on_unverified_account_revokes_prehijack(self, db):
        """Someone registered the address with a password first: the verified
        owner gets the account, the first registrant loses password + sessions."""
        _, factory, tables = db
        victim = await _seed_user(factory, tables, "owner@x.com", verified=False)
        uid = await _login(factory, tables, _google("g-1", "owner@x.com", True))
        assert uid == victim
        user = await _user(factory, tables, victim)
        assert user.password_hash is None and user.email_verified
        assert user.provider == "google"
        assert await _rows(factory, tables["sessions"]) == []

    def test_google_exchange_reports_email_verified(self):
        info = {"sub": "g-1", "email": "A@x.com", "name": "A", "email_verified": True}
        attrs = _run_google_exchange(info)
        assert attrs["email_verified"] is True and attrs["provider_id"] == "g-1"

    @pytest.mark.parametrize("claim", [False, None, "false", "yes", 1])
    def test_google_exchange_non_true_claim_is_unverified(self, claim):
        info = {"sub": "g-1", "email": "a@x.com", "name": "A"}
        if claim is not None:
            info["email_verified"] = claim
        attrs = _run_google_exchange(info)
        assert attrs["email_verified"] is False
        assert identity_from_oauth_attrs(attrs).email_verified is False

    def test_google_exchange_string_true_claim(self):
        info = {"sub": "g-1", "email": "a@x.com", "email_verified": "true"}
        assert _run_google_exchange(info)["email_verified"] is True


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------


class TestGitHub:
    def test_uses_primary_verified_email_not_public_profile_email(self):
        user = {"id": 42, "login": "octo", "email": "victim@x.com"}  # public, unverifiable
        emails = [
            {"email": "victim@x.com", "primary": False, "verified": False},
            {"email": "Octo@Real.com", "primary": True, "verified": True},
        ]
        attrs = _run_github_exchange(user, emails)
        assert attrs["email"] == "octo@real.com"
        assert attrs["email_verified"] is True
        assert attrs["provider_id"] == "42"

    def test_unverified_primary_email_rejected(self):
        user = {"id": 42, "login": "octo", "email": "victim@x.com"}
        emails = [{"email": "victim@x.com", "primary": True, "verified": False}]
        with pytest.raises(ValueError, match="verified"):
            _run_github_exchange(user, emails)

    def test_verified_but_not_primary_rejected(self):
        emails = [
            {"email": "p@x.com", "primary": True, "verified": False},
            {"email": "s@x.com", "primary": False, "verified": True},
        ]
        with pytest.raises(ValueError):
            _run_github_exchange({"id": 1, "login": "o"}, emails)

    @pytest.mark.parametrize("payload", [None, {}, "x", [{"primary": "true", "verified": "true", "email": "a@x"}]])
    def test_select_verified_email_rejects_malformed(self, payload):
        assert oauth_service.select_github_verified_email(payload) is None

    async def test_github_verified_identity_links_and_relogs_by_sub(self, db):
        _, factory, tables = db
        victim = await _seed_user(factory, tables, "octo@real.com", verified=True)
        ident = identity_from_oauth_attrs({
            "provider": "github", "provider_id": "42", "email": "octo@real.com",
            "email_verified": True, "name": "Octo",
        })
        assert await _login(factory, tables, ident) == victim
        assert await _login(factory, tables, ident) == victim
        assert len(await _rows(factory, tables["identities"])) == 1

    async def test_github_identity_without_verified_flag_cannot_link(self, db):
        _, factory, tables = db
        await _seed_user(factory, tables, "victim@x.com", verified=True)
        ident = identity_from_oauth_attrs({
            "provider": "github", "provider_id": "666", "email": "victim@x.com", "name": "Evil",
        })
        with pytest.raises(IdentityLinkError):
            await _login(factory, tables, ident)


# ---------------------------------------------------------------------------
# SAML
# ---------------------------------------------------------------------------


class TestSaml:
    async def test_saml_can_claim_existing_verified_account(self, db):
        _, factory, tables = db
        victim = await _seed_user(factory, tables, "emp@corp.com", verified=True, password_hash=None)
        assert await _login(factory, tables, _saml("emp-1", "emp@corp.com")) == victim
        (ident,) = await _rows(factory, tables["identities"])
        assert ident.provider_sub == "https://idp.example.com|emp-1"

    async def test_saml_cannot_claim_existing_unverified_account(self, db):
        _, factory, tables = db
        victim = await _seed_user(factory, tables, "emp@corp.com", verified=False)
        with pytest.raises(IdentityLinkError) as err:
            await _login(factory, tables, _saml("emp-1", "emp@corp.com"))
        assert str(err.value) == "Sign-in failed."
        assert await _rows(factory, tables["identities"]) == []
        user = await _user(factory, tables, victim)
        assert user.password_hash == "hash" and not user.email_verified
        assert len(await _rows(factory, tables["sessions"])) == 1

    async def test_saml_new_user_is_provisioned_and_relogs_by_nameid(self, db):
        _, factory, tables = db
        uid = await _login(factory, tables, _saml("emp-1", "new@corp.com"))
        assert (await _user(factory, tables, uid)).email_verified
        assert await _login(factory, tables, _saml("emp-1", "new@corp.com")) == uid

    def test_saml_without_nameid_is_rejected(self):
        with pytest.raises(ValueError):
            identity_from_saml_attrs(
                {"provider": "saml", "name_id": "", "email": "a@corp.com", "name": "A"},
                idp_entity_id="https://idp",
            )

    def test_saml_subject_scoped_to_issuer(self):
        a = identity_from_saml_attrs({"provider": "saml", "name_id": "u1", "email": "a@c.com"}, "idp-a")
        b = identity_from_saml_attrs({"provider": "saml", "name_id": "u1", "email": "a@c.com"}, "idp-b")
        assert a.provider_sub != b.provider_sub


# ---------------------------------------------------------------------------
# Email/password signup
# ---------------------------------------------------------------------------


class TestPasswordSignup:
    async def test_signup_cannot_take_over_oauth_account(self, db):
        _, factory, tables = db
        uid = await _login(factory, tables, _google("g-1", "owner@x.com", True))
        with pytest.raises(EmailAlreadyRegistered):
            await register_password_user("Owner@X.com", "Evil", "evilhash", tables["users"], factory)
        user = await _user(factory, tables, uid)
        assert user.password_hash is None and user.provider == "google"
        assert len(await _rows(factory, tables["users"])) == 1

    async def test_signup_creates_unverified_user(self, db):
        _, factory, tables = db
        uid = await register_password_user("n@x.com", "N", "h", tables["users"], factory)
        user = await _user(factory, tables, uid)
        assert user.provider == "email" and user.password_hash == "h"
        assert not user.email_verified and user.org_id


# ---------------------------------------------------------------------------
# Legacy backfill
# ---------------------------------------------------------------------------


class TestBackfill:
    async def test_backfill_is_idempotent_and_preserves_sub_login(self, db):
        engine, factory, tables = db
        async with factory() as s:
            async with s.begin():
                await s.execute(insert(tables["users"]), [
                    {"id": "u-g", "email": "g@x.com", "name": "G", "provider": "google",
                     "provider_id": "g-9", "password_hash": None},
                    {"id": "u-s", "email": "s@x.com", "name": "S", "provider": "okta",
                     "provider_id": None, "password_hash": None},
                    {"id": "u-e", "email": "e@x.com", "name": "E", "provider": "email",
                     "provider_id": None, "password_hash": "h"},
                ])
        for _ in range(2):
            async with engine.begin() as conn:
                await backfill_legacy_identities(conn)
        idents = await _rows(factory, tables["identities"])
        assert [(i.user_id, i.provider_sub, i.email_verified) for i in idents] == [("u-g", "g-9", False)]
        assert await _login(factory, tables, _google("g-9", "g@x.com", False)) == "u-g"
        assert (await _user(factory, tables, "u-s")).email_verified
        assert not (await _user(factory, tables, "u-e")).email_verified


# ---------------------------------------------------------------------------
# httpx mocks
# ---------------------------------------------------------------------------


def _resp(payload: Any) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _mock_client(post_payload: Any, get_payloads: dict[str, Any]) -> MagicMock:
    client = MagicMock()
    client.post = AsyncMock(return_value=_resp(post_payload))

    async def _get(url: str, **_: Any) -> MagicMock:
        return _resp(get_payloads[url])

    client.get = AsyncMock(side_effect=_get)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _run(coro: Any) -> Any:
    import asyncio
    return asyncio.run(coro)


def _run_github_exchange(user: dict, emails: Any) -> dict:
    ctx = _mock_client(
        {"access_token": "test-token"},
        {oauth_service.GITHUB_USER_URL: user, oauth_service.GITHUB_EMAILS_URL: emails},
    )
    with patch.object(oauth_service.httpx, "AsyncClient", return_value=ctx):
        return _run(oauth_service.github_exchange_code("code"))


def _run_google_exchange(info: dict) -> dict:
    ctx = _mock_client({"access_token": "test-token"}, {oauth_service.GOOGLE_USERINFO_URL: info})
    with patch.object(oauth_service.httpx, "AsyncClient", return_value=ctx):
        return _run(oauth_service.google_exchange_code("code", "verifier"))
