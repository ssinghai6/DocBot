"""Provider identity linking — DOCBOT-1617 (closes SEC-09).

SEC-09: every login path used to resolve the user by email alone
(`jit_provision_user`). Anyone able to make *any* provider assert a victim's
email — an unverified Google address, a GitHub account with an unverified
secondary email, a mis-configured SAML IdP — was logged straight into the
victim's account.

This module replaces that with an explicit identity table:

    user_identities(id, user_id → users.id, provider, provider_sub,
                    email, email_verified, created_at)
    UNIQUE(provider, provider_sub)

Resolution order for a provider login (`resolve_provider_login`):

1. Look up ``(provider, provider_sub)``. Found → that user, done. The email
   the provider sends today is irrelevant: the stable subject id is the key.
2. Not found, but a user with the same email exists → link ONLY if
     * OAuth (google/github): the provider asserts the email is verified;
     * SAML: the *existing account's* email is verified.
   Otherwise reject with a generic `IdentityLinkError` (no detail about
   whether the account exists).
   When a verified OAuth identity links to an account whose own email was
   never verified (e.g. an email/password signup — there is no verification
   flow), the account is treated as possibly pre-hijacked: its password is
   cleared and its sessions revoked, so whoever registered the address
   first cannot keep a foothold in the verified owner's account.
3. No user with that email → create one (personal org + user + identity),
   but only for a verified OAuth email or a SAML assertion. An OAuth login
   with an unverified email never creates or claims anything.

Email/password signup (`register_password_user`) is insert-only: it can
never return, update or attach to an existing user, OAuth or otherwise.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    String,
    Table,
    UniqueConstraint,
    delete,
    func,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.exc import IntegrityError

from api.auth_service import insert_new_user
from api.utils.email_canonical import canonical_email, canonical_is_same_mailbox

logger = logging.getLogger(__name__)

OAUTH_PROVIDERS: frozenset[str] = frozenset({"google", "github"})
SAML_PROVIDERS: frozenset[str] = frozenset({"saml", "okta", "azure_ad"})

GENERIC_SIGNIN_ERROR = "Sign-in failed."


class IdentityLinkError(ValueError):
    """Login refused. The message is deliberately generic (no account oracle)."""

    def __init__(self, reason: str = "") -> None:
        super().__init__(GENERIC_SIGNIN_ERROR)
        # Internal-only reason for logs; never surfaced to the client.
        self.reason = reason


class EmailAlreadyRegistered(ValueError):
    """Email/password signup hit an email that already has an account."""


class ProviderIdentity(BaseModel):
    """What a provider asserted about the person signing in."""

    provider: str = Field(min_length=1)
    provider_sub: str = Field(min_length=1)
    email: str = Field(min_length=3)
    email_verified: bool = False
    name: str = Field(min_length=1)

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: str) -> str:
        return value.lower().strip()

    @field_validator("provider_sub")
    @classmethod
    def _strip_sub(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("provider_sub must not be blank")
        return stripped

    @property
    def is_saml(self) -> bool:
        return self.provider in SAML_PROVIDERS


def identity_from_oauth_attrs(attrs: dict) -> ProviderIdentity:
    """Build a ProviderIdentity from github/google_exchange_code() output."""
    return ProviderIdentity(
        provider=attrs["provider"],
        provider_sub=str(attrs.get("provider_id") or ""),
        email=attrs["email"],
        email_verified=attrs.get("email_verified") is True,
        name=attrs.get("name") or attrs["email"].split("@")[0],
    )


def identity_from_saml_attrs(attrs: dict, idp_entity_id: Optional[str] = None) -> ProviderIdentity:
    """Build a ProviderIdentity from process_acs() output.

    The subject is the NameID scoped to the IdP entity id, so swapping the
    configured IdP can never make a new IdP's NameID collide with an old one.
    SAML ``email_verified`` is left False: whether an existing account can be
    claimed depends on that account's own verified flag, not the assertion.
    """
    issuer = idp_entity_id if idp_entity_id is not None else os.getenv("SAML_IDP_ENTITY_ID", "")
    name_id = str(attrs.get("name_id") or "").strip()
    return ProviderIdentity(
        provider=attrs["provider"],
        provider_sub=f"{issuer}|{name_id}" if name_id else "",
        email=attrs["email"],
        email_verified=False,
        name=attrs.get("name") or attrs["email"].split("@")[0],
    )


# ---------------------------------------------------------------------------
# Schema + idempotent migrations
# ---------------------------------------------------------------------------


def register_user_identities_table(metadata: Any) -> Table:
    """Define `user_identities` on the shared metadata (created by init_db)."""
    return Table(
        "user_identities",
        metadata,
        Column("id", String, primary_key=True),
        Column(
            "user_id",
            String,
            ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        Column("provider", String, nullable=False),
        Column("provider_sub", String, nullable=False),
        Column("email", String),
        Column("email_verified", Boolean, nullable=False, server_default="false"),
        Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
        UniqueConstraint("provider", "provider_sub", name="uq_user_identities_provider_sub"),
    )


# Same ALTER ... ADD COLUMN IF NOT EXISTS idiom as the rest of init_db():
# a no-op on a fresh DB (the Table definition already declares the column).
USER_IDENTITY_COLUMN_MIGRATIONS: list[str] = [
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified BOOLEAN NOT NULL DEFAULT false",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS email_canonical TEXT",  # DOCBOT-2332
]

# DOCBOT-2332: created after `backfill_email_canonical`, so legacy collisions
# (left NULL there) can never fail boot. NULLs are distinct in a UNIQUE index.
EMAIL_CANONICAL_INDEX_DDL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_users_email_canonical ON users (email_canonical)"
)


async def backfill_email_canonical(conn: Any) -> int:
    """Idempotently fill ``users.email_canonical`` for rows that lack it.

    Oldest account wins a canonical; a later legacy account whose canonical is
    already taken (e.g. ``a.b@gmail.com`` next to ``ab@gmail.com``) keeps NULL
    and stays usable — it just doesn't block anyone. Returns that skip count.
    """
    taken = {
        row[0] for row in (await conn.execute(text(
            "SELECT email_canonical FROM users WHERE email_canonical IS NOT NULL"
        ))).fetchall()
    }
    pending = (await conn.execute(text(
        "SELECT id, email FROM users WHERE email_canonical IS NULL ORDER BY created_at, id"
    ))).fetchall()
    collisions = 0
    for user_id, email in pending:
        canon = canonical_email(email)
        if canon in taken:
            collisions += 1
            continue
        taken.add(canon)
        await conn.execute(
            text("UPDATE users SET email_canonical = :c WHERE id = :id"),
            {"c": canon, "id": user_id},
        )
    if collisions:
        logger.warning(
            "identity: %d legacy user(s) share a canonical email with an older account; "
            "left email_canonical NULL", collisions,
        )
    return collisions


def _email_match(users_table: Any, email: str) -> Any:
    """Exact email, or the same canonical mailbox when the column exists."""
    cond = users_table.c.email == email
    if "email_canonical" in users_table.c:
        cond = cond | (users_table.c.email_canonical == canonical_email(email))
    return cond


async def backfill_legacy_identities(conn: Any) -> None:
    """Idempotently carry pre-1617 accounts over to the identity model.

    * GitHub/Google users that already stored a ``provider_id`` get a
      ``user_identities`` row, so they keep signing in by subject id. Their
      email is NOT marked verified (Google's flag was never checked before).
    * SAML-provisioned users without a password had their email asserted by
      the configured IdP, so their account email is marked verified —
      otherwise every existing SSO user would be locked out on next login.
    """
    await conn.execute(text(
        "INSERT INTO user_identities (id, user_id, provider, provider_sub, email, email_verified) "
        "SELECT 'legacy:' || id, id, provider, provider_id, email, false FROM users "
        "WHERE provider IN ('github', 'google') "
        "AND provider_id IS NOT NULL AND provider_id <> '' "
        "ON CONFLICT DO NOTHING"
    ))
    await conn.execute(text(
        "UPDATE users SET email_verified = true "
        "WHERE provider IN ('saml', 'okta', 'azure_ad') "
        "AND password_hash IS NULL AND email_verified = false"
    ))


# ---------------------------------------------------------------------------
# Provider login
# ---------------------------------------------------------------------------


async def resolve_provider_login(
    identity: ProviderIdentity,
    users_table: Any,
    identities_table: Any,
    user_sessions_table: Any,
    async_session_factory: Any,
    api_keys_table: Any = None,
) -> str:
    """Return the user_id for a provider login, linking/creating per policy.

    Raises `IdentityLinkError` (generic message) when the login is refused.
    """
    now = datetime.now(timezone.utc)
    try:
        async with async_session_factory() as session:
            async with session.begin():
                user_id = await _find_by_subject(session, identity, users_table, identities_table, now)
                if user_id:
                    return user_id

                # DOCBOT-2332: exact email first; link by canonical mailbox only
                # where aliases provably share one inbox (Gmail). Elsewhere a
                # canonical match must not link (takeover), so the new account
                # is created without a canonical (see insert below).
                existing = (await session.execute(
                    select(users_table).where(users_table.c.email == identity.email)
                )).fetchone()
                canonical_taken = False
                if existing is None and "email_canonical" in users_table.c:
                    alias = (await session.execute(
                        select(users_table).where(_email_match(users_table, identity.email))
                        .order_by(users_table.c.created_at).limit(1)
                    )).fetchone()
                    if alias is not None and canonical_is_same_mailbox(identity.email):
                        existing = alias
                    elif alias is not None:
                        canonical_taken = True

                if existing is not None:
                    return await _link_existing(
                        session, identity, existing, users_table,
                        identities_table, user_sessions_table, now, api_keys_table,
                    )

                if not identity.is_saml and not identity.email_verified:
                    raise IdentityLinkError("unverified provider email, no account")

                attrs = {
                    "email": identity.email,
                    "name": identity.name,
                    "provider": identity.provider,
                    "provider_id": identity.provider_sub,
                }
                user_id = await insert_new_user(
                    session, users_table, attrs, email_verified=True,
                    store_canonical=not canonical_taken,
                )
                await _insert_identity(session, identities_table, identity, user_id, verified=True)
                logger.info("identity: new user %s via %s", user_id, identity.provider)
                # DOCBOT-1706: provider-verified email, so both steps at once.
                from api.funnel_service import record_funnel_event
                for event in ("signup", "email_verified"):
                    record_funnel_event(event, user_id=user_id, props={"provider": identity.provider})
                return user_id
    except IntegrityError as exc:
        # Concurrent first login for the same subject/email. Fail closed; a
        # retry resolves via the row the winning request committed.
        raise IdentityLinkError("integrity conflict during provisioning") from exc


async def _find_by_subject(
    session: Any,
    identity: ProviderIdentity,
    users_table: Any,
    identities_table: Any,
    now: datetime,
) -> Optional[str]:
    row = (await session.execute(
        select(identities_table.c.user_id).where(
            identities_table.c.provider == identity.provider,
            identities_table.c.provider_sub == identity.provider_sub,
        )
    )).fetchone()
    if row is None:
        return None
    user = (await session.execute(
        select(users_table.c.id).where(users_table.c.id == row.user_id)
    )).fetchone()
    if user is None:
        raise IdentityLinkError("identity points at a missing user")
    await session.execute(
        update(users_table)
        .where(users_table.c.id == row.user_id)
        .values(name=identity.name, last_login_at=now)
    )
    logger.info("identity: returning user %s via %s", row.user_id, identity.provider)
    return row.user_id


async def _link_existing(
    session: Any,
    identity: ProviderIdentity,
    existing: Any,
    users_table: Any,
    identities_table: Any,
    user_sessions_table: Any,
    now: datetime,
    api_keys_table: Any = None,
) -> str:
    account_verified = bool(getattr(existing, "email_verified", False))

    if identity.is_saml:
        if not account_verified:
            raise IdentityLinkError("SAML claim of an unverified existing account")
    elif not identity.email_verified:
        raise IdentityLinkError("unverified provider email cannot link")

    await _insert_identity(
        session, identities_table, identity, existing.id,
        verified=identity.email_verified or identity.is_saml,
    )

    update_vals: dict[str, Any] = {"last_login_at": now}
    if not account_verified:
        # Verified OAuth owner claims an account nobody ever proved owned
        # the address: drop whatever credentials the first registrant set.
        update_vals.update(email_verified=True, password_hash=None, provider=identity.provider)
        await session.execute(
            delete(user_sessions_table).where(user_sessions_table.c.user_id == existing.id)
        )
        revoked = await _revoke_claimed_account_keys(session, api_keys_table, existing, now)
        logger.warning(
            "identity: verified %s identity claimed unverified account %s; "
            "password cleared, sessions revoked, %d API key(s) revoked",
            identity.provider, existing.id, revoked,
        )
    await session.execute(
        update(users_table).where(users_table.c.id == existing.id).values(**update_vals)
    )
    logger.info("identity: linked %s identity to user %s", identity.provider, existing.id)
    return existing.id


async def _revoke_claimed_account_keys(session: Any, keys: Any, existing: Any, now: datetime) -> int:
    """DOCBOT-4013: revoke every live key of the claimed account's org and any
    key the account created, so a pre-hijacker loses /v1 access too."""
    if keys is None:  # caller has no api_keys table (tests of other tables)
        return 0
    scope = keys.c.created_by == existing.id
    org_id = getattr(existing, "org_id", None)
    if org_id:
        scope = scope | (keys.c.org_id == org_id)
    result = await session.execute(
        update(keys).where(keys.c.revoked_at.is_(None), scope).values(revoked_at=now)
    )
    return result.rowcount or 0


async def _insert_identity(
    session: Any,
    identities_table: Any,
    identity: ProviderIdentity,
    user_id: str,
    verified: bool,
) -> None:
    await session.execute(
        insert(identities_table).values(
            id=str(uuid.uuid4()),
            user_id=user_id,
            provider=identity.provider,
            provider_sub=identity.provider_sub,
            email=identity.email,
            email_verified=verified,
        )
    )


# ---------------------------------------------------------------------------
# Email/password signup
# ---------------------------------------------------------------------------


async def register_password_user(
    email: str,
    name: str,
    password_hash: str,
    users_table: Any,
    async_session_factory: Any,
) -> str:
    """Create a brand-new email/password user. Never touches an existing one.

    Raises `EmailAlreadyRegistered` if any account (OAuth, SAML or password)
    already holds the email — including when a concurrent insert wins the
    race between the existence check and the INSERT.
    """
    email = email.lower().strip()
    attrs = {"email": email, "name": name, "provider": "email", "password_hash": password_hash}
    try:
        async with async_session_factory() as session:
            async with session.begin():
                taken = (await session.execute(
                    select(users_table.c.id).where(_email_match(users_table, email)).limit(1)
                )).fetchone()
                if taken is not None:
                    raise EmailAlreadyRegistered(email)
                # DOCBOT-1713: the signup route mints + emails a verification
                # link (endpoint changes this flag once proven). Until then
                # the account stays unverified — exactly what the verified-
                # --email policy above keys off.
                return await insert_new_user(session, users_table, attrs, email_verified=False)
    except IntegrityError as exc:
        raise EmailAlreadyRegistered(email) from exc
