"""Email verification (Resend) + Turnstile bot protection — DOCBOT-1713.

Email/password signup previously inserted `users.email_verified = false` with
no way to ever flip it (the comment DOCBOT-1617 left in
``identity_service.register_password_user``). This module adds two
dormant-by-default flows, selected purely by environment:

* **Email verification** — when ``RESEND_API_KEY`` AND ``EMAIL_FROM`` are set,
  ``POST /api/auth/register`` mints a single-use expiring verification token
  (SHA-256-hashed at rest; the raw token exists only in the email), sends the
  link via Resend's plain HTTP API (``api.resend.com/emails``, no SDK) and
  does NOT hand out a login session. ``POST /api/auth/verify`` consumes the
  token atomically (conditional UPDATE on ``used_at IS NULL`` — no TOCTOU)
  and flips ``users.email_verified = true`` — the exact state DOCBOT-1617
  reads for OAuth/SAML identity linking, never a parallel mechanism. While
  unverified, ``/api/auth/login`` answers 403 with a clear message and
  re-sends the link at most once per ``RESEND_COOLDOWN_SECONDS``. Tokens
  expire after ``LINK_TTL_HOURS``; the raw token is never logged.

  With those keys unset everything skips: signup keeps today's insert-only
  behavior (`email_verified = false`, immediate session cookie, no 403 gate
  — the gate evaluates ``is_email_verification_enabled()`` at request time).
  Accounts registered while dormant keep flag False exactly like today;
  once the founder adds the keys, their login path re-sends a link so they
  verify instead of being silently locked out.

* **Turnstile bot protection** — when ``TURNSTILE_SECRET_KEY`` is set, signup
  must carry a token minted by the widget rendered under
  ``NEXT_PUBLIC_TURNSTILE_SITE_KEY``; it is verified server-side against
  Cloudflare's siteverify endpoint BEFORE any user row is created. Unset →
  validation is skipped, zero behavior change.

Security notes: raw tokens are 256-bit urlsafe secrets stored only as
SHA-256 hashes and never logged; a Cloudflare outage fails signup closed
(400) rather than letting the gate lapse. The DOCBOT-1617 pre-hijack defense
is untouched: dormant-mode password accounts stay unverified, so a verified
OAuth identity claiming the address still clears the password and revokes
sessions.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx
from sqlalchemy import (
    Column,
    DateTime,
    String,
    Table,
    func,
    insert,
    select,
    update,
)

logger = logging.getLogger(__name__)

RESEND_API_URL = "https://api.resend.com/emails"
TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"

LINK_TTL_HOURS = 24
RESEND_COOLDOWN_SECONDS = 60

TOKEN_BYTES = 32
_APP_BASE_FALLBACK = "https://docbot.app"

# Public: routes re-raise this verbatim so unknown/expired/reused links get
# one identical, oracle-free message.
GENERIC_TOKEN_ERROR = "This verification link is invalid or has expired. Sign in to send a new one."
_TURNSTILE_MISSING = "Please complete the bot check to create an account."
_TURNSTILE_FAILED = "Bot check failed. Please try again."
_TURNSTILE_UNAVAILABLE = "Bot check is temporarily unavailable. Please try again."


class VerificationTokenInvalid(ValueError):
    """Raw token unknown/expired/already used. Message is generic (no oracle)."""

    def __init__(self, message: str = GENERIC_TOKEN_ERROR) -> None:
        super().__init__(message)


class TurnstileValidationError(ValueError):
    """Turnstile token missing/invalid/unverifiable — signup must fail."""


# ---------------------------------------------------------------------------
# Schema (created with the rest of init_db via metadata.create_all)
# ---------------------------------------------------------------------------


def register_email_verification_tokens_table(metadata: Any) -> Table:
    """Define `email_verification_tokens` on the shared metadata (init_db).

    Only the SHA-256 hash is stored: the raw token exists exclusively in the
    email we send. ``expires_at`` kills links after LINK_TTL_HOURS; ``used_at``
    is the single-use marker, consumed with a conditional UPDATE so two
    concurrent clicks of the same link resolve to exactly one success.
    """
    return Table(
        "email_verification_tokens",
        metadata,
        Column("token_hash", String, primary_key=True),
        Column("user_id", String, nullable=False, index=True),
        Column("email", String, nullable=False),
        Column("expires_at", DateTime(timezone=True), nullable=False),
        Column("used_at", DateTime(timezone=True)),
        Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    )


# ---------------------------------------------------------------------------
# Feature toggles (dormant by default — zero prod behavior change)
# ---------------------------------------------------------------------------


def is_email_verification_enabled() -> bool:
    """Resend + EMAIL_FROM set → live flow. Unset → skip everything."""
    return bool((os.getenv("RESEND_API_KEY") or "").strip()) and bool(
        (os.getenv("EMAIL_FROM") or "").strip()
    )


def is_turnstile_enabled() -> bool:
    return bool((os.getenv("TURNSTILE_SECRET_KEY") or "").strip())


def login_requires_verification(user: Any) -> bool:
    """Login gate for POST /api/auth/login.

    Dormant RESEND config, or any non-email account (OAuth/SAML users proved
    email ownership through their provider; DOCBOT-1617 keeps its own
    verified state for them) → False: exactly today's behavior.
    """
    if not is_email_verification_enabled():
        return False
    if getattr(user, "provider", "") != "email":
        return False
    return not bool(getattr(user, "email_verified", False))


# ---------------------------------------------------------------------------
# Token minting / consumption
# ---------------------------------------------------------------------------


def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _as_aware(dt: datetime) -> datetime:
    """SQLite (aiosqlite) parses even timezone-aware columns as naive UTC."""
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def generate_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


async def mint_verification_token(
    user_id: str,
    email: str,
    tokens_table: Any,
    async_session_factory: Any,
    *,
    ttl_hours: int = LINK_TTL_HOURS,
) -> str:
    """Store a fresh single-use token (hashed at rest); return the raw value.

    The raw value is printable exactly once — at mint time, straight into the
    outbound email. Call sites must never log it.
    """
    raw = generate_token()
    async with async_session_factory() as session:
        async with session.begin():
            await session.execute(
                insert(tokens_table).values(
                    token_hash=_hash_token(raw),
                    user_id=user_id,
                    email=email.lower().strip(),
                    expires_at=datetime.now(timezone.utc) + timedelta(hours=ttl_hours),
                )
            )
    return raw


async def consume_verification_token(
    raw_token: str,
    tokens_table: Any,
    users_table: Any,
    async_session_factory: Any,
) -> str:
    """Atomically verify a token and mark its user's email verified.

    Single-use is enforced with a conditional UPDATE guarded on ``used_at IS
    NULL`` in the same transaction that flips ``users.email_verified`` — the
    email bound at mint (``row.email``) must still equal the account's email,
    so a re-registered address cannot be verified by a stale link.

    Raises `VerificationTokenInvalid` (generic detail) for unknown, expired,
    reused, orphaned, or user/email-mismatched tokens. Returns the verified
    email.
    """
    raw = (raw_token or "").strip()
    if not raw or len(raw) > 512:
        raise VerificationTokenInvalid()

    token_hash = _hash_token(raw)
    now = datetime.now(timezone.utc)
    async with async_session_factory() as session:
        async with session.begin():
            row = (await session.execute(
                select(
                    tokens_table.c.user_id,
                    tokens_table.c.email,
                    tokens_table.c.expires_at,
                ).where(tokens_table.c.token_hash == token_hash)
            )).fetchone()
            if row is None or _as_aware(row.expires_at) < now:
                raise VerificationTokenInvalid()

            # Conditional single-use UPDATE: the "used_at IS NULL" predicate
            # makes two concurrent clicks of the same link claim exactly one.
            claimed = (await session.execute(
                update(tokens_table)
                .where(tokens_table.c.token_hash == token_hash)
                .where(tokens_table.c.used_at.is_(None))
                .values(used_at=now)
            )).rowcount
            if not claimed:
                raise VerificationTokenInvalid()

            user = (await session.execute(
                select(users_table.c.email, users_table.c.org_id).where(
                    users_table.c.id == row.user_id,
                    users_table.c.email == row.email,
                )
            )).fetchone()
            if user is None:
                raise VerificationTokenInvalid()

            await session.execute(
                update(users_table)
                .where(users_table.c.id == row.user_id)
                .values(email_verified=True)
            )
    from api.funnel_service import record_funnel_event  # DOCBOT-1706

    record_funnel_event("email_verified", user_id=row.user_id)
    await _grant_on_verify(row.user_id, user.org_id)

    logger.info("email-verification: marked email verified for user %s", row.user_id)
    return user.email


async def _grant_on_verify(user_id: str, org_id: Optional[str]) -> None:
    """DOCBOT-2332: free credits land on verify, not signup. Best-effort —
    a credit-store hiccup must not fail verification (``reserve`` retries
    the grant lazily); the ledger's UNIQUE(kind, ref) keeps it once."""
    from sqlalchemy.exc import SQLAlchemyError

    from api import credit_service

    billing_org = credit_service.billing_org_for(user_id, org_id)
    if not billing_org:
        return
    try:
        await credit_service.grant_free_credits(billing_org)
    except SQLAlchemyError as exc:
        logger.warning("email-verification: credit grant failed for user %s: %s", user_id, exc.__class__.__name__)


# ---------------------------------------------------------------------------
# Resend delivery
# ---------------------------------------------------------------------------


def _frontend_base_url() -> str:
    """Frontend base for emailed links; mirrors oauth_service's helper."""
    from api.oauth_service import _frontend_url

    return _frontend_url() or _APP_BASE_FALLBACK


def verification_link(raw_token: str) -> str:
    """The <a href> target. DOCBOT-1712: /verify POSTs ?token= to
    POST /api/auth/verify — the link alone never verifies (no GET
    auto-consume; email scanner prefetch can't flip the flag). The legacy
    /chat?verify_token= landing still works for links already sent."""
    return f"{_frontend_base_url()}/verify?token={raw_token}"


def _email_payload(email: str, raw_token: str) -> dict[str, Any]:
    link = verification_link(raw_token)
    text = (
        "Welcome to DocBot!\n\n"
        "Confirm your email address by opening this link:\n"
        f"{link}\n\n"
        f"This link expires in {LINK_TTL_HOURS} hours and can be used once.\n"
        "If you didn't create a DocBot account, you can ignore this email.\n"
    )
    html = (
        '<p style="font-family:Arial,Helvetica,sans-serif;font-size:14px;color:#111;">'
        "Welcome to DocBot!</p>"
        '<p style="font-family:Arial,Helvetica,sans-serif;font-size:14px;color:#111;">'
        "Confirm your email address:</p>"
        '<p><a href="LINK_PLACEHOLDER" style="font-family:Arial,Helvetica,sans-serif;'
        'font-size:14px;color:#06b6d4;">Verify my email</a></p>'
        '<p style="font-family:Arial,Helvetica,sans-serif;font-size:12px;color:#666;">'
        f"This link expires in {LINK_TTL_HOURS} hours and can be used once. "
        "If you didn't create a DocBot account, you can ignore this email.</p>"
    ).replace("LINK_PLACEHOLDER", link)
    return {
        "from": (os.getenv("EMAIL_FROM") or "").strip(),
        "to": [email],
        "subject": "Verify your DocBot email",
        "text": text,
        "html": html,
    }


async def send_verification_email(email: str, raw_token: str) -> None:
    """POST the verification link via Resend's plain HTTP API.

    Raises on misconfiguration or a Resend error so callers can log; retries
    are the caller's problem (signup keeps its 202 or falls back, login-path
    re-sends are throttled by `resend_verification_if_due`).
    """
    api_key = (os.getenv("RESEND_API_KEY") or "").strip()
    payload = _email_payload(email, raw_token)
    if not api_key or not payload["from"]:
        raise ValueError("Email verification is not configured (RESEND_API_KEY / EMAIL_FROM).")

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            RESEND_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
    resp.raise_for_status()


async def issue_verification_token_and_send(
    user_id: str,
    email: str,
    tokens_table: Any,
    async_session_factory: Any,
) -> bool:
    """Signup-path wrapper: mint a token, then best-effort send.

    Resend being down must not 500 the signup — the user row already exists,
    so the response can still say "check your inbox" and the login path
    re-sends when the unverified user tries to sign in. Returns whether an
    email actually went out.
    """
    try:
        raw = await mint_verification_token(user_id, email, tokens_table, async_session_factory)
        await send_verification_email(email, raw)
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("email-verification: send failed for user %s: %s", user_id, exc.__class__.__name__)
        return False
    logger.info("email-verification: link sent to user %s", user_id)
    return True


_send_tasks: set[asyncio.Task[bool]] = set()


def _log_send_failure(task: "asyncio.Task[bool]") -> None:
    _send_tasks.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.warning("email-verification: background send crashed: %s", task.exception().__class__.__name__)


def issue_verification_in_background(
    user_id: str,
    email: str,
    tokens_table: Any,
    async_session_factory: Any,
) -> None:
    """DOCBOT-2332: signup must not await Resend — otherwise a new address
    answers measurably slower than an existing one (registration oracle)."""
    task = asyncio.get_running_loop().create_task(
        issue_verification_token_and_send(user_id, email, tokens_table, async_session_factory)
    )
    _send_tasks.add(task)
    task.add_done_callback(_log_send_failure)


async def resend_verification_if_due(
    user_id: str,
    email: str,
    tokens_table: Any,
    async_session_factory: Any,
) -> None:
    """Login-path re-send for an unverified password user, per-cooldown.

    Tokens are hashed at rest, so the emailed value is unrecoverable — every
    notice needs a fresh mint. ``created_at`` of the newest row throttles how
    often we hit Resend for hammering on /api/auth/login; (DOCBOT-1622 IP
    rate limiting guards the endpoint itself).
    """
    if not is_email_verification_enabled():
        return
    async with async_session_factory() as session:
        latest = (await session.execute(
            select(tokens_table.c.created_at)
            .where(tokens_table.c.user_id == user_id)
            .order_by(tokens_table.c.created_at.desc())
            .limit(1)
        )).fetchone()
    if latest is not None and _as_aware(latest.created_at) > (
        datetime.now(timezone.utc) - timedelta(seconds=RESEND_COOLDOWN_SECONDS)
    ):
        return
    await issue_verification_token_and_send(user_id, email, tokens_table, async_session_factory)


# ---------------------------------------------------------------------------
# Turnstile bot protection (signup only)
# ---------------------------------------------------------------------------


async def verify_turnstile_token(token: str, remote_ip: Optional[str]) -> bool:
    """Server-side check against Cloudflare siteverify. Raises on failure.

    Failure covers invalid, missing, and unverifiable (network / HTTP error)
    — callers fail closed, so an outage blocks new signups but can never
    weaken the gate. The secret is never logged.
    """
    if not is_turnstile_enabled():
        raise TurnstileValidationError(_TURNSTILE_FAILED)
    secret = (os.getenv("TURNSTILE_SECRET_KEY") or "").strip()
    candidate = (token or "").strip()
    if not secret or not candidate:
        raise TurnstileValidationError(_TURNSTILE_MISSING)

    form: dict[str, Any] = {"secret": secret, "response": candidate}
    if remote_ip:
        form["remoteip"] = remote_ip

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(TURNSTILE_VERIFY_URL, data=form)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("email-verification: turnstile siteverify unavailable: %s", exc.__class__.__name__)
        raise TurnstileValidationError(_TURNSTILE_UNAVAILABLE) from exc

    if resp.json().get("success") is not True:
        logger.warning(
            "email-verification: turnstile siteverify rejected a token (ip=%s)",
            bool(remote_ip),
        )
        raise TurnstileValidationError(_TURNSTILE_FAILED)
    return True


async def enforce_turnstile(
    token: Optional[str],
    remote_ip: Optional[str] = None,
) -> None:
    """Signup-gate entrypoint for POST /api/auth/register.

    No-op (dormant) unless TURNSTILE_SECRET_KEY is set; otherwise raises
    `TurnstileValidationError`, which the route translates to HTTP 400.
    """
    if not is_turnstile_enabled():
        return
    await verify_turnstile_token(token or "", remote_ip)
