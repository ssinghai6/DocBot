"""Developer API key management — DOCBOT-1530.

DocBot's only auth mechanism so far is the session-cookie path (SAML SSO /
OAuth / email+password, enforced via ``api/rbac_service.py``'s
``require_role()``). That path is fine for the Next.js frontend but unusable
for a programmatic caller — there is no browser to hold a cookie.

This module adds a second, parallel auth mechanism: long-lived API keys,
scoped to an organization from creation (DOCBOT-1524's ``organizations``
table now exists, so there is no excuse to ship an unscoped key that would
need a retrofit later — see CLAUDE.md's DOCBOT-1530 ticket and the research
that preceded it).

Design mirrors the established ``register_<x>_table`` / ``wire_<x>_store``
pattern used throughout the service layer (``trace_service.py``,
``eval_service.py``, ``org_service.py``, ``lineage_service.py``):

  * ``register_api_keys_table(metadata)`` — defines the table once at
    import time from ``api/index.py``.
  * ``wire_api_key_store(table, async_session_factory)`` — injects the live
    table + session factory from the FastAPI lifespan.
  * ``create_api_key`` / ``revoke_api_key`` / ``validate_api_key`` /
    ``list_api_keys`` — the CRUD surface.
  * ``require_api_key(min_scope)`` — a FastAPI dependency factory, parallel
    to (never replacing) ``require_role()``. Reads ``Authorization: Bearer
    <key>``, validates the key, checks scope, and attaches ``org_id`` to
    ``request.state`` so downstream handlers scope queries exactly like the
    session-cookie path does post-DOCBOT-1524 (see
    ``api/rbac_service.py``'s ``get_user_org_id()``).

Security notes
---------------
The raw key is never stored. ``create_api_key`` returns it exactly once (the
standard API-key UX — think Stripe/GitHub PAT creation) and only its SHA-256
hash is persisted. ``key_prefix`` (the first several characters, e.g.
``dk_live_a1b2cd34``) is stored in the clear purely so the management UI can
show *which* key a row refers to without ever being able to show the whole
thing again. Never log the raw key; `logger.info` below only ever logs the
key id / prefix.

Scopes
------
A small, fixed scope set mapping onto the existing RBAC role hierarchy
(viewer < analyst < admin) rather than a granular per-route permission
system:

    hybrid:read         — viewer-equivalent. Read/query the hybrid chat pipeline.
    lineage:read        — viewer-equivalent. Read persisted answer lineage.
    autopilot:run       — analyst-equivalent. Trigger multi-step Autopilot runs.
    connectors:manage   — analyst-equivalent. List/register/sync marketplace connectors.

``require_api_key(min_scope)`` checks that the key was issued exactly that
scope (simple set membership — no role-hierarchy inference), matching the
"don't over-engineer a granular permission system" guidance. A key can hold
any subset of ``API_KEY_SCOPES``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy import Column, DateTime, String, Table, Text, select
from sqlalchemy import insert as sa_insert
from sqlalchemy import update as sa_update

from api.connectors.rate_limiter import RateLimiter
from api.rbac_service import UserRole

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scope set — mirrors the RBAC role hierarchy without building a second,
# granular permission system.
# ---------------------------------------------------------------------------

API_KEY_SCOPES: dict[str, UserRole] = {
    "hybrid:read": UserRole.viewer,
    "lineage:read": UserRole.viewer,
    "autopilot:run": UserRole.analyst,
    "connectors:manage": UserRole.analyst,
}

KEY_PREFIX_LEN = 16  # chars of the raw key shown in the management UI, e.g. "dk_live_a1b2c3d4"
_RAW_KEY_BYTES = 32  # secrets.token_urlsafe(32) -> 43 url-safe chars of entropy


# ---------------------------------------------------------------------------
# Table definition
# ---------------------------------------------------------------------------


def register_api_keys_table(metadata: Any) -> Table:
    """Define the `api_keys` table on shared metadata.

    Called once at import time from `api/index.py`, mirroring every other
    `register_*_table` helper.
    """
    return Table(
        "api_keys",
        metadata,
        Column("id", String, primary_key=True),
        Column("org_id", String, nullable=False, index=True),
        Column("key_hash", String, nullable=False, unique=True, index=True),
        Column("key_prefix", String, nullable=False),
        Column("name", String, nullable=False),
        Column("scopes", Text, nullable=False),  # JSON array of scope strings
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("last_used_at", DateTime(timezone=True), nullable=True),
        Column("revoked_at", DateTime(timezone=True), nullable=True),
    )


# ---------------------------------------------------------------------------
# Module-level references — set by wire_api_key_store()
# ---------------------------------------------------------------------------

_api_keys_table: Optional[Table] = None
_async_session_factory: Any = None
_audit_log_table: Any = None

# Per-key-id rate limiters for /v1/* routes (DOCBOT-1530 reuses the existing
# connector token-bucket primitive rather than building a new limiter).
_rate_limiters: dict[str, RateLimiter] = {}
V1_RATE_LIMIT_CALLS_PER_SECOND = 5.0


def wire_api_key_store(table: Table, async_session_factory: Any, audit_log_table: Any = None) -> None:
    """Inject table + session references. Called once from index.py lifespan.

    ``audit_log_table`` is optional so existing unit tests that only need
    key CRUD (not the `require_api_key` dependency's audit side-effect) can
    wire this module without also standing up the audit_log table.
    """
    global _api_keys_table, _async_session_factory, _audit_log_table
    _api_keys_table = table
    _async_session_factory = async_session_factory
    _audit_log_table = audit_log_table


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class APIKeyRecord:
    """An issued API key's metadata. Never carries the raw key or its hash."""

    id: str
    org_id: str
    key_prefix: str
    name: str
    scopes: list[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None

    @property
    def is_revoked(self) -> bool:
        return self.revoked_at is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "org_id": self.org_id,
            "key_prefix": self.key_prefix,
            "name": self.name,
            "scopes": self.scopes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
        }


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _row_to_record(row: Any) -> APIKeyRecord:
    try:
        scopes = json.loads(row.scopes) if row.scopes else []
    except (TypeError, ValueError):
        scopes = []
    return APIKeyRecord(
        id=row.id,
        org_id=row.org_id,
        key_prefix=row.key_prefix,
        name=row.name,
        scopes=scopes,
        created_at=row.created_at,
        last_used_at=row.last_used_at,
        revoked_at=row.revoked_at,
    )


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def _require_wired() -> None:
    if _api_keys_table is None or _async_session_factory is None:
        raise RuntimeError(
            "api_key_service not wired — call wire_api_key_store() from the app lifespan first."
        )


async def create_api_key(
    org_id: str,
    scopes: list[str],
    name: str,
) -> tuple[APIKeyRecord, str]:
    """Issue a new API key for ``org_id``.

    Returns ``(record, raw_key)``. ``raw_key`` is the only time the plaintext
    key is ever available — persist it client-side immediately, it cannot be
    retrieved again. Only its SHA-256 hash is stored.

    Raises ``ValueError`` if ``scopes`` contains anything outside
    ``API_KEY_SCOPES`` or is empty.
    """
    _require_wired()
    if not scopes:
        raise ValueError("At least one scope is required.")
    invalid = [s for s in scopes if s not in API_KEY_SCOPES]
    if invalid:
        raise ValueError(f"Unknown scope(s): {invalid}. Valid scopes: {sorted(API_KEY_SCOPES)}")

    raw_key = f"dk_live_{secrets.token_urlsafe(_RAW_KEY_BYTES)}"
    key_hash = _hash_key(raw_key)
    key_prefix = raw_key[:KEY_PREFIX_LEN]
    key_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    async with _async_session_factory() as session:
        async with session.begin():
            await session.execute(
                sa_insert(_api_keys_table).values(
                    id=key_id,
                    org_id=org_id,
                    key_hash=key_hash,
                    key_prefix=key_prefix,
                    name=name,
                    scopes=json.dumps(scopes),
                    created_at=now,
                    last_used_at=None,
                    revoked_at=None,
                )
            )

    logger.info("api_key_service: issued key %s (prefix=%s, org=%s)", key_id, key_prefix, org_id)

    record = APIKeyRecord(
        id=key_id,
        org_id=org_id,
        key_prefix=key_prefix,
        name=name,
        scopes=scopes,
        created_at=now,
        last_used_at=None,
        revoked_at=None,
    )
    return record, raw_key


async def revoke_api_key(key_id: str, org_id: Optional[str] = None) -> bool:
    """Revoke a key by id. Idempotent — revoking an already-revoked key is a no-op.

    If ``org_id`` is given, the revoke is scoped to that org (a caller may
    only revoke keys belonging to their own org) and a key belonging to a
    different org is treated as not found.

    Returns True if a key was found (and newly revoked, or already revoked),
    False if no matching key exists.
    """
    _require_wired()
    conditions = [_api_keys_table.c.id == key_id]
    if org_id is not None:
        conditions.append(_api_keys_table.c.org_id == org_id)

    async with _async_session_factory() as session:
        async with session.begin():
            result = await session.execute(
                select(_api_keys_table.c.id, _api_keys_table.c.revoked_at).where(*conditions)
            )
            row = result.first()
            if row is None:
                return False
            if row.revoked_at is None:
                await session.execute(
                    sa_update(_api_keys_table)
                    .where(_api_keys_table.c.id == key_id)
                    .values(revoked_at=datetime.now(timezone.utc))
                )
    logger.info("api_key_service: revoked key %s", key_id)
    return True


async def validate_api_key(raw_key: str) -> Optional[APIKeyRecord]:
    """Hash ``raw_key``, look it up, and return its record if valid.

    Returns ``None`` if the key is unknown or revoked. On success, updates
    ``last_used_at`` (fire-and-forget is unnecessary here — this is already
    on the hot auth path, and the update is a single indexed-PK write).
    """
    _require_wired()
    key_hash = _hash_key(raw_key)

    async with _async_session_factory() as session:
        async with session.begin():
            result = await session.execute(
                select(_api_keys_table).where(_api_keys_table.c.key_hash == key_hash)
            )
            row = result.first()
            if row is None:
                return None
            if row.revoked_at is not None:
                return None

            await session.execute(
                sa_update(_api_keys_table)
                .where(_api_keys_table.c.id == row.id)
                .values(last_used_at=datetime.now(timezone.utc))
            )

    return _row_to_record(row)


async def list_api_keys(org_id: str) -> list[APIKeyRecord]:
    """List all keys (including revoked) for an org, newest first."""
    _require_wired()
    async with _async_session_factory() as session:
        result = await session.execute(
            select(_api_keys_table)
            .where(_api_keys_table.c.org_id == org_id)
            .order_by(_api_keys_table.c.created_at.desc())
        )
        rows = result.all()
    return [_row_to_record(r) for r in rows]


# ---------------------------------------------------------------------------
# FastAPI auth dependency — parallel to, never replacing, require_role()
# ---------------------------------------------------------------------------


def _get_rate_limiter(key_id: str) -> RateLimiter:
    limiter = _rate_limiters.get(key_id)
    if limiter is None:
        limiter = RateLimiter(calls_per_second=V1_RATE_LIMIT_CALLS_PER_SECOND)
        _rate_limiters[key_id] = limiter
    return limiter


def require_api_key(min_scope: str):
    """FastAPI dependency factory for the `/v1/*` developer API.

    Validates ``Authorization: Bearer <key>``, checks the key carries
    ``min_scope``, applies a per-key token-bucket rate limit (reusing
    ``api/connectors/rate_limiter.py``'s primitive), and attaches
    ``org_id``/``api_key_id`` to ``request.state`` for downstream handlers.

    Inject as: ``key = Depends(require_api_key("autopilot:run"))``

    Raises HTTP 401 for a missing/invalid/revoked key, 403 for an
    insufficiently-scoped key.
    """

    async def _dependency(
        request: Request,
        authorization: Optional[str] = Header(default=None),
    ) -> APIKeyRecord:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(
                status_code=401,
                detail="Missing or malformed Authorization header. Expected 'Bearer <api_key>'.",
            )
        raw_key = authorization.split(" ", 1)[1].strip()
        if not raw_key:
            raise HTTPException(status_code=401, detail="Empty API key.")

        record = await validate_api_key(raw_key)
        if record is None:
            raise HTTPException(status_code=401, detail="Invalid or revoked API key.")

        if min_scope not in record.scopes:
            raise HTTPException(
                status_code=403,
                detail=f"API key missing required scope: {min_scope}.",
            )

        async with _get_rate_limiter(record.id):
            pass

        request.state.org_id = record.org_id
        request.state.api_key_id = record.id

        from api.audit_service import log_event, AuditEventType, get_client_ip
        log_event(
            AuditEventType.api_key_use,
            _audit_log_table,
            _async_session_factory,
            user_id=None,
            detail=f"{request.method} {request.url.path}",
            metadata={
                "api_key_id": record.id,
                "scope": min_scope,
                "ip_address": get_client_ip(request),
            },
            org_id=record.org_id,
        )

        return record

    return _dependency
