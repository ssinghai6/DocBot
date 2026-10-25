"""Caller scope — who owns the rows a request may touch (DOCBOT-1601, SEC-02).

Small, shared authorization primitive. Other tickets (DOCBOT-1603 connection
scoping, DOCBOT-1611 session-bound routes) import exactly these names:

``CallerScope``
    Frozen dataclass: ``user_id``, ``org_id``, ``anon_token``, ``is_admin``.

``get_caller_scope(request)``  (FastAPI dependency)
    Derives the scope from the ``docbot_session`` cookie, falling back to the
    anonymous owner cookie. Raises 401 for anonymous callers when
    ``AUTH_REQUIRED=true``; in open/demo mode anonymous callers get an
    anon-scoped ``CallerScope`` instead.

``resolve_caller_scope(request, require_auth=False)``
    Same derivation without the 401, for write paths (upload, demo init,
    EDGAR ingest) that must stamp an owner but whose auth gating is owned by
    another ticket.

``scope_filter(table, scope)``
    SQLAlchemy WHERE clause restricting ``table`` to rows the caller owns.

``owner_values(scope)``
    Column values to stamp on INSERT so the row is later visible to its
    creator (and only its creator / org admin).

Ownership rules (``scope_filter``)
----------------------------------
* Logged-in non-admin: ``user_id = me`` AND (when the caller has an org)
  ``org_id = my org``.
* Logged-in admin with an org: every row in their org (``org_id = my org``),
  including unowned legacy rows backfilled into that org. Never cross-org.
  An admin without an org falls back to the non-admin rule.
* Anonymous with an owner cookie: ``anon_owner = digest(cookie)`` AND
  ``user_id IS NULL``. Tables without an ``anon_owner`` column match nothing.
* Anonymous without an owner cookie: matches nothing (``false()``).

NULL-owner rows (created before DOCBOT-1601: ``user_id`` and ``anon_owner``
both NULL) therefore match no anonymous or non-admin caller; they are only
visible to an admin of the org they were backfilled into.

Anonymous owner cookie
----------------------
``docbot_anon`` is a 256-bit random, httpOnly, SameSite=Lax cookie issued by
the HTTP middleware (``prime_anon_cookie`` / ``attach_anon_cookie``) on the
first ``/api/`` request. Only its SHA-256 digest is stored in the database
(``sessions.anon_owner``) and carried in ``CallerScope.anon_token``, so a
database leak does not hand out working owner cookies. The raw value never
appears in logs or LLM context.

Non-owners must receive 404 (not 403) so the existence of another caller's
row is not disclosed — use ``scope_filter`` in the lookup itself rather than
fetching and then comparing.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import HTTPException, Request
from sqlalchemy import and_, false
from sqlalchemy.sql.elements import ColumnElement

SESSION_COOKIE_NAME = "docbot_session"
ANON_COOKIE_NAME = "docbot_anon"
ANON_COOKIE_MAX_AGE_SECONDS = 30 * 24 * 3600

# secrets.token_urlsafe(32) -> 43 url-safe chars. Accept a bounded range so a
# trivially short attacker-chosen value can never become an owner identity.
_ANON_COOKIE_RE = re.compile(r"^[A-Za-z0-9_-]{32,128}$")
_STATE_ATTR = "docbot_anon_cookie"


@dataclass(frozen=True)
class CallerScope:
    """The ownership identity of the current request.

    Attributes:
        user_id: ``users.id`` of the logged-in caller, else None.
        org_id: the caller's organization (``users.org_id``), else None.
        anon_token: SHA-256 hex digest of the anonymous owner cookie for an
            anonymous caller, else None. Never the raw cookie value.
        is_admin: True when the logged-in caller's role is ``admin`` (org
            admin; scope stays inside ``org_id``).
    """

    user_id: Optional[str]
    org_id: Optional[str]
    anon_token: Optional[str]
    is_admin: bool = False

    @property
    def is_anonymous(self) -> bool:
        return self.user_id is None


# ---------------------------------------------------------------------------
# Anonymous owner cookie
# ---------------------------------------------------------------------------


def _valid_anon_cookie(value: Optional[str]) -> Optional[str]:
    if value and _ANON_COOKIE_RE.match(value):
        return value
    return None


def anon_token_digest(raw_cookie: str) -> str:
    """Digest stored in ``anon_owner`` columns for a raw anon cookie value."""
    return hashlib.sha256(raw_cookie.encode("utf-8")).hexdigest()


def prime_anon_cookie(request: Request) -> Optional[str]:
    """Ensure this request has an anonymous owner identity.

    Called by the HTTP middleware BEFORE the handler runs. When the browser
    has no valid ``docbot_anon`` cookie, a fresh token is generated and put
    on ``request.state`` so a handler that creates a row in this very request
    can stamp it. Returns the new token (to be set on the response via
    ``attach_anon_cookie``) or None when the caller already has one.
    """
    if _valid_anon_cookie(request.cookies.get(ANON_COOKIE_NAME)):
        return None
    token = secrets.token_urlsafe(32)
    setattr(request.state, _STATE_ATTR, token)
    return token


def attach_anon_cookie(response: Any, token: str) -> None:
    """Set the anonymous owner cookie on an outgoing response."""
    response.set_cookie(
        key=ANON_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=os.getenv("APP_BASE_URL", "").startswith("https"),
        samesite="lax",
        max_age=ANON_COOKIE_MAX_AGE_SECONDS,
        path="/",
    )


def _raw_anon_cookie(request: Request) -> Optional[str]:
    cookie = _valid_anon_cookie(request.cookies.get(ANON_COOKIE_NAME))
    if cookie:
        return cookie
    return _valid_anon_cookie(getattr(request.state, _STATE_ATTR, None))


# ---------------------------------------------------------------------------
# Scope resolution
# ---------------------------------------------------------------------------


def scope_from_user(user: Any) -> CallerScope:
    """Build a CallerScope from a ``users`` row."""
    org_id = getattr(user, "org_id", None)
    return CallerScope(
        user_id=str(user.id),
        org_id=str(org_id) if org_id else None,
        anon_token=None,
        is_admin=str(getattr(user, "role", "") or "").lower() == "admin",
    )


async def resolve_caller_scope(request: Request, *, require_auth: bool = False) -> CallerScope:
    """Derive the caller's scope from cookies.

    Args:
        request: the incoming request.
        require_auth: when True, an anonymous caller gets HTTP 401.
    """
    from api.rbac_service import resolve_session_user

    user = await resolve_session_user(request.cookies.get(SESSION_COOKIE_NAME))
    if user is not None:
        return scope_from_user(user)
    if require_auth:
        raise HTTPException(status_code=401, detail="Authentication required.")
    raw = _raw_anon_cookie(request)
    return CallerScope(
        user_id=None,
        org_id=None,
        anon_token=anon_token_digest(raw) if raw else None,
        is_admin=False,
    )


async def get_caller_scope(request: Request) -> CallerScope:
    """FastAPI dependency: the caller's scope; 401 for anonymous under AUTH_REQUIRED.

    Usage::

        @app.get("/api/thing/{id}")
        async def get_thing(id: str, scope: CallerScope = Depends(get_caller_scope)):
            row = await db.execute(select(t).where(t.c.id == id, scope_filter(t, scope)))
            if row is None: raise HTTPException(404)
    """
    from api.auth_service import is_auth_enforcement_active

    return await resolve_caller_scope(request, require_auth=is_auth_enforcement_active())


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------


def scope_filter(table: Any, scope: CallerScope) -> ColumnElement[bool]:
    """WHERE clause limiting ``table`` to rows owned by ``scope``.

    ``table`` must have ``user_id`` and ``org_id`` columns; an ``anon_owner``
    column is required for anonymous callers to match anything.
    """
    cols = table.c
    if scope.user_id is not None:
        if scope.is_admin and scope.org_id:
            return cols.org_id == scope.org_id
        if scope.org_id:
            return and_(cols.user_id == scope.user_id, cols.org_id == scope.org_id)
        return cols.user_id == scope.user_id
    if scope.anon_token and "anon_owner" in cols:
        return and_(cols.anon_owner == scope.anon_token, cols.user_id.is_(None))
    return false()


def owner_values(scope: CallerScope) -> dict[str, Optional[str]]:
    """Owner columns to stamp on INSERT (``user_id``, ``org_id``, ``anon_owner``)."""
    if scope.user_id is not None:
        return {"user_id": scope.user_id, "org_id": scope.org_id, "anon_owner": None}
    return {"user_id": None, "org_id": None, "anon_owner": scope.anon_token}


# Idempotent DDL for pre-existing databases (fresh DBs get the column from
# the Table definition). Run from init_db().
ANON_OWNER_COLUMN_MIGRATIONS: list[str] = [
    "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS anon_owner TEXT",
    "CREATE INDEX IF NOT EXISTS ix_sessions_anon_owner ON sessions (anon_owner)",
    # DOCBOT-1635: anonymous DB connections (connect / CSV / SQLite / demo).
    "ALTER TABLE db_connections ADD COLUMN IF NOT EXISTS anon_owner TEXT",
    "CREATE INDEX IF NOT EXISTS ix_db_connections_anon_owner ON db_connections (anon_owner)",
]
