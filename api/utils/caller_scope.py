"""Caller scope — who is making this request, for row-level ownership checks.

Shared interface (DOCBOT-1601 / DOCBOT-1603):

    CallerScope(user_id, org_id, anon_token, is_admin)
    get_caller_scope(request)  -> CallerScope   (FastAPI dependency)
    scope_filter(table, scope) -> SQLAlchemy boolean clause

NOTE: this is the minimal implementation shipped with DOCBOT-1603 so the
connection-scoping work does not block on DOCBOT-1601. If both land, the
DOCBOT-1601 version of this module wins on conflict — the interface above
is identical by agreement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import Request
from sqlalchemy import Table, and_, false, or_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.elements import ColumnElement

logger = logging.getLogger(__name__)

ANON_COOKIE_NAME = "docbot_anon"
SESSION_COOKIE_NAME = "docbot_session"


@dataclass(frozen=True)
class CallerScope:
    """Identity of the caller for ownership predicates.

    - ``user_id``    logged-in user (session cookie), else None.
    - ``org_id``     caller's org (user's org, or an API key's org), else None.
    - ``anon_token`` opaque anonymous-browser token, else None.
    - ``is_admin``   org admin — may see every row in ``org_id``.
    """

    user_id: Optional[str] = None
    org_id: Optional[str] = None
    anon_token: Optional[str] = None
    is_admin: bool = False

    @property
    def is_anonymous(self) -> bool:
        return self.user_id is None and self.org_id is None


async def get_caller_scope(request: Request) -> CallerScope:
    """FastAPI dependency: resolve the caller's scope from the request.

    Resolves the session cookie to a user row independently of
    AUTH_REQUIRED — ownership must hold in open/demo mode too. Never raises:
    an invalid/expired session simply yields an anonymous scope.
    """
    anon_token = request.cookies.get(ANON_COOKIE_NAME) or None
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return CallerScope(anon_token=anon_token)

    from api import rbac_service
    from api.auth_service import get_user_from_session

    if rbac_service._async_session_factory is None:
        return CallerScope(anon_token=anon_token)
    try:
        user = await get_user_from_session(
            token,
            rbac_service._user_sessions_table,
            rbac_service._users_table,
            rbac_service._async_session_factory,
        )
    except SQLAlchemyError as exc:
        logger.warning("get_caller_scope: session lookup failed (%s)", type(exc).__name__)
        return CallerScope(anon_token=anon_token)
    if user is None:
        return CallerScope(anon_token=anon_token)
    return scope_from_user(user, anon_token=anon_token)


def scope_from_user(user: Any, anon_token: Optional[str] = None) -> CallerScope:
    """Build a scope from a ``users`` row."""
    return CallerScope(
        user_id=str(user.id),
        org_id=getattr(user, "org_id", None),
        anon_token=anon_token,
        is_admin=str(getattr(user, "role", "") or "").lower() == "admin",
    )


def scope_from_api_key(key: Any) -> CallerScope:
    """Build a scope for a `/v1/*` API-key caller: org-only, no user."""
    return CallerScope(org_id=getattr(key, "org_id", None))


def scope_filter(table: Table, scope: CallerScope) -> ColumnElement[bool]:
    """Strict "rows this caller owns" predicate for a table.

    Uses the table's ``user_id`` / ``org_id`` columns when present:
      - org admin            → ``org_id = scope.org_id``
      - logged-in user       → ``user_id = scope.user_id`` (and, when the
                               caller has an org, the row's org is that org
                               or unset)
      - org-only (API key)   → ``org_id = scope.org_id``
      - anonymous            → only unowned rows (``user_id IS NULL AND
                               org_id IS NULL``)
    A table with neither column matches nothing.
    """
    cols = table.c
    has_user = "user_id" in cols
    has_org = "org_id" in cols

    if scope.is_admin and scope.org_id and has_org:
        return cols.org_id == scope.org_id

    if scope.user_id and has_user:
        clause = cols.user_id == scope.user_id
        if scope.org_id and has_org:
            clause = and_(clause, or_(cols.org_id == scope.org_id, cols.org_id.is_(None)))
        return clause

    if scope.org_id and not scope.user_id:
        return cols.org_id == scope.org_id if has_org else false()

    parts = []
    if has_user:
        parts.append(cols.user_id.is_(None))
    if has_org:
        parts.append(cols.org_id.is_(None))
    return and_(*parts) if parts else false()
