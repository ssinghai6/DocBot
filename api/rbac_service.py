"""Role-Based Access Control — DOCBOT-603.

Three roles (least to most privileged):
    viewer   — can read query results; cannot create/remove DB connections
    analyst  — can run queries and manage their own DB connections (default for JIT-provisioned users)
    admin    — full access including audit log, user management, all connections

Usage in route handlers:
    from api.rbac_service import require_role, require_admin, UserRole

    @app.get("/admin/audit-log")
    async def audit_log(user=Depends(require_admin())):
        ...

When auth enforcement is not active (AUTH_REQUIRED env var is not set to
"true"), viewer/analyst role checks pass automatically so the app remains
usable without login — ideal for demos and local development.

Exceptions that are enforced in EVERY mode (DOCBOT-1610 / DOCBOT-1612):
    require_admin()        — admin routes always need a logged-in admin;
                             handlers scope to require_admin_org_id(user)
    require_platform_admin() — cross-org data (/admin/metrics) additionally
                             needs users.id in PLATFORM_ADMIN_USER_IDS
    require_sandbox_user() — raw code execution needs a logged-in analyst+
                             unless ALLOW_ANONYMOUS_SANDBOX=true (dev only;
                             ignored on Railway/production — DOCBOT-1607)

Startup guard: assert_safe_auth_config() refuses to boot when
MULTI_TENANT=true without AUTH_REQUIRED=true.
"""

from __future__ import annotations

import logging
import os
from enum import IntEnum
from typing import Any, Optional

from fastapi import Cookie, Depends, HTTPException

logger = logging.getLogger(__name__)


class UserRole(IntEnum):
    """Role hierarchy: higher int = more privileged."""
    viewer = 1
    analyst = 2
    admin = 3

    @classmethod
    def from_str(cls, value: str) -> "UserRole":
        try:
            return cls[value.lower()]
        except KeyError:
            return cls.analyst  # safe default


_ROLE_ORDER = [r.name for r in sorted(UserRole, key=lambda r: r.value)]


def _role_gte(user_role: str, required: UserRole) -> bool:
    """Return True if user_role is at least as privileged as required."""
    return UserRole.from_str(user_role).value >= required.value


async def _resolve_user_with_role(docbot_session: Optional[str], min_role: UserRole) -> Any:
    """Resolve the session cookie to a user row and check its role.

    Raises 401 when there is no cookie or the session is invalid/expired,
    and 403 when the user's role is below ``min_role``. Never skips — the
    open-mode decision is made by the caller.
    """
    if not docbot_session:
        raise HTTPException(status_code=401, detail="Authentication required.")

    from api.auth_service import get_user_from_session
    user = await get_user_from_session(
        docbot_session,
        _user_sessions_table,
        _users_table,
        _async_session_factory,
    )
    if not user:
        raise HTTPException(status_code=401, detail="Session expired or invalid.")

    if not _role_gte(user.role, min_role):
        raise HTTPException(
            status_code=403,
            detail=f"Insufficient permissions. Required: {min_role.name}, your role: {user.role}.",
        )

    return user


def require_role(min_role: UserRole, *, enforce_in_open_mode: bool = False):
    """FastAPI dependency factory.

    Returns a dependency that:
    - Skips enforcement when AUTH_REQUIRED is not set (open/demo mode),
      UNLESS ``enforce_in_open_mode=True`` (see ``require_admin()``)
    - Reads the session cookie and resolves the user
    - Raises HTTP 401 if not authenticated
    - Raises HTTP 403 if the user's role is below min_role

    Inject as: user = Depends(require_role(UserRole.analyst))

    DOCBOT-1524: the returned user is the full `users` table row, which
    carries `org_id` alongside `role` — route handlers that must scope a
    query to the requesting user's organization read `user.org_id`
    directly (see `get_user_org_id()` below for the None-safe accessor
    used when auth enforcement may be off).
    """
    async def _dependency(
        docbot_session: Optional[str] = Cookie(default=None),
        # Tables and factory injected via module-level references set by wire_rbac().
    ) -> Optional[Any]:
        from api.auth_service import is_auth_enforcement_active

        # Open mode: non-admin checks pass so the demo stays usable without login.
        if not enforce_in_open_mode and not is_auth_enforcement_active():
            return None

        return await _resolve_user_with_role(docbot_session, min_role)

    return _dependency


def require_admin():
    """Admin-only dependency — DOCBOT-1610 (SEC-01).

    Unlike ``require_role()``, this NEVER honours open/demo mode: anonymous
    callers get 401 and non-admins get 403 regardless of AUTH_REQUIRED.
    Admin routes (user/role management, API-key mint/revoke, audit log,
    platform metrics) are privilege-granting surfaces, so "open mode" must
    not mean "anyone is admin". The owner key is deliberately NOT accepted
    here either — admin access always requires a logged-in admin session.

    Always returns a user row (never None).
    """
    return require_role(UserRole.admin, enforce_in_open_mode=True)


def require_admin_org_id(user: Any) -> str:
    """Return the admin's org_id, failing CLOSED when it is missing — DOCBOT-1610.

    Every signup now owns a personal org and is its admin (DOCBOT-1702), so
    "is admin" no longer implies "may see everything". Admin routes must
    always scope to the caller's org. An admin row with no org_id (legacy /
    pre-backfill) gets 403 rather than an unscoped, cross-org view.
    """
    org_id = get_user_org_id(user)
    if not org_id:
        raise HTTPException(
            status_code=403,
            detail="Your account is not attached to an organization.",
        )
    return org_id


def platform_admin_ids() -> frozenset[str]:
    """User ids allowed to see platform-wide data (PLATFORM_ADMIN_USER_IDS).

    Comma-separated users.id values; default empty = nobody. Ids (server-
    generated UUIDs) are used instead of emails because an email can be
    pre-claimed by an attacker before the operator signs up (SEC-09),
    whereas a registrant cannot choose their id. Read per request.
    """
    raw = os.getenv("PLATFORM_ADMIN_USER_IDS", "")
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def require_platform_admin():
    """Platform-operator dependency for cross-org data (e.g. /admin/metrics).

    Requires a logged-in admin in every mode (like ``require_admin()``) AND
    the user's id in PLATFORM_ADMIN_USER_IDS. Org admins who are not
    operators get 403. Default (env unset) denies everyone.
    """
    async def _dependency(
        docbot_session: Optional[str] = Cookie(default=None),
    ) -> Any:
        user = await _resolve_user_with_role(docbot_session, UserRole.admin)
        if str(getattr(user, "id", "")) not in platform_admin_ids():
            raise HTTPException(status_code=403, detail="Platform operator access required.")
        return user

    return _dependency


class RoleChangeResult:
    """Outcome of ``update_user_role_in_org``."""

    __slots__ = ("user_id", "email", "new_role")

    def __init__(self, user_id: str, email: str, new_role: str) -> None:
        self.user_id = user_id
        self.email = email
        self.new_role = new_role


async def update_user_role_in_org(caller: Any, target_user_id: str, new_role: str) -> RoleChangeResult:
    """Org-scoped role change — DOCBOT-1610.

    - 400 on an unknown role name
    - 403 when the caller has no org (see ``require_admin_org_id``)
    - 404 when the target does not exist OR belongs to another org (same
      response either way, so org membership is not disclosed)
    - 409 when the change would leave the org with zero admins (covers an
      admin demoting themselves while being the last admin)

    The lookup, admin count and UPDATE run in one transaction and the UPDATE
    also carries the org predicate (defence in depth).
    """
    from sqlalchemy import and_, select, update

    valid_roles = {r.name for r in UserRole}
    if new_role not in valid_roles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role '{new_role}'. Must be one of: {', '.join(sorted(valid_roles))}",
        )
    org_id = require_admin_org_id(caller)
    users = _users_table
    in_org = and_(users.c.id == target_user_id, users.c.org_id == org_id)

    async with _async_session_factory() as db:
        async with db.begin():
            target = (await db.execute(select(users).where(in_org))).fetchone()
            if target is None:
                raise HTTPException(status_code=404, detail="User not found.")

            demoting_admin = target.role == UserRole.admin.name and new_role != UserRole.admin.name
            if demoting_admin:
                # Lock the org's admin rows so two concurrent demotions cannot
                # both see "2 admins" and leave zero (FOR UPDATE can't be used
                # with an aggregate, so count the locked rows in Python).
                admin_rows = (await db.execute(
                    select(users.c.id).where(
                        and_(users.c.org_id == org_id, users.c.role == UserRole.admin.name)
                    ).with_for_update()
                )).fetchall()
                if len(admin_rows) <= 1:
                    raise HTTPException(
                        status_code=409,
                        detail="Cannot remove the last admin of the organization.",
                    )

            await db.execute(update(users).where(in_org).values(role=new_role))
    return RoleChangeResult(user_id=target_user_id, email=target.email, new_role=new_role)


def is_deployed_environment() -> bool:
    """True on any hosted deploy (Railway) or an explicit production env — DOCBOT-1607.

    Railway injects RAILWAY_ENVIRONMENT / RAILWAY_ENVIRONMENT_NAME into every
    service; ENVIRONMENT / APP_ENV = prod|production covers other hosts.
    """
    if os.getenv("RAILWAY_ENVIRONMENT", "").strip() or os.getenv("RAILWAY_ENVIRONMENT_NAME", "").strip():
        return True
    for name in ("ENVIRONMENT", "APP_ENV"):
        if os.getenv(name, "").strip().lower() in ("prod", "production"):
            return True
    return False


def _anonymous_sandbox_flag_set() -> bool:
    return os.getenv("ALLOW_ANONYMOUS_SANDBOX", "").strip().lower() in ("1", "true", "yes", "on")


def anonymous_sandbox_allowed() -> bool:
    """True only when ALLOW_ANONYMOUS_SANDBOX is explicitly enabled — DOCBOT-1612.

    DEV-ONLY: never honored on a deployed environment (DOCBOT-1607) — on
    Railway / production the flag is ignored and a warning is logged at boot.
    Read at request time so it can be toggled in tests. Default: deny.
    """
    return _anonymous_sandbox_flag_set() and not is_deployed_environment()


def require_sandbox_user():
    """Dependency for raw code execution (POST /api/sandbox/execute) — DOCBOT-1612.

    Arbitrary Python in an E2B VM is the most expensive and most abusable
    surface, so it requires a logged-in analyst+ in EVERY mode by default:

    - AUTH_REQUIRED=true                       → analyst+ session required
    - open mode, ALLOW_ANONYMOUS_SANDBOX unset → analyst+ session required
    - open mode, ALLOW_ANONYMOUS_SANDBOX=true  → anonymous allowed (local dev only)

    Returns the user row, or None only in the explicit anonymous-dev case.
    """
    async def _dependency(
        docbot_session: Optional[str] = Cookie(default=None),
    ) -> Optional[Any]:
        from api.auth_service import is_auth_enforcement_active

        if not is_auth_enforcement_active() and anonymous_sandbox_allowed():
            return None
        return await _resolve_user_with_role(docbot_session, UserRole.analyst)

    return _dependency


class UnsafeAuthConfigError(RuntimeError):
    """Raised at startup when the auth configuration is unsafe to serve."""


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def assert_safe_auth_config() -> None:
    """Startup guard — DOCBOT-1610.

    A multi-tenant deployment (MULTI_TENANT=true) must never run in open
    mode: open mode skips every non-admin role check and leaves session data
    unscoped, which is a cross-tenant data leak. Refuse to boot instead.
    Anonymous sandbox execution is also refused in multi-tenant mode.

    Raises:
        UnsafeAuthConfigError: with an actionable message.
    """
    if _anonymous_sandbox_flag_set() and is_deployed_environment():
        # DOCBOT-1607: don't brick a prod boot over a stray dev flag — it is
        # already ignored by anonymous_sandbox_allowed(); make it visible.
        logger.warning(
            "ALLOW_ANONYMOUS_SANDBOX is set on a deployed environment and is IGNORED "
            "(dev-only). Remove it from this environment."
        )
    if not _env_true("MULTI_TENANT"):
        return

    from api.auth_service import is_auth_enforcement_active

    if not is_auth_enforcement_active():
        raise UnsafeAuthConfigError(
            "Refusing to start: MULTI_TENANT=true requires AUTH_REQUIRED=true. "
            "Open/demo mode skips role checks and cannot isolate tenants. "
            "Set AUTH_REQUIRED=true or unset MULTI_TENANT."
        )
    if _anonymous_sandbox_flag_set():
        raise UnsafeAuthConfigError(
            "Refusing to start: ALLOW_ANONYMOUS_SANDBOX=true is not permitted "
            "when MULTI_TENANT=true. Unset ALLOW_ANONYMOUS_SANDBOX."
        )


def get_user_org_id(user: Optional[Any]) -> Optional[str]:
    """None-safe accessor for the requesting user's org_id — DOCBOT-1524.

    `user` is whatever `require_role()` returned: `None` when auth
    enforcement is off (open/demo mode, unscoped by design), or the full
    `users` row otherwise. Route handlers scoping an admin query to the
    requesting user's organization should use this instead of reading
    `user.org_id` directly, so callers never need their own None check.
    """
    if user is None:
        return None
    return getattr(user, "org_id", None)


async def resolve_session_user(docbot_session: Optional[str]) -> Optional[Any]:
    """Return the user row for a session cookie value, or None — DOCBOT-1601.

    Never raises for a missing/expired/unknown token (callers treat None as
    anonymous). Returns None when RBAC has not been wired yet (e.g. a test
    client built without the app lifespan). Database errors propagate so a
    broken auth backend fails closed instead of silently downgrading a
    logged-in caller to anonymous.
    """
    if not docbot_session or _user_sessions_table is None or _users_table is None:
        return None
    from api.auth_service import get_user_from_session
    return await get_user_from_session(
        docbot_session,
        _user_sessions_table,
        _users_table,
        _async_session_factory,
    )


# ---------------------------------------------------------------------------
# Wire-up — called once from index.py after tables are created
# ---------------------------------------------------------------------------

_users_table: Any = None
_user_sessions_table: Any = None
_async_session_factory: Any = None


def wire_rbac(users_table: Any, user_sessions_table: Any, async_session_factory: Any) -> None:
    """Inject table references into module-level vars used by require_role."""
    global _users_table, _user_sessions_table, _async_session_factory
    _users_table = users_table
    _user_sessions_table = user_sessions_table
    _async_session_factory = async_session_factory
