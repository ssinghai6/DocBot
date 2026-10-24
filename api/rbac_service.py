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

Exception enforced in EVERY mode (DOCBOT-1610):
    require_admin()        — admin routes always need a logged-in admin

Startup guard: assert_safe_auth_config() refuses to boot when
MULTI_TENANT=true without AUTH_REQUIRED=true.
"""

from __future__ import annotations

import os
from enum import IntEnum
from typing import Any, Optional

from fastapi import Cookie, Depends, HTTPException


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


class UnsafeAuthConfigError(RuntimeError):
    """Raised at startup when the auth configuration is unsafe to serve."""


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def assert_safe_auth_config() -> None:
    """Startup guard — DOCBOT-1610.

    A multi-tenant deployment (MULTI_TENANT=true) must never run in open
    mode: open mode skips every non-admin role check and leaves session data
    unscoped, which is a cross-tenant data leak. Refuse to boot instead.

    Raises:
        UnsafeAuthConfigError: with an actionable message.
    """
    if not _env_true("MULTI_TENANT"):
        return

    from api.auth_service import is_auth_enforcement_active

    if not is_auth_enforcement_active():
        raise UnsafeAuthConfigError(
            "Refusing to start: MULTI_TENANT=true requires AUTH_REQUIRED=true. "
            "Open/demo mode skips role checks and cannot isolate tenants. "
            "Set AUTH_REQUIRED=true or unset MULTI_TENANT."
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
