"""Role-Based Access Control — DOCBOT-603.

Three roles (least to most privileged):
    viewer   — can read query results; cannot create/remove DB connections
    analyst  — can run queries and manage their own DB connections (default for JIT-provisioned users)
    admin    — full access including audit log, user management, all connections

Usage in route handlers:
    from api.rbac_service import require_role, UserRole

    @app.get("/admin/audit-log")
    async def audit_log(user=Depends(require_role(UserRole.admin))):
        ...

When auth enforcement is not active (AUTH_REQUIRED env var is not set to
"true"), all role checks pass automatically so the app remains usable
without login — ideal for demos and local development.
"""

from __future__ import annotations

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


def require_role(min_role: UserRole):
    """FastAPI dependency factory.

    Returns a dependency that:
    - Skips enforcement when AUTH_REQUIRED is not set (open/demo mode)
    - Reads the session cookie and resolves the user
    - Raises HTTP 401 if not authenticated (and SAML is on)
    - Raises HTTP 403 if the user's role is below min_role

    Inject as: user = Depends(require_role(UserRole.admin))
    """
    async def _dependency(
        docbot_session: Optional[str] = Cookie(default=None),
        # Tables and factory injected at call site — passed via closures from index.py
        # We use module-level references set by wire_rbac() below.
    ) -> Optional[Any]:
        from api.auth_service import is_auth_enforcement_active

        # When auth enforcement is not active the app runs in open mode — all checks pass
        if not is_auth_enforcement_active():
            return None

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

    return _dependency


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
