"""Unit tests for api/rbac_service.py — DOCBOT-603.

All tests are CI-safe: no network calls, no real DB.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# UserRole
# ---------------------------------------------------------------------------

class TestUserRole:
    def test_three_roles_defined(self):
        from api.rbac_service import UserRole
        assert {r.name for r in UserRole} == {"viewer", "analyst", "admin"}

    def test_hierarchy_order(self):
        from api.rbac_service import UserRole
        assert UserRole.viewer < UserRole.analyst < UserRole.admin

    def test_from_str_known_role(self):
        from api.rbac_service import UserRole
        assert UserRole.from_str("admin") == UserRole.admin
        assert UserRole.from_str("VIEWER") == UserRole.viewer
        assert UserRole.from_str("Analyst") == UserRole.analyst

    def test_from_str_unknown_defaults_to_analyst(self):
        from api.rbac_service import UserRole
        assert UserRole.from_str("superuser") == UserRole.analyst
        assert UserRole.from_str("") == UserRole.analyst


# ---------------------------------------------------------------------------
# _role_gte
# ---------------------------------------------------------------------------

class TestRoleGte:
    def test_same_role_passes(self):
        from api.rbac_service import _role_gte, UserRole
        assert _role_gte("analyst", UserRole.analyst) is True

    def test_higher_role_passes(self):
        from api.rbac_service import _role_gte, UserRole
        assert _role_gte("admin", UserRole.analyst) is True
        assert _role_gte("admin", UserRole.viewer) is True

    def test_lower_role_fails(self):
        from api.rbac_service import _role_gte, UserRole
        assert _role_gte("viewer", UserRole.analyst) is False
        assert _role_gte("analyst", UserRole.admin) is False
        assert _role_gte("viewer", UserRole.admin) is False


# ---------------------------------------------------------------------------
# require_role — dependency behaviour
# ---------------------------------------------------------------------------

class TestRequireRole:
    @pytest.mark.asyncio
    async def test_passes_when_saml_not_configured(self):
        """In non-SSO mode every request passes regardless of cookie presence."""
        from api.rbac_service import require_role, UserRole
        dep = require_role(UserRole.admin)
        with patch("api.auth_service.is_saml_configured", return_value=False):
            result = await dep(docbot_session=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_raises_401_when_no_cookie_and_saml_on(self):
        from api.rbac_service import require_role, UserRole
        from fastapi import HTTPException
        dep = require_role(UserRole.viewer)
        with patch("api.auth_service.is_saml_configured", return_value=True):
            with pytest.raises(HTTPException) as exc_info:
                await dep(docbot_session=None)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_raises_401_when_session_expired(self):
        from api.rbac_service import require_role, UserRole, wire_rbac
        from fastapi import HTTPException

        wire_rbac(MagicMock(), MagicMock(), MagicMock())
        dep = require_role(UserRole.viewer)

        with patch("api.auth_service.is_saml_configured", return_value=True), \
             patch("api.auth_service.get_user_from_session", new_callable=AsyncMock, return_value=None):
            with pytest.raises(HTTPException) as exc_info:
                await dep(docbot_session="expired-token")
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_raises_403_when_role_too_low(self):
        from api.rbac_service import require_role, UserRole, wire_rbac
        from fastapi import HTTPException

        fake_user = MagicMock()
        fake_user.role = "viewer"
        wire_rbac(MagicMock(), MagicMock(), MagicMock())
        dep = require_role(UserRole.admin)

        with patch("api.auth_service.is_saml_configured", return_value=True), \
             patch("api.auth_service.get_user_from_session", new_callable=AsyncMock, return_value=fake_user):
            with pytest.raises(HTTPException) as exc_info:
                await dep(docbot_session="valid-token")
        assert exc_info.value.status_code == 403
        assert "admin" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_returns_user_when_role_sufficient(self):
        from api.rbac_service import require_role, UserRole, wire_rbac

        fake_user = MagicMock()
        fake_user.role = "analyst"
        wire_rbac(MagicMock(), MagicMock(), MagicMock())
        dep = require_role(UserRole.analyst)

        with patch("api.auth_service.is_saml_configured", return_value=True), \
             patch("api.auth_service.get_user_from_session", new_callable=AsyncMock, return_value=fake_user):
            result = await dep(docbot_session="valid-token")
        assert result is fake_user

    @pytest.mark.asyncio
    async def test_admin_can_access_admin_route(self):
        from api.rbac_service import require_role, UserRole, wire_rbac

        fake_admin = MagicMock()
        fake_admin.role = "admin"
        wire_rbac(MagicMock(), MagicMock(), MagicMock())
        dep = require_role(UserRole.admin)

        with patch("api.auth_service.is_saml_configured", return_value=True), \
             patch("api.auth_service.get_user_from_session", new_callable=AsyncMock, return_value=fake_admin):
            result = await dep(docbot_session="admin-token")
        assert result.role == "admin"


# ---------------------------------------------------------------------------
# wire_rbac
# ---------------------------------------------------------------------------

class TestWireRbac:
    def test_wire_sets_module_vars(self):
        from api.rbac_service import wire_rbac
        import api.rbac_service as rbac_mod

        fake_users = MagicMock()
        fake_sessions = MagicMock()
        fake_factory = MagicMock()

        wire_rbac(fake_users, fake_sessions, fake_factory)

        assert rbac_mod._users_table is fake_users
        assert rbac_mod._user_sessions_table is fake_sessions
        assert rbac_mod._async_session_factory is fake_factory
