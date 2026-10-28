"""DOCBOT-1610 (SEC-01): admin routes require auth in EVERY mode.

Before this ticket `require_role()` returned None when AUTH_REQUIRED was not
"true", so anonymous callers could mint /v1 API keys and promote users to
admin. These tests pin the fix:

- every admin-only route returns 401 to an anonymous caller in open mode
  (and in enforced mode);
- a logged-in non-admin gets 403 in open mode;
- the owner key does NOT unlock admin routes;
- the route list below is exhaustive — a new /admin/* route (or any route
  wired to the admin dependency) that is not listed here fails the sweep;
- MULTI_TENANT=true without AUTH_REQUIRED=true refuses to boot.

CI-safe: no network, no real DB. The app's lifespan is not run (TestClient
is used without a context manager), and the admin dependency rejects before
any DB access.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import api.index as idx
from api import rbac_service
from api.utils import request_guard


# Every route that depends on the admin role. (method, path-template, sample path, json body)
ADMIN_ROUTES = [
    ("GET", "/admin/audit-log", "/admin/audit-log", None),
    ("GET", "/admin/users", "/admin/users", None),
    ("PATCH", "/admin/users/{user_id}/role", "/admin/users/u-123/role", {"role": "admin"}),
    ("GET", "/admin/api-keys", "/admin/api-keys", None),
    ("POST", "/admin/api-keys", "/admin/api-keys", {"name": "pwn", "scopes": ["hybrid:read"]}),
    ("DELETE", "/admin/api-keys/{key_id}", "/admin/api-keys/k-123", None),
    ("GET", "/admin/metrics", "/admin/metrics", None),
    ("GET", "/admin/funnel", "/admin/funnel", None),
]

_ADMIN_DEP_CALLS = {idx._rbac_admin.dependency, idx._rbac_platform_admin.dependency}


@pytest.fixture(autouse=True)
def _open_mode(monkeypatch):
    """Open/demo mode: AUTH_REQUIRED unset, no guards that could mask a 401."""
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    monkeypatch.delenv("MULTI_TENANT", raising=False)
    monkeypatch.setattr(request_guard, "RATE_LIMIT_PER_MIN", 0)
    request_guard.reset_state()
    yield
    request_guard.reset_state()


@pytest.fixture
def client():
    return TestClient(idx.app, raise_server_exceptions=False)


def _send(client, method, path, body, cookies=None, **kwargs):
    for name, value in (cookies or {}).items():
        client.cookies.set(name, value)
    if body is None:
        return client.request(method, path, **kwargs)
    return client.request(method, path, json=body, **kwargs)


def _route_uses_admin_dep(route: APIRoute) -> bool:
    stack = list(route.dependant.dependencies)
    while stack:
        dep = stack.pop()
        if dep.call in _ADMIN_DEP_CALLS:
            return True
        stack.extend(dep.dependencies)
    return False


# ---------------------------------------------------------------------------
# Anonymous → 401, per route
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,template,path,body", ADMIN_ROUTES, ids=[f"{m} {t}" for m, t, _, _ in ADMIN_ROUTES])
def test_anonymous_gets_401_in_open_mode(client, method, template, path, body):
    resp = _send(client, method, path, body)
    assert resp.status_code == 401, f"{method} {path} returned {resp.status_code} to an anonymous caller in open mode"


@pytest.mark.parametrize("method,template,path,body", ADMIN_ROUTES, ids=[f"{m} {t}" for m, t, _, _ in ADMIN_ROUTES])
def test_anonymous_gets_401_when_auth_required(client, monkeypatch, method, template, path, body):
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    resp = _send(client, method, path, body)
    assert resp.status_code == 401


@pytest.mark.parametrize("method,template,path,body", ADMIN_ROUTES, ids=[f"{m} {t}" for m, t, _, _ in ADMIN_ROUTES])
def test_owner_key_does_not_unlock_admin(client, monkeypatch, method, template, path, body):
    monkeypatch.setattr(request_guard, "OWNER_KEY", "test-owner-key")
    resp = _send(client, method, path, body, headers={"x-owner-key": "test-owner-key"})
    assert resp.status_code == 401


@pytest.mark.parametrize("method,template,path,body", ADMIN_ROUTES, ids=[f"{m} {t}" for m, t, _, _ in ADMIN_ROUTES])
def test_invalid_session_gets_401_in_open_mode(client, method, template, path, body):
    with patch("api.auth_service.get_user_from_session", new_callable=AsyncMock, return_value=None):
        resp = _send(client, method, path, body, cookies={"docbot_session": "expired"})
    assert resp.status_code == 401


@pytest.mark.parametrize("method,template,path,body", ADMIN_ROUTES, ids=[f"{m} {t}" for m, t, _, _ in ADMIN_ROUTES])
def test_non_admin_gets_403_in_open_mode(client, method, template, path, body):
    analyst = MagicMock()
    analyst.role = "analyst"
    with patch("api.auth_service.get_user_from_session", new_callable=AsyncMock, return_value=analyst):
        resp = _send(client, method, path, body, cookies={"docbot_session": "analyst-token"})
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Sweep: the list above is exhaustive
# ---------------------------------------------------------------------------

def test_admin_route_list_is_exhaustive():
    listed = {(m, t) for m, t, _, _ in ADMIN_ROUTES}
    found = set()
    for route in idx.app.routes:
        if not isinstance(route, APIRoute):
            continue
        if route.path.startswith("/admin") or _route_uses_admin_dep(route):
            for method in route.methods:
                found.add((method, route.path))
    assert found == listed, (
        f"Admin route set changed. Unlisted: {sorted(found - listed)}; stale: {sorted(listed - found)}. "
        "Add new admin routes to ADMIN_ROUTES (and guard them with _rbac_admin)."
    )


def test_every_admin_path_uses_admin_dependency():
    for route in idx.app.routes:
        if isinstance(route, APIRoute) and route.path.startswith("/admin"):
            assert _route_uses_admin_dep(route), f"{route.path} is not guarded by require_admin()"


# ---------------------------------------------------------------------------
# require_admin dependency, directly
# ---------------------------------------------------------------------------

class TestRequireAdmin:
    @pytest.mark.asyncio
    async def test_open_mode_no_cookie_401(self):
        dep = rbac_service.require_admin()
        with patch("api.auth_service.is_auth_enforcement_active", return_value=False):
            with pytest.raises(Exception) as exc_info:
                await dep(docbot_session=None)
        assert getattr(exc_info.value, "status_code", None) == 401

    @pytest.mark.asyncio
    async def test_open_mode_admin_session_returns_user(self):
        admin = MagicMock()
        admin.role = "admin"
        rbac_service.wire_rbac(MagicMock(), MagicMock(), MagicMock())
        dep = rbac_service.require_admin()
        with patch("api.auth_service.is_auth_enforcement_active", return_value=False), \
             patch("api.auth_service.get_user_from_session", new_callable=AsyncMock, return_value=admin):
            result = await dep(docbot_session="admin-token")
        assert result is admin

    @pytest.mark.asyncio
    async def test_plain_require_role_still_open_in_open_mode(self):
        """Non-admin routes keep the documented open-mode behaviour."""
        dep = rbac_service.require_role(rbac_service.UserRole.viewer)
        with patch("api.auth_service.is_auth_enforcement_active", return_value=False):
            assert await dep(docbot_session=None) is None


# ---------------------------------------------------------------------------
# MULTI_TENANT boot guard
# ---------------------------------------------------------------------------

class TestMultiTenantBootGuard:
    def test_multi_tenant_without_auth_refuses(self, monkeypatch):
        monkeypatch.setenv("MULTI_TENANT", "true")
        monkeypatch.delenv("AUTH_REQUIRED", raising=False)
        with pytest.raises(rbac_service.UnsafeAuthConfigError, match="AUTH_REQUIRED=true"):
            rbac_service.assert_safe_auth_config()

    def test_multi_tenant_with_auth_false_refuses(self, monkeypatch):
        monkeypatch.setenv("MULTI_TENANT", "1")
        monkeypatch.setenv("AUTH_REQUIRED", "false")
        with pytest.raises(rbac_service.UnsafeAuthConfigError):
            rbac_service.assert_safe_auth_config()

    def test_multi_tenant_with_auth_boots(self, monkeypatch):
        monkeypatch.setenv("MULTI_TENANT", "true")
        monkeypatch.setenv("AUTH_REQUIRED", "true")
        rbac_service.assert_safe_auth_config()  # no raise

    def test_single_tenant_open_mode_boots(self, monkeypatch):
        monkeypatch.delenv("MULTI_TENANT", raising=False)
        monkeypatch.delenv("AUTH_REQUIRED", raising=False)
        rbac_service.assert_safe_auth_config()  # no raise

    @pytest.mark.asyncio
    async def test_lifespan_aborts_before_db_init(self, monkeypatch):
        monkeypatch.setenv("MULTI_TENANT", "true")
        monkeypatch.delenv("AUTH_REQUIRED", raising=False)
        init_db = AsyncMock()
        monkeypatch.setattr(idx, "init_db", init_db)
        with pytest.raises(rbac_service.UnsafeAuthConfigError):
            async with idx.lifespan(idx.app):
                pass
        init_db.assert_not_called()
