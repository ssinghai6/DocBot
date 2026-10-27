"""DOCBOT-1612: /api/demo/init IP limit + /api/sandbox/execute auth.

CI-safe: no network, no real DB, no E2B. Heavy service calls are mocked;
the app lifespan is not run.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api.index as idx
from api import rbac_service
from api.sandbox_service import SandboxResult
from api.utils import request_guard


class _FakeClient:
    def __init__(self, host):
        self.host = host


class _FakeRequest:
    def __init__(self, ip="1.2.3.4", forwarded=None, headers=None, cookies=None):
        hdrs = dict(headers or {})
        if forwarded is not None:
            hdrs["x-forwarded-for"] = forwarded
        self.headers = hdrs
        self.query_params = {}
        self.cookies = dict(cookies or {})
        self.client = _FakeClient(ip)
        self.state = SimpleNamespace()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    monkeypatch.delenv("MULTI_TENANT", raising=False)
    monkeypatch.delenv("ALLOW_ANONYMOUS_SANDBOX", raising=False)
    for _name in ("RAILWAY_ENVIRONMENT", "RAILWAY_ENVIRONMENT_NAME", "ENVIRONMENT", "APP_ENV"):
        monkeypatch.delenv(_name, raising=False)
    monkeypatch.setattr(request_guard, "RATE_LIMIT_PER_MIN", 0)
    monkeypatch.setattr(request_guard, "SESSION_BUDGET", 0)
    monkeypatch.setattr(request_guard, "USER_DAILY_BUDGET", 0)
    monkeypatch.setattr(request_guard, "OWNER_KEY", "")
    monkeypatch.setattr(request_guard, "TRUSTED_PROXY_HOPS", 1)
    request_guard.reset_state()
    yield
    request_guard.reset_state()


@pytest.fixture
def client():
    return TestClient(idx.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# client_ip — TRUSTED_PROXY_HOPS
# ---------------------------------------------------------------------------

class TestClientIp:
    def test_default_is_one_trusted_hop(self):
        assert request_guard.TRUSTED_PROXY_HOPS == 1  # fixture mirrors the env default

    def test_zero_hops_ignores_header(self, monkeypatch):
        monkeypatch.setattr(request_guard, "TRUSTED_PROXY_HOPS", 0)
        req = _FakeRequest(ip="10.0.0.1", forwarded="203.0.113.7")
        assert request_guard.client_ip(req) == "10.0.0.1"

    def test_one_trusted_hop_takes_right_most(self, monkeypatch):
        monkeypatch.setattr(request_guard, "TRUSTED_PROXY_HOPS", 1)
        req = _FakeRequest(forwarded="6.6.6.6, 203.0.113.7")  # 6.6.6.6 is client-spoofed
        assert request_guard.client_ip(req) == "203.0.113.7"

    def test_two_trusted_hops(self, monkeypatch):
        monkeypatch.setattr(request_guard, "TRUSTED_PROXY_HOPS", 2)
        req = _FakeRequest(forwarded="6.6.6.6, 203.0.113.7, 76.76.21.1")
        assert request_guard.client_ip(req) == "203.0.113.7"

    def test_spoofed_left_most_entries_never_used(self, monkeypatch):
        for hops, chain, want in [
            (1, "1.1.1.1, 2.2.2.2, 203.0.113.7", "203.0.113.7"),
            (2, "1.1.1.1, 2.2.2.2, 203.0.113.7, 76.76.21.1", "203.0.113.7"),
        ]:
            monkeypatch.setattr(request_guard, "TRUSTED_PROXY_HOPS", hops)
            assert request_guard.client_ip(_FakeRequest(forwarded=chain)) == want

    def test_hops_larger_than_chain_clamps_to_left_most(self, monkeypatch):
        monkeypatch.setattr(request_guard, "TRUSTED_PROXY_HOPS", 5)
        req = _FakeRequest(forwarded="203.0.113.7")
        assert request_guard.client_ip(req) == "203.0.113.7"

    def test_empty_header_falls_back_to_peer(self):
        req = _FakeRequest(ip="198.51.100.2", forwarded=" , ")
        assert request_guard.client_ip(req) == "198.51.100.2"


# ---------------------------------------------------------------------------
# check_demo_init_limit
# ---------------------------------------------------------------------------

class TestDemoInitLimit:
    def test_defaults_are_non_zero(self):
        # Safe default: limit is ON without any env configuration.
        assert request_guard.DEMO_INIT_PER_IP_PER_HOUR > 0
        assert request_guard.DEMO_INIT_GLOBAL_PER_HOUR > 0

    async def test_per_ip_limit(self, monkeypatch):
        monkeypatch.setattr(request_guard, "DEMO_INIT_PER_IP_PER_HOUR", 3)
        monkeypatch.setattr(request_guard, "DEMO_INIT_GLOBAL_PER_HOUR", 0)
        req = _FakeRequest(ip="9.9.9.9")
        for _ in range(3):
            await request_guard.check_demo_init_limit(req)
        with pytest.raises(HTTPException) as exc:
            await request_guard.check_demo_init_limit(req)
        assert exc.value.status_code == 429
        assert int(exc.value.headers["Retry-After"]) >= 1

    async def test_other_ip_unaffected(self, monkeypatch):
        monkeypatch.setattr(request_guard, "DEMO_INIT_PER_IP_PER_HOUR", 1)
        monkeypatch.setattr(request_guard, "DEMO_INIT_GLOBAL_PER_HOUR", 0)
        await request_guard.check_demo_init_limit(_FakeRequest(ip="9.9.9.9"))
        await request_guard.check_demo_init_limit(_FakeRequest(ip="8.8.8.8"))  # no raise

    async def test_global_cap_stops_xff_rotation(self, monkeypatch):
        monkeypatch.setattr(request_guard, "DEMO_INIT_PER_IP_PER_HOUR", 100)
        monkeypatch.setattr(request_guard, "DEMO_INIT_GLOBAL_PER_HOUR", 5)
        for i in range(5):
            await request_guard.check_demo_init_limit(_FakeRequest(forwarded=f"10.1.0.{i}"))
        with pytest.raises(HTTPException) as exc:
            await request_guard.check_demo_init_limit(_FakeRequest(forwarded="10.1.0.99"))
        assert exc.value.status_code == 429

    async def test_rejected_call_does_not_consume_global_quota(self, monkeypatch):
        monkeypatch.setattr(request_guard, "DEMO_INIT_PER_IP_PER_HOUR", 1)
        monkeypatch.setattr(request_guard, "DEMO_INIT_GLOBAL_PER_HOUR", 2)
        noisy = _FakeRequest(ip="9.9.9.9")
        await request_guard.check_demo_init_limit(noisy)
        for _ in range(5):
            with pytest.raises(HTTPException):
                await request_guard.check_demo_init_limit(noisy)
        await request_guard.check_demo_init_limit(_FakeRequest(ip="8.8.8.8"))  # global slot still free

    async def test_window_expiry(self, monkeypatch):
        monkeypatch.setattr(request_guard, "DEMO_INIT_PER_IP_PER_HOUR", 1)
        monkeypatch.setattr(request_guard, "DEMO_INIT_GLOBAL_PER_HOUR", 0)
        clock = [3600.0 * 1000]
        monkeypatch.setattr(request_guard, "time", SimpleNamespace(time=lambda: clock[0]))
        req = _FakeRequest()
        await request_guard.check_demo_init_limit(req)
        with pytest.raises(HTTPException):
            await request_guard.check_demo_init_limit(req)
        clock[0] += request_guard.DEMO_INIT_WINDOW_SEC + 1
        await request_guard.check_demo_init_limit(req)  # window rolled over

    async def test_disabled_at_zero(self, monkeypatch):
        monkeypatch.setattr(request_guard, "DEMO_INIT_PER_IP_PER_HOUR", 0)
        monkeypatch.setattr(request_guard, "DEMO_INIT_GLOBAL_PER_HOUR", 0)
        for _ in range(500):
            await request_guard.check_demo_init_limit(_FakeRequest())

    async def test_owner_bypasses(self, monkeypatch):
        monkeypatch.setattr(request_guard, "OWNER_KEY", "owner-secret")
        monkeypatch.setattr(request_guard, "DEMO_INIT_PER_IP_PER_HOUR", 1)
        req = _FakeRequest(cookies={request_guard.OWNER_COOKIE_NAME: request_guard.owner_cookie_value()})
        for _ in range(10):
            await request_guard.check_demo_init_limit(req)

    def test_route_returns_429_after_limit(self, client, monkeypatch):
        monkeypatch.setattr(request_guard, "DEMO_INIT_PER_IP_PER_HOUR", 2)
        with patch("api.demo_service.init_demo_session", new_callable=AsyncMock, return_value={"ok": True}) as init:
            codes = [client.post("/api/demo/init").status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        assert init.await_count == 2  # the limited call never reached the service


# ---------------------------------------------------------------------------
# /api/sandbox/execute auth
# ---------------------------------------------------------------------------

_OK_RESULT = SandboxResult(stdout="2", stderr="", charts=[], error=None, execution_time_ms=1)


def _post_sandbox(client, cookie=None):
    if cookie:
        client.cookies.set("docbot_session", cookie)
    return client.post("/api/sandbox/execute", json={"code": "print(1+1)"})


class TestSandboxAuth:
    def test_anonymous_open_mode_401(self, client):
        with patch("api.sandbox_service.run_python", new_callable=AsyncMock, return_value=_OK_RESULT) as run:
            resp = _post_sandbox(client)
        assert resp.status_code == 401
        run.assert_not_called()

    def test_anonymous_auth_required_401(self, client, monkeypatch):
        monkeypatch.setenv("AUTH_REQUIRED", "true")
        with patch("api.sandbox_service.run_python", new_callable=AsyncMock, return_value=_OK_RESULT) as run:
            resp = _post_sandbox(client)
        assert resp.status_code == 401
        run.assert_not_called()

    def test_owner_key_alone_is_not_an_account(self, client, monkeypatch):
        monkeypatch.setattr(request_guard, "OWNER_KEY", "owner-secret")
        with patch("api.sandbox_service.run_python", new_callable=AsyncMock, return_value=_OK_RESULT) as run:
            resp = client.post(
                "/api/sandbox/execute", json={"code": "print(1)"}, headers={"x-owner-key": "owner-secret"}
            )
        assert resp.status_code == 401
        run.assert_not_called()

    def test_explicit_dev_flag_allows_anonymous_in_open_mode(self, client, monkeypatch):
        monkeypatch.setenv("ALLOW_ANONYMOUS_SANDBOX", "true")
        with patch("api.sandbox_service.run_python", new_callable=AsyncMock, return_value=_OK_RESULT):
            resp = _post_sandbox(client)
        assert resp.status_code == 200
        assert resp.json()["stdout"] == "2"

    def test_dev_flag_ignored_when_auth_required(self, client, monkeypatch):
        monkeypatch.setenv("ALLOW_ANONYMOUS_SANDBOX", "true")
        monkeypatch.setenv("AUTH_REQUIRED", "true")
        with patch("api.sandbox_service.run_python", new_callable=AsyncMock, return_value=_OK_RESULT) as run:
            resp = _post_sandbox(client)
        assert resp.status_code == 401
        run.assert_not_called()

    def test_viewer_gets_403(self, client):
        viewer = MagicMock()
        viewer.role = "viewer"
        with patch("api.auth_service.get_user_from_session", new_callable=AsyncMock, return_value=viewer), \
             patch("api.sandbox_service.run_python", new_callable=AsyncMock, return_value=_OK_RESULT) as run:
            resp = _post_sandbox(client, cookie="viewer-token")
        assert resp.status_code == 403
        run.assert_not_called()

    def test_analyst_can_execute(self, client):
        analyst = MagicMock()
        analyst.role = "analyst"
        analyst.id = "user-1"
        with patch("api.auth_service.get_user_from_session", new_callable=AsyncMock, return_value=analyst), \
             patch("api.sandbox_service.run_python", new_callable=AsyncMock, return_value=_OK_RESULT):
            resp = _post_sandbox(client, cookie="analyst-token")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Boot guard extension
# ---------------------------------------------------------------------------

def test_multi_tenant_refuses_anonymous_sandbox(monkeypatch):
    monkeypatch.setenv("MULTI_TENANT", "true")
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("ALLOW_ANONYMOUS_SANDBOX", "true")
    with pytest.raises(rbac_service.UnsafeAuthConfigError, match="ALLOW_ANONYMOUS_SANDBOX"):
        rbac_service.assert_safe_auth_config()


@pytest.mark.parametrize("value,expected", [("", False), ("false", False), ("0", False), ("true", True), ("1", True)])
def test_anonymous_sandbox_flag_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("ALLOW_ANONYMOUS_SANDBOX", value)
    assert rbac_service.anonymous_sandbox_allowed() is expected
