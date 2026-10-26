"""DOCBOT-1607 (SEC-04): E2B sandbox lockdown.

CI-safe: E2B is replaced by an in-memory fake module; no network.
Covers: anon 401 / dev flag ignored on deployed envs, org/user metering
(not session_id), create kwargs (timeout + internet off), kill() on
success / exception / timeout, pip egress policy, usage-ledger metering.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import api.index as idx
from api import rbac_service
from api.sandbox_service import SandboxResult
from api.utils import request_guard, sandbox_policy

_OK = SandboxResult(stdout="ok", stderr="", charts=[], error=None, execution_time_ms=1)
_ENV_NAMES = ("RAILWAY_ENVIRONMENT", "RAILWAY_ENVIRONMENT_NAME", "ENVIRONMENT", "APP_ENV")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("AUTH_REQUIRED", "MULTI_TENANT", "ALLOW_ANONYMOUS_SANDBOX",
                 "SANDBOX_PIP_INTERNET", "E2B_SANDBOX_TEMPLATE", *_ENV_NAMES):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(request_guard, "RATE_LIMIT_PER_MIN", 0)
    monkeypatch.setattr(request_guard, "SESSION_BUDGET", 0)
    monkeypatch.setattr(request_guard, "USER_DAILY_BUDGET", 0)
    monkeypatch.setattr(request_guard, "OWNER_KEY", "")
    monkeypatch.setattr(request_guard, "SANDBOX_RUNS_PER_HOUR", 60)
    request_guard.reset_state()
    yield
    request_guard.reset_state()


@pytest.fixture
def client():
    return TestClient(idx.app, raise_server_exceptions=False)


def _analyst(org_id: Any = "org-1", user_id: str = "user-1") -> MagicMock:
    u = MagicMock()
    u.role = "analyst"
    u.id = user_id
    u.org_id = org_id
    return u


def _plain_request(headers: dict | None = None) -> MagicMock:
    req = MagicMock()
    req.headers = dict(headers or {})
    req.query_params = {}
    req.cookies = {}
    req.client.host = "9.9.9.9"
    return req


def _post(client, cookie=None, session_id=None):
    body = {"code": "print(1)"}
    if session_id:
        body["session_id"] = session_id
    if cookie:
        client.cookies.set("docbot_session", cookie)
    return client.post("/api/sandbox/execute", json=body)


# ---------------------------------------------------------------------------
# Auth: signed-in analyst+ required, dev flag never honored when deployed
# ---------------------------------------------------------------------------

class TestAuth:
    def test_anonymous_open_mode_401(self, client):
        with patch("api.sandbox_service.run_python", new_callable=AsyncMock, return_value=_OK) as run:
            resp = _post(client)
        assert resp.status_code == 401
        run.assert_not_called()

    @pytest.mark.parametrize("name,value", [
        ("RAILWAY_ENVIRONMENT", "production"),
        ("RAILWAY_ENVIRONMENT_NAME", "staging"),
        ("ENVIRONMENT", "production"),
        ("APP_ENV", "prod"),
    ])
    def test_dev_flag_ignored_on_deployed_env(self, client, monkeypatch, name, value):
        monkeypatch.setenv("ALLOW_ANONYMOUS_SANDBOX", "true")
        monkeypatch.setenv(name, value)
        assert rbac_service.anonymous_sandbox_allowed() is False
        with patch("api.sandbox_service.run_python", new_callable=AsyncMock, return_value=_OK) as run:
            resp = _post(client)
        assert resp.status_code == 401
        run.assert_not_called()

    def test_dev_flag_still_works_locally(self, monkeypatch):
        monkeypatch.setenv("ALLOW_ANONYMOUS_SANDBOX", "true")
        assert rbac_service.anonymous_sandbox_allowed() is True

    def test_boot_warns_but_does_not_crash_on_deployed_flag(self, monkeypatch, caplog):
        monkeypatch.setenv("ALLOW_ANONYMOUS_SANDBOX", "true")
        monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
        rbac_service.assert_safe_auth_config()
        assert "IGNORED" in caplog.text

    def test_multi_tenant_still_refuses_flag_even_when_deployed(self, monkeypatch):
        monkeypatch.setenv("ALLOW_ANONYMOUS_SANDBOX", "true")
        monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
        monkeypatch.setenv("MULTI_TENANT", "true")
        monkeypatch.setenv("AUTH_REQUIRED", "true")
        with pytest.raises(rbac_service.UnsafeAuthConfigError):
            rbac_service.assert_safe_auth_config()

    def test_route_runs_user_code_without_pip_egress(self, client):
        with patch("api.auth_service.get_user_from_session", new_callable=AsyncMock, return_value=_analyst()), \
             patch("api.sandbox_service.run_python", new_callable=AsyncMock, return_value=_OK) as run:
            resp = _post(client, cookie="tok")
        assert resp.status_code == 200
        assert run.await_args.kwargs["allow_pip_egress"] is False


# ---------------------------------------------------------------------------
# Metering by org / user, not session_id
# ---------------------------------------------------------------------------

class TestMetering:
    def test_org_quota_ignores_session_rotation(self, client, monkeypatch):
        monkeypatch.setattr(request_guard, "SANDBOX_RUNS_PER_HOUR", 2)
        with patch("api.auth_service.get_user_from_session", new_callable=AsyncMock, return_value=_analyst()), \
             patch("api.sandbox_service.run_python", new_callable=AsyncMock, return_value=_OK) as run:
            codes = [_post(client, cookie="tok", session_id=f"s-{i}").status_code for i in range(3)]
        assert codes == [200, 200, 429]
        assert run.await_count == 2

    def test_org_members_share_quota(self, client, monkeypatch):
        monkeypatch.setattr(request_guard, "SANDBOX_RUNS_PER_HOUR", 1)
        users = iter([_analyst(user_id="a"), _analyst(user_id="b")])
        with patch("api.auth_service.get_user_from_session", new_callable=AsyncMock,
                   side_effect=lambda *_a, **_k: next(users)), \
             patch("api.sandbox_service.run_python", new_callable=AsyncMock, return_value=_OK):
            assert _post(client, cookie="a").status_code == 200
            assert _post(client, cookie="b").status_code == 429

    def test_key_falls_back_to_user_then_ip(self):
        req = _plain_request()
        assert request_guard.sandbox_principal_key(_analyst(org_id="o"), req) == "org:o"
        assert request_guard.sandbox_principal_key(_analyst(org_id=None, user_id="u"), req) == "user:u"
        assert request_guard.sandbox_principal_key(None, req) == "ip:9.9.9.9"

    async def test_user_daily_budget_charged(self, monkeypatch):
        calls: List[tuple] = []
        async def _record(uid, ep):
            calls.append((uid, ep))
        monkeypatch.setattr(request_guard, "enforce_user_budget", _record)
        await request_guard.enforce_sandbox_quota(_plain_request(), _analyst(user_id="u-7"))
        assert calls == [("u-7", "sandbox")]

    async def test_owner_bypasses(self, monkeypatch):
        monkeypatch.setattr(request_guard, "OWNER_KEY", "k")
        monkeypatch.setattr(request_guard, "SANDBOX_RUNS_PER_HOUR", 1)
        req = _plain_request()
        req.cookies = {request_guard.OWNER_COOKIE_NAME: request_guard.owner_cookie_value()}
        for _ in range(3):
            await request_guard.enforce_sandbox_quota(req, _analyst())


# ---------------------------------------------------------------------------
# E2B lifecycle — create kwargs, kill() in finally, metering
# ---------------------------------------------------------------------------

class _Logs:
    def __init__(self):
        self.stdout = ["hi\n"]
        self.stderr: list = []


class _Execution:
    def __init__(self):
        self.logs = _Logs()
        self.error = None


class _MissingModule:
    name = "ModuleNotFoundError"
    value = "No module named 'prophet'"


class _FakeSandbox:
    created: List[dict] = []
    instances: List["_FakeSandbox"] = []
    mode = "ok"  # ok | boom | hang | missing
    release = threading.Event()

    def __init__(self):
        self.killed = 0
        self.installed = False

    @classmethod
    def create(cls, template=None, timeout=None, allow_internet_access=True, **opts):
        kwargs: dict = {"timeout": timeout, "allow_internet_access": allow_internet_access}
        if template is not None:
            kwargs["template"] = template
        cls.created.append(kwargs)
        sb = cls()
        cls.instances.append(sb)
        return sb

    def run_code(self, code, **_kw):
        if _FakeSandbox.mode == "boom":
            raise RuntimeError("boom")
        if _FakeSandbox.mode == "hang":
            _FakeSandbox.release.wait(5)
        ex = _Execution()
        if _FakeSandbox.mode == "missing" and not self.installed:
            ex.error = _MissingModule()
        return ex

    def kill(self):
        self.killed += 1

    class files:  # noqa: N801 - mimic SDK attribute
        @staticmethod
        def write(path, data):
            return None


@pytest.fixture
def fake_e2b(monkeypatch):
    _FakeSandbox.created = []
    _FakeSandbox.instances = []
    _FakeSandbox.mode = "ok"
    _FakeSandbox.release = threading.Event()
    mod = types.ModuleType("e2b_code_interpreter")
    mod.Sandbox = _FakeSandbox
    monkeypatch.setitem(sys.modules, "e2b_code_interpreter", mod)
    monkeypatch.setenv("E2B_API_KEY", "e2b_test_key")
    from api import sandbox_service

    metered: List[float] = []
    monkeypatch.setattr(sandbox_service, "meter_sandbox", lambda s: metered.append(s))
    yield sandbox_service, metered
    _FakeSandbox.release.set()


_DEFAULT_KW = {"timeout": 60, "allow_internet_access": False}


class TestLifecycle:
    def test_create_kwargs_timeout_and_internet_off(self, fake_e2b):
        svc, _ = fake_e2b
        res = svc._run_in_sandbox("print(1)")
        assert res.error is None
        assert _FakeSandbox.created == [_DEFAULT_KW]

    def test_kill_and_meter_on_success(self, fake_e2b):
        svc, metered = fake_e2b
        svc._run_in_sandbox("print(1)")
        assert _FakeSandbox.instances[0].killed == 1
        assert len(metered) == 1

    def test_kill_and_meter_on_exception(self, fake_e2b):
        svc, metered = fake_e2b
        _FakeSandbox.mode = "boom"
        with pytest.raises(RuntimeError):
            svc._run_in_sandbox("print(1)")
        assert _FakeSandbox.instances[0].killed == 1
        assert len(metered) == 1

    def test_csv_kill_on_success_and_exception(self, fake_e2b):
        svc, metered = fake_e2b
        svc._run_csv_in_sandbox_sync("print(1)", "/home/user/x.csv", b"a\n1\n")
        _FakeSandbox.mode = "boom"
        with pytest.raises(RuntimeError):
            svc._run_csv_in_sandbox_sync("print(1)", "/home/user/x.csv", b"a\n1\n")
        assert [sb.killed for sb in _FakeSandbox.instances] == [1, 1]
        assert _FakeSandbox.created == [_DEFAULT_KW, _DEFAULT_KW]
        assert len(metered) == 2

    def test_kill_on_timeout_without_waiting_for_worker(self, fake_e2b):
        svc, metered = fake_e2b
        _FakeSandbox.mode = "hang"

        async def _main():
            res = await svc.run_python("print(1)", timeout_seconds=1)
            killed_at_timeout = _FakeSandbox.instances[0].killed
            _FakeSandbox.release.set()
            await asyncio.sleep(0.3)
            return res, killed_at_timeout

        res, killed_at_timeout = asyncio.run(_main())
        assert "timed out" in (res.error or "")
        assert killed_at_timeout == 1  # killed by the async side on timeout
        assert _FakeSandbox.instances[0].killed == 1  # finally kill is idempotent
        assert len(metered) == 1

    def test_csv_kill_on_timeout(self, fake_e2b, monkeypatch):
        svc, _ = fake_e2b
        monkeypatch.setenv("SANDBOX_PIP_INTERNET", "false")  # no install budget on the deadline
        _FakeSandbox.mode = "hang"

        async def _main():
            res = await svc._run_csv_in_sandbox("print(1)", "/home/user/x.csv", b"a\n", timeout_seconds=1)
            killed = _FakeSandbox.instances[0].killed
            _FakeSandbox.release.set()
            await asyncio.sleep(0.3)
            return res, killed

        res, killed = asyncio.run(_main())
        assert "timed out" in (res.error or "")
        assert killed == 1

    def test_user_code_never_gets_internet_or_pip(self, fake_e2b, monkeypatch):
        svc, _ = fake_e2b
        installs: List[Any] = []
        monkeypatch.setattr(svc, "_install_sandbox_packages", lambda sb, pk: installs.append(pk))
        svc._run_in_sandbox("import statsmodels\nprint(1)", allow_pip_egress=False)
        assert _FakeSandbox.created[0]["allow_internet_access"] is False
        assert installs == []

    def test_internal_codegen_gets_egress_only_for_missing_packages(self, fake_e2b, monkeypatch):
        svc, _ = fake_e2b
        installs: List[Any] = []
        monkeypatch.setattr(svc, "_install_sandbox_packages", lambda sb, pk: installs.append(pk))
        svc._run_in_sandbox("import sklearn\nprint(1)", allow_pip_egress=True)  # pre-baked
        svc._run_in_sandbox("import statsmodels\nprint(1)", allow_pip_egress=True)
        assert [c["allow_internet_access"] for c in _FakeSandbox.created] == [False, True]
        assert installs == [["statsmodels"]]

    def test_pip_egress_kill_switch(self, fake_e2b, monkeypatch):
        svc, _ = fake_e2b
        monkeypatch.setenv("SANDBOX_PIP_INTERNET", "false")
        monkeypatch.setattr(svc, "_install_sandbox_packages", lambda sb, pk: None)
        svc._run_in_sandbox("import statsmodels\nprint(1)", allow_pip_egress=True)
        assert _FakeSandbox.created[0]["allow_internet_access"] is False

    def test_custom_template(self, fake_e2b, monkeypatch):
        svc, _ = fake_e2b
        monkeypatch.setenv("E2B_SANDBOX_TEMPLATE", "docbot-ml")
        svc._run_in_sandbox("print(1)")
        assert _FakeSandbox.created[0]["template"] == "docbot-ml"


class TestLifetimeAndModuleRetry:
    """DOCBOT-2318: lifetime covers install + run; codegen module retry gets egress."""

    @pytest.fixture
    def pip(self, fake_e2b, monkeypatch):
        svc, _ = fake_e2b
        installs: List[str] = []

        def _fake_pip(sb, pkg):
            installs.append(pkg)
            sb.installed = True
            return True

        monkeypatch.setattr(svc, "_pip_install_in_sandbox", _fake_pip)
        return installs

    def test_raw_user_code_lifetime_60_no_internet(self, fake_e2b):
        svc, _ = fake_e2b
        asyncio.run(svc.run_python("print(1)", timeout_seconds=25, allow_pip_egress=False))
        assert _FakeSandbox.created == [_DEFAULT_KW]

    def test_codegen_lifetime_exceeds_exec_timeout(self, fake_e2b):
        svc, _ = fake_e2b
        asyncio.run(svc.run_python("print(1)", timeout_seconds=60, allow_pip_egress=True))
        asyncio.run(svc._run_csv_in_sandbox("print(1)", "/home/user/x.csv", b"a\n", timeout_seconds=60))
        assert [c["timeout"] for c in _FakeSandbox.created] == [150, 150]
        assert all(c["allow_internet_access"] is False for c in _FakeSandbox.created)

    def test_policy_lifetime(self, monkeypatch):
        assert sandbox_policy.lifetime_seconds(allow_pip_egress=False, exec_timeout_seconds=120) == 60
        assert sandbox_policy.lifetime_seconds(allow_pip_egress=True, exec_timeout_seconds=30) == 120
        monkeypatch.setenv("SANDBOX_PIP_INTERNET", "false")
        assert sandbox_policy.lifetime_seconds(allow_pip_egress=True, exec_timeout_seconds=30) == 60
        assert sandbox_policy.lifetime_seconds(allow_pip_egress=True, exec_timeout_seconds=75) == 75

    def test_codegen_module_retry_uses_fresh_egress_sandbox(self, fake_e2b, pip):
        svc, metered = fake_e2b
        _FakeSandbox.mode = "missing"
        res = svc._run_in_sandbox("print(1)", allow_pip_egress=True, lifetime_seconds=115)
        assert res.error is None
        assert _FakeSandbox.created == [
            {"timeout": 115, "allow_internet_access": False},
            {"timeout": 115, "allow_internet_access": True},
        ]
        assert pip == ["prophet"]
        assert [sb.killed for sb in _FakeSandbox.instances] == [1, 1]
        assert len(metered) == 1

    def test_csv_module_retry_uses_fresh_egress_sandbox(self, fake_e2b, pip):
        svc, _ = fake_e2b
        _FakeSandbox.mode = "missing"
        res = svc._run_csv_in_sandbox_sync("print(1)", "/home/user/x.csv", b"a\n1\n")
        assert res.error is None
        assert [c["allow_internet_access"] for c in _FakeSandbox.created] == [False, True]
        assert [sb.killed for sb in _FakeSandbox.instances] == [1, 1]

    def test_raw_user_code_never_gets_module_retry(self, fake_e2b, pip):
        svc, _ = fake_e2b
        _FakeSandbox.mode = "missing"
        res = svc._run_in_sandbox("print(1)", allow_pip_egress=False)
        assert "ModuleNotFoundError" in (res.error or "")
        assert _FakeSandbox.created == [_DEFAULT_KW]
        assert pip == []

    def test_module_retry_respects_kill_switch(self, fake_e2b, pip, monkeypatch):
        svc, _ = fake_e2b
        monkeypatch.setenv("SANDBOX_PIP_INTERNET", "false")
        _FakeSandbox.mode = "missing"
        res = svc._run_in_sandbox("print(1)", allow_pip_egress=True)
        res_csv = svc._run_csv_in_sandbox_sync("print(1)", "/home/user/x.csv", b"a\n")
        assert "ModuleNotFoundError" in (res.error or "")
        assert "ModuleNotFoundError" in (res_csv.error or "")
        assert [c["allow_internet_access"] for c in _FakeSandbox.created] == [False, False]
        assert pip == []


class TestPolicy:
    def test_unsupported_kwargs_are_dropped(self):
        class Old:
            @classmethod
            def create(cls, timeout=None):
                return (cls, timeout)

        assert sandbox_policy.create_kwargs(Old.create, allow_internet=False) == {"timeout": 60}

    def test_holder_kills_late_created_sandbox(self):
        holder = sandbox_policy.SandboxHolder()
        holder.kill()  # timeout fired before create() returned
        sb = MagicMock()
        holder.set(sb)
        sb.kill.assert_called_once()
        holder.kill()
        sb.kill.assert_called_once()

    def test_holder_kill_swallows_errors(self):
        holder = sandbox_policy.SandboxHolder()
        sb = MagicMock()
        sb.kill.side_effect = RuntimeError("net down")
        holder.set(sb)
        holder.kill()  # must not raise


class TestSseKeepalive:
    """DOCBOT-2318: long CSV sandbox runs keep the SSE stream alive."""

    def test_yields_keepalives_until_task_done(self):
        from api import sandbox_service as svc

        async def _run():
            task = asyncio.ensure_future(asyncio.sleep(0.05, result="ok"))
            chunks = [c async for c in svc._keepalive_until(task, interval=0.01)]
            return chunks, task.result()

        chunks, result = asyncio.run(_run())
        assert result == "ok"
        assert chunks and all(c == svc.SSE_KEEPALIVE for c in chunks)
        assert not svc.SSE_KEEPALIVE.startswith("data:")  # clients ignore it

    def test_closing_stream_cancels_task(self):
        from api import sandbox_service as svc

        async def _run():
            task = asyncio.ensure_future(asyncio.sleep(10))
            gen = svc._keepalive_until(task, interval=0.01)
            await gen.__anext__()
            await gen.aclose()
            await asyncio.sleep(0)
            return task.cancelled()

        assert asyncio.run(_run())
