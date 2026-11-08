"""DOCBOT-2305: every 401/403/429/503 the frontend gate classifier handles is JSON with `detail`.

The frontend (src/lib/httpGate.ts) reads `detail` (a string, or the structured
dict for 402/503 capacity) to show the backend's own message. A plain-text
body would surface as a generic error, so pin the wire shape here.
CI-safe: no network, no DB, app lifespan not run.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import api.index as idx
from api.utils import request_guard
from api.utils.spend_guard import CAPACITY_ERROR


@pytest.fixture(autouse=True)
def _guards(monkeypatch):
    for name in ("RATE_LIMIT_PER_MIN", "SESSION_BUDGET", "USER_DAILY_BUDGET",
                 "EDGAR_PER_IP_PER_HOUR", "EDGAR_PER_USER_PER_HOUR",
                 "SANDBOX_RUNS_PER_HOUR", "LOGIN_FAILURES_PER_WINDOW"):
        monkeypatch.setattr(request_guard, name, 0)
    monkeypatch.setattr(request_guard, "OWNER_KEY", "")
    monkeypatch.setattr(request_guard, "PUBLIC_DEMO_MODE", False)
    request_guard.reset_state()
    yield
    request_guard.reset_state()


def _assert_string_detail(resp, status: int) -> str:
    assert resp.status_code == status
    assert resp.headers["content-type"].startswith("application/json")
    detail = resp.json()["detail"]
    assert isinstance(detail, str) and detail
    return detail


# ── abuse_guard middleware short-circuits (real app) ─────────────────────────


def test_ip_rate_limit_429_is_json_detail(monkeypatch):
    monkeypatch.setattr(request_guard, "RATE_LIMIT_PER_MIN", 1)
    client = TestClient(idx.app, raise_server_exceptions=False)
    client.post("/api/__nope__")
    resp = client.post("/api/__nope__")
    assert "Rate limit exceeded" in _assert_string_detail(resp, 429)
    assert resp.headers.get("retry-after")


def test_public_demo_block_403_is_json_detail(monkeypatch):
    monkeypatch.setattr(request_guard, "PUBLIC_DEMO_MODE", True)
    client = TestClient(idx.app, raise_server_exceptions=False)
    resp = client.post("/api/upload")
    assert "disabled in the public demo" in _assert_string_detail(resp, 403)


# ── guard HTTPExceptions as FastAPI serialises them ──────────────────────────


def _guard_app() -> FastAPI:
    app = FastAPI()

    @app.post("/session-budget")
    async def session_budget() -> dict:
        for _ in range(2):
            await request_guard.enforce_session_budget("s1", "chat")
        return {}

    @app.post("/user-budget")
    async def user_budget() -> dict:
        for _ in range(2):
            await request_guard.enforce_user_budget("u1", "chat")
        return {}

    @app.post("/edgar")
    async def edgar(request: Request) -> dict:
        with patch.object(request_guard, "_caller_scope", AsyncMock(return_value=type("S", (), {"user_id": None})())), \
             patch.object(request_guard, "_meter"):
            for _ in range(2):
                await request_guard.check_edgar_limit(request, "edgar_search")
        return {}

    @app.post("/sandbox")
    async def sandbox(request: Request) -> dict:
        for _ in range(2):
            await request_guard.enforce_sandbox_quota(request, None)
        return {}

    @app.post("/login-throttle")
    async def login_throttle() -> dict:
        await request_guard.record_login_failure("a@b.co")
        await request_guard.check_login_throttle("a@b.co")
        return {}

    @app.post("/demo-cap")
    async def demo_cap() -> dict:
        with patch("api.usage_service.anonymous_spend_today_micro_usd", AsyncMock(return_value=10**12)):
            await request_guard.enforce_demo_spend_cap()
        return {}

    return app


@pytest.mark.parametrize("path, knob, needle", [
    ("/session-budget", "SESSION_BUDGET", "Session budget exhausted"),
    ("/user-budget", "USER_DAILY_BUDGET", "Daily free-tier limit"),
    ("/edgar", "EDGAR_PER_IP_PER_HOUR", "SEC EDGAR limit reached"),
    ("/sandbox", "SANDBOX_RUNS_PER_HOUR", "Sandbox limit reached"),
    ("/login-throttle", "LOGIN_FAILURES_PER_WINDOW", "Too many failed sign-in attempts"),
    ("/demo-cap", None, "sign in (free)"),
])
def test_guard_429s_are_json_string_detail(monkeypatch, path, knob, needle):
    if knob:
        monkeypatch.setattr(request_guard, knob, 1)
    resp = TestClient(_guard_app()).post(path)
    assert needle in _assert_string_detail(resp, 429)


def test_capacity_503_is_structured_detail():
    from fastapi import HTTPException

    app = FastAPI()

    @app.post("/cap")
    async def cap() -> dict:
        raise HTTPException(status_code=503, detail=CAPACITY_ERROR)

    resp = TestClient(app).post("/cap")
    assert resp.status_code == 503
    assert resp.json()["detail"] == {"error": "platform_capacity", "message": CAPACITY_ERROR["message"]}
