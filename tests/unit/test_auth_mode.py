"""DOCBOT-1711 — APP_AUTH_MODE open|required, demo principal, DEMO_DAILY_USD_CAP.

No DB, no network: the TestClient never runs the app lifespan, and every
lookup that would touch Postgres (session cookie -> user, demo session
existence, session ownership, the spend query) is monkeypatched.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import AsyncMock

import pytest
from cryptography.fernet import Fernet

if not os.environ.get("DB_ENCRYPTION_KEY"):
    os.environ["DB_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql://fake:fake@localhost:5432/testdb"

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import MetaData  # noqa: E402

import api.index as idx  # noqa: E402
from api import usage_service  # noqa: E402
from api.auth_service import is_auth_enforcement_active  # noqa: E402
from api.utils import caller_scope, request_guard  # noqa: E402

DEMO_SID = "demo-11111111-1111-1111-1111-111111111111"
DEMO_CID = "demo-22222222-2222-2222-2222-222222222222"
PASSED_GUARD = 418  # _require_session is patched to raise this: the auth guard let the call through


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.delenv("APP_AUTH_MODE", raising=False)
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    monkeypatch.delenv("DEMO_DAILY_USD_CAP", raising=False)
    monkeypatch.setattr(idx.request_guard, "is_demo_blocked", lambda *_a, **_k: False)
    monkeypatch.setattr(idx.request_guard, "check_ip_rate_limit", AsyncMock(return_value=None))
    monkeypatch.setattr(caller_scope, "is_server_demo_session", AsyncMock(side_effect=lambda sid: sid == DEMO_SID))
    monkeypatch.setattr(usage_service, "anonymous_spend_today_micro_usd", AsyncMock(return_value=0))

    async def _past_guard(*_a: object, **_k: object) -> None:
        raise HTTPException(status_code=PASSED_GUARD, detail="past auth guard")

    monkeypatch.setattr(idx, "_require_session", _past_guard)
    monkeypatch.setattr(idx, "_require_connection", AsyncMock(return_value=None))
    request_guard.reset_state()
    yield TestClient(idx.app, raise_server_exceptions=False)
    request_guard.reset_state()


def _login(monkeypatch: pytest.MonkeyPatch) -> None:
    user = SimpleNamespace(id="user-1", org_id="org-1", role="analyst")
    monkeypatch.setattr("api.rbac_service.resolve_session_user", AsyncMock(return_value=user))


def _hybrid(client: TestClient, sid: str = DEMO_SID, cid: str = DEMO_CID) -> int:
    return client.post(
        "/api/hybrid/chat", json={"question": "q", "session_id": sid, "connection_id": cid}
    ).status_code


# ---------------------------------------------------------------------------
# Mode resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode,legacy,expected", [
    (None, None, False),          # default: open
    ("open", None, False),
    ("required", None, True),
    ("REQUIRED", None, True),
    (None, "true", True),         # AUTH_REQUIRED alias
    ("open", "true", True),       # stricter setting wins
    ("requird", None, True),      # typo fails closed
])
def test_auth_mode_resolution(
    monkeypatch: pytest.MonkeyPatch, mode: str | None, legacy: str | None, expected: bool
) -> None:
    for name, value in (("APP_AUTH_MODE", mode), ("AUTH_REQUIRED", legacy)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert is_auth_enforcement_active() is expected


# ---------------------------------------------------------------------------
# Required mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env", [("APP_AUTH_MODE", "required"), ("AUTH_REQUIRED", "true")])
def test_required_anon_demo_chat_allowed_own_data_refused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, env: tuple[str, str]
) -> None:
    monkeypatch.setenv(*env)
    assert _hybrid(client) == PASSED_GUARD
    for method, path, kwargs in [
        ("POST", "/api/upload", {"files": [("files", ("a.pdf", b"%PDF", "application/pdf"))]}),
        ("POST", "/api/db/connect", {"json": {"dialect": "postgresql", "host": "h", "database": "d"}}),
        ("POST", "/api/db/upload/csv", {"files": [("file", ("a.csv", b"a,b\n1,2", "text/csv"))]}),
        ("GET", "/api/sessions", {}),
        ("POST", "/api/edgar/ingest", {"json": {"cik": "1", "accession_number": "a", "primary_document": "p"}}),
    ]:
        resp = client.request(method, path, **kwargs)
        assert resp.status_code == 401, (path, resp.status_code, resp.text[:200])
        assert resp.json()["detail"].startswith("Sign in (free)"), path


def test_required_demo_prefix_must_exist_server_side(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_AUTH_MODE", "required")
    forged = "demo-not-a-real-session"
    assert _hybrid(client, sid=forged) == 401
    assert _hybrid(client, cid="user-connection-id") == 401  # demo session + own DB
    assert client.post("/api/chat", json={"session_id": forged, "message": "q"}).status_code == 401
    assert client.post("/api/chat", json={"session_id": DEMO_SID, "message": "q"}).status_code == PASSED_GUARD


def test_required_demo_id_cannot_be_smuggled_in_query(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """DOCBOT-4014: a demo id in the query string must not vouch for a POST that uses other ids."""
    monkeypatch.setenv("APP_AUTH_MODE", "required")
    upload = client.post(
        f"/api/upload?session_id={DEMO_SID}", files=[("files", ("a.pdf", b"%PDF", "application/pdf"))]
    )
    assert upload.status_code == 401, upload.text[:200]
    own = client.post(f"/api/chat?session_id={DEMO_SID}", json={"session_id": "my-own-session", "message": "q"})
    assert own.status_code == 401, own.text[:200]
    assert _hybrid(client, sid="my-own-session", cid=DEMO_CID) == 401
    legit = client.post(f"/api/chat?session_id={DEMO_SID}", json={"session_id": DEMO_SID, "message": "q"})
    assert legit.status_code == PASSED_GUARD
    assert client.post("/api/chat", json={"session_id": DEMO_SID, "message": "q"}).status_code == PASSED_GUARD


def test_required_anon_demo_init_ok(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_AUTH_MODE", "required")
    monkeypatch.setattr(
        "api.demo_service.init_demo_session",
        AsyncMock(return_value={"session_id": DEMO_SID, "connection_id": DEMO_CID}),
    )
    resp = client.post("/api/demo/init")
    assert resp.status_code == 200, resp.text
    assert resp.json()["session_id"] == DEMO_SID


def test_required_logged_in_user_ok(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_AUTH_MODE", "required")
    _login(monkeypatch)
    assert _hybrid(client, sid="user-session", cid="user-conn") == PASSED_GUARD


def test_open_mode_unchanged(client: TestClient) -> None:
    assert _hybrid(client, sid="user-session", cid="user-conn") == PASSED_GUARD
    # Upload reaches validation (no files -> 422), never an auth 401.
    assert client.post("/api/upload").status_code == 422


# ---------------------------------------------------------------------------
# DEMO_DAILY_USD_CAP
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["open", "required"])
def test_demo_cap_exceeded_blocks_anon_demo_only(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("APP_AUTH_MODE", mode)
    monkeypatch.setenv("DEMO_DAILY_USD_CAP", "10")
    monkeypatch.setattr(usage_service, "anonymous_spend_today_micro_usd", AsyncMock(return_value=10_000_001))
    init = client.post("/api/demo/init")
    assert init.status_code == 429
    assert init.json()["detail"] == request_guard.DEMO_BUSY_DETAIL
    assert _hybrid(client) == 429
    if mode == "open":  # anonymous non-demo traffic is not the demo
        assert _hybrid(client, sid="own-session", cid="own-conn") == PASSED_GUARD
    _login(monkeypatch)
    assert _hybrid(client) == PASSED_GUARD


def test_demo_cap_under_limit_or_disabled(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(usage_service, "anonymous_spend_today_micro_usd", AsyncMock(return_value=10_000_000))
    assert _hybrid(client) == PASSED_GUARD  # exactly at the cap is not over it
    monkeypatch.setattr(usage_service, "anonymous_spend_today_micro_usd", AsyncMock(return_value=10**12))
    monkeypatch.setenv("DEMO_DAILY_USD_CAP", "0")
    assert _hybrid(client) == PASSED_GUARD


class _FakeResult:
    def __init__(self, value: int) -> None:
        self._value = value

    def scalar_one(self) -> int:
        return self._value


class _FakeSession:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def execute(self, stmt: object) -> _FakeResult:
        self._calls.append(str(stmt))
        return _FakeResult(1234)


@pytest.mark.asyncio
async def test_anonymous_spend_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    clock = [1000.0]
    monkeypatch.setattr(usage_service, "_usage_table", usage_service.register_usage_events_table(MetaData()))
    monkeypatch.setattr(usage_service, "_async_session_factory", lambda: _FakeSession(calls))
    monkeypatch.setattr(usage_service.time, "monotonic", lambda: clock[0])
    usage_service.reset_anon_spend_cache()
    try:
        assert await usage_service.anonymous_spend_today_micro_usd() == 1234
        clock[0] += 59
        assert await usage_service.anonymous_spend_today_micro_usd() == 1234
        assert len(calls) == 1
        assert "org_id IS NULL" in calls[0] and "user_id IS NULL" in calls[0]
        clock[0] += 2
        await usage_service.anonymous_spend_today_micro_usd()
        assert len(calls) == 2
    finally:
        usage_service.reset_anon_spend_cache()
