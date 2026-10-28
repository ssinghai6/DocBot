"""DOCBOT-1622 — generic auth errors, per-account login throttle, breached-password check.

SQLite + in-memory rate-limit store; HIBP is always mocked. No network.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from cryptography.fernet import Fernet

if not os.environ.get("DB_ENCRYPTION_KEY"):
    os.environ["DB_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql://fake:fake@localhost:5432/testdb"

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, insert  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

import api.index as idx  # noqa: E402
from api import email_verification_service as evs  # noqa: E402
from api import oauth_service  # noqa: E402
from api.oauth_service import hash_password, is_password_breached  # noqa: E402
from api.utils import request_guard  # noqa: E402

PASSWORD = "Sup3r-secret-pass!"
TABLES = [idx.organizations_table, idx.users_table, idx.user_sessions_table, idx.audit_log_table]


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    path = tmp_path / "auth.db"
    sync_engine = create_engine(f"sqlite:///{path}")
    idx.metadata.create_all(sync_engine, tables=TABLES)
    with sync_engine.begin() as conn:
        conn.execute(insert(idx.users_table).values(
            id="alice", email="alice@example.com", name="Alice", provider="email",
            password_hash=hash_password(PASSWORD), email_verified=True, org_id="org-x",
        ))
        conn.execute(insert(idx.users_table).values(
            id="gina", email="gina@example.com", name="Gina", provider="google",
            password_hash=None, email_verified=True, org_id="org-x",
        ))
    sync_engine.dispose()
    factory = async_sessionmaker(create_async_engine(f"sqlite+aiosqlite:///{path}"), expire_on_commit=False)
    monkeypatch.setattr(idx, "async_session_factory", factory)
    monkeypatch.setattr(idx.request_guard, "is_demo_blocked", lambda *_a, **_k: False)
    monkeypatch.setattr(idx.request_guard, "check_ip_rate_limit", AsyncMock(return_value=None))
    monkeypatch.setattr(oauth_service, "is_password_breached", AsyncMock(return_value=False))
    request_guard.reset_state()
    yield TestClient(idx.app)
    request_guard.reset_state()


def _login(client: TestClient, email: str, password: str) -> Any:
    return client.post("/api/auth/login", json={"email": email, "password": password})


# ---------------------------------------------------------------------------
# Generic errors
# ---------------------------------------------------------------------------


def test_login_errors_identical_for_unknown_user_wrong_password_and_oauth_account(client):
    unknown = _login(client, "nobody@example.com", PASSWORD)
    wrong = _login(client, "alice@example.com", "wrong-password")
    oauth_only = _login(client, "gina@example.com", PASSWORD)
    assert unknown.status_code == wrong.status_code == oauth_only.status_code == 401
    assert unknown.json() == wrong.json() == oauth_only.json() == {"detail": "Invalid email or password."}


def test_login_success_still_works(client):
    resp = _login(client, "Alice@Example.com ", PASSWORD)
    assert resp.status_code == 200
    assert "docbot_session" in resp.cookies


def test_register_existing_email_is_generic(client, monkeypatch):
    monkeypatch.setattr(evs, "is_email_verification_enabled", lambda: False)
    resp = client.post("/api/auth/register", json={"email": "alice@example.com", "password": PASSWORD, "accepted_terms": True})
    assert resp.status_code == 400
    assert "exists" not in resp.json()["detail"].lower()
    assert resp.json()["detail"] == idx.REGISTER_FAILED_DETAIL


def test_register_existing_email_mirrors_verification_response(client, monkeypatch):
    monkeypatch.setattr(evs, "is_email_verification_enabled", lambda: True)
    resp = client.post("/api/auth/register", json={"email": "alice@example.com", "password": PASSWORD, "accepted_terms": True})
    assert resp.status_code == 202
    assert resp.json() == {"status": "verification_required", "email": "alice@example.com"}


# ---------------------------------------------------------------------------
# Per-account throttle
# ---------------------------------------------------------------------------


def test_sixth_failure_returns_429_even_with_correct_password(client):
    for _ in range(5):
        assert _login(client, "alice@example.com", "wrong-password").status_code == 401
    locked = _login(client, "ALICE@example.com", PASSWORD)
    assert locked.status_code == 429
    assert int(locked.headers["Retry-After"]) > 0
    # Other accounts are unaffected.
    assert _login(client, "nobody@example.com", "x").status_code == 401


def test_unknown_account_is_throttled_too(client):
    for _ in range(5):
        assert _login(client, "ghost@example.com", "x").status_code == 401
    assert _login(client, "ghost@example.com", "x").status_code == 429


def test_successful_logins_do_not_count(client):
    for _ in range(6):
        assert _login(client, "alice@example.com", PASSWORD).status_code == 200


# ---------------------------------------------------------------------------
# Breached-password check (HIBP k-anonymity)
# ---------------------------------------------------------------------------


def test_register_rejects_breached_password(client, monkeypatch):
    monkeypatch.setattr(oauth_service, "is_password_breached", AsyncMock(return_value=True))
    resp = client.post("/api/auth/register", json={"email": "new@example.com", "password": "password123", "accepted_terms": True})
    assert resp.status_code == 422
    assert resp.json()["detail"] == oauth_service.BREACHED_PASSWORD_MESSAGE


def _mock_hibp(monkeypatch: pytest.MonkeyPatch, *, text: str = "", exc: Exception | None = None) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    resp.raise_for_status.return_value = None
    http = MagicMock()
    http.get = AsyncMock(side_effect=exc) if exc else AsyncMock(return_value=resp)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=http)
    ctx.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(oauth_service.httpx, "AsyncClient", MagicMock(return_value=ctx))
    return http


async def test_hibp_match_detected_and_only_prefix_sent(monkeypatch):
    digest = hashlib.sha1(b"hunter22").hexdigest().upper()
    http = _mock_hibp(monkeypatch, text=f"0000000000000000000000000000000000A:0\r\n{digest[5:]}:42\r\n")
    assert await is_password_breached("hunter22") is True
    url = http.get.call_args.args[0]
    assert url.endswith("/range/" + digest[:5])
    assert digest[5:] not in url


async def test_hibp_no_match_or_padding_row_allowed(monkeypatch):
    digest = hashlib.sha1(b"hunter22").hexdigest().upper()
    _mock_hibp(monkeypatch, text=f"{digest[5:]}:0\r\nFFFF:3\r\n")
    assert await is_password_breached("hunter22") is False


async def test_hibp_outage_fails_open(monkeypatch):
    _mock_hibp(monkeypatch, exc=httpx.ConnectTimeout("down"))
    assert await is_password_breached("hunter22") is False


# ---------------------------------------------------------------------------
# DOCBOT-1707: Terms / Privacy acceptance is required server-side
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extra", [{}, {"accepted_terms": False}, {"accepted_terms": None}, {"accepted_terms": "yes"}])
def test_register_requires_accepted_terms(client, extra):
    body = {"email": "terms@example.com", "password": PASSWORD, **extra}
    resp = client.post("/api/auth/register", json=body)
    assert resp.status_code == 422
    if extra.get("accepted_terms", False) is False:
        assert resp.json()["detail"] == idx.TERMS_REQUIRED_DETAIL
    # Nothing was created: the same address still registers once accepted.
    ok = client.post(
        "/api/auth/register",
        json={"email": "terms@example.com", "password": PASSWORD, "accepted_terms": True},
    )
    assert ok.status_code in (200, 202)


def test_register_with_accepted_terms_creates_account(client, monkeypatch):
    monkeypatch.setattr(evs, "is_email_verification_enabled", lambda: False)
    resp = client.post(
        "/api/auth/register",
        json={"email": "terms@example.com", "password": PASSWORD, "accepted_terms": True},
    )
    assert resp.status_code == 200
    assert "docbot_session" in resp.cookies
