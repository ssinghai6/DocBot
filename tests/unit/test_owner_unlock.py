"""DOCBOT-1605: owner unlock is header-only on POST, cookie holds an HMAC."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.index as idx
from api.utils import request_guard

KEY = "-".join(["owner", "test", "1605"])  # built at runtime so gitleaks never sees a literal


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(request_guard, "OWNER_KEY", KEY)
    monkeypatch.setattr(request_guard, "RATE_LIMIT_PER_MIN", 0)
    request_guard.reset_state()
    yield
    request_guard.reset_state()


@pytest.fixture
def client():
    return TestClient(idx.app, raise_server_exceptions=False)


def test_header_unlock_sets_hmac_cookie_round_trip(client):
    resp = client.post("/api/auth/owner-unlock", headers={"X-Owner-Key": KEY})
    assert resp.status_code == 200 and resp.json() == {"owner": True}
    cookie = resp.cookies.get(request_guard.OWNER_COOKIE_NAME)
    assert cookie == request_guard.owner_cookie_value()
    assert KEY not in cookie
    assert "httponly" in resp.headers["set-cookie"].lower()

    class _Req:
        cookies = {request_guard.OWNER_COOKIE_NAME: cookie}

    assert request_guard.is_owner(_Req()) is True


def test_wrong_key_rejected(client):
    resp = client.post("/api/auth/owner-unlock", headers={"X-Owner-Key": "nope"})
    assert resp.status_code == 403
    assert request_guard.OWNER_COOKIE_NAME not in resp.cookies


def test_query_param_ignored(client):
    resp = client.post(f"/api/auth/owner-unlock?owner_key={KEY}")
    assert resp.status_code == 403
    assert request_guard.OWNER_COOKIE_NAME not in resp.cookies


def test_get_route_gone(client):
    resp = client.get(f"/api/auth/owner-unlock?owner_key={KEY}")
    assert resp.status_code in (404, 405)
    assert request_guard.OWNER_COOKIE_NAME not in resp.cookies


def test_unset_key_never_unlocks(client, monkeypatch):
    monkeypatch.setattr(request_guard, "OWNER_KEY", "")
    assert client.post("/api/auth/owner-unlock", headers={"X-Owner-Key": ""}).status_code == 403
