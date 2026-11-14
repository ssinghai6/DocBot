"""DOCBOT-2514: WRITE_FREEZE pauses every API write (owner too) during the cutover."""

import pytest
from fastapi.testclient import TestClient

import api.index as idx
from api.utils import request_guard


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(request_guard, "OWNER_KEY", "owner-key-for-tests")
    # No `with` block: skip lifespan (DB init); the middleware still runs.
    return TestClient(idx.app)


@pytest.mark.parametrize("method,path,frozen", [
    ("POST", "/api/demo/init", True),
    ("DELETE", "/api/session/x", True),
    ("POST", "/v1/hybrid/chat", True),
    ("POST", "/api/billing/webhook", True),  # Stripe retries on 503
    ("GET", "/api/does-not-exist-2514", False),
    ("OPTIONS", "/api/demo/init", False),
    ("POST", "/not-api", False),
])
def test_is_write_frozen(monkeypatch, method, path, frozen):
    monkeypatch.setattr(request_guard, "WRITE_FREEZE", True)
    assert request_guard.is_write_frozen(method, path) is frozen


def test_off_by_default(monkeypatch):
    monkeypatch.setattr(request_guard, "WRITE_FREEZE", False)
    assert request_guard.is_write_frozen("POST", "/api/demo/init") is False


def test_frozen_write_gets_503_even_for_owner(client, monkeypatch):
    monkeypatch.setattr(request_guard, "WRITE_FREEZE", True)
    client.cookies.set(request_guard.OWNER_COOKIE_NAME, request_guard.owner_cookie_value())
    r = client.post("/api/demo/init")
    assert r.status_code == 503
    assert r.headers["retry-after"] == "600"
    assert r.headers["content-security-policy"]  # security headers still applied


def test_reads_pass_during_freeze(client, monkeypatch):
    monkeypatch.setattr(request_guard, "WRITE_FREEZE", True)
    assert client.get("/api/does-not-exist-2514").status_code == 404
