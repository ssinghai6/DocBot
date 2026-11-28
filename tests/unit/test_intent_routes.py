"""DOCBOT-2728 — POST /api/route-intent: auth guard, request/response models,
and the service wiring. The service itself is stubbed (no DB, no network)."""

from __future__ import annotations

import os

import pytest
from cryptography.fernet import Fernet

if not os.environ.get("DB_ENCRYPTION_KEY"):
    os.environ["DB_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql://fake:fake@localhost:5432/testdb"

from fastapi.testclient import TestClient  # noqa: E402

import api.index as idx  # noqa: E402
from api.intent_models import RouteIntentRequest, RouteIntentResponse  # noqa: E402
from api.routes import intent_routes  # noqa: E402

READY = RouteIntentResponse(route="autopilot", source="sql", task="forecast", data_ready=True, confidence=0.8)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, list[RouteIntentRequest]]:
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    monkeypatch.setenv("APP_AUTH_MODE", "open")
    seen: list[RouteIntentRequest] = []

    async def fake_route_intent(req, scope, **_tables):
        seen.append(req)
        return READY

    monkeypatch.setattr(intent_routes, "route_intent", fake_route_intent)
    return TestClient(idx.app), seen


def test_route_registered_with_guard() -> None:
    paths = {getattr(r, "path", None) for r in idx.app.routes}
    paths |= {getattr(r, "path", None) for r in intent_routes.router.routes}
    assert "/api/route-intent" in paths


def test_open_mode_returns_router_verdict(env) -> None:
    client, seen = env
    resp = client.post("/api/route-intent", json={
        "question": "where will diesel be in 2030?",
        "history": [{"role": "user", "content": "hi"}],
        "connection_id": "c1",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json() == READY.model_dump()
    assert seen[0].connection_id == "c1" and seen[0].session_id is None


@pytest.mark.parametrize("body", [{"question": ""}, {}, {"question": "x" * 4001}])
def test_rejects_bad_request(env, body: dict) -> None:
    assert env[0].post("/api/route-intent", json=body).status_code == 422


def test_anonymous_gets_401_when_auth_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    resp = TestClient(idx.app).post("/api/route-intent", json={"question": "forecast revenue"})
    assert resp.status_code == 401


def test_response_model_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError):
        RouteIntentResponse(route="slow", source="sql", task="forecast", data_ready=True, confidence=0.5)
    with pytest.raises(ValueError):
        RouteIntentResponse(route="fast", source="sql", task="lookup", data_ready=True, confidence=1.5)
