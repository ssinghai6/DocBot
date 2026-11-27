"""DOCBOT-3901: every /v1 error uses the envelope; key in URL -> 400 key_in_url.

No network, no DB: API-key validation and the connector store are mocked.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet

if not os.environ.get("DB_ENCRYPTION_KEY"):
    os.environ["DB_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql://fake:fake@localhost:5432/testdb"

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import api.index as idx  # noqa: E402
from api import api_key_service  # noqa: E402
from api.api_key_service import APIKeyRecord  # noqa: E402
from api.utils import request_guard, v1_contract  # noqa: E402

TYPES = {"invalid_request", "authentication", "permission", "not_found", "rate_limit", "server"}


def _assert_envelope(resp, status: int, type_: str, code: str) -> dict:
    assert resp.status_code == status, resp.text
    err = resp.json()["error"]
    assert set(err) == {"type", "code", "message", "request_id"}
    assert (err["type"], err["code"]) == (type_, code)
    assert err["type"] in TYPES and err["message"]
    assert err["request_id"] == resp.headers["X-Request-Id"]
    assert err["request_id"].startswith("req_")
    return err


# ── Real app: auth errors, key_in_url, middleware short-circuits ─────────────


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(request_guard, "check_ip_rate_limit", AsyncMock(return_value=None))
    return TestClient(idx.app, raise_server_exceptions=False)


def test_missing_key_is_authentication_envelope(client: TestClient) -> None:
    _assert_envelope(client.get("/v1/connectors"), 401, "authentication", "missing_key")


def test_invalid_key_is_authentication_envelope(client: TestClient) -> None:
    with patch.object(api_key_service, "validate_api_key", new=AsyncMock(return_value=None)):
        resp = client.get("/v1/connectors", headers={"Authorization": "Bearer dk_live_nope"})
    _assert_envelope(resp, 401, "authentication", "invalid_key")


def test_missing_scope_is_permission_envelope(client: TestClient) -> None:
    rec = APIKeyRecord(id="k1", org_id="o1", key_prefix="dk_live_abc", name="k", scopes=["hybrid:read"])
    with patch.object(api_key_service, "validate_api_key", new=AsyncMock(return_value=rec)):
        resp = client.get("/v1/connectors", headers={"Authorization": "Bearer dk_live_abc"})
    _assert_envelope(resp, 403, "permission", "missing_scope")


@pytest.mark.parametrize("query", [
    "api_key=dk_live_abc", "key=x", "token=x", "access_token=x", "foo=dk_test_abc",
])
def test_key_in_query_string_is_400_key_in_url(client: TestClient, query: str) -> None:
    resp = client.get(f"/v1/connectors?{query}", headers={"Authorization": "Bearer dk_live_abc"})
    _assert_envelope(resp, 400, "invalid_request", "key_in_url")


def test_ordinary_query_string_is_not_key_in_url() -> None:
    assert not v1_contract.key_in_query(b"limit=10&cursor=abc")


def test_write_freeze_short_circuit_uses_envelope_on_v1(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(request_guard, "is_write_frozen", lambda *_a: True)
    err = _assert_envelope(client.post("/v1/hybrid/chat", json={}), 503, "server", "service_unavailable")
    assert "maintenance" in err["message"]
    # /api keeps the legacy shape.
    assert "detail" in client.post("/api/chat", json={}).json()


def test_ip_rate_limit_short_circuit_uses_envelope_on_v1(client: TestClient, monkeypatch) -> None:
    async def limited(_req):
        raise HTTPException(status_code=429, detail="Rate limit exceeded", headers={"Retry-After": "7"})

    monkeypatch.setattr(request_guard, "check_ip_rate_limit", limited)
    resp = client.post("/v1/hybrid/chat", json={})
    _assert_envelope(resp, 429, "rate_limit", "rate_limited")
    assert resp.headers["Retry-After"] == "7"


def test_unknown_v1_path_is_not_found_envelope(client: TestClient) -> None:
    _assert_envelope(client.get("/v1/does-not-exist"), 404, "not_found", "resource_not_found")


def test_api_paths_keep_detail_shape(client: TestClient) -> None:
    resp = client.get("/api/does-not-exist-3901")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Not Found"}
    assert "X-Request-Id" not in resp.headers


# ── Handler mapping on a minimal app (shared contract for other agents) ──────


class _Body(BaseModel):
    n: int


def _mini_app() -> TestClient:
    app = FastAPI()
    v1_contract.install_v1_contract(app)

    @app.get("/v1/raise/{status}/{code}")
    async def raise_dict(status: int, code: str):
        raise HTTPException(status_code=status, detail={"code": code, "message": f"boom {code}"})

    @app.get("/v1/plain")
    async def raise_plain():
        raise HTTPException(status_code=502, detail="An internal error occurred.")

    @app.post("/v1/body")
    async def body(b: _Body):
        return {"n": b.n}

    @app.post("/api/body")
    async def api_body(b: _Body):
        return {"n": b.n}

    @app.get("/v1/crash")
    async def crash():
        raise RuntimeError("secret internals")

    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("status,type_,code", [
    (400, "invalid_request", "invalid_body"),
    (401, "authentication", "expired_key"),
    (402, "invalid_request", "insufficient_credits"),
    (403, "permission", "test_key_limit"),
    (404, "not_found", "resource_not_found"),
    (409, "invalid_request", "idempotency_in_progress"),
    (429, "rate_limit", "test_allowance_exhausted"),
    (500, "server", "internal_error"),
    (503, "server", "service_unavailable"),
])
def test_dict_detail_keeps_code_and_maps_status(status: int, type_: str, code: str) -> None:
    err = _assert_envelope(_mini_app().get(f"/v1/raise/{status}/{code}"), status, type_, code)
    assert err["message"] == f"boom {code}"


def test_string_detail_gets_default_code() -> None:
    _assert_envelope(_mini_app().get("/v1/plain"), 502, "server", "pipeline_failed")


def test_validation_error_is_envelope_on_v1_only() -> None:
    client = _mini_app()
    err = _assert_envelope(client.post("/v1/body", json={"n": "x"}), 422, "invalid_request", "validation_failed")
    assert err["message"].startswith("n:")
    assert isinstance(client.post("/api/body", json={"n": "x"}).json()["detail"], list)


def test_unhandled_exception_is_server_envelope_without_internals() -> None:
    err = _assert_envelope(_mini_app().get("/v1/crash"), 500, "server", "internal_error")
    assert "secret" not in err["message"]


def test_quota_payload_error_field_becomes_code() -> None:
    req = MagicMock()
    req.state.request_id = "req_x"
    resp = v1_contract.v1_error_response(req, 402, {"error": "account_frozen", "message": "Frozen."})
    assert resp.status_code == 402
    assert b'"code":"account_frozen"' in resp.body
