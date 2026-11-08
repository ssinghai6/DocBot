"""DOCBOT-1803 — packs, Stripe Checkout / Payment Link, manual credit grant.

No network: Stripe is reached only through a patched ``httpx.AsyncClient``;
the credit store is a mock (real-ledger idempotency lives in
tests/integration/test_credit_service.py).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import billing_service, credit_service
from api.routes import billing_routes

PACKS = [
    {"id": "starter", "name": "Starter", "credits": 10000, "price_usd": 10,
     "stripe_price_id": "price_Starter1"},
    {"id": "linkpack", "name": "Link", "credits": 5000, "price_usd": 5,
     "payment_link_url": "https://buy.stripe.com/test_abc123"},
]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("STRIPE_SECRET_KEY", "STRIPE_PACKS_JSON", "PLATFORM_ADMIN_USER_IDS", "AUTH_REQUIRED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FRONTEND_URL", "https://app.example.com")


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(billing_routes.router)
    return TestClient(app, raise_server_exceptions=False)


def _user(role: str = "admin", org_id: str = "org-1", uid: str = "u-1") -> SimpleNamespace:
    return SimpleNamespace(id=uid, role=role, org_id=org_id, email="buyer@example.com")


def _as(user: Any):
    return patch("api.auth_service.get_user_from_session", new_callable=AsyncMock, return_value=user)


def _post(client: TestClient, path: str, body: dict[str, Any]) -> httpx.Response:
    client.cookies.set("docbot_session", "tok")
    return client.post(path, json=body)


# ── packs ──────────────────────────────────────────────────────────────────


class TestPacks:
    def test_defaults_when_unset_are_not_purchasable(self, client):
        data = client.get("/api/billing/packs").json()
        assert [p["id"] for p in data["packs"]] == ["starter", "growth", "scale"]
        assert all(p["purchasable"] is False for p in data["packs"])

    def test_env_parsed_and_stripe_ids_hidden(self, client, monkeypatch):
        monkeypatch.setenv("STRIPE_PACKS_JSON", json.dumps(PACKS))
        resp = client.get("/api/billing/packs")
        data = resp.json()
        assert data["packs"][0]["price_cents"] == 1000
        # starter needs the secret key; the payment link pack is buyable without it
        assert [p["purchasable"] for p in data["packs"]] == [False, True]
        assert "price_Starter1" not in resp.text and "buy.stripe.com" not in resp.text

    def test_secret_key_makes_price_pack_purchasable(self, client, monkeypatch):
        monkeypatch.setenv("STRIPE_PACKS_JSON", json.dumps(PACKS))
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_x")
        assert all(p["purchasable"] for p in client.get("/api/billing/packs").json()["packs"])

    @pytest.mark.parametrize("raw", ["not json", "{}", "[]", '[{"id":"BAD"}]',
                                     json.dumps([PACKS[0], PACKS[0]])])
    def test_invalid_env_falls_back_to_defaults(self, monkeypatch, raw):
        monkeypatch.setenv("STRIPE_PACKS_JSON", raw)
        assert [p.id for p in billing_service.load_pack_configs()] == ["starter", "growth", "scale"]


# ── checkout ───────────────────────────────────────────────────────────────


class _FakeStripe:
    """Stands in for httpx.AsyncClient; records the single POST."""

    calls: list[dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> "_FakeStripe":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def post(self, url: str, data: dict[str, str], headers: dict[str, str]) -> httpx.Response:
        _FakeStripe.calls.append({"url": url, "data": data, "headers": headers})
        return httpx.Response(200, json={
            "id": "cs_test_abc123",
            "url": "https://checkout.stripe.com/c/pay/cs_test_abc123",
            "expires_at": 1_900_000_000,
        })


class TestCheckout:
    def test_anonymous_401(self, client):
        assert client.post("/api/billing/checkout", json={"pack_id": "starter"}).status_code == 401

    def test_non_admin_403(self, client, monkeypatch):
        monkeypatch.setenv("STRIPE_PACKS_JSON", json.dumps(PACKS))
        with _as(_user(role="analyst")):
            assert _post(client, "/api/billing/checkout", {"pack_id": "linkpack"}).status_code == 403

    def test_client_org_id_rejected(self, client):
        with _as(_user()):
            resp = _post(client, "/api/billing/checkout", {"pack_id": "starter", "org_id": "victim"})
        assert resp.status_code == 422

    def test_disabled_503(self, client):
        with _as(_user()):
            resp = _post(client, "/api/billing/checkout", {"pack_id": "starter"})
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Payments are not enabled yet."

    def test_unknown_pack_404(self, client):
        with _as(_user()):
            assert _post(client, "/api/billing/checkout", {"pack_id": "nope"}).status_code == 404

    def test_checkout_session_built_server_side(self, client, monkeypatch):
        monkeypatch.setenv("STRIPE_PACKS_JSON", json.dumps(PACKS))
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_secret")
        _FakeStripe.calls = []
        with _as(_user(org_id="org-42")), patch.object(billing_service.httpx, "AsyncClient", _FakeStripe):
            resp = _post(client, "/api/billing/checkout", {"pack_id": "starter", "return_path": "/chat"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["url"] == "https://checkout.stripe.com/c/pay/cs_test_abc123"
        assert resp.json()["mode"] == "checkout"
        (call,) = _FakeStripe.calls
        assert call["url"] == "https://api.stripe.com/v1/checkout/sessions"
        assert call["headers"]["Authorization"] == "Bearer sk_test_secret"
        d = call["data"]
        assert d["mode"] == "payment"
        assert d["line_items[0][price]"] == "price_Starter1"
        assert d["client_reference_id"] == "org-42" and d["metadata[org_id]"] == "org-42"
        assert d["success_url"] == "https://app.example.com/chat?checkout=success"
        assert d["cancel_url"] == "https://app.example.com/chat?checkout=cancel"
        assert d["customer_email"] == "buyer@example.com"

    def test_checkout_default_return_path(self, client, monkeypatch):
        monkeypatch.setenv("STRIPE_PACKS_JSON", json.dumps(PACKS))
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_secret")
        _FakeStripe.calls = []
        with _as(_user()), patch.object(billing_service.httpx, "AsyncClient", _FakeStripe):
            _post(client, "/api/billing/checkout", {"pack_id": "starter"})
        assert _FakeStripe.calls[0]["data"]["success_url"] == (
            "https://app.example.com/settings/billing?checkout=success"
        )

    def test_stripe_error_502(self, client, monkeypatch):
        monkeypatch.setenv("STRIPE_PACKS_JSON", json.dumps(PACKS))
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_secret")

        class _Down(_FakeStripe):
            async def post(self, *a: Any, **k: Any) -> httpx.Response:
                return httpx.Response(401, json={"error": {"message": "bad key"}})

        with _as(_user()), patch.object(billing_service.httpx, "AsyncClient", _Down):
            assert _post(client, "/api/billing/checkout", {"pack_id": "starter"}).status_code == 502

    def test_payment_link_appends_client_reference_id(self, client, monkeypatch):
        monkeypatch.setenv("STRIPE_PACKS_JSON", json.dumps(PACKS))
        with _as(_user(org_id="org-7")):
            resp = _post(client, "/api/billing/checkout", {"pack_id": "linkpack"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["mode"] == "payment_link"
        parts = urlsplit(body["url"])
        assert f"{parts.scheme}://{parts.netloc}{parts.path}" == "https://buy.stripe.com/test_abc123"
        q = parse_qs(parts.query)
        assert q["client_reference_id"] == ["org-7"]
        assert q["prefilled_email"] == ["buyer@example.com"]


# ── manual grant ───────────────────────────────────────────────────────────


GRANT = {"org_id": "org-9", "credits": 10000, "reference": "pi_3Abc"}


class TestManualGrant:
    def test_org_admin_not_platform_admin_403(self, client, monkeypatch):
        monkeypatch.setenv("PLATFORM_ADMIN_USER_IDS", "someone-else")
        with _as(_user(uid="u-1")):
            assert _post(client, "/admin/credits/grant", GRANT).status_code == 403

    def test_anonymous_401(self, client):
        assert client.post("/admin/credits/grant", json=GRANT).status_code == 401

    def test_platform_admin_grants_and_replay_is_duplicate(self, client, monkeypatch):
        monkeypatch.setenv("PLATFORM_ADMIN_USER_IDS", "u-op")
        monkeypatch.setattr(credit_service, "is_wired", lambda: True)
        grant = AsyncMock(side_effect=[
            credit_service.GrantResult(duplicate=False, balance_credits=10500),
            credit_service.GrantResult(duplicate=True, balance_credits=10500),
        ])
        monkeypatch.setattr(credit_service, "grant_purchase", grant)
        with _as(_user(uid="u-op")):
            first = _post(client, "/admin/credits/grant", GRANT).json()
            second = _post(client, "/admin/credits/grant", GRANT).json()
        assert first["duplicate"] is False and second["duplicate"] is True
        assert grant.await_args.args[:3] == ("org-9", 10000, "pi_3Abc")

    def test_unknown_org_404(self, client, monkeypatch):
        monkeypatch.setenv("PLATFORM_ADMIN_USER_IDS", "u-op")
        monkeypatch.setattr(credit_service, "is_wired", lambda: True)
        monkeypatch.setattr(credit_service, "grant_purchase", AsyncMock(side_effect=credit_service.UnknownOrg("org-9")))
        with _as(_user(uid="u-op")):
            assert _post(client, "/admin/credits/grant", GRANT).status_code == 404

    def test_unwired_store_503(self, client, monkeypatch):
        monkeypatch.setenv("PLATFORM_ADMIN_USER_IDS", "u-op")
        monkeypatch.setattr(credit_service, "is_wired", lambda: False)
        with _as(_user(uid="u-op")):
            assert _post(client, "/admin/credits/grant", GRANT).status_code == 503

    @pytest.mark.parametrize("bad", [{**GRANT, "credits": 0}, {**GRANT, "reference": ""}, {"org_id": "o"}])
    def test_validation_422(self, client, monkeypatch, bad):
        monkeypatch.setenv("PLATFORM_ADMIN_USER_IDS", "u-op")
        with _as(_user(uid="u-op")):
            assert _post(client, "/admin/credits/grant", bad).status_code == 422
