"""DOCBOT-2334 — Stripe webhook: signature check + idempotent credit grant.

Real SQLite ledger (``UNIQUE(kind, ref)``), signed payloads built locally with a
runtime-generated secret. No network.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI

from api import billing_service, credit_service
from api.billing_models import LedgerEntryKind
from api.routes import billing_routes
from tests.integration.test_credit_service import _balance, _ledger, _make_store

SECRET = "whsec_" + secrets.token_hex(16)
SESSION_ID = "cs_test_" + secrets.token_hex(8)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", SECRET)
    monkeypatch.setenv("BILLING_FREE_MONTHLY_CREDITS", "500")
    yield
    credit_service._tables = None
    credit_service._session_factory = None
    credit_service._granted_period.clear()


@pytest.fixture
async def store(tmp_path):
    engine, tables, factory = await _make_store(f"sqlite+aiosqlite:///{tmp_path / 'credits.db'}")
    yield tables, factory
    await engine.dispose()


@pytest.fixture
async def client():
    app = FastAPI()
    app.include_router(billing_routes.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c


def _event(event_type: str = "checkout.session.completed", **obj) -> bytes:
    session = {
        "id": SESSION_ID,
        "payment_status": "paid",
        "client_reference_id": "org-1",
        "metadata": {"org_id": "org-1", "pack_id": "starter", "credits": "10000", "user_id": "u-1"},
    }
    session.update(obj)
    return json.dumps({"id": "evt_1", "type": event_type, "data": {"object": session}}).encode()


def _sig(payload: bytes, ts: int | None = None, secret: str = SECRET) -> str:
    ts = int(time.time()) if ts is None else ts
    mac = hmac.new(secret.encode(), f"{ts}.".encode() + payload, hashlib.sha256).hexdigest()
    return f"t={ts},v1={mac}"


async def _post(client, payload: bytes, sig: str | None):
    headers = {"Content-Type": "application/json"}
    if sig is not None:
        headers["Stripe-Signature"] = sig
    return await client.post(billing_service.WEBHOOK_PATH, content=payload, headers=headers)


async def _purchases(factory, tables):
    return [r for r in await _ledger(factory, tables) if r.kind == LedgerEntryKind.purchase.value]


async def test_valid_signature_grants_credits_and_records_paid(store, client):
    tables, factory = store
    payload = _event()
    with patch("api.funnel_service.record_paid") as paid:
        resp = await _post(client, payload, _sig(payload))
    assert resp.status_code == 200
    assert (await _balance(factory, tables)).balance_credits == 10_500  # 500 free + 10k pack
    rows = await _purchases(factory, tables)
    assert [r.ref for r in rows] == [SESSION_ID]
    paid.assert_called_once_with("u-1", "org-1", {"pack_id": "starter", "credits": 10000})


async def test_any_v1_signature_may_match(store, client):
    payload = _event()
    ts = int(time.time())
    good = _sig(payload, ts).split(",")[1]
    resp = await _post(client, payload, f"t={ts},v1={'0' * 64},{good}")
    assert resp.status_code == 200


@pytest.mark.parametrize(
    "sig",
    [
        None,
        "",
        "t=abc,v1=deadbeef",
        "v1=" + "0" * 64,
    ],
)
async def test_missing_or_malformed_signature_is_400(store, client, sig):
    tables, factory = store
    resp = await _post(client, _event(), sig)
    assert resp.status_code == 400
    assert await _purchases(factory, tables) == []


async def test_wrong_secret_or_tampered_body_is_400(store, client):
    tables, factory = store
    payload = _event()
    other = "whsec_" + secrets.token_hex(16)
    assert (await _post(client, payload, _sig(payload, secret=other))).status_code == 400
    tampered = _event(metadata={"org_id": "org-1", "credits": "999999"})
    assert (await _post(client, tampered, _sig(payload))).status_code == 400
    assert await _purchases(factory, tables) == []


async def test_stale_timestamp_is_400(store, client):
    tables, factory = store
    payload = _event()
    stale = int(time.time()) - billing_service.WEBHOOK_TOLERANCE_SEC - 5
    resp = await _post(client, payload, _sig(payload, stale))
    assert resp.status_code == 400
    assert await _purchases(factory, tables) == []


async def test_replay_is_idempotent(store, client):
    tables, factory = store
    payload = _event()
    with patch("api.funnel_service.record_paid") as paid:
        first = await _post(client, payload, _sig(payload))
        again = await _post(client, payload, _sig(payload))
        # async_payment_succeeded for the same session must not double-grant either
        async_payload = _event("checkout.session.async_payment_succeeded")
        third = await _post(client, async_payload, _sig(async_payload))
    assert first.status_code == again.status_code == third.status_code == 200
    assert (await _balance(factory, tables)).balance_credits == 10_500
    assert len(await _purchases(factory, tables)) == 1
    paid.assert_called_once()


@pytest.mark.parametrize(
    "payload",
    [
        _event("payment_intent.succeeded"),
        _event("customer.created"),
        _event(payment_status="unpaid"),  # delayed method; async_payment_succeeded follows
        _event(metadata={}, client_reference_id="org-1"),  # Payment Link: manual grant
    ],
)
async def test_ignored_events_are_200_noop(store, client, payload):
    tables, factory = store
    resp = await _post(client, payload, _sig(payload))
    assert resp.status_code == 200
    assert await _purchases(factory, tables) == []


async def test_unknown_org_acks_without_ledger_row(store, client):
    tables, factory = store
    payload = _event(metadata={"org_id": "org-missing", "credits": "100"})
    resp = await _post(client, payload, _sig(payload))
    assert resp.status_code == 200
    assert await _ledger(factory, tables, org="org-missing") == []


async def test_unset_secret_refuses(store, client, monkeypatch):
    tables, factory = store
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET")
    payload = _event()
    resp = await _post(client, payload, _sig(payload))
    assert resp.status_code == 503
    assert await _purchases(factory, tables) == []

