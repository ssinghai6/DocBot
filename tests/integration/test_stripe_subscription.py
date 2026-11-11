"""DOCBOT-1810 — Pro subscription lifecycle + dunning via signed Stripe webhooks.

Real SQLite (credit ledger + org_subscriptions), locally signed payloads, no network.
"""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import MetaData, select

from api import credit_service, subscription_service
from api.billing_models import LedgerEntryKind
from api.routes import billing_routes
from tests.integration.test_credit_service import _balance, _ledger, _make_store
from tests.integration.test_stripe_webhook import SECRET, _post, _sig

SUB = "sub_test_1"
T0 = int(time.time())


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", SECRET)
    monkeypatch.setenv("BILLING_FREE_MONTHLY_CREDITS", "500")
    yield
    credit_service._tables = None
    credit_service._session_factory = None
    credit_service._granted_period.clear()
    subscription_service._table = None
    subscription_service._session_factory = None


@pytest.fixture
async def store(tmp_path):
    engine, tables, factory = await _make_store(f"sqlite+aiosqlite:///{tmp_path / 'credits.db'}")
    md = MetaData()
    sub_table = subscription_service.register_subscription_table(md)
    async with engine.begin() as conn:
        await conn.run_sync(md.create_all)
    subscription_service.wire_subscription_store(sub_table, factory)
    yield tables, factory, sub_table
    await engine.dispose()


@pytest.fixture
async def client():
    app = FastAPI()
    app.include_router(billing_routes.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c


def _sub_event(etype: str, status: str = "active", created: int = T0, **extra) -> bytes:
    obj = {
        "id": SUB, "object": "subscription", "status": status, "customer": "cus_1",
        "metadata": {"org_id": "org-1", "user_id": "u-1"},
        "items": {"data": [{"current_period_end": T0 + 30 * 86400}]},
    }
    obj.update(extra)
    return json.dumps({"id": f"evt_{etype}_{created}", "type": etype, "created": created,
                       "data": {"object": obj}}).encode()


def _invoice_event(etype: str, inv_id: str = "in_1", reason: str = "subscription_create",
                   created: int = T0, meta: dict | None = None) -> bytes:
    # Post-2025-03 API shape: subscription under parent.subscription_details.
    obj = {
        "id": inv_id, "object": "invoice", "customer": "cus_1", "billing_reason": reason,
        "parent": {"subscription_details": {
            "subscription": SUB, "metadata": {"org_id": "org-1", "user_id": "u-1"} if meta is None else meta,
        }},
    }
    return json.dumps({"id": f"evt_{inv_id}_{created}", "type": etype, "created": created,
                       "data": {"object": obj}}).encode()


async def _send(client, payload: bytes):
    resp = await _post(client, payload, _sig(payload))
    assert resp.status_code == 200, resp.text
    return resp


async def _status(factory, sub_table):
    async with factory() as s:
        row = (await s.execute(select(sub_table).where(sub_table.c.org_id == "org-1"))).first()
    return row.status if row else None


async def _pro_grants(factory, tables):
    return [r for r in await _ledger(factory, tables) if r.kind == LedgerEntryKind.purchase.value]


async def test_full_lifecycle_active_past_due_canceled(store, client):
    tables, factory, sub_table = store
    with patch("api.funnel_service.record_paid") as paid:
        await _send(client, _sub_event("customer.subscription.created", "incomplete", T0))
        assert await _status(factory, sub_table) == "canceled"  # no access until paid
        await _send(client, _invoice_event("invoice.paid", created=T0 + 1))
        await _send(client, _sub_event("customer.subscription.updated", "active", T0 + 2))
    assert await _status(factory, sub_table) == "active"
    assert (await _balance(factory, tables)).balance_credits == 500 + 25_000
    paid.assert_called_once_with("u-1", "org-1", {"plan": "pro", "credits": 25_000})

    # Renewal fails: past_due but still Pro during Stripe's retry window.
    await _send(client, _invoice_event("invoice.payment_failed", "in_2", "subscription_cycle", T0 + 10))
    assert await _status(factory, sub_table) == "past_due"
    assert subscription_service.plan_for("past_due") == "pro"
    assert len(await _pro_grants(factory, tables)) == 1

    # Retries exhausted -> Stripe cancels -> free plan; granted credits kept.
    await _send(client, _sub_event("customer.subscription.deleted", "canceled", T0 + 20))
    assert await _status(factory, sub_table) == "canceled"
    assert subscription_service.plan_for("canceled") == "free"
    assert (await _balance(factory, tables)).balance_credits == 25_500


async def test_invoice_paid_is_idempotent_per_invoice(store, client):
    tables, factory, _ = store
    payload = _invoice_event("invoice.paid")
    await _send(client, payload)
    await _send(client, payload)  # Stripe redelivery
    await _send(client, _invoice_event("invoice.paid", "in_2", "subscription_cycle", T0 + 5))
    assert [r.ref for r in await _pro_grants(factory, tables)] == ["in_1", "in_2"]
    assert (await _balance(factory, tables)).balance_credits == 500 + 50_000


async def test_proration_invoice_grants_nothing(store, client):
    tables, factory, sub_table = store
    await _send(client, _invoice_event("invoice.paid", "in_p", "subscription_update"))
    assert await _pro_grants(factory, tables) == []
    assert await _status(factory, sub_table) == "active"


async def test_out_of_order_event_does_not_regress_state(store, client):
    _, factory, sub_table = store
    await _send(client, _invoice_event("invoice.paid", created=T0 + 50))
    await _send(client, _invoice_event("invoice.payment_failed", "in_old", "subscription_cycle", T0 + 10))
    assert await _status(factory, sub_table) == "active"


async def test_org_resolved_from_stored_subscription_when_metadata_missing(store, client):
    tables, factory, sub_table = store
    await _send(client, _sub_event("customer.subscription.created", "active", T0))
    await _send(client, _invoice_event("invoice.paid", "in_9", "subscription_cycle", T0 + 1, meta={}))
    assert [r.ref for r in await _pro_grants(factory, tables)] == ["in_9"]


async def test_unknown_subscription_and_one_off_invoice_are_noops(store, client):
    tables, factory, sub_table = store
    await _send(client, _invoice_event("invoice.paid", meta={}))  # never-seen subscription, no metadata
    one_off = json.dumps({"id": "evt_x", "type": "invoice.paid", "created": T0,
                          "data": {"object": {"id": "in_x", "billing_reason": "manual"}}}).encode()
    await _send(client, one_off)
    assert await _pro_grants(factory, tables) == []
    assert await _status(factory, sub_table) is None


async def test_subscription_checkout_session_not_credited_as_pack(store, client):
    tables, factory, _ = store
    payload = json.dumps({"id": "evt_cs", "type": "checkout.session.completed", "data": {"object": {
        "id": "cs_test_sub", "mode": "subscription", "payment_status": "paid",
        "client_reference_id": "org-1", "metadata": {"org_id": "org-1"},
    }}}).encode()
    await _send(client, payload)
    assert await _pro_grants(factory, tables) == []


async def test_unwired_store_returns_503_so_stripe_retries(store, client):
    subscription_service._table = None
    payload = _invoice_event("invoice.paid")
    resp = await _post(client, payload, _sig(payload))
    assert resp.status_code == 503
