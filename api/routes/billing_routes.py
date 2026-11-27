"""Billing routes — DOCBOT-1803 (F-4: new routes live here).

* ``GET  /api/billing/packs``    — public pack catalogue (no Stripe ids).
* ``POST /api/billing/checkout`` — org admin (every signup is admin of its
  personal org, DOCBOT-1702). ``org_id`` comes from the session only.
* ``POST /api/billing/webhook`` — Stripe webhook (DOCBOT-2334). No session;
  authenticated by ``Stripe-Signature`` only. Exempt from the IP limiter.
* ``POST /api/billing/subscribe`` / ``GET /api/billing/subscription`` — Pro
  plan checkout (mode=subscription) and status, org admin (DOCBOT-1810).
* ``POST /admin/credits/grant``  — platform operator; manual credit for the
  Payment Link flow, idempotent on ``reference``.

Logic lives in ``api.billing_service`` / ``api.credit_service``.
"""

from __future__ import annotations

from datetime import timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.exc import SQLAlchemyError

from api import billing_service, credit_service, funnel_service, subscription_service
from api.billing_models import (
    AdminCreditGrantRequest,
    AdminCreditGrantResponse,
    CheckoutSessionRequest,
    CheckoutSessionResponse,
    CreditPackCatalog,
    SubscriptionCheckoutRequest,
    SubscriptionStatus,
)
from api.rbac_service import require_admin, require_admin_org_id, require_platform_admin

router = APIRouter(tags=["billing"])

# Module-level so tests/unit/test_admin_auth_guard.py can match them by identity.
_rbac_admin = Depends(require_admin())
_rbac_platform_admin = Depends(require_platform_admin())

PAYMENTS_DISABLED = "Payments are not enabled yet."


@router.get("/api/billing/packs", response_model=CreditPackCatalog)
async def list_packs() -> CreditPackCatalog:
    return billing_service.catalog()


@router.post("/api/billing/checkout", response_model=CheckoutSessionResponse)
async def create_checkout(body: CheckoutSessionRequest, user: Any = _rbac_admin) -> CheckoutSessionResponse:
    org_id = require_admin_org_id(user)
    try:
        return await billing_service.checkout_url(
            body.pack_id, org_id, getattr(user, "email", None), body.return_path, getattr(user, "id", None)
        )
    except billing_service.PackNotFound:
        raise HTTPException(status_code=404, detail="Unknown credit pack.") from None
    except billing_service.PaymentsDisabled:
        raise HTTPException(status_code=503, detail=PAYMENTS_DISABLED) from None
    except billing_service.PaymentProviderError:
        raise HTTPException(status_code=502, detail="Payment provider error. Try again shortly.") from None


@router.post("/api/billing/subscribe", response_model=CheckoutSessionResponse)
async def create_subscription_checkout(
    body: SubscriptionCheckoutRequest, user: Any = _rbac_admin
) -> CheckoutSessionResponse:
    org_id = require_admin_org_id(user)
    current = await subscription_service.get_subscription(org_id)
    if current and subscription_service.plan_for(current["status"]) == "pro":
        raise HTTPException(status_code=409, detail="This workspace is already on Pro.")
    try:
        return await billing_service.subscription_checkout_url(
            org_id, getattr(user, "email", None), body.return_path, getattr(user, "id", None)
        )
    except billing_service.PaymentsDisabled:
        raise HTTPException(status_code=503, detail=PAYMENTS_DISABLED) from None
    except billing_service.PaymentProviderError:
        raise HTTPException(status_code=502, detail="Payment provider error. Try again shortly.") from None


@router.get("/api/billing/subscription", response_model=SubscriptionStatus)
async def get_subscription(user: Any = _rbac_admin) -> SubscriptionStatus:
    row = await subscription_service.get_subscription(require_admin_org_id(user))
    if row is None:
        return SubscriptionStatus(plan="free")
    period_end = row["current_period_end"]
    if period_end is not None and period_end.tzinfo is None:  # SQLite drops tzinfo
        period_end = period_end.replace(tzinfo=timezone.utc)
    return SubscriptionStatus(
        plan=subscription_service.plan_for(row["status"]), status=row["status"], current_period_end=period_end
    )


async def _handle_subscription_event(event: dict) -> None:
    sub_event = subscription_service.parse_event(event)
    if sub_event is None:
        return
    if not (credit_service.is_wired() and subscription_service.is_wired()):
        raise HTTPException(status_code=503, detail="Credit store unavailable.")  # Stripe retries
    try:
        await subscription_service.apply_event(sub_event)
    except credit_service.UnknownOrg:
        billing_service.logger.error("billing webhook: unknown org for subscription %s", sub_event.subscription_id)
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="Credit store unavailable.") from None


@router.post(billing_service.WEBHOOK_PATH)
async def stripe_webhook(request: Request) -> dict[str, bool]:
    payload = await request.body()
    try:
        event = billing_service.verify_webhook(payload, request.headers.get("stripe-signature"))
    except billing_service.WebhookNotConfigured:
        raise HTTPException(status_code=503, detail="Webhook not configured.") from None
    except billing_service.InvalidSignature:
        raise HTTPException(status_code=400, detail="Invalid signature.") from None
    if event.get("type") in subscription_service.SUBSCRIPTION_EVENTS:
        await _handle_subscription_event(event)
        return {"received": True}
    grant = billing_service.checkout_grant(event)
    if grant is None:
        return {"received": True}
    if not credit_service.is_wired():
        raise HTTPException(status_code=503, detail="Credit store unavailable.")  # Stripe retries
    try:
        result = await credit_service.grant_purchase(
            grant.org_id, grant.credits, grant.session_id, f"Stripe checkout ({grant.pack_id or 'pack'})"
        )
    except credit_service.UnknownOrg:
        # Retrying can't fix it; ack so Stripe stops, operator reconciles from the log.
        billing_service.logger.error("billing webhook: unknown org for session %s", grant.session_id)
        return {"received": True}
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="Credit store unavailable.") from None
    if not result.duplicate:
        funnel_service.record_paid(
            grant.user_id, grant.org_id, {"pack_id": grant.pack_id, "credits": grant.credits}
        )
    return {"received": True}


@router.post("/admin/credits/grant", response_model=AdminCreditGrantResponse)
async def grant_credits(body: AdminCreditGrantRequest, _user: Any = _rbac_platform_admin) -> AdminCreditGrantResponse:
    if not credit_service.is_wired():
        raise HTTPException(status_code=503, detail="Credit store unavailable.")
    try:
        result = await credit_service.grant_purchase(
            body.org_id, body.credits, body.reference, f"Manual grant ({body.reference})"
        )
    except credit_service.UnknownOrg:
        raise HTTPException(status_code=404, detail="Unknown org.") from None
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="Credit store unavailable.") from None
    return AdminCreditGrantResponse(
        org_id=body.org_id,
        credits=body.credits,
        reference=body.reference,
        duplicate=result.duplicate,
        balance_credits=result.balance_credits,
    )
