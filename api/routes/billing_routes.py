"""Billing routes — DOCBOT-1803 (F-4: new routes live here).

* ``GET  /api/billing/packs``    — public pack catalogue (no Stripe ids).
* ``POST /api/billing/checkout`` — org admin (every signup is admin of its
  personal org, DOCBOT-1702). ``org_id`` comes from the session only.
* ``POST /admin/credits/grant``  — platform operator; manual credit for the
  Payment Link flow, idempotent on ``reference``.

Logic lives in ``api.billing_service`` / ``api.credit_service``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from api import billing_service, credit_service
from api.billing_models import (
    AdminCreditGrantRequest,
    AdminCreditGrantResponse,
    CheckoutSessionRequest,
    CheckoutSessionResponse,
    CreditPackCatalog,
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
            body.pack_id, org_id, getattr(user, "email", None), body.return_path
        )
    except billing_service.PackNotFound:
        raise HTTPException(status_code=404, detail="Unknown credit pack.") from None
    except billing_service.PaymentsDisabled:
        raise HTTPException(status_code=503, detail=PAYMENTS_DISABLED) from None
    except billing_service.PaymentProviderError:
        raise HTTPException(status_code=502, detail="Payment provider error. Try again shortly.") from None


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
