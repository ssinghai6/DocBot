"""Credit packs + Stripe Checkout / Payment Link URLs — DOCBOT-1803 (EPIC-18).

Dormant until configured. Two ways to sell a pack, chosen per pack:

* **Checkout** — ``STRIPE_SECRET_KEY`` set and the pack has ``stripe_price_id``:
  a Checkout Session is created server-side (plain httpx, no SDK) with
  ``client_reference_id`` and ``metadata.org_id`` taken from the caller's
  session, never from the request body.
* **Payment Link** — no secret key but the pack has ``payment_link_url``: the
  link is returned with ``client_reference_id=<org_id>`` appended so the
  operator can match the Stripe payment to an org and credit it with
  ``POST /admin/credits/grant``.

Neither -> :class:`PaymentsDisabled` (HTTP 503). Packs come from
``STRIPE_PACKS_JSON``; unset/invalid -> :data:`DEFAULT_PACKS`, which are never
purchasable ("coming soon").
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from pydantic import ValidationError

from api.billing_models import CheckoutSessionResponse, CreditPackCatalog, CreditPackConfig
from api.credit_service import free_monthly_credits

logger = logging.getLogger(__name__)

STRIPE_CHECKOUT_URL = "https://api.stripe.com/v1/checkout/sessions"
WEBHOOK_PATH = "/api/billing/webhook"
DEFAULT_RETURN_PATH = "/settings/billing"

DEFAULT_PACKS: tuple[CreditPackConfig, ...] = (
    CreditPackConfig(id="starter", name="Starter", credits=10_000, price_cents=1_000),
    CreditPackConfig(id="growth", name="Growth", credits=50_000, price_cents=4_500, highlight=True),
    CreditPackConfig(id="scale", name="Scale", credits=200_000, price_cents=16_000),
)


class PackNotFound(Exception):
    pass


class PaymentsDisabled(Exception):
    pass


class PaymentProviderError(Exception):
    pass


def _secret_key() -> str:
    return os.getenv("STRIPE_SECRET_KEY", "").strip()


def load_pack_configs() -> list[CreditPackConfig]:
    """Parse ``STRIPE_PACKS_JSON`` (a JSON list), read per call."""
    raw = os.getenv("STRIPE_PACKS_JSON", "").strip()
    if not raw:
        return list(DEFAULT_PACKS)
    try:
        items = json.loads(raw)
        if not isinstance(items, list) or not items:
            raise ValueError("STRIPE_PACKS_JSON must be a non-empty JSON list")
        packs = [CreditPackConfig.model_validate(item) for item in items]
        CreditPackCatalog(packs=[p.to_public() for p in packs], free_monthly_credits=0)  # unique ids
        return packs
    except (ValueError, ValidationError) as exc:
        # Misconfig must not take the billing page down; fall back to non-buyable defaults.
        logger.error("billing_service: invalid STRIPE_PACKS_JSON (%s) — using defaults", type(exc).__name__)
        return list(DEFAULT_PACKS)


def is_purchasable(pack: CreditPackConfig) -> bool:
    return bool(pack.payment_link_url) or bool(pack.stripe_price_id and _secret_key())


def catalog() -> CreditPackCatalog:
    packs = load_pack_configs()
    return CreditPackCatalog(
        packs=[p.to_public(purchasable=is_purchasable(p)) for p in packs],
        free_monthly_credits=free_monthly_credits(),
    )


def _with_query(url: str, params: dict[str, str]) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    query.update(params)
    return urlunsplit(parts._replace(query=urlencode(query)))


def _frontend_url(path: str) -> str:
    from api.oauth_service import _frontend_url as base

    return base() + path


async def _create_checkout_session(
    pack: CreditPackConfig, org_id: str, email: Optional[str], return_path: str, user_id: Optional[str]
) -> CheckoutSessionResponse:
    form: dict[str, str] = {
        "mode": "payment",
        "line_items[0][price]": pack.stripe_price_id or "",
        "line_items[0][quantity]": "1",
        "client_reference_id": org_id,
        "metadata[org_id]": org_id,
        "metadata[pack_id]": pack.id,
        "metadata[credits]": str(pack.credits),
        "metadata[user_id]": user_id or "",
        "success_url": _frontend_url(_with_query(return_path, {"checkout": "success"})),
        "cancel_url": _frontend_url(_with_query(return_path, {"checkout": "cancel"})),
    }
    if email:
        form["customer_email"] = email
    return await _post_checkout(form)


async def _post_checkout(form: dict[str, str]) -> CheckoutSessionResponse:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                STRIPE_CHECKOUT_URL,
                data=form,
                headers={"Authorization": f"Bearer {_secret_key()}"},
            )
    except httpx.HTTPError as exc:
        raise PaymentProviderError(type(exc).__name__) from None
    if resp.status_code != 200:
        # Never log the response body wholesale — it can echo request params.
        logger.error("billing_service: stripe checkout failed status=%d", resp.status_code)
        raise PaymentProviderError(f"stripe status {resp.status_code}")
    body = resp.json()
    expires = body.get("expires_at")
    return CheckoutSessionResponse(
        url=body["url"],
        mode="checkout",
        session_id=body.get("id"),
        expires_at=datetime.fromtimestamp(expires, tz=timezone.utc) if expires else None,
    )


async def checkout_url(
    pack_id: str,
    org_id: str,
    email: Optional[str],
    return_path: Optional[str] = None,
    user_id: Optional[str] = None,
) -> CheckoutSessionResponse:
    """Return where to send the buyer for ``pack_id``. ``org_id`` must come from the session."""
    pack = next((p for p in load_pack_configs() if p.id == pack_id), None)
    if pack is None:
        raise PackNotFound(pack_id)
    if pack.stripe_price_id and _secret_key():
        return await _create_checkout_session(pack, org_id, email, return_path or DEFAULT_RETURN_PATH, user_id)
    if pack.payment_link_url:
        params = {"client_reference_id": org_id}
        if email:
            params["prefilled_email"] = email
        return CheckoutSessionResponse(url=_with_query(pack.payment_link_url, params), mode="payment_link")
    raise PaymentsDisabled(pack_id)


async def subscription_checkout_url(
    org_id: str, email: Optional[str], return_path: Optional[str] = None, user_id: Optional[str] = None
) -> CheckoutSessionResponse:
    """DOCBOT-1810: Checkout Session (mode=subscription) for Pro. ``org_id`` must come from the session.

    ``subscription_data.metadata`` puts org/user on the Subscription so every
    later ``customer.subscription.*`` / ``invoice.*`` webhook can be mapped
    back to the org (``api/subscription_service.py``).
    """
    from api.subscription_service import pro_price_id

    price = pro_price_id()
    if not price or not _secret_key():
        raise PaymentsDisabled("pro")
    path = return_path or DEFAULT_RETURN_PATH
    form: dict[str, str] = {
        "mode": "subscription",
        "line_items[0][price]": price,
        "line_items[0][quantity]": "1",
        "client_reference_id": org_id,
        "metadata[org_id]": org_id,
        "subscription_data[metadata][org_id]": org_id,
        "subscription_data[metadata][user_id]": user_id or "",
        "success_url": _frontend_url(_with_query(path, {"checkout": "success"})),
        "cancel_url": _frontend_url(_with_query(path, {"checkout": "cancel"})),
    }
    if email:
        form["customer_email"] = email
    return await _post_checkout(form)


# ---------------------------------------------------------------------------
# Webhook (DOCBOT-2334)
# ---------------------------------------------------------------------------

WEBHOOK_TOLERANCE_SEC = 300
# checkout.session.completed can arrive unpaid for delayed methods; the async
# event follows once funds land. Same session id -> same ledger ref.
CREDIT_EVENTS = frozenset({"checkout.session.completed", "checkout.session.async_payment_succeeded"})


class WebhookNotConfigured(Exception):
    pass


class InvalidSignature(Exception):
    pass


def _webhook_secret() -> str:
    return os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()


def verify_webhook(payload: bytes, sig_header: Optional[str], now: Optional[float] = None) -> dict:
    """Verify ``Stripe-Signature`` (t=..,v1=..) over ``"{t}.{payload}"`` and return the event."""
    secret = _webhook_secret()
    if not secret:
        raise WebhookNotConfigured()
    timestamp: Optional[int] = None
    signatures: list[str] = []
    for part in (sig_header or "").split(","):
        key, _, value = part.strip().partition("=")
        if key == "t" and value.isdigit():
            timestamp = int(value)
        elif key == "v1":
            signatures.append(value)
    if timestamp is None or not signatures:
        raise InvalidSignature("malformed header")
    current = time.time() if now is None else now
    if abs(current - timestamp) > WEBHOOK_TOLERANCE_SEC:
        raise InvalidSignature("timestamp outside tolerance")
    expected = hmac.new(
        secret.encode(), f"{timestamp}.".encode() + payload, hashlib.sha256
    ).hexdigest()
    if not any(hmac.compare_digest(expected, sig) for sig in signatures):
        raise InvalidSignature("no matching signature")
    try:
        event = json.loads(payload)
    except ValueError:
        raise InvalidSignature("bad json") from None
    if not isinstance(event, dict):
        raise InvalidSignature("bad json")
    return event


@dataclass(frozen=True)
class CheckoutGrant:
    session_id: str
    org_id: str
    user_id: Optional[str]
    credits: int
    pack_id: Optional[str]


def checkout_grant(event: dict) -> Optional[CheckoutGrant]:
    """The credit grant a verified event implies, or None (ignored event / unpaid / not ours).

    ``org_id`` / ``credits`` / ``user_id`` are the metadata ``_create_checkout_session``
    set server-side; ``client_reference_id`` (org id) is the fallback for org.
    """
    if event.get("type") not in CREDIT_EVENTS:
        return None
    obj = (event.get("data") or {}).get("object") or {}
    if obj.get("mode") == "subscription":
        return None  # DOCBOT-1810: credited per invoice.paid, not per checkout
    if obj.get("payment_status") != "paid":
        return None
    meta = obj.get("metadata") or {}
    session_id = obj.get("id") or ""
    org_id = meta.get("org_id") or obj.get("client_reference_id") or ""
    try:
        credits = int(meta.get("credits") or 0)
    except (TypeError, ValueError):
        credits = 0
    if not session_id.startswith("cs_") or not org_id or credits <= 0:
        # e.g. a Payment Link session (no credits metadata): operator grants manually.
        logger.warning("billing_service: paid checkout session not auto-creditable (missing metadata)")
        return None
    return CheckoutGrant(session_id, org_id, meta.get("user_id") or None, credits, meta.get("pack_id"))


__all__ = [
    "CheckoutGrant",
    "InvalidSignature",
    "WebhookNotConfigured",
    "checkout_grant",
    "verify_webhook",
    "DEFAULT_PACKS",
    "PackNotFound",
    "PaymentProviderError",
    "PaymentsDisabled",
    "catalog",
    "checkout_url",
    "subscription_checkout_url",
    "is_purchasable",
    "load_pack_configs",
]
