"""Billing API contracts (Pydantic v2) — DOCBOT-1800, EPIC-18.

Mirrored 1:1 (same field names) by ``src/lib/schemas/billing.ts`` (Zod).
``tests/unit/test_billing_models.py`` fails if a field drifts between them.

Units — chosen deliberately, read before adding fields
------------------------------------------------------
* **Credits are integers.** 1 credit = $0.001 retail (EPIC-18 model). Every
  balance, grant, pack size, ledger amount and debit is an ``int`` number of
  credits. Debits round *up*: ``ceil(cost_usd × MARKUP × 1000)``
  (:func:`credits_for_cost`). Integers are exact in Postgres ``BIGINT``,
  Python and JSON (well under 2^53), so no Decimal-as-string juggling on the
  frontend.
* **Raw provider cost is integer micro-USD** (``cost_micro_usd``,
  1 USD = 1_000_000). Matches ``RequestMeter`` in
  ``api/utils/billing_context.py``. Cost is internal (margin data) and only
  appears on server-side models (``UsageEventRecord``), never on
  customer-facing responses.
* **Fiat prices are integer cents** (``price_cents``) + ISO-4217 lowercase
  ``currency`` — the same shape Stripe uses.
* All timestamps are timezone-aware (``AwareDatetime``) and serialize as
  ISO-8601 with offset, which the Zod side validates with
  ``z.iso.datetime({ offset: true })``.

Server-only models (``CreditPackConfig``, ``UsageEventRecord``,
``AdminCreditGrant*``) have no Zod mirror on purpose — they carry Stripe ids,
margin data, or are operator-only.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal
from enum import Enum
from typing import Annotated, Literal, Optional

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Unit constants
# ---------------------------------------------------------------------------

MICRO_USD_PER_USD = 1_000_000
CREDITS_PER_USD = 1_000  # 1 credit = $0.001 retail
MICRO_USD_PER_CREDIT = MICRO_USD_PER_USD // CREDITS_PER_USD  # 1_000

Credits = Annotated[int, Field(ge=0, description="Whole credits (1 credit = $0.001 retail)")]
SignedCredits = Annotated[int, Field(description="Signed whole credits; negative = debit")]
MicroUSD = Annotated[int, Field(ge=0, description="Integer micro-USD (1 USD = 1_000_000)")]

_PACK_ID_RE = r"^[a-z0-9][a-z0-9_-]{0,63}$"


def credits_for_cost(cost_micro_usd: int, markup: Decimal | int | str) -> int:
    """Credits to debit for a raw provider cost: ``ceil(cost_usd × markup × 1000)``.

    Integer/Decimal math only. Any non-zero cost debits at least 1 credit.
    """
    if cost_micro_usd < 0:
        raise ValueError("cost_micro_usd must be non-negative")
    m = Decimal(str(markup))
    if not m.is_finite() or m <= 0:
        raise ValueError("markup must be a positive finite number")
    credits = Decimal(cost_micro_usd) * m / Decimal(MICRO_USD_PER_CREDIT)
    return int(math.ceil(credits))


# ---------------------------------------------------------------------------
# Enums (string-valued so JSON and Zod enums match exactly)
# ---------------------------------------------------------------------------


class LedgerEntryKind(str, Enum):
    """Kinds in the append-only ``credit_ledger`` (``UNIQUE(kind, ref)``).

    ``ref`` per kind (DOCBOT-1802/1804):
      free_grant      -> ``"{org_id}:{YYYY-MM}"`` (one lazy grant per month)
      purchase        -> Stripe ``checkout_session_id``
      usage           -> usage reservation / request id
      refund          -> Stripe ``charge`` / refund id (negative amount)
      dispute_hold    -> Stripe dispute id (freezes the org)
      dispute_release -> Stripe dispute id
      adjustment      -> admin action id (manual grant / correction)
    """

    free_grant = "free_grant"
    purchase = "purchase"
    usage = "usage"
    refund = "refund"
    dispute_hold = "dispute_hold"
    dispute_release = "dispute_release"
    adjustment = "adjustment"


class UsageEventKind(str, Enum):
    llm = "llm"
    sandbox = "sandbox"
    upload = "upload"
    request = "request"  # DOCBOT-1623: admitted call to a rate-limited route


class KeySource(str, Enum):
    platform = "platform"
    byok = "byok"  # EPIC-19, post-submit — 0 LLM credits debited


class QuotaErrorCode(str, Enum):
    insufficient_credits = "insufficient_credits"  # HTTP 402
    demo_quota_exceeded = "demo_quota_exceeded"  # HTTP 429 -> signup prompt
    rate_limited = "rate_limited"  # HTTP 429
    email_unverified = "email_unverified"  # HTTP 403 — F-2: verify before credits
    account_frozen = "account_frozen"  # HTTP 402 — open dispute
    platform_capacity = "platform_capacity"  # HTTP 503 — DOCBOT-1615 kill switch


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


# ---------------------------------------------------------------------------
# Balance
# ---------------------------------------------------------------------------


class CreditBalance(_Contract):
    """``GET /api/billing/balance`` — the caller's org balance."""

    org_id: str
    balance_credits: Credits
    reserved_credits: Credits = 0
    available_credits: Credits
    free_monthly_credits: Credits = Field(description="Size of the monthly free grant")
    free_grant_period: Optional[str] = Field(
        default=None,
        pattern=r"^\d{4}-(0[1-9]|1[0-2])$",
        description="YYYY-MM of the most recent free grant, null if never granted",
    )
    frozen: bool = False
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def _available_matches(self) -> "CreditBalance":
        if self.available_credits != max(self.balance_credits - self.reserved_credits, 0):
            raise ValueError("available_credits must equal max(balance - reserved, 0)")
        return self


# ---------------------------------------------------------------------------
# Packs / catalog
# ---------------------------------------------------------------------------


class CreditPack(_Contract):
    """Public view of a prepaid credit pack (no Stripe ids)."""

    id: str = Field(pattern=_PACK_ID_RE)
    name: str = Field(min_length=1, max_length=80)
    credits: int = Field(gt=0)
    price_cents: int = Field(gt=0)
    currency: str = Field(default="usd", pattern=r"^[a-z]{3}$")
    description: Optional[str] = Field(default=None, max_length=280)
    highlight: bool = False
    purchasable: bool = Field(
        default=False,
        description="False -> show as 'coming soon' with no buy action (payments not enabled)",
    )


class CreditPackConfig(CreditPack):
    """Server-only: one entry of ``STRIPE_PACKS_JSON`` (DOCBOT-1803).

    Accepts ``price_usd`` (e.g. ``9.99``) as an alternative to ``price_cents``.
    A pack is buyable via Checkout (``stripe_price_id`` + ``STRIPE_SECRET_KEY``)
    or via a Stripe Payment Link (``payment_link_url``).
    """

    stripe_price_id: Optional[str] = Field(default=None, pattern=r"^price_[A-Za-z0-9]+$")
    payment_link_url: Optional[str] = Field(
        default=None, max_length=512, pattern=r"^https://buy\.stripe\.com/[A-Za-z0-9_/-]+$"
    )

    @model_validator(mode="before")
    @classmethod
    def _price_usd_to_cents(cls, data: object) -> object:
        if isinstance(data, dict) and "price_usd" in data:
            data = dict(data)
            usd = data.pop("price_usd")
            if "price_cents" not in data:
                data["price_cents"] = int((Decimal(str(usd)) * 100).to_integral_value())
        return data

    def to_public(self, purchasable: bool = False) -> CreditPack:
        public = self.model_dump(exclude={"stripe_price_id", "payment_link_url"})
        public["purchasable"] = purchasable
        return CreditPack.model_validate(public)


class CreditPackCatalog(_Contract):
    """``GET /api/billing/packs``."""

    packs: list[CreditPack]
    currency: str = Field(default="usd", pattern=r"^[a-z]{3}$")
    free_monthly_credits: Credits
    credit_usd_value: Literal["0.001"] = "0.001"

    @field_validator("packs")
    @classmethod
    def _unique_ids(cls, packs: list[CreditPack]) -> list[CreditPack]:
        ids = [p.id for p in packs]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate pack id")
        return packs


# ---------------------------------------------------------------------------
# Checkout
# ---------------------------------------------------------------------------

_SAFE_RETURN_PATH = re.compile(r"^/(?![/\\])[A-Za-z0-9/_\-?=&.%]*$")


class CheckoutSessionRequest(_Contract):
    """``POST /api/billing/checkout`` (admin). ``org_id`` is NEVER accepted
    from the client — it comes from the server-side Principal and is set on
    ``client_reference_id`` + ``metadata.org_id`` (DOCBOT-1803). ``extra=forbid``
    makes a smuggled ``org_id`` a 422."""

    pack_id: str = Field(pattern=_PACK_ID_RE)
    return_path: Optional[str] = Field(
        default=None,
        max_length=512,
        description="Same-origin relative path to resume after purchase (DOCBOT-1805)",
    )

    @field_validator("return_path")
    @classmethod
    def _relative_only(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not _SAFE_RETURN_PATH.match(v):
            raise ValueError("return_path must be a same-origin relative path")
        return v


class CheckoutSessionResponse(_Contract):
    """Where to send the browser: a Checkout Session or a Payment Link (DOCBOT-1803)."""

    url: str = Field(pattern=r"^https://(checkout|buy)\.stripe\.com/")
    mode: Literal["checkout", "payment_link"]
    session_id: Optional[str] = Field(default=None, pattern=r"^cs_(test|live)_[A-Za-z0-9]+$")
    expires_at: Optional[AwareDatetime] = None


class AdminCreditGrantRequest(_Contract):
    """``POST /admin/credits/grant`` (platform admin) — manual top-up for the
    Payment Link flow. ``reference`` is the idempotency key (e.g. the Stripe
    payment/checkout id): re-posting it never double-credits."""

    org_id: str = Field(min_length=1, max_length=255)
    credits: int = Field(gt=0, le=100_000_000)
    reference: str = Field(min_length=1, max_length=255)


class AdminCreditGrantResponse(_Contract):
    org_id: str
    credits: int
    reference: str
    duplicate: bool = Field(description="True when this reference was already granted (no-op)")
    balance_credits: int


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class LedgerEntry(_Contract):
    """One row of the append-only credit ledger, as shown to the org."""

    id: str
    org_id: str
    kind: LedgerEntryKind
    amount_credits: SignedCredits
    balance_after_credits: int
    ref: str = Field(min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=280)
    actor_user_id: Optional[str] = None
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _sign_matches_kind(self) -> "LedgerEntry":
        positive = {LedgerEntryKind.free_grant, LedgerEntryKind.purchase}
        negative = {LedgerEntryKind.usage, LedgerEntryKind.refund}
        if self.kind in positive and self.amount_credits < 0:
            raise ValueError(f"{self.kind.value} entries must be non-negative")
        if self.kind in negative and self.amount_credits > 0:
            raise ValueError(f"{self.kind.value} entries must be non-positive")
        return self


class LedgerPage(_Contract):
    entries: list[LedgerEntry]
    next_cursor: Optional[str] = None


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


class UsageEvent(_Contract):
    """Customer-facing usage row (no raw cost / margin)."""

    id: str
    org_id: str
    user_id: Optional[str] = None
    kind: UsageEventKind
    key_source: KeySource = KeySource.platform
    run_id: Optional[str] = None
    model: Optional[str] = None
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    sandbox_ms: int = Field(default=0, ge=0)
    credits_debited: Credits = 0
    created_at: AwareDatetime


class UsageEventRecord(UsageEvent):
    """Server-only: what DOCBOT-1801 writes to ``usage_events``."""

    cost_micro_usd: MicroUSD = 0

    def to_public(self) -> UsageEvent:
        return UsageEvent.model_validate(self.model_dump(exclude={"cost_micro_usd"}))


class UsageBreakdownItem(_Contract):
    kind: UsageEventKind
    event_count: int = Field(ge=0)
    credits_used: Credits


class UsageSummary(_Contract):
    """``GET /api/usage/summary`` — current period roll-up for the meter UI."""

    org_id: str
    period_start: AwareDatetime
    period_end: AwareDatetime
    credits_used: Credits
    credits_granted: Credits
    credits_purchased: Credits
    event_count: int = Field(ge=0)
    by_kind: list[UsageBreakdownItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _period_order(self) -> "UsageSummary":
        if self.period_end <= self.period_start:
            raise ValueError("period_end must be after period_start")
        return self


class UsagePage(_Contract):
    events: list[UsageEvent]
    next_cursor: Optional[str] = None


# ---------------------------------------------------------------------------
# Stripe webhook + errors
# ---------------------------------------------------------------------------


class StripeWebhookAck(_Contract):
    """Body of the 2xx returned only after the ledger txn commits (DOCBOT-1804)."""

    received: Literal[True] = True
    event_id: str = Field(pattern=r"^evt_[A-Za-z0-9]+$")
    duplicate: bool = False
    handled: bool = Field(default=True, description="False for ignored event types")


class QuotaErrorPayload(_Contract):
    """``HTTPException.detail`` for 402 / 429 / 403-unverified / 503-capacity.

    Frontend (DOCBOT-1805) switches on ``error``: ``insufficient_credits`` ->
    OutOfCreditsModal, ``demo_quota_exceeded`` -> signup prompt.
    """

    error: QuotaErrorCode
    message: str = Field(min_length=1, max_length=500)
    required_credits: Optional[Credits] = None
    available_credits: Optional[Credits] = None
    retry_after_seconds: Optional[int] = Field(default=None, ge=0)
    action_path: Optional[str] = Field(
        default=None,
        description="Relative in-app path for the CTA, e.g. /settings/billing or /signup",
    )

    @field_validator("action_path")
    @classmethod
    def _relative_only(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not _SAFE_RETURN_PATH.match(v):
            raise ValueError("action_path must be a same-origin relative path")
        return v


class QuotaErrorResponse(_Contract):
    """Full JSON body FastAPI emits for ``HTTPException(detail=payload.model_dump(mode="json"))``."""

    detail: QuotaErrorPayload


__all__ = [
    "AdminCreditGrantRequest",
    "AdminCreditGrantResponse",
    "CREDITS_PER_USD",
    "MICRO_USD_PER_CREDIT",
    "MICRO_USD_PER_USD",
    "CheckoutSessionRequest",
    "CheckoutSessionResponse",
    "CreditBalance",
    "CreditPack",
    "CreditPackCatalog",
    "CreditPackConfig",
    "KeySource",
    "LedgerEntry",
    "LedgerEntryKind",
    "LedgerPage",
    "QuotaErrorCode",
    "QuotaErrorPayload",
    "QuotaErrorResponse",
    "StripeWebhookAck",
    "UsageBreakdownItem",
    "UsageEvent",
    "UsageEventKind",
    "UsageEventRecord",
    "UsagePage",
    "UsageSummary",
    "credits_for_cost",
]
