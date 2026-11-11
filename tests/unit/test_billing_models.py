"""DOCBOT-1800 — billing Pydantic contracts + Zod mirror drift check."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from api import billing_models as bm
from api.billing_models import (
    CheckoutSessionRequest,
    CheckoutSessionResponse,
    CreditBalance,
    CreditPack,
    CreditPackCatalog,
    CreditPackConfig,
    LedgerEntry,
    LedgerEntryKind,
    LedgerPage,
    QuotaErrorCode,
    QuotaErrorPayload,
    QuotaErrorResponse,
    StripeWebhookAck,
    SubscriptionCheckoutRequest,
    SubscriptionStatus,
    UsageBreakdownItem,
    UsageEvent,
    UsageEventKind,
    UsageEventRecord,
    UsagePage,
    UsageSummary,
    credits_for_cost,
)

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
TS_FILE = Path(__file__).resolve().parents[2] / "src" / "lib" / "schemas" / "billing.ts"


def _roundtrip(model: BaseModel) -> None:
    payload = json.loads(model.model_dump_json())
    again = type(model).model_validate(payload)
    assert again == model


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


class TestUnits:
    def test_constants(self) -> None:
        assert bm.MICRO_USD_PER_CREDIT == 1_000
        assert bm.CREDITS_PER_USD == 1_000

    def test_credits_for_cost_ceil(self) -> None:
        # $0.0123 raw, 3x markup -> $0.0369 retail -> 36.9 credits -> 37
        assert credits_for_cost(12_300, Decimal("3")) == 37
        assert credits_for_cost(1, 1) == 1  # any non-zero cost debits >= 1
        assert credits_for_cost(0, 3) == 0
        assert credits_for_cost(1_000_000, "2.5") == 2_500

    @pytest.mark.parametrize("cost,markup", [(-1, 3), (10, 0), (10, -1), (10, "NaN")])
    def test_credits_for_cost_rejects(self, cost: int, markup: object) -> None:
        with pytest.raises(ValueError):
            credits_for_cost(cost, markup)  # noqa


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _balance(**kw: object) -> CreditBalance:
    data: dict[str, object] = dict(
        org_id="org1",
        balance_credits=500,
        reserved_credits=120,
        available_credits=380,
        free_monthly_credits=100,
        free_grant_period="2026-10",
        updated_at=NOW,
    )
    data.update(kw)
    return CreditBalance.model_validate(data)


class TestBalance:
    def test_roundtrip(self) -> None:
        _roundtrip(_balance())

    def test_available_must_match(self) -> None:
        with pytest.raises(ValidationError):
            _balance(available_credits=999)

    def test_available_floors_at_zero(self) -> None:
        b = _balance(balance_credits=10, reserved_credits=50, available_credits=0)
        assert b.available_credits == 0

    def test_negative_balance_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _balance(balance_credits=-1)

    def test_naive_datetime_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _balance(updated_at=datetime(2026, 10, 24))

    def test_bad_period(self) -> None:
        with pytest.raises(ValidationError):
            _balance(free_grant_period="2026-13")

    def test_float_credits_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _balance(balance_credits=1.5)

    def test_extra_field_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            _balance(cost_micro_usd=5)


class TestPacks:
    def test_catalog_roundtrip(self) -> None:
        cat = CreditPackCatalog(
            packs=[CreditPack(id="pack_20", name="Starter", credits=20_000, price_cents=2_000)],
            free_monthly_credits=100,
        )
        _roundtrip(cat)
        assert cat.credit_usd_value == "0.001"

    def test_duplicate_ids_rejected(self) -> None:
        p = CreditPack(id="a", name="A", credits=1, price_cents=1)
        with pytest.raises(ValidationError):
            CreditPackCatalog(packs=[p, p], free_monthly_credits=0)

    @pytest.mark.parametrize("bad", ["", "UPPER", "../x", "a" * 65])
    def test_pack_id_pattern(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            CreditPack(id=bad, name="x", credits=1, price_cents=1)

    def test_zero_price_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CreditPack(id="x", name="x", credits=1, price_cents=0)

    def test_config_to_public_drops_stripe_id(self) -> None:
        cfg = CreditPackConfig(
            id="pack_20", name="Starter", credits=20_000, price_cents=2_000, stripe_price_id="price_abc123"
        )
        pub = cfg.to_public()
        assert type(pub) is CreditPack
        assert "stripe_price_id" not in pub.model_dump()
        assert "price_abc123" not in pub.model_dump_json()
        assert pub.purchasable is False

    def test_config_price_usd_and_payment_link(self) -> None:
        cfg = CreditPackConfig.model_validate(
            {"id": "p1", "name": "P1", "credits": 1000, "price_usd": 9.99,
             "payment_link_url": "https://buy.stripe.com/test_abc"}
        )
        assert cfg.price_cents == 999 and cfg.stripe_price_id is None
        assert "buy.stripe.com" not in cfg.to_public(purchasable=True).model_dump_json()

    def test_config_rejects_foreign_payment_link(self) -> None:
        with pytest.raises(ValidationError):
            CreditPackConfig(id="p", name="p", credits=1, price_cents=1, payment_link_url="https://evil.com/x")

    def test_config_parses_env_json(self) -> None:
        raw = '[{"id":"p1","name":"P1","credits":1000,"price_cents":500,"stripe_price_id":"price_X1"}]'
        packs = [CreditPackConfig.model_validate(x) for x in json.loads(raw)]
        assert packs[0].currency == "usd"


class TestCheckout:
    def test_request_roundtrip(self) -> None:
        _roundtrip(CheckoutSessionRequest(pack_id="pack_20", return_path="/chat?resume=1"))
        _roundtrip(CheckoutSessionRequest(pack_id="pack_20"))

    def test_client_cannot_send_org_id(self) -> None:
        with pytest.raises(ValidationError):
            CheckoutSessionRequest.model_validate({"pack_id": "pack_20", "org_id": "victim"})

    @pytest.mark.parametrize(
        "bad",
        ["https://evil.com", "//evil.com", "/\\evil.com", "javascript:alert(1)", "chat", "/a b"],
    )
    def test_open_redirect_rejected(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            CheckoutSessionRequest(pack_id="pack_20", return_path=bad)

    def test_response(self) -> None:
        r = CheckoutSessionResponse(
            url="https://checkout.stripe.com/c/pay/cs_test_abc",
            mode="checkout",
            session_id="cs_test_abc123",
            expires_at=NOW + timedelta(hours=1),
        )
        _roundtrip(r)
        _roundtrip(CheckoutSessionResponse(url="https://buy.stripe.com/abc?x=1", mode="payment_link"))

    def test_response_rejects_foreign_url(self) -> None:
        with pytest.raises(ValidationError):
            CheckoutSessionResponse(url="https://evil.com/x", mode="checkout")


class TestLedger:
    def _entry(self, kind: LedgerEntryKind, amount: int) -> LedgerEntry:
        return LedgerEntry(
            id="l1",
            org_id="org1",
            kind=kind,
            amount_credits=amount,
            balance_after_credits=100,
            ref="cs_test_1",
            created_at=NOW,
        )

    def test_roundtrip(self) -> None:
        page = LedgerPage(entries=[self._entry(LedgerEntryKind.purchase, 20_000)], next_cursor="abc")
        _roundtrip(page)
        assert json.loads(page.model_dump_json())["entries"][0]["kind"] == "purchase"

    @pytest.mark.parametrize(
        "kind,amount",
        [
            (LedgerEntryKind.purchase, -1),
            (LedgerEntryKind.free_grant, -1),
            (LedgerEntryKind.usage, 5),
            (LedgerEntryKind.refund, 5),
        ],
    )
    def test_sign_enforced(self, kind: LedgerEntryKind, amount: int) -> None:
        with pytest.raises(ValidationError):
            self._entry(kind, amount)

    def test_adjustment_any_sign(self) -> None:
        self._entry(LedgerEntryKind.adjustment, -50)
        self._entry(LedgerEntryKind.adjustment, 50)


class TestUsage:
    def _record(self) -> UsageEventRecord:
        return UsageEventRecord(
            id="e1",
            org_id="org1",
            user_id="u1",
            kind=UsageEventKind.llm,
            model="llama-3.3-70b",
            prompt_tokens=100,
            completion_tokens=50,
            credits_debited=3,
            cost_micro_usd=987,
            created_at=NOW,
        )

    def test_record_to_public_hides_cost(self) -> None:
        pub = self._record().to_public()
        assert type(pub) is UsageEvent
        assert "cost_micro_usd" not in pub.model_dump_json()
        assert pub.key_source.value == "platform"
        _roundtrip(pub)

    def test_public_rejects_cost(self) -> None:
        data = self._record().model_dump()
        with pytest.raises(ValidationError):
            UsageEvent.model_validate(data)

    def test_summary(self) -> None:
        s = UsageSummary(
            org_id="org1",
            period_start=NOW,
            period_end=NOW + timedelta(days=30),
            credits_used=40,
            credits_granted=100,
            credits_purchased=0,
            event_count=4,
            by_kind=[UsageBreakdownItem(kind=UsageEventKind.llm, event_count=4, credits_used=40)],
        )
        _roundtrip(s)
        _roundtrip(UsagePage(events=[self._record().to_public()]))

    def test_summary_period_order(self) -> None:
        with pytest.raises(ValidationError):
            UsageSummary(
                org_id="o",
                period_start=NOW,
                period_end=NOW,
                credits_used=0,
                credits_granted=0,
                credits_purchased=0,
                event_count=0,
            )


class TestWebhookAndErrors:
    def test_ack(self) -> None:
        _roundtrip(StripeWebhookAck(event_id="evt_123", duplicate=True))
        with pytest.raises(ValidationError):
            StripeWebhookAck(event_id="bogus")
        with pytest.raises(ValidationError):
            StripeWebhookAck.model_validate({"received": False, "event_id": "evt_1"})

    def test_quota_error(self) -> None:
        p = QuotaErrorPayload(
            error=QuotaErrorCode.insufficient_credits,
            message="Out of credits",
            required_credits=40,
            available_credits=12,
            action_path="/settings/billing",
        )
        body = QuotaErrorResponse(detail=p)
        _roundtrip(body)
        assert json.loads(body.model_dump_json())["detail"]["error"] == "insufficient_credits"

    def test_quota_error_action_path_relative(self) -> None:
        with pytest.raises(ValidationError):
            QuotaErrorPayload(error="rate_limited", message="x", action_path="https://evil.com")


# ---------------------------------------------------------------------------
# Zod mirror drift check
# ---------------------------------------------------------------------------

PUBLIC_MODELS: list[type[BaseModel]] = [
    CreditBalance,
    CreditPack,
    CreditPackCatalog,
    CheckoutSessionRequest,
    CheckoutSessionResponse,
    SubscriptionCheckoutRequest,
    SubscriptionStatus,
    LedgerEntry,
    LedgerPage,
    UsageEvent,
    UsageBreakdownItem,
    UsageSummary,
    UsagePage,
    StripeWebhookAck,
    QuotaErrorPayload,
    QuotaErrorResponse,
]
ENUMS = [bm.LedgerEntryKind, bm.UsageEventKind, bm.KeySource, bm.QuotaErrorCode]


def _ts_block(source: str, name: str) -> str:
    m = re.search(rf"export const {name}Schema = (.*?)(?=\nexport |\Z)", source, re.S)
    assert m, f"{name}Schema missing from billing.ts"
    return m.group(1)


class TestZodMirror:
    source = TS_FILE.read_text()

    @pytest.mark.parametrize("model", PUBLIC_MODELS, ids=lambda m: m.__name__)
    def test_fields_identical(self, model: type[BaseModel]) -> None:
        block = _ts_block(self.source, model.__name__)
        ts_fields = set(re.findall(r"^\s{2,4}([a-z_]+):", block, re.M))
        assert ts_fields == set(model.model_fields), model.__name__

    @pytest.mark.parametrize("enum", ENUMS, ids=lambda e: e.__name__)
    def test_enum_values_identical(self, enum: type) -> None:
        block = _ts_block(self.source, enum.__name__)
        ts_values = set(re.findall(r'"([a-z_]+)"', block))
        assert ts_values == {e.value for e in enum}

    def test_server_only_models_not_mirrored(self) -> None:
        assert "CreditPackConfigSchema" not in self.source
        assert "UsageEventRecordSchema" not in self.source
        assert "stripe_price_id" not in self.source
        assert "payment_link_url" not in self.source
        assert "cost_micro_usd" not in self.source

    def test_no_any(self) -> None:
        assert not re.search(r"\bany\b", re.sub(r"//.*", "", self.source))
