"""DOCBOT-1800 — billing_context ContextVars, KeyContext masking, RequestMeter."""

from __future__ import annotations

import asyncio
import copy
import logging
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from typing import AsyncIterator

import pytest

from api.utils import billing_context as bc
from api.utils.billing_context import (
    KeyContext,
    Principal,
    RequestMeter,
    bind_key_context,
    bind_principal,
    capture_billing_context,
    copy_ctx_run,
    current_key_source,
    get_key_context,
    get_meter,
    get_principal,
    metered_request,
    reset_meter,
    reset_principal,
    set_meter,
    set_principal,
    use_billing_context,
    usd_to_micro_usd,
)

CANARY = "gsk_CANARYcanary0123456789abcdefWXYZ"


def _principal(**kw: object) -> Principal:
    base: dict[str, object] = dict(
        user_id="u1", org_id="org1", role="analyst", auth_method="session", email_verified=True
    )
    base.update(kw)
    return Principal(**base)


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------


class TestPrincipal:
    def test_defaults_are_anonymous(self) -> None:
        p = Principal(user_id=None, org_id=None)
        assert not p.is_authenticated
        assert not p.can_spend_credits

    def test_verified_user_can_spend(self) -> None:
        assert _principal().can_spend_credits

    def test_unverified_cannot_spend(self) -> None:
        assert not _principal(email_verified=False).can_spend_credits

    def test_demo_cannot_spend(self) -> None:
        p = _principal(auth_method="demo", is_demo=True)
        assert not p.is_authenticated
        assert not p.can_spend_credits

    def test_frozen(self) -> None:
        p = _principal()
        with pytest.raises(AttributeError):
            p.org_id = "other"


# ---------------------------------------------------------------------------
# KeyContext masking
# ---------------------------------------------------------------------------


class TestKeyContextMasking:
    def test_repr_and_str_never_contain_key(self) -> None:
        k = KeyContext("groq", "byok", CANARY, org_id="org1")
        for rendered in (repr(k), str(k), f"{k}", f"{k!r}", f"{k:>200}", "%s" % k, "%r" % (k,)):
            assert CANARY not in rendered
            assert CANARY[:-4] not in rendered
        assert "****WXYZ" in repr(k)
        assert "byok" in repr(k)

    def test_short_key_has_no_hint(self) -> None:
        k = KeyContext("groq", "byok", "short-key")
        assert k.key_hint == "****"
        assert "-key" not in repr(k)

    def test_no_key(self) -> None:
        k = KeyContext("groq")
        assert k.key_hint == "<none>"
        assert not k.has_key
        assert k.reveal() is None
        assert k.key_source == "platform"

    def test_reveal_returns_raw(self) -> None:
        assert KeyContext("groq", "byok", CANARY).reveal() == CANARY

    def test_inside_containers(self) -> None:
        k = KeyContext("groq", "byok", CANARY)
        assert CANARY not in repr([k])
        assert CANARY not in repr({"k": k})
        assert CANARY not in repr(bc.BillingContextSnapshot(None, k, None))

    def test_logging_does_not_leak(self, caplog: pytest.LogCaptureFixture) -> None:
        k = KeyContext("groq", "byok", CANARY)
        with caplog.at_level(logging.INFO):
            logging.getLogger("test.billing").info("key ctx %s %r", k, k)
        assert CANARY not in caplog.text

    def test_pickle_refused(self) -> None:
        k = KeyContext("groq", "byok", CANARY)
        with pytest.raises(TypeError):
            pickle.dumps(k)

    def test_copy_shares_instance(self) -> None:
        k = KeyContext("groq", "byok", CANARY)
        assert copy.copy(k) is k
        assert copy.deepcopy({"k": k})["k"] is k

    def test_immutable_and_no_dict(self) -> None:
        k = KeyContext("groq", "byok", CANARY)
        with pytest.raises(AttributeError):
            k.provider = "x"
        with pytest.raises(AttributeError):
            del k.provider
        assert not hasattr(k, "__dict__")

    def test_invalid_source(self) -> None:
        with pytest.raises(ValueError):
            KeyContext("groq", "stolen")

    def test_current_key_source(self) -> None:
        assert current_key_source() == "platform"
        with bind_key_context(KeyContext("groq", "byok", CANARY)):
            assert current_key_source() == "byok"
            assert get_key_context() is not None
        assert get_key_context() is None


# ---------------------------------------------------------------------------
# RequestMeter
# ---------------------------------------------------------------------------


class TestRequestMeter:
    def test_usd_to_micro_rounds_up_and_is_exact(self) -> None:
        assert usd_to_micro_usd(0.1) == 100_000
        assert usd_to_micro_usd("0.0000001") == 1
        assert usd_to_micro_usd(Decimal("1.5")) == 1_500_000
        assert usd_to_micro_usd(0) == 0

    @pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf"), "abc"])
    def test_usd_to_micro_rejects(self, bad: object) -> None:
        with pytest.raises(ValueError):
            usd_to_micro_usd(bad)

    def test_accumulates(self) -> None:
        m = RequestMeter()
        m.record_llm(100, 50, 0.001)
        m.record_llm(10, 5, "0.0005")
        m.record_sandbox(1.2341, cost_usd=0.002)
        m.record_upload(2048)
        s = m.snapshot()
        assert s.llm_calls == 2
        assert s.total_tokens == 165
        assert s.cost_micro_usd == 1_000 + 500 + 2_000
        assert s.sandbox_ms == 1235
        assert s.sandbox_seconds == pytest.approx(1.235)
        assert (s.uploads, s.upload_bytes) == (1, 2048)

    def test_rejects_negative(self) -> None:
        m = RequestMeter()
        with pytest.raises(ValueError):
            m.record_llm(-1, 0)
        with pytest.raises(ValueError):
            m.record_sandbox(-1.0)
        with pytest.raises(ValueError):
            m.record_upload(-5)

    def test_snapshot_is_immutable_copy(self) -> None:
        m = RequestMeter()
        s1 = m.snapshot()
        m.record_llm(1, 1, 0)
        assert s1.llm_calls == 0
        with pytest.raises(AttributeError):
            s1.llm_calls = 5

    def test_thread_safe_under_contention(self) -> None:
        m = RequestMeter()
        n_threads, per_thread = 16, 500

        def work() -> None:
            for _ in range(per_thread):
                m.record_llm(1, 2, "0.000001")

        threads = [threading.Thread(target=work) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        s = m.snapshot()
        total = n_threads * per_thread
        assert s.llm_calls == total
        assert s.prompt_tokens == total
        assert s.completion_tokens == 2 * total
        assert s.cost_micro_usd == total


# ---------------------------------------------------------------------------
# ContextVar accessors
# ---------------------------------------------------------------------------


class TestAccessors:
    def test_set_and_reset_tokens(self) -> None:
        assert get_principal() is None
        tok = set_principal(_principal())
        assert get_principal() is not None
        reset_principal(tok)
        assert get_principal() is None

        m = RequestMeter()
        mtok = set_meter(m)
        assert get_meter() is m
        reset_meter(mtok)
        assert get_meter() is None

    def test_bind_resets_on_exception(self) -> None:
        with pytest.raises(RuntimeError):
            with bind_principal(_principal()):
                raise RuntimeError("boom")
        assert get_principal() is None

    def test_metered_request_fresh_meter(self) -> None:
        with metered_request() as m:
            assert get_meter() is m
        assert get_meter() is None

    def test_capture_and_use_snapshot(self) -> None:
        p = _principal()
        with bind_principal(p), metered_request() as m:
            snap = capture_billing_context()
        assert get_principal() is None
        with use_billing_context(snap):
            assert get_principal() is p
            assert get_meter() is m
        assert get_principal() is None and get_meter() is None


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------


def _record_in_worker() -> tuple[str | None, int]:
    p = get_principal()
    meter = get_meter()
    assert meter is not None
    meter.record_llm(10, 5, "0.001")
    meter.record_sandbox(0.5)
    return (p.org_id if p else None), threading.get_ident()


class TestPropagation:
    def test_plain_thread_pool_does_not_propagate(self) -> None:
        """Documents WHY copy_ctx_run exists."""
        with bind_principal(_principal()):
            with ThreadPoolExecutor(max_workers=1) as pool:
                seen = pool.submit(get_principal).result()
        assert seen is None

    def test_copy_ctx_run_thread_pool(self) -> None:
        main_ident = threading.get_ident()
        with bind_principal(_principal(org_id="orgA")), metered_request() as m:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(copy_ctx_run(_record_in_worker)) for _ in range(8)]
                results = [f.result() for f in futures]
        assert all(org == "orgA" for org, _ in results)
        assert any(ident != main_ident for _, ident in results)
        s = m.snapshot()
        assert s.llm_calls == 8
        assert s.cost_micro_usd == 8_000
        assert s.sandbox_ms == 4_000

    def test_copy_ctx_run_passes_args(self) -> None:
        def add(a: int, b: int, *, c: int) -> int:
            return a + b + c

        assert copy_ctx_run(add, 1, 2, c=3)() == 6

    def test_same_wrapper_reusable_concurrently(self) -> None:
        barrier = threading.Barrier(4)

        def wait_then_read() -> str | None:
            barrier.wait(timeout=5)
            p = get_principal()
            return p.org_id if p else None

        with bind_principal(_principal(org_id="orgR")):
            runner = copy_ctx_run(wait_then_read)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: runner(), range(4)))
        assert results == ["orgR"] * 4

    def test_worker_rebind_does_not_leak(self) -> None:
        def rebind() -> str | None:
            set_principal(_principal(org_id="evil"))
            p = get_principal()
            return p.org_id if p else None

        with bind_principal(_principal(org_id="orgA")):
            assert copy_ctx_run(rebind)() == "evil"
            p = get_principal()
            assert p is not None and p.org_id == "orgA"

    def test_context_captured_at_wrap_time(self) -> None:
        with bind_principal(_principal(org_id="first")):
            runner = copy_ctx_run(get_principal)
        with bind_principal(_principal(org_id="second")):
            p = runner()
        assert p is not None and p.org_id == "first"

    async def test_run_in_executor(self) -> None:
        loop = asyncio.get_running_loop()
        with bind_principal(_principal(org_id="orgE")), metered_request() as m:
            org, _ = await loop.run_in_executor(None, copy_ctx_run(_record_in_worker))
        assert org == "orgE"
        assert m.snapshot().llm_calls == 1

    async def test_asyncio_tasks_share_meter_and_isolate_principal(self) -> None:
        async def handler(org: str) -> tuple[str | None, int]:
            with bind_principal(_principal(org_id=org)), metered_request() as m:
                loop = asyncio.get_running_loop()
                sub = [
                    asyncio.create_task(asyncio.to_thread(_record_in_worker)),
                    loop.run_in_executor(None, copy_ctx_run(_record_in_worker)),
                ]
                await asyncio.sleep(0)
                results = await asyncio.gather(*sub)
                orgs = {r[0] for r in results}
                assert orgs == {org}
                return get_principal().org_id if get_principal() else None, m.snapshot().llm_calls

        results = await asyncio.gather(*(handler(f"org{i}") for i in range(10)))
        assert results == [(f"org{i}", 2) for i in range(10)]
        assert get_principal() is None

    async def test_sse_generator_rebinds_snapshot(self) -> None:
        async def stream(snap: bc.BillingContextSnapshot) -> AsyncIterator[str | None]:
            with use_billing_context(snap):
                for _ in range(3):
                    meter = get_meter()
                    assert meter is not None
                    meter.record_llm(1, 1, 0)
                    p = get_principal()
                    yield p.org_id if p else None

        with bind_principal(_principal(org_id="orgS")), metered_request() as m:
            snap = capture_billing_context()
        # Request-scoped bindings are gone by the time the body streams.
        assert get_principal() is None
        chunks = [c async for c in stream(snap)]
        assert chunks == ["orgS"] * 3
        assert m.snapshot().llm_calls == 3
