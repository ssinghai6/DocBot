"""DOCBOT-1801 — usage ledger + metering hooks.

Pure unit tests: no network, no DB. The usage_events / llm_calls queues are
drained in-process and inspected directly.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import queue
import sys
import types

import pytest
from sqlalchemy import MetaData, create_engine, inspect

from api import llm_trace_service, usage_service
from api.utils import llm_provider
from api.utils.billing_context import (
    KeyContext,
    Principal,
    RequestMeter,
    bind_key_context,
    bind_principal,
    copy_ctx_run,
    get_principal,
    metered_request,
)

CANARY_KEY = "gsk_CANARY_DO_NOT_LEAK_7f3a9c1e5b2d4f60"

PRINCIPAL = Principal(
    user_id="user-1",
    org_id="org-1",
    role="analyst",
    auth_method="session",
    email_verified=True,
    session_id="sess-1",
)


def _drain(q: "queue.Queue") -> list:
    rows = []
    while True:
        try:
            rows.append(q.get_nowait())
        except queue.Empty:
            return rows


@pytest.fixture(autouse=True)
def _clean_queues(monkeypatch):
    _drain(usage_service._queue)
    _drain(llm_trace_service._queue)
    monkeypatch.setattr(llm_provider, "_trace_sink", llm_trace_service.enqueue_call)
    monkeypatch.setattr("api.utils.langsmith_tracing.emit_llm_run", lambda payload: None)
    yield
    _drain(usage_service._queue)
    _drain(llm_trace_service._queue)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_usage_events_table_schema_and_index():
    md = MetaData()
    table = usage_service.register_usage_events_table(md)
    llm_trace_service.register_llm_calls_table(md)
    engine = create_engine("sqlite://")
    md.create_all(engine)
    md.create_all(engine)  # idempotent
    insp = inspect(engine)
    cols = {c["name"] for c in insp.get_columns("usage_events")}
    assert {
        "id", "org_id", "user_id", "session_id", "kind", "provider", "model",
        "prompt_tokens", "completion_tokens", "cost_micro_usd", "sandbox_ms",
        "bytes", "key_source", "created_at",
    } <= cols
    assert str(table.c.cost_micro_usd.type) == "BIGINT"
    idx = {i["name"]: i["column_names"] for i in insp.get_indexes("usage_events")}
    assert idx["ix_usage_events_org_created"] == ["org_id", "created_at"]
    llm_cols = {c["name"] for c in insp.get_columns("llm_calls")}
    assert {"org_id", "user_id", "key_source"} <= llm_cols


def test_llm_calls_migrations_are_guarded():
    for ddl in usage_service.LLM_CALLS_BILLING_MIGRATIONS:
        assert "IF NOT EXISTS" in ddl
        assert "llm_calls" in ddl


# ---------------------------------------------------------------------------
# Executor propagation
# ---------------------------------------------------------------------------


def test_context_propagates_through_executor():
    async def _main():
        loop = asyncio.get_running_loop()
        with bind_principal(PRINCIPAL), metered_request() as meter:
            seen = await loop.run_in_executor(
                None,
                copy_ctx_run(
                    lambda: (get_principal(), usage_service.meter_sandbox(0.25)),
                ),
            )
            return seen[0], meter.snapshot()

    principal, snap = asyncio.run(_main())
    assert principal is PRINCIPAL
    assert snap.sandbox_ms == 250
    rows = _drain(usage_service._queue)
    assert rows[0]["org_id"] == "org-1"
    assert rows[0]["user_id"] == "user-1"


def test_sandbox_service_executor_sites_use_copy_ctx_run():
    import inspect as pyinspect

    from api import sandbox_service

    import re

    src = pyinspect.getsource(sandbox_service)
    sites = re.findall(r"run_in_executor\(\s*None,\s*([A-Za-z_]+)\(", src)
    assert len(sites) == 3
    assert all(fn == "copy_ctx_run" for fn in sites), sites
    assert "run_in_executor(" not in pyinspect.getsource(llm_provider)


# ---------------------------------------------------------------------------
# Sandbox metering on exception
# ---------------------------------------------------------------------------


class _BoomSandbox:
    killed = False

    @classmethod
    def create(cls):
        return cls()

    def run_code(self, code):
        raise RuntimeError("sandbox exploded")

    def kill(self):
        _BoomSandbox.killed = True

    class files:  # noqa: N801 - mimic SDK attribute
        @staticmethod
        def write(path, data):
            return None


@pytest.fixture
def fake_e2b(monkeypatch):
    mod = types.ModuleType("e2b_code_interpreter")
    mod.Sandbox = _BoomSandbox
    monkeypatch.setitem(sys.modules, "e2b_code_interpreter", mod)
    monkeypatch.setenv("E2B_API_KEY", "e2b_test_key")
    from api import sandbox_service

    monkeypatch.setattr(sandbox_service, "_install_sandbox_packages", lambda *a, **k: None)
    return sandbox_service


def test_sandbox_ms_recorded_on_exception(fake_e2b):
    with bind_principal(PRINCIPAL), metered_request() as meter:
        with pytest.raises(RuntimeError):
            fake_e2b._run_in_sandbox("print(1)")
        snap = meter.snapshot()
    assert _BoomSandbox.killed
    rows = [r for r in _drain(usage_service._queue) if r["kind"] == "sandbox"]
    assert len(rows) == 1
    assert rows[0]["provider"] == "e2b"
    assert rows[0]["org_id"] == "org-1"
    assert rows[0]["sandbox_ms"] == snap.sandbox_ms
    assert snap.sandbox_ms >= 0
    assert rows[0]["cost_micro_usd"] == usage_service.sandbox_cost_micro_usd(rows[0]["sandbox_ms"])


def test_csv_sandbox_ms_recorded_on_exception(fake_e2b):
    with metered_request():
        with pytest.raises(RuntimeError):
            fake_e2b._run_csv_in_sandbox_sync("print(1)", "/home/user/x.csv", b"a,b\n1,2\n")
    rows = [r for r in _drain(usage_service._queue) if r["kind"] == "sandbox"]
    assert len(rows) == 1


def test_run_python_timeout_still_meters_on_request_meter(fake_e2b, monkeypatch):
    import time as _time

    def _slow(code, **_kwargs):
        start = _time.monotonic()
        try:
            _time.sleep(0.3)
        finally:
            usage_service.meter_sandbox(_time.monotonic() - start)

    monkeypatch.setattr(fake_e2b, "_run_in_sandbox", _slow)

    async def _main():
        with bind_principal(PRINCIPAL), metered_request() as meter:
            res = await fake_e2b.run_python("x", timeout_seconds=0)
            await asyncio.sleep(0.5)  # worker finishes after the timeout
            return res, meter.snapshot()

    res, snap = asyncio.run(_main())
    assert "timed out" in (res.error or "")
    assert snap.sandbox_ms >= 300  # worker thread shared the request meter


# ---------------------------------------------------------------------------
# llm_calls enrichment + canary key never leaks
# ---------------------------------------------------------------------------


def _log_one_call():
    llm_provider._log_llm_call(
        provider="groq",
        model=llm_provider.GROQ_MODEL,
        latency_ms=12.0,
        success=True,
        fallback_triggered=False,
        caller="unit_test",
        input_tokens=1000,
        output_tokens=500,
    )


def test_llm_calls_payload_enriched_from_contextvars():
    key_ctx = KeyContext(provider="groq", key_source="byok", api_key=CANARY_KEY, org_id="org-1")
    with bind_principal(PRINCIPAL), bind_key_context(key_ctx), metered_request() as meter:
        _log_one_call()
        snap = meter.snapshot()

    trace_rows = _drain(llm_trace_service._queue)
    assert len(trace_rows) == 1
    payload = trace_rows[0]
    assert payload["org_id"] == "org-1"
    assert payload["user_id"] == "user-1"
    assert payload["key_source"] == "byok"

    usage_rows = _drain(usage_service._queue)
    assert len(usage_rows) == 1
    row = usage_rows[0]
    assert row["kind"] == "llm"
    assert row["key_source"] == "byok"
    assert row["prompt_tokens"] == 1000 and row["completion_tokens"] == 500
    assert isinstance(row["cost_micro_usd"], int) and row["cost_micro_usd"] > 0
    assert row["run_id"] == payload["run_id"]

    assert snap.llm_calls == 1
    assert snap.prompt_tokens == 1000
    assert snap.cost_micro_usd == row["cost_micro_usd"]


def test_llm_calls_persist_maps_attribution(monkeypatch):
    captured = {}

    class _Sess:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def begin(self):
            return self

        async def execute(self, stmt):
            captured.update(stmt.compile().params)

    md = MetaData()
    table = llm_trace_service.register_llm_calls_table(md)
    monkeypatch.setattr(llm_trace_service, "_llm_calls_table", table)
    monkeypatch.setattr(llm_trace_service, "_async_session_factory", lambda: _Sess())
    asyncio.run(
        llm_trace_service._persist_one(
            {"run_id": "r", "llm_provider": "groq", "llm_model": "m", "llm_success": True,
             "org_id": "org-1", "user_id": "user-1", "key_source": "platform"}
        )
    )
    assert captured["org_id"] == "org-1"
    assert captured["user_id"] == "user-1"
    assert captured["key_source"] == "platform"


def test_canary_key_never_in_rows_or_logs(caplog):
    key_ctx = KeyContext(provider="groq", key_source="platform", api_key=CANARY_KEY)
    caplog.set_level(logging.DEBUG)
    with bind_principal(PRINCIPAL), bind_key_context(key_ctx), metered_request():
        _log_one_call()
        usage_service.meter_upload(1234, pages=3)
        usage_service.meter_sandbox(0.01)

    trace_rows = _drain(llm_trace_service._queue)
    usage_rows = _drain(usage_service._queue)
    assert trace_rows and len(usage_rows) == 3
    blob = repr(trace_rows) + repr(usage_rows)
    assert CANARY_KEY not in blob
    assert CANARY_KEY[-12:] not in blob
    for rec in caplog.records:
        assert CANARY_KEY not in rec.getMessage()
        assert CANARY_KEY not in repr(rec.__dict__)


def test_failed_call_without_tokens_not_metered():
    with metered_request() as meter:
        llm_provider._log_llm_call(
            provider="groq", model="m", latency_ms=1.0, success=False,
            fallback_triggered=True, caller="t",
        )
        assert meter.snapshot().llm_calls == 0
    assert _drain(usage_service._queue) == []


def test_no_context_defaults_to_platform_and_null_org():
    _log_one_call()
    row = _drain(usage_service._queue)[0]
    assert row["org_id"] is None
    assert row["key_source"] == "platform"


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------


def test_upload_metered_with_bytes_and_pages():
    with bind_principal(PRINCIPAL), metered_request() as meter:
        usage_service.meter_upload(4096, pages=7, session_id="upload-sess")
        snap = meter.snapshot()
    row = _drain(usage_service._queue)[0]
    assert row["kind"] == "upload"
    assert row["bytes"] == 4096 and row["pages"] == 7
    assert row["session_id"] == "upload-sess"
    assert snap.uploads == 1 and snap.upload_bytes == 4096


# ---------------------------------------------------------------------------
# SSE generator rebind
# ---------------------------------------------------------------------------


def test_sse_generator_rebinds_billing_context():
    meter = RequestMeter()

    async def event_stream():
        yield get_principal()
        usage_service.meter_upload(10)
        yield get_principal()

    with bind_principal(PRINCIPAL), metered_request(meter):
        wrapped = usage_service.bind_billing_stream(event_stream())
    # The request-scoped binding is gone; iterate in a fresh, empty context
    # (as Starlette does after the dependency has reset its tokens).
    assert get_principal() is None

    async def _consume():
        return [p async for p in wrapped]

    seen = contextvars.Context().run(asyncio.run, _consume())
    assert seen == [PRINCIPAL, PRINCIPAL]
    assert meter.snapshot().uploads == 1
    assert get_principal() is None
    row = _drain(usage_service._queue)[0]
    assert row["org_id"] == "org-1"


def test_unwrapped_generator_loses_context():
    async def event_stream():
        yield get_principal()

    with bind_principal(PRINCIPAL):
        gen = event_stream()

    async def _consume():
        return [p async for p in gen]

    assert contextvars.Context().run(asyncio.run, _consume()) == [None]


# ---------------------------------------------------------------------------
# Writer: failures never propagate
# ---------------------------------------------------------------------------


def test_writer_survives_db_failure(monkeypatch, caplog):
    from sqlalchemy.exc import OperationalError

    calls = []

    async def _boom(row):
        calls.append(row)
        raise OperationalError("insert", {}, Exception("db down"))

    monkeypatch.setattr(usage_service, "_persist_one", _boom)

    async def _main():
        task = asyncio.create_task(usage_service.writer_loop())
        usage_service.record_usage("upload", size_bytes=1)
        usage_service.record_usage("upload", size_bytes=2)
        usage_service._queue.put_nowait(usage_service._SHUTDOWN_SENTINEL)
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(_main())
    assert len(calls) == 2
    assert any("failed to persist" in r.getMessage() for r in caplog.records)


def test_full_queue_drops_without_raising(monkeypatch):
    monkeypatch.setattr(usage_service, "_queue", queue.Queue(maxsize=1))
    usage_service.record_usage("upload", size_bytes=1)
    usage_service.record_usage("upload", size_bytes=2)  # dropped, no raise
    assert usage_service._queue.qsize() == 1


def test_unknown_kind_ignored():
    usage_service.record_usage("bogus")
    assert _drain(usage_service._queue) == []
