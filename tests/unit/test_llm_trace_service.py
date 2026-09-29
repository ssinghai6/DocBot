"""Unit tests for api.llm_trace_service — LLM call log persistence (DOCBOT-1501).

Uses an in-memory SQLite async engine so no live PostgreSQL is required
(mirrors tests/integration/test_db_pipeline.py's approach for other tables).
"""

from __future__ import annotations

import asyncio
import queue as queue_module

import pytest
from sqlalchemy import MetaData, select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from api import llm_trace_service


@pytest.fixture
async def wired_store():
    """Create an in-memory SQLite-backed llm_calls table and wire the module."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    metadata = MetaData()
    table = llm_trace_service.register_llm_calls_table(metadata)

    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    llm_trace_service.wire_llm_trace_store(table, session_factory)

    yield table, session_factory

    # Reset module globals so tests don't leak state into each other.
    llm_trace_service._llm_calls_table = None
    llm_trace_service._async_session_factory = None
    await engine.dispose()


def _drain_queue():
    """Empty the module-level queue between tests (it's a singleton)."""
    while True:
        try:
            llm_trace_service._queue.get_nowait()
        except queue_module.Empty:
            break


@pytest.fixture(autouse=True)
def clean_queue():
    _drain_queue()
    yield
    _drain_queue()


# ---------------------------------------------------------------------------
# Table registration
# ---------------------------------------------------------------------------


class TestRegisterTable:
    def test_table_has_expected_columns(self):
        metadata = MetaData()
        table = llm_trace_service.register_llm_calls_table(metadata)
        col_names = {c.name for c in table.columns}
        assert col_names == {
            "id", "run_id", "provider", "model", "caller", "latency_ms",
            "input_tokens", "output_tokens", "estimated_cost_usd", "success",
            "fallback_triggered", "error_message", "created_at",
        }

    def test_table_name(self):
        metadata = MetaData()
        table = llm_trace_service.register_llm_calls_table(metadata)
        assert table.name == "llm_calls"


# ---------------------------------------------------------------------------
# enqueue_call — never blocks, never raises
# ---------------------------------------------------------------------------


class TestEnqueueCall:
    def test_enqueue_does_not_raise_when_unwired(self):
        llm_trace_service._llm_calls_table = None
        llm_trace_service._async_session_factory = None
        llm_trace_service.enqueue_call({"run_id": "r1", "llm_provider": "groq"})
        # Should have landed in the queue regardless of wiring state.
        item = llm_trace_service._queue.get_nowait()
        assert item["run_id"] == "r1"

    def test_enqueue_full_queue_does_not_raise(self, monkeypatch):
        small_queue = queue_module.Queue(maxsize=1)
        monkeypatch.setattr(llm_trace_service, "_queue", small_queue)
        llm_trace_service.enqueue_call({"run_id": "first"})
        # Queue is now full — this must be dropped silently, not raise.
        llm_trace_service.enqueue_call({"run_id": "second"})
        assert small_queue.qsize() == 1


# ---------------------------------------------------------------------------
# Persistence via the writer loop
# ---------------------------------------------------------------------------


class TestWriterLoop:
    @pytest.mark.asyncio
    async def test_enqueued_payload_is_persisted(self, wired_store):
        table, session_factory = wired_store

        llm_trace_service.enqueue_call({
            "run_id": "run-abc",
            "llm_provider": "groq",
            "llm_model": "openai/gpt-oss-20b",
            "llm_caller": "sql_gen",
            "llm_latency_ms": 123.0,
            "llm_input_tokens": 10,
            "llm_output_tokens": 5,
            "llm_estimated_cost_usd": 0.001,
            "llm_success": True,
            "llm_fallback_triggered": False,
        })

        task = llm_trace_service.start_writer()
        await asyncio.sleep(0.05)  # let the writer drain the queue
        await llm_trace_service.stop_writer()

        async with session_factory() as session:
            result = await session.execute(select(table))
            rows = result.fetchall()

        assert len(rows) == 1
        row = rows[0]
        assert row.run_id == "run-abc"
        assert row.provider == "groq"
        assert row.model == "openai/gpt-oss-20b"
        assert row.caller == "sql_gen"
        assert row.success is True
        assert row.fallback_triggered is False
        assert row.input_tokens == 10
        assert row.output_tokens == 5

    @pytest.mark.asyncio
    async def test_persist_failure_does_not_crash_loop(self, wired_store, monkeypatch):
        table, session_factory = wired_store

        calls = {"n": 0}
        original = llm_trace_service._persist_one

        async def _flaky(payload):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated DB failure")
            await original(payload)

        monkeypatch.setattr(llm_trace_service, "_persist_one", _flaky)

        llm_trace_service.enqueue_call({"run_id": "will-fail", "llm_provider": "groq"})
        llm_trace_service.enqueue_call({"run_id": "will-succeed", "llm_provider": "groq"})

        llm_trace_service.start_writer()
        await asyncio.sleep(0.05)
        await llm_trace_service.stop_writer()

        async with session_factory() as session:
            result = await session.execute(select(table))
            rows = result.fetchall()

        # The failed write is dropped; the loop keeps running for the next one.
        run_ids = {r.run_id for r in rows}
        assert "will-succeed" in run_ids
        assert "will-fail" not in run_ids

    @pytest.mark.asyncio
    async def test_unwired_store_is_a_noop(self):
        # No wire_llm_trace_store() call — writer must not raise.
        llm_trace_service._llm_calls_table = None
        llm_trace_service._async_session_factory = None
        llm_trace_service.enqueue_call({"run_id": "no-store"})
        llm_trace_service.start_writer()
        await asyncio.sleep(0.02)
        await llm_trace_service.stop_writer()  # must return cleanly

    @pytest.mark.asyncio
    async def test_events_enqueued_before_writer_starts_are_not_dropped(self, wired_store):
        """Startup ordering: index.py's lifespan calls wire_llm_trace_store()
        then start_writer() — but nothing prevents an LLM call (and its
        enqueue_call) from happening in between, or even before either. The
        queue must hold those events rather than drop them, and the writer
        must drain everything once started."""
        table, session_factory = wired_store

        # Simulate calls arriving before the writer task exists.
        for i in range(5):
            llm_trace_service.enqueue_call({"run_id": f"pre-start-{i}", "llm_provider": "groq"})

        assert llm_trace_service._writer_task is None

        llm_trace_service.start_writer()
        await asyncio.sleep(0.05)
        await llm_trace_service.stop_writer()

        async with session_factory() as session:
            result = await session.execute(select(table))
            rows = result.fetchall()

        run_ids = {r.run_id for r in rows}
        assert run_ids == {f"pre-start-{i}" for i in range(5)}

    @pytest.mark.asyncio
    async def test_stop_writer_drains_pending_queue_before_returning(self, wired_store):
        """stop_writer() enqueues a shutdown sentinel at the back of the
        FIFO queue — every real event enqueued before shutdown was requested
        must be persisted before the writer loop exits, not dropped."""
        table, session_factory = wired_store

        llm_trace_service.start_writer()
        for i in range(20):
            llm_trace_service.enqueue_call({"run_id": f"drain-{i}", "llm_provider": "groq"})
        # No sleep here — stop_writer() is called immediately, while the
        # queue may still hold unprocessed items.
        await llm_trace_service.stop_writer()

        async with session_factory() as session:
            result = await session.execute(select(table))
            rows = result.fetchall()

        run_ids = {r.run_id for r in rows}
        assert run_ids == {f"drain-{i}" for i in range(20)}

    @pytest.mark.asyncio
    async def test_missing_table_degrades_gracefully_does_not_crash_loop(self, wired_store, monkeypatch):
        """DOCBOT-1502 review item: if the llm_calls table doesn't exist yet
        against a given DB connection (e.g. an old pooled connection during a
        Railway mid-deploy window), the insert raises a DB error — the
        writer must log and drop it, not crash, and must keep processing
        subsequent events. Simulated by pointing the module's table
        reference at a table object with a name the DB doesn't recognize."""
        table, session_factory = wired_store

        from sqlalchemy import Column, MetaData, String, Table as SATable

        bogus_metadata = MetaData()
        bogus_table = SATable(
            "llm_calls_does_not_exist", bogus_metadata,
            Column("id", String, primary_key=True),
            Column("run_id", String),
        )
        monkeypatch.setattr(llm_trace_service, "_llm_calls_table", bogus_table)

        llm_trace_service.enqueue_call({"run_id": "against-missing-table", "llm_provider": "groq"})
        llm_trace_service.start_writer()
        await asyncio.sleep(0.05)
        # Must return cleanly — a missing table must not hang or crash the loop.
        await llm_trace_service.stop_writer()

        # Restore the real table and confirm the loop (a fresh one, started
        # again) still works normally afterwards — proves the failure didn't
        # corrupt any shared state.
        monkeypatch.setattr(llm_trace_service, "_llm_calls_table", table)
        llm_trace_service.enqueue_call({"run_id": "after-recovery", "llm_provider": "groq"})
        llm_trace_service.start_writer()
        await asyncio.sleep(0.05)
        await llm_trace_service.stop_writer()

        async with session_factory() as session:
            result = await session.execute(select(table))
            rows = result.fetchall()
        run_ids = {r.run_id for r in rows}
        assert "after-recovery" in run_ids
        assert "against-missing-table" not in run_ids


# ---------------------------------------------------------------------------
# get_call_stats — read path for DOCBOT-1502
# ---------------------------------------------------------------------------


class TestGetCallStats:
    @pytest.mark.asyncio
    async def test_returns_persisted_rows(self, wired_store):
        table, session_factory = wired_store

        llm_trace_service.enqueue_call({
            "run_id": "stats-run",
            "llm_provider": "gemini",
            "llm_model": "gemini-2.5-flash",
            "llm_caller": "hybrid_synthesis",
            "llm_latency_ms": 42.0,
            "llm_success": True,
            "llm_fallback_triggered": True,
        })
        llm_trace_service.start_writer()
        await asyncio.sleep(0.05)
        await llm_trace_service.stop_writer()

        stats = await llm_trace_service.get_call_stats()
        assert len(stats) == 1
        assert stats[0]["run_id"] == "stats-run"
        assert stats[0]["provider"] == "gemini"
        assert stats[0]["fallback_triggered"] is True

    @pytest.mark.asyncio
    async def test_empty_when_unwired(self):
        llm_trace_service._llm_calls_table = None
        llm_trace_service._async_session_factory = None
        stats = await llm_trace_service.get_call_stats()
        assert stats == []
