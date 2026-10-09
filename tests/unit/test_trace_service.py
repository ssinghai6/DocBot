"""Unit tests for api.trace_service — agent trace logging + feedback (DOCBOT-1512).

Uses an in-memory SQLite async engine so no live PostgreSQL is required —
mirrors tests/unit/test_llm_trace_service.py's approach for the sibling
llm_calls table.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import MetaData, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api import trace_service


@pytest.fixture
async def wired_store():
    """Create an in-memory SQLite-backed agent_traces table and wire the module."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    metadata = MetaData()
    table = trace_service.register_agent_traces_table(metadata)

    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    trace_service.wire_trace_store(table, session_factory)

    yield table, session_factory

    # Reset module globals so tests don't leak state into each other.
    trace_service._agent_traces_table = None
    trace_service._async_session_factory = None
    await engine.dispose()


async def _drain_pending_writes() -> None:
    """Await every in-flight fire-and-forget write task before asserting."""
    pending = list(trace_service._pending_writes)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


# ---------------------------------------------------------------------------
# Table registration
# ---------------------------------------------------------------------------


class TestRegisterTable:
    def test_table_has_expected_columns(self):
        metadata = MetaData()
        table = trace_service.register_agent_traces_table(metadata)
        col_names = {c.name for c in table.columns}
        assert col_names == {
            "id", "run_id", "session_id", "pipeline", "question",
            "intent_classified", "tool_chosen", "plan_steps_json",
            "retrieved_refs_json", "final_answer", "latency_ms", "cost_usd",
            "error", "user_feedback", "implicit_signal", "created_at",
        }

    def test_table_name(self):
        metadata = MetaData()
        table = trace_service.register_agent_traces_table(metadata)
        assert table.name == "agent_traces"


# ---------------------------------------------------------------------------
# log_trace — never raises, never blocks on the DB write
# ---------------------------------------------------------------------------


class TestLogTrace:
    @pytest.mark.asyncio
    async def test_returns_a_trace_id_when_unwired(self):
        trace_service._agent_traces_table = None
        trace_service._async_session_factory = None
        trace_id = await trace_service.log_trace(
            run_id="r1", session_id="s1", pipeline="chat", question="hello?",
        )
        assert isinstance(trace_id, str) and len(trace_id) > 0

    @pytest.mark.asyncio
    async def test_persists_a_row(self, wired_store):
        table, session_factory = wired_store

        trace_id = await trace_service.log_trace(
            run_id="run-abc",
            session_id="sess-1",
            pipeline="hybrid",
            question="What drove the Q4 revenue miss?",
            intent_classified="hybrid",
            tool_chosen="sql+rag",
            plan_steps=["classify_intent", "retrieve_docs", "run_sql_pipeline"],
            retrieved_refs=[{"source": "10-K.pdf", "page": 12}],
            final_answer="Revenue missed due to...",
            latency_ms=842.3,
            cost_usd=0.0021,
        )
        await _drain_pending_writes()

        async with session_factory() as session:
            result = await session.execute(select(table).where(table.c.id == trace_id))
            row = result.first()

        assert row is not None
        assert row.run_id == "run-abc"
        assert row.session_id == "sess-1"
        assert row.pipeline == "hybrid"
        assert row.intent_classified == "hybrid"
        assert row.tool_chosen == "sql+rag"
        assert row.latency_ms == 842
        assert row.cost_usd == pytest.approx(0.0021)
        assert row.user_feedback is None

    @pytest.mark.asyncio
    async def test_never_raises_when_db_write_fails(self, wired_store, monkeypatch):
        table, session_factory = wired_store

        async def _boom(values):
            raise RuntimeError("simulated DB failure")

        monkeypatch.setattr(trace_service, "_insert_trace", _boom)

        # Must not raise even though the (awaited-via-task) insert blows up.
        trace_id = await trace_service.log_trace(
            run_id="r1", session_id="s1", pipeline="chat", question="hi",
        )
        await _drain_pending_writes()
        assert isinstance(trace_id, str)

    @pytest.mark.asyncio
    async def test_truncates_long_final_answer(self, wired_store):
        table, session_factory = wired_store
        long_answer = "x" * 10_000

        trace_id = await trace_service.log_trace(
            run_id="r1", session_id="s1", pipeline="autopilot", question="q",
            final_answer=long_answer,
        )
        await _drain_pending_writes()

        async with session_factory() as session:
            result = await session.execute(select(table).where(table.c.id == trace_id))
            row = result.first()

        assert len(row.final_answer) == trace_service._MAX_ANSWER_LEN

    @pytest.mark.asyncio
    async def test_plan_steps_and_refs_persist_as_json(self, wired_store):
        table, session_factory = wired_store

        trace_id = await trace_service.log_trace(
            run_id="r1", session_id="s1", pipeline="autopilot", question="q",
            plan_steps=["step a", "step b"],
            retrieved_refs=[{"source": "doc.pdf"}],
        )
        await _drain_pending_writes()

        async with session_factory() as session:
            result = await session.execute(select(table).where(table.c.id == trace_id))
            row = result.first()

        import json
        assert json.loads(row.plan_steps_json) == ["step a", "step b"]
        assert json.loads(row.retrieved_refs_json) == [{"source": "doc.pdf"}]

    @pytest.mark.asyncio
    async def test_no_running_loop_is_a_noop_not_a_crash(self, wired_store, monkeypatch):
        """If scheduling the background write raises RuntimeError (no running
        event loop), log_trace must swallow it and still return a usable id."""
        def _raise_runtime_error():
            raise RuntimeError("no running event loop")

        monkeypatch.setattr(asyncio, "get_running_loop", _raise_runtime_error)

        trace_id = await trace_service.log_trace(
            run_id="r1", session_id="s1", pipeline="chat", question="q",
        )
        assert isinstance(trace_id, str) and len(trace_id) > 0


# ---------------------------------------------------------------------------
# record_feedback — validation + found/not-found
# ---------------------------------------------------------------------------


class TestRecordFeedback:
    @pytest.mark.asyncio
    async def test_rejects_invalid_feedback_value(self, wired_store):
        with pytest.raises(ValueError):
            await trace_service.record_feedback("some-id", "sideways")

    @pytest.mark.asyncio
    async def test_returns_false_for_nonexistent_trace(self, wired_store):
        found = await trace_service.record_feedback("does-not-exist", "up")
        assert found is False

    @pytest.mark.asyncio
    async def test_returns_false_when_unwired(self):
        trace_service._agent_traces_table = None
        trace_service._async_session_factory = None
        found = await trace_service.record_feedback("any-id", "up")
        assert found is False

    @pytest.mark.asyncio
    async def test_updates_existing_trace_and_returns_true(self, wired_store):
        table, session_factory = wired_store

        trace_id = await trace_service.log_trace(
            run_id="r1", session_id="s1", pipeline="chat", question="q",
        )
        await _drain_pending_writes()

        found = await trace_service.record_feedback(trace_id, "down")
        assert found is True

        async with session_factory() as session:
            result = await session.execute(select(table).where(table.c.id == trace_id))
            row = result.first()
        assert row.user_feedback == "down"

    @pytest.mark.asyncio
    async def test_rejects_invalid_feedback_before_touching_db(self, wired_store):
        """Validation happens before any DB call — 'sideways' must never be
        written, even if a row with this id exists."""
        table, session_factory = wired_store
        trace_id = await trace_service.log_trace(
            run_id="r1", session_id="s1", pipeline="chat", question="q",
        )
        await _drain_pending_writes()

        with pytest.raises(ValueError):
            await trace_service.record_feedback(trace_id, "sideways")

        async with session_factory() as session:
            result = await session.execute(select(table).where(table.c.id == trace_id))
            row = result.first()
        assert row.user_feedback is None
