"""Unit tests for api.eval_service — LLM-judge offline eval harness (DOCBOT-1519).

Uses an in-memory SQLite async engine for agent_traces/eval_judgments (same
approach as tests/unit/test_trace_service.py) and mocks
api.utils.llm_provider.call_llm so no live LLM call is ever made.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import MetaData, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api import eval_service, trace_service


@pytest.fixture
async def wired_stores():
    """Wire both agent_traces and eval_judgments against one in-memory engine."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    metadata = MetaData()
    traces_table = trace_service.register_agent_traces_table(metadata)
    judgments_table = eval_service.register_eval_judgments_table(metadata)

    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    trace_service.wire_trace_store(traces_table, session_factory)
    eval_service.wire_eval_store(judgments_table, session_factory)

    yield traces_table, judgments_table, session_factory

    trace_service._agent_traces_table = None
    trace_service._async_session_factory = None
    eval_service._eval_judgments_table = None
    eval_service._async_session_factory = None
    await engine.dispose()


async def _drain_pending_writes() -> None:
    pending = list(trace_service._pending_writes)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


# ---------------------------------------------------------------------------
# Table registration
# ---------------------------------------------------------------------------


class TestRegisterTable:
    def test_table_has_expected_columns(self):
        metadata = MetaData()
        table = eval_service.register_eval_judgments_table(metadata)
        col_names = {c.name for c in table.columns}
        assert col_names == {
            "id", "trace_id", "tool_selection_verdict", "tool_selection_reason",
            "evidence_support_verdict", "evidence_support_reason", "judged_at",
        }

    def test_table_name(self):
        metadata = MetaData()
        table = eval_service.register_eval_judgments_table(metadata)
        assert table.name == "eval_judgments"


# ---------------------------------------------------------------------------
# sample_recent_traces
# ---------------------------------------------------------------------------


class TestSampleRecentTraces:
    @pytest.mark.asyncio
    async def test_returns_empty_list_when_unwired(self):
        trace_service._agent_traces_table = None
        trace_service._async_session_factory = None
        eval_service._async_session_factory = None
        result = await eval_service.sample_recent_traces()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_expected_shape(self, wired_stores):
        traces_table, judgments_table, session_factory = wired_stores

        trace_id = await trace_service.log_trace(
            run_id="r1", session_id="s1", pipeline="autopilot",
            question="What drove the Q4 revenue miss?",
            intent_classified="analytical", tool_chosen="doc_search",
            retrieved_refs=[{"source": "10-K.pdf", "page": 7}],
            final_answer="Revenue missed due to...",
        )
        await _drain_pending_writes()

        result = await eval_service.sample_recent_traces(pipeline="autopilot")

        assert len(result) == 1
        row = result[0]
        assert row["id"] == trace_id
        assert row["pipeline"] == "autopilot"
        assert row["question"] == "What drove the Q4 revenue miss?"
        assert row["tool_chosen"] == "doc_search"

    @pytest.mark.asyncio
    async def test_filters_by_pipeline(self, wired_stores):
        await trace_service.log_trace(
            run_id="r1", session_id="s1", pipeline="autopilot", question="q1",
        )
        await trace_service.log_trace(
            run_id="r2", session_id="s2", pipeline="hybrid", question="q2",
        )
        await _drain_pending_writes()

        result = await eval_service.sample_recent_traces(pipeline="hybrid")
        assert len(result) == 1
        assert result[0]["pipeline"] == "hybrid"

    @pytest.mark.asyncio
    async def test_respects_limit(self, wired_stores):
        for i in range(5):
            await trace_service.log_trace(
                run_id=f"r{i}", session_id="s1", pipeline="chat", question=f"q{i}",
            )
        await _drain_pending_writes()

        result = await eval_service.sample_recent_traces(limit=2)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# judge_trace
# ---------------------------------------------------------------------------


SAMPLE_TRACE = {
    "id": "trace-1",
    "question": "What was Q4 net income?",
    "intent_classified": "db_chat",
    "tool_chosen": "run_sql_pipeline",
    "retrieved_refs_json": '[{"source": "financials.db"}]',
    "final_answer": "Q4 net income was $325M.",
}


class TestJudgeTrace:
    @pytest.mark.asyncio
    async def test_returns_well_formed_verdict_dict(self, monkeypatch):
        async def _fake_call_llm(prompt, **kwargs):
            return (
                '{"tool_selection_verdict": "correct", '
                '"tool_selection_reason": "SQL pipeline is right for a numeric lookup.", '
                '"evidence_support_verdict": "supported", '
                '"evidence_support_reason": "The figure matches the retrieved row."}'
            )

        monkeypatch.setattr("api.utils.llm_provider.call_llm", _fake_call_llm)

        result = await eval_service.judge_trace(SAMPLE_TRACE)

        assert result["trace_id"] == "trace-1"
        assert result["tool_selection_verdict"] == "correct"
        assert "SQL pipeline" in result["tool_selection_reason"]
        assert result["evidence_support_verdict"] == "supported"
        assert "matches" in result["evidence_support_reason"]

    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self, monkeypatch):
        async def _fake_call_llm(prompt, **kwargs):
            return (
                '```json\n'
                '{"tool_selection_verdict": "incorrect", '
                '"tool_selection_reason": "Wrong tool.", '
                '"evidence_support_verdict": "unsupported", '
                '"evidence_support_reason": "No matching evidence."}'
                '\n```'
            )

        monkeypatch.setattr("api.utils.llm_provider.call_llm", _fake_call_llm)

        result = await eval_service.judge_trace(SAMPLE_TRACE)
        assert result["tool_selection_verdict"] == "incorrect"
        assert result["evidence_support_verdict"] == "unsupported"

    @pytest.mark.asyncio
    async def test_clamps_out_of_vocabulary_verdict_to_uncertain(self, monkeypatch):
        async def _fake_call_llm(prompt, **kwargs):
            return (
                '{"tool_selection_verdict": "maybe", '
                '"tool_selection_reason": "Hard to tell.", '
                '"evidence_support_verdict": "supported", '
                '"evidence_support_reason": "Looks fine."}'
            )

        monkeypatch.setattr("api.utils.llm_provider.call_llm", _fake_call_llm)

        result = await eval_service.judge_trace(SAMPLE_TRACE)
        assert result["tool_selection_verdict"] == "uncertain"
        assert result["evidence_support_verdict"] == "supported"

    @pytest.mark.asyncio
    async def test_never_raises_on_llm_call_exception(self, monkeypatch):
        async def _boom(prompt, **kwargs):
            raise RuntimeError("Groq and Gemini both down")

        monkeypatch.setattr("api.utils.llm_provider.call_llm", _boom)

        result = await eval_service.judge_trace(SAMPLE_TRACE)

        assert result["trace_id"] == "trace-1"
        assert result["tool_selection_verdict"] == "uncertain"
        assert result["evidence_support_verdict"] == "uncertain"
        assert "judge call failed" in result["tool_selection_reason"]
        assert "judge call failed" in result["evidence_support_reason"]

    @pytest.mark.asyncio
    async def test_never_raises_on_malformed_json_response(self, monkeypatch):
        async def _fake_call_llm(prompt, **kwargs):
            return "not json at all"

        monkeypatch.setattr("api.utils.llm_provider.call_llm", _fake_call_llm)

        result = await eval_service.judge_trace(SAMPLE_TRACE)

        assert result["tool_selection_verdict"] == "uncertain"
        assert result["evidence_support_verdict"] == "uncertain"
        assert "judge call failed" in result["tool_selection_reason"]


# ---------------------------------------------------------------------------
# run_eval_batch
# ---------------------------------------------------------------------------


class TestRunEvalBatch:
    @pytest.mark.asyncio
    async def test_returns_empty_summary_with_no_traces(self, wired_stores):
        summary = await eval_service.run_eval_batch(pipeline="autopilot")
        assert summary["sampled"] == 0
        assert summary["tool_selection_counts"] == {}
        assert summary["evidence_support_counts"] == {}
        assert summary["flagged"] == []
        assert summary["estimated_cost_usd"] == 0.0

    @pytest.mark.asyncio
    async def test_aggregates_verdicts_from_judge_trace(self, wired_stores, monkeypatch):
        traces_table, judgments_table, session_factory = wired_stores

        for i in range(3):
            await trace_service.log_trace(
                run_id=f"r{i}", session_id="s1", pipeline="autopilot", question=f"q{i}",
            )
        await _drain_pending_writes()

        verdicts = [
            {"trace_id": "t0", "tool_selection_verdict": "correct",
             "tool_selection_reason": "ok", "evidence_support_verdict": "supported",
             "evidence_support_reason": "ok"},
            {"trace_id": "t1", "tool_selection_verdict": "incorrect",
             "tool_selection_reason": "wrong tool", "evidence_support_verdict": "unsupported",
             "evidence_support_reason": "no evidence"},
            {"trace_id": "t2", "tool_selection_verdict": "uncertain",
             "tool_selection_reason": "unclear", "evidence_support_verdict": "supported",
             "evidence_support_reason": "ok"},
        ]
        calls = iter(verdicts)

        async def _fake_judge_trace(trace):
            return next(calls)

        monkeypatch.setattr(eval_service, "judge_trace", _fake_judge_trace)

        summary = await eval_service.run_eval_batch(pipeline="autopilot", limit=10)

        assert summary["sampled"] == 3
        assert summary["tool_selection_counts"] == {"correct": 1, "incorrect": 1, "uncertain": 1}
        assert summary["evidence_support_counts"] == {"supported": 2, "unsupported": 1}
        # Only the fully-clean correct/supported judgment is excluded from flagged.
        assert len(summary["flagged"]) == 2
        flagged_ids = {j["trace_id"] for j in summary["flagged"]}
        assert flagged_ids == {"t1", "t2"}

    @pytest.mark.asyncio
    async def test_persists_judgments(self, wired_stores, monkeypatch):
        traces_table, judgments_table, session_factory = wired_stores

        await trace_service.log_trace(
            run_id="r1", session_id="s1", pipeline="hybrid", question="q",
        )
        await _drain_pending_writes()

        async def _fake_judge_trace(trace):
            return {
                "trace_id": trace["id"],
                "tool_selection_verdict": "correct",
                "tool_selection_reason": "fine",
                "evidence_support_verdict": "supported",
                "evidence_support_reason": "fine",
            }

        monkeypatch.setattr(eval_service, "judge_trace", _fake_judge_trace)

        summary = await eval_service.run_eval_batch(pipeline="hybrid", limit=10)
        assert summary["sampled"] == 1

        async with session_factory() as session:
            result = await session.execute(select(judgments_table))
            rows = result.all()
        assert len(rows) == 1
        assert rows[0].tool_selection_verdict == "correct"
        assert rows[0].evidence_support_verdict == "supported"
