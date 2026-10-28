"""DOCBOT-2310: Autopilot timeouts still answer, and the stream always ends
with ``done``. The LangGraph app is replaced by a scripted fake; no network."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import api.autopilot_service as ap
from api.utils.step_summary import build_step_summary

STEP = {"step": "Fetch revenue", "tool": "sql_query", "result": "rows", "row_count": 3}


def _events(chunks):
    return [json.loads(c[len("data: "):]) for c in chunks if c.startswith("data: ")]


class _FakeApp:
    def __init__(self, script):
        self._script = script

    def astream(self, *_a, **_kw):
        return self._script()


async def _run(script, synth=None):
    kwargs = {k: MagicMock() for k in (
        "db_connections_table", "schema_cache_table", "query_history_table",
        "query_embeddings_table", "session_artifacts_table", "table_embeddings_table",
        "async_session_factory",
    )}
    with patch.object(ap, "_build_graph", return_value=_FakeApp(script)), \
         patch.object(ap, "make_executor_node", return_value=MagicMock()), \
         patch.object(ap, "_synthesizer_node", synth or AsyncMock(return_value={"final_answer": "Partial answer"})):
        return _events([c async for c in ap.run_autopilot(
            "q", "sess", "conn", "Generalist", expert_personas={}, vector_stores={},
            has_db=True, **kwargs,
        )])


@pytest.mark.asyncio
async def test_overall_timeout_synthesizes_partial_results_and_emits_done(monkeypatch):
    monkeypatch.setattr(ap, "TOTAL_TIMEOUT_S", 0.2)

    async def script():
        yield {"planner": {"plan": ["Fetch revenue", "Fetch costs", "Compare"]}}
        yield {"executor": {"steps_completed": [dict(STEP)], "iteration": 1}}
        await asyncio.sleep(10)  # second wave hangs past the deadline
        yield {}

    synth = AsyncMock(return_value={"final_answer": "Partial answer"})
    events = await _run(script, synth)

    synth.assert_awaited_once()
    state = synth.await_args.args[0]
    assert [s["step"] for s in state["steps_completed"]] == ["Fetch revenue"]
    assert state["timed_out"] is True
    answer = next(e for e in events if e["type"] == "answer")["content"]
    assert "Stopped early after 1 of 3 steps" in answer and "Partial answer" in answer
    assert events[-1]["type"] == "done"
    assert sum(e["type"] == "done" for e in events) == 1


@pytest.mark.asyncio
async def test_partial_synthesis_failure_falls_back_to_step_summaries(monkeypatch):
    monkeypatch.setattr(ap, "TOTAL_TIMEOUT_S", 0.1)

    async def script():
        yield {"planner": {"plan": ["Fetch revenue", "Compare"]}}
        yield {"executor": {"steps_completed": [dict(STEP)]}}
        await asyncio.sleep(10)
        yield {}

    events = await _run(script, AsyncMock(side_effect=RuntimeError("llm down")))
    answer = next(e for e in events if e["type"] == "answer")["content"]
    assert "Queried" in answer and "1 of 2 steps" in answer
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_exception_mid_run_still_emits_done_without_leaking():
    async def script():
        yield {"planner": {"plan": ["Fetch revenue"]}}
        raise RuntimeError("secret db password in traceback")

    events = await _run(script)
    err = next(e for e in events if e["type"] == "error")
    assert "secret" not in err["content"]
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_step_timeout_marks_failed_step_and_run_continues(monkeypatch):
    monkeypatch.setattr(ap, "STEP_TIMEOUT_S", 0.1)

    async def _select(step, **_kw):
        if "revenue" in step:
            await asyncio.sleep(10)
        return "unsupported"

    node = ap.make_executor_node(
        db_connections_table=MagicMock(), schema_cache_table=MagicMock(),
        query_history_table=MagicMock(), query_embeddings_table=MagicMock(),
        session_artifacts_table=MagicMock(), table_embeddings_table=MagicMock(),
        async_session_factory=MagicMock(), expert_personas={}, vector_stores={},
    )
    plan = ["Fetch revenue by region", "Fetch costs by region"]
    state = {
        "question": "q", "session_id": "s", "connection_id": "c", "persona": "Generalist",
        "run_id": "r", "plan": plan, "steps_completed": [], "iteration": 0,
        "has_db": True, "has_docs": False, "has_csv": False,
        "data_results": {}, "degradation_reasons": [],
    }
    with patch.object(ap, "_select_tool_llm", side_effect=_select):
        update = await node(state)

    slow, fast = update["steps_completed"]
    assert slow["failure_reason"] == "step_timeout"
    assert build_step_summary(slow) == "Step failed: timed out"
    assert fast["step"] == "Fetch costs by region"  # sibling still ran
    assert any(r.startswith("step_timeout:") for r in update["degradation_reasons"])
