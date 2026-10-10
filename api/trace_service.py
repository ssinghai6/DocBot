"""Agent trace logging + feedback capture — DOCBOT-1512.

Persists one row per agent routing/answer decision (autopilot, hybrid, plain
doc chat) so future eval / fine-tuning work has real production data to work
from, plus a thumbs up/down feedback channel keyed by trace id.

Design notes
------------
This mirrors the ``register_lineage_table`` / ``wire_lineage_store`` /
``emit_lineage`` pattern already used by ``api/lineage_service.py`` (itself
modeled on ``api/llm_trace_service.py``'s ``register_llm_calls_table`` /
``wire_llm_trace_store``):

  * ``register_agent_traces_table(metadata)`` defines the table once at
    import time from ``api/index.py``.
  * ``wire_trace_store(table, async_session_factory)`` injects the live
    table + session factory from the FastAPI lifespan.
  * ``log_trace(...)`` is the hot-path write. It is declared ``async def``
    (per spec) but never *awaits* the actual insert: it mints the trace id
    synchronously (cheap, can't fail), schedules the DB write as a
    fire-and-forget ``asyncio.create_task``, and returns the id immediately.
    Callers can thread the returned id straight into an SSE "done" event
    without blocking the response on a DB round-trip, and the function never
    raises — any DB error is caught and logged inside the background task.
    A strong reference to each task is kept in ``_pending_writes`` so it is
    not garbage-collected mid-flight (same trick as lineage_service).
  * ``record_feedback(...)`` is a plain awaited call from a dedicated,
    already-validated API route (not a latency-sensitive streaming path), so
    it performs a normal blocking update and is allowed to raise ValueError
    on a bad ``feedback`` value.

No prompt/response text beyond the (length-capped) final answer is stored,
and nothing here touches SQL validation, credentials, or PII masking —
callers are expected to pass already-masked text (as the three call sites
already do for what they stream to the client).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, Table, Text, func
from sqlalchemy import insert as sa_insert
from sqlalchemy import update as sa_update

logger = logging.getLogger(__name__)

# Keep persisted question/answer text bounded — these are observability rows,
# not a second copy of the full conversation.
_MAX_QUESTION_LEN = 2000
_MAX_ANSWER_LEN = 4000

VALID_FEEDBACK_VALUES = {"up", "down"}


# ---------------------------------------------------------------------------
# Table definition
# ---------------------------------------------------------------------------


def register_agent_traces_table(metadata: Any) -> Table:
    """Define the agent_traces table on shared metadata.

    Called once at import time from ``api/index.py``, mirroring
    ``register_llm_calls_table`` / ``register_lineage_table``.
    """
    return Table(
        "agent_traces",
        metadata,
        Column("id", String, primary_key=True),  # UUID, minted by log_trace()
        Column("run_id", String, index=True),  # shared trace id across a multi-step run
        Column("session_id", String, index=True),
        Column("pipeline", String, nullable=False, index=True),  # autopilot | hybrid | db_chat | chat
        Column("question", Text, nullable=False),
        Column("intent_classified", String),
        Column("tool_chosen", String),
        Column("plan_steps_json", Text),  # JSON array, nullable
        Column("retrieved_refs_json", Text),  # JSON array (citations/sources), nullable
        Column("final_answer", Text),  # truncated to _MAX_ANSWER_LEN
        Column("latency_ms", Integer),
        Column("cost_usd", Float),
        Column("error", Text),
        Column("user_feedback", String),  # up | down, nullable
        Column("implicit_signal", String),  # reserved for future implicit-feedback signals
        Column(
            "created_at",
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
            index=True,
        ),
    )


# ---------------------------------------------------------------------------
# Module-level wiring
# ---------------------------------------------------------------------------

_agent_traces_table: Optional[Table] = None
_async_session_factory: Any = None
# Strong refs so fire-and-forget writes are not garbage collected mid-flight.
_pending_writes: set[asyncio.Task] = set()


def wire_trace_store(table: Table, async_session_factory: Any) -> None:
    """Inject table + session factory references. Called once from index.py lifespan."""
    global _agent_traces_table, _async_session_factory
    _agent_traces_table = table
    _async_session_factory = async_session_factory


def get_agent_traces_table() -> Optional[Table]:
    """Read-only accessor for the live ``agent_traces`` table reference.

    Added for DOCBOT-1519's eval harness (``api/eval_service.py``), which is
    a read-only consumer of this table and needs it to build a ``select()``
    query. Returns None if the store hasn't been wired yet (e.g. unit tests,
    no DB configured) — callers must treat that as "no data available", not
    an error.
    """
    return _agent_traces_table


# ---------------------------------------------------------------------------
# Write path — fire-and-forget
# ---------------------------------------------------------------------------


async def _insert_trace(values: dict) -> None:
    """Background task body — the only place a DB error can surface, and it
    never propagates past this function."""
    if _agent_traces_table is None or _async_session_factory is None:
        return
    try:
        async with _async_session_factory() as session:
            async with session.begin():
                await session.execute(sa_insert(_agent_traces_table).values(**values))
    except Exception as exc:  # observability must never break a request
        logger.warning("trace_service: failed to persist trace %s: %s", values.get("id"), exc)


async def log_trace(
    run_id: Optional[str],
    session_id: Optional[str],
    pipeline: str,
    question: str,
    *,
    intent_classified: Optional[str] = None,
    tool_chosen: Optional[str] = None,
    plan_steps: Optional[list] = None,
    retrieved_refs: Optional[list] = None,
    final_answer: Optional[str] = None,
    latency_ms: Optional[float] = None,
    cost_usd: Optional[float] = None,
    error: Optional[str] = None,
) -> str:
    """Fire-and-forget agent trace log. Never raises.

    Mints and returns the trace id immediately — the actual DB insert is
    scheduled as a background task and is not awaited here, so this call
    never blocks a streaming response on a DB round-trip.
    """
    trace_id = str(uuid.uuid4())
    try:
        truncated_question = (question or "")[:_MAX_QUESTION_LEN]
        truncated_answer = final_answer[:_MAX_ANSWER_LEN] if final_answer else None
        values = {
            "id": trace_id,
            "run_id": run_id,
            "session_id": session_id,
            "pipeline": pipeline,
            "question": truncated_question,
            "intent_classified": intent_classified,
            "tool_chosen": tool_chosen,
            "plan_steps_json": json.dumps(plan_steps, default=str) if plan_steps is not None else None,
            "retrieved_refs_json": (
                json.dumps(retrieved_refs, default=str) if retrieved_refs is not None else None
            ),
            "final_answer": truncated_answer,
            "latency_ms": int(latency_ms) if latency_ms is not None else None,
            "cost_usd": cost_usd,
            "error": error,
        }
        try:
            task = asyncio.get_running_loop().create_task(_insert_trace(values))
            _pending_writes.add(task)
            task.add_done_callback(_pending_writes.discard)
        except RuntimeError:
            # No running event loop (e.g. sync test code) — skip persistence,
            # the caller still gets a usable trace id back.
            logger.debug("trace_service: no running event loop, trace not persisted")
    except Exception as exc:  # never let tracing break the caller
        logger.warning("trace_service: log_trace failed to schedule write: %s", exc)
    return trace_id


# ---------------------------------------------------------------------------
# Feedback — called from a validated, non-streaming API route
# ---------------------------------------------------------------------------


async def record_feedback(trace_id: str, feedback: str) -> bool:
    """Record a thumbs up/down for a trace. Returns True if a row was updated.

    Raises ValueError if ``feedback`` is not 'up' or 'down' — this path is
    called from an already-validated route, so a bad value here is a
    programming error worth surfacing rather than swallowing.
    """
    if feedback not in VALID_FEEDBACK_VALUES:
        raise ValueError(f"feedback must be 'up' or 'down', got {feedback!r}")

    if _agent_traces_table is None or _async_session_factory is None:
        return False

    async with _async_session_factory() as session:
        async with session.begin():
            result = await session.execute(
                sa_update(_agent_traces_table)
                .where(_agent_traces_table.c.id == trace_id)
                .values(user_feedback=feedback)
            )
    return (result.rowcount or 0) > 0
