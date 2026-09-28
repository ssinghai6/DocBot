"""LLM call log persistence — DOCBOT-1501.

Persists one row per LLM call (provider, model, run_id, caller, tokens,
latency, cost, success/fallback) to PostgreSQL so multi-step Autopilot /
Deep Research / hybrid investigations can be reconstructed after the fact —
not just grepped out of stdout.

Design notes
------------
``api/utils/llm_provider.py`` is the single choke point for every LLM call
(``call_llm`` / ``chat_completion`` / ``chat_completion_stream`` /
``log_external_llm_call``) and already builds a structured JSON payload per
call in ``_log_llm_call``. That function runs from a mix of contexts:

  * the FastAPI event loop thread directly (e.g. ``chat_completion_stream``
    called inline inside an ``async def`` generator), and
  * a plain worker thread with **no** running event loop (functions invoked
    via ``loop.run_in_executor(...)``).

To persist from both without blocking either, ``_log_llm_call`` hands its
payload to :func:`enqueue_call`, a synchronous, non-blocking call backed by
a stdlib ``queue.Queue`` (thread-safe, no event loop required). A single
background asyncio task (:func:`writer_loop`, started from the FastAPI
lifespan) drains the queue and performs the actual async DB insert. A full
queue drops the oldest-pending write with a warning rather than applying
backpressure to the request path — tracing must never slow down or break a
real LLM call.

No prompt/response text, credentials, or PII are ever enqueued — only the
metadata fields already present in the llm_call log payload.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Table,
    Text,
    func,
    select,
)
from sqlalchemy import insert as sa_insert

logger = logging.getLogger(__name__)

_MAX_QUEUE_SIZE = 2000

# Sentinel used to unblock the writer loop's blocking queue.get() on shutdown.
_SHUTDOWN_SENTINEL = object()


# ---------------------------------------------------------------------------
# Table definition
# ---------------------------------------------------------------------------


def register_llm_calls_table(metadata) -> Table:
    """Define the llm_calls table on shared metadata.

    Called once at import time from ``api/index.py``, mirroring the
    ``register_connector_tables`` / ``register_commerce_tables`` pattern.
    """
    llm_calls = Table(
        "llm_calls",
        metadata,
        Column("id", String, primary_key=True),               # UUID
        Column("run_id", String, nullable=False, index=True), # shared trace id across a multi-step run
        Column("provider", String, nullable=False),           # groq | gemini
        Column("model", String, nullable=False),
        Column("caller", String, index=True),                 # e.g. "autopilot_planner", "sql_gen"
        Column("latency_ms", Float),
        Column("input_tokens", Integer),
        Column("output_tokens", Integer),
        Column("estimated_cost_usd", Float),
        Column("success", Boolean, nullable=False),
        Column("fallback_triggered", Boolean, nullable=False, server_default="false"),
        Column("error_message", Text),                        # sanitized, truncated — no payload/creds
        Column(
            "created_at",
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
            index=True,
        ),
    )
    return llm_calls


# ---------------------------------------------------------------------------
# Module-level wiring
# ---------------------------------------------------------------------------

_llm_calls_table: Optional[Table] = None
_async_session_factory: Any = None
_queue: "queue.Queue[Any]" = queue.Queue(maxsize=_MAX_QUEUE_SIZE)
_writer_task: Optional[asyncio.Task] = None


def wire_llm_trace_store(llm_calls_table: Table, async_session_factory: Any) -> None:
    """Inject table + session references. Called once from index.py lifespan."""
    global _llm_calls_table, _async_session_factory
    _llm_calls_table = llm_calls_table
    _async_session_factory = async_session_factory


# ---------------------------------------------------------------------------
# Enqueue (called synchronously from llm_provider — must never block/raise)
# ---------------------------------------------------------------------------


def enqueue_call(payload: dict) -> None:
    """Non-blocking enqueue of one llm_call payload for background persistence.

    Safe to call from any thread (worker thread or the event loop thread) —
    never awaits, never raises. Drops the call with a warning if the queue
    is full (a saturated queue means the writer is falling behind; better to
    lose a trace row than to slow down or crash a real LLM call).
    """
    try:
        _queue.put_nowait(dict(payload))
    except queue.Full:
        logger.warning("llm_trace_service: queue full, dropping llm_call trace row")
    except Exception as exc:  # never let tracing break the caller
        logger.debug("llm_trace_service: enqueue failed (%s)", exc)


# ---------------------------------------------------------------------------
# Background writer
# ---------------------------------------------------------------------------


async def _persist_one(payload: dict) -> None:
    if _llm_calls_table is None or _async_session_factory is None:
        return

    row = {
        "id": str(uuid.uuid4()),
        "run_id": payload.get("run_id") or "unknown",
        "provider": payload.get("llm_provider") or payload.get("provider") or "unknown",
        "model": payload.get("llm_model") or payload.get("model") or "unknown",
        "caller": payload.get("llm_caller") or payload.get("caller"),
        "latency_ms": payload.get("llm_latency_ms") or payload.get("latency_ms"),
        "input_tokens": payload.get("llm_input_tokens") or payload.get("input_tokens"),
        "output_tokens": payload.get("llm_output_tokens") or payload.get("output_tokens"),
        "estimated_cost_usd": payload.get("llm_estimated_cost_usd") or payload.get("estimated_cost_usd"),
        "success": bool(payload.get("llm_success", payload.get("success", True))),
        "fallback_triggered": bool(
            payload.get("llm_fallback_triggered", payload.get("fallback_triggered", False))
        ),
        "error_message": _sanitize_error(payload.get("error_message")),
    }

    async with _async_session_factory() as session:
        async with session.begin():
            await session.execute(sa_insert(_llm_calls_table).values(**row))


def _sanitize_error(message: Optional[str]) -> Optional[str]:
    """Truncate error text and never let it carry payload/credential content.

    Callers only ever pass short exception-type strings here, but cap
    defensively since this lands in a persisted table.
    """
    if not message:
        return None
    return str(message)[:500]


async def writer_loop() -> None:
    """Drain the queue and persist rows one at a time.

    Runs as a long-lived background asyncio task (started from the FastAPI
    lifespan). Uses ``asyncio.to_thread`` for the blocking ``queue.get()``
    so it never busy-polls or blocks the event loop. A single failed insert
    is logged and dropped — it must never crash the loop.
    """
    logger.info("llm_trace_service: writer loop started")
    while True:
        payload = await asyncio.to_thread(_queue.get)
        if payload is _SHUTDOWN_SENTINEL:
            break
        try:
            await _persist_one(payload)
        except Exception as exc:
            logger.warning("llm_trace_service: failed to persist llm_call row: %s", exc)
    logger.info("llm_trace_service: writer loop stopped")


def start_writer() -> asyncio.Task:
    """Start the background writer task. Idempotent — returns the existing
    task if already running."""
    global _writer_task
    if _writer_task is None or _writer_task.done():
        _writer_task = asyncio.create_task(writer_loop())
    return _writer_task


async def stop_writer(timeout: float = 5.0) -> None:
    """Signal the writer loop to stop and wait for it to drain/exit."""
    global _writer_task
    if _writer_task is None:
        return
    try:
        _queue.put_nowait(_SHUTDOWN_SENTINEL)
    except queue.Full:
        pass
    try:
        await asyncio.wait_for(_writer_task, timeout=timeout)
    except (asyncio.TimeoutError, Exception) as exc:
        logger.warning("llm_trace_service: writer did not stop cleanly (%s)", exc)
    _writer_task = None


# ---------------------------------------------------------------------------
# Read path — DOCBOT-1502 (cost/latency dashboard) reads through this
# ---------------------------------------------------------------------------


async def get_call_stats(since: Optional[datetime] = None) -> list[dict[str, Any]]:
    """Return raw llm_calls rows (metadata only) since a given time.

    Kept intentionally thin — DOCBOT-1502's aggregation (per-day, per-caller,
    P50/P95 latency) lives in ``api/metrics_service.py``, which calls this
    for the raw rows and aggregates in Python (row volume is small enough
    for a solo-deploy container; no need for a window-function query yet).
    """
    if _llm_calls_table is None or _async_session_factory is None:
        return []

    stmt = select(_llm_calls_table)
    if since is not None:
        stmt = stmt.where(_llm_calls_table.c.created_at >= since)

    async with _async_session_factory() as session:
        result = await session.execute(stmt)
        rows = result.all()

    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({
            "id": r.id,
            "run_id": r.run_id,
            "provider": r.provider,
            "model": r.model,
            "caller": r.caller,
            "latency_ms": r.latency_ms,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "estimated_cost_usd": r.estimated_cost_usd,
            "success": r.success,
            "fallback_triggered": r.fallback_triggered,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })
    return out
