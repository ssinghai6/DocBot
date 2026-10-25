"""Usage ledger + metering hooks — DOCBOT-1801 (EPIC-18).

Builds on the DOCBOT-1800 contracts in ``api/utils/billing_context.py``:

* Every metered unit of work (an LLM call, an E2B sandbox run, a file upload)
  is recorded twice:
    1. on the request's ``RequestMeter`` (if one is bound) so DOCBOT-1802 can
       settle the credit reservation from ``meter.snapshot()``; and
    2. as one append-only row in ``usage_events`` for the per-org ledger.
* Attribution (``org_id`` / ``user_id`` / ``session_id`` / ``key_source``) is
  read from the ContextVars — callers never thread it through signatures.
* Only ``KeyContext.key_source`` is ever read. ``KeyContext.reveal()`` is
  never called here, so key material cannot reach a row or a log line.

Writes never block or break the request path: rows go onto a bounded,
thread-safe ``queue.Queue`` (callable from worker threads with no event
loop) and a single background task drains it into Postgres, mirroring
``api/llm_trace_service.py``. A full queue or a failed insert is logged and
dropped.

Money is integer micro-USD end to end (``cost_micro_usd`` BIGINT).
"""

from __future__ import annotations

import asyncio
import logging
import math
import queue
import uuid
from decimal import Decimal
from typing import Any, AsyncIterator, Optional

from sqlalchemy import BigInteger, Column, DateTime, Index, Integer, String, Table, func
from sqlalchemy import insert as sa_insert
from sqlalchemy.exc import SQLAlchemyError

from api.utils.billing_context import (
    capture_billing_context,
    current_key_source,
    get_meter,
    get_principal,
    usd_to_micro_usd,
    use_billing_context,
)

logger = logging.getLogger(__name__)

_MAX_QUEUE_SIZE = 5000
_SHUTDOWN_SENTINEL = object()

USAGE_KINDS = ("llm", "sandbox", "upload")

# Approximate E2B list price for the default code-interpreter sandbox
# (2 vCPU x $0.000014/vCPU-s). Integer micro-USD per second; margin data only,
# never shown to customers. Revisit when E2B pricing changes.
E2B_MICRO_USD_PER_SECOND = 28

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# DOCBOT-1801: billing attribution on the pre-existing llm_calls table. No-op
# on a fresh DB (register_llm_calls_table already declares the columns).
LLM_CALLS_BILLING_MIGRATIONS: list[str] = [
    "ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS org_id TEXT",
    "ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS user_id TEXT",
    "ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS key_source TEXT",
    "CREATE INDEX IF NOT EXISTS ix_llm_calls_org_id ON llm_calls (org_id)",
]


def register_usage_events_table(metadata: Any) -> Table:
    """Define ``usage_events`` on the shared metadata (created by create_all)."""
    return Table(
        "usage_events",
        metadata,
        Column("id", String, primary_key=True),  # UUID
        Column("org_id", String),
        Column("user_id", String),
        Column("session_id", String),
        Column("run_id", String),  # joins to llm_calls.run_id
        Column("kind", String, nullable=False),  # llm | sandbox | upload
        Column("provider", String),
        Column("model", String),
        Column("prompt_tokens", Integer, nullable=False, server_default="0"),
        Column("completion_tokens", Integer, nullable=False, server_default="0"),
        Column("cost_micro_usd", BigInteger, nullable=False, server_default="0"),
        Column("sandbox_ms", BigInteger, nullable=False, server_default="0"),
        Column("bytes", BigInteger, nullable=False, server_default="0"),
        Column("pages", Integer),  # uploads only, when known
        Column("key_source", String, nullable=False, server_default="platform"),
        Column(
            "created_at",
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
        ),
        Index("ix_usage_events_org_created", "org_id", "created_at"),
    )


# ---------------------------------------------------------------------------
# Wiring + background writer
# ---------------------------------------------------------------------------

_usage_table: Optional[Table] = None
_async_session_factory: Any = None
_queue: "queue.Queue[Any]" = queue.Queue(maxsize=_MAX_QUEUE_SIZE)
_writer_task: Optional[asyncio.Task] = None


def wire_usage_store(usage_table: Table, async_session_factory: Any) -> None:
    """Inject table + session factory. Called once from the index.py lifespan."""
    global _usage_table, _async_session_factory
    _usage_table = usage_table
    _async_session_factory = async_session_factory


def _enqueue(row: dict) -> None:
    try:
        _queue.put_nowait(row)
    except queue.Full:
        logger.warning("usage_service: queue full, dropping usage_events row kind=%s", row.get("kind"))


async def _persist_one(row: dict) -> None:
    if _usage_table is None or _async_session_factory is None:
        return
    async with _async_session_factory() as session:
        async with session.begin():
            await session.execute(sa_insert(_usage_table).values(**row))


async def writer_loop() -> None:
    """Drain the queue into ``usage_events``. A failed insert is dropped."""
    logger.info("usage_service: writer loop started")
    while True:
        row = await asyncio.to_thread(_queue.get)
        if row is _SHUTDOWN_SENTINEL:
            break
        try:
            await _persist_one(row)
        except (SQLAlchemyError, OSError) as exc:
            logger.warning("usage_service: failed to persist usage row: %s", type(exc).__name__)
        except Exception as exc:  # keep the writer alive; metering must not die silently
            logger.warning("usage_service: unexpected persist error: %s", type(exc).__name__)
    logger.info("usage_service: writer loop stopped")


def start_writer() -> asyncio.Task:
    """Start the background writer. Idempotent."""
    global _writer_task
    if _writer_task is None or _writer_task.done():
        _writer_task = asyncio.create_task(writer_loop())
    return _writer_task


async def stop_writer(timeout: float = 5.0) -> None:
    global _writer_task
    if _writer_task is None:
        return
    try:
        _queue.put_nowait(_SHUTDOWN_SENTINEL)
    except queue.Full:
        logger.warning("usage_service: queue full at shutdown; writer cancelled")
        _writer_task.cancel()
    try:
        await asyncio.wait_for(_writer_task, timeout=timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        logger.warning("usage_service: writer did not stop cleanly")
    _writer_task = None


# ---------------------------------------------------------------------------
# Attribution (ContextVars -> plain fields; never key material)
# ---------------------------------------------------------------------------


def billing_attribution() -> dict[str, Optional[str]]:
    """``org_id`` / ``user_id`` / ``session_id`` / ``key_source`` for the current context."""
    principal = get_principal()
    return {
        "org_id": principal.org_id if principal else None,
        "user_id": principal.user_id if principal else None,
        "session_id": principal.session_id if principal else None,
        "key_source": current_key_source(),
    }


def record_usage(
    kind: str,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_micro_usd: int = 0,
    sandbox_ms: int = 0,
    size_bytes: int = 0,
    pages: Optional[int] = None,
    session_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    """Queue one ``usage_events`` row attributed from the ContextVars. Never raises."""
    if kind not in USAGE_KINDS:
        logger.warning("usage_service: unknown usage kind %r ignored", kind)
        return
    attr = billing_attribution()
    row = {
        "id": str(uuid.uuid4()),
        "org_id": attr["org_id"],
        "user_id": attr["user_id"],
        "session_id": session_id or attr["session_id"],
        "run_id": run_id,
        "kind": kind,
        "provider": provider,
        "model": model,
        "prompt_tokens": max(int(prompt_tokens or 0), 0),
        "completion_tokens": max(int(completion_tokens or 0), 0),
        "cost_micro_usd": max(int(cost_micro_usd or 0), 0),
        "sandbox_ms": max(int(sandbox_ms or 0), 0),
        "bytes": max(int(size_bytes or 0), 0),
        "pages": pages,
        "key_source": attr["key_source"],
    }
    _enqueue(row)


# ---------------------------------------------------------------------------
# Metering hooks
# ---------------------------------------------------------------------------


def meter_llm_call(
    *,
    provider: str,
    model: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    estimated_cost_usd: Optional[float],
    run_id: Optional[str] = None,
) -> None:
    """Record one LLM call on the request meter + usage ledger. Never raises."""
    prompt = max(int(input_tokens or 0), 0)
    completion = max(int(output_tokens or 0), 0)
    try:
        micro = usd_to_micro_usd(estimated_cost_usd) if estimated_cost_usd else 0
    except ValueError:
        micro = 0
    meter = get_meter()
    if meter is not None:
        meter.record_llm(prompt, completion, Decimal(micro).scaleb(-6))
    record_usage(
        "llm",
        provider=provider,
        model=model,
        prompt_tokens=prompt,
        completion_tokens=completion,
        cost_micro_usd=micro,
        run_id=run_id,
    )


def sandbox_cost_micro_usd(sandbox_ms: int) -> int:
    return int(math.ceil(max(sandbox_ms, 0) * E2B_MICRO_USD_PER_SECOND / 1000))


def meter_sandbox(elapsed_seconds: float, *, model: str = "code-interpreter") -> None:
    """Record E2B wall-clock time. Call from a ``finally``. Never raises."""
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        elapsed_seconds = 0.0
    sandbox_ms = int(math.ceil(elapsed_seconds * 1000))
    micro = sandbox_cost_micro_usd(sandbox_ms)
    meter = get_meter()
    if meter is not None:
        meter.record_sandbox(elapsed_seconds, Decimal(micro).scaleb(-6))
    record_usage("sandbox", provider="e2b", model=model, sandbox_ms=sandbox_ms, cost_micro_usd=micro)


def meter_upload(size_bytes: int, *, pages: Optional[int] = None, session_id: Optional[str] = None) -> None:
    """Record one accepted upload (bytes, pages when known). Never raises."""
    size = max(int(size_bytes or 0), 0)
    meter = get_meter()
    if meter is not None:
        meter.record_upload(size)
    record_usage("upload", size_bytes=size, pages=pages, session_id=session_id)


# ---------------------------------------------------------------------------
# SSE generator rebind
# ---------------------------------------------------------------------------


def bind_billing_stream(agen: AsyncIterator[Any]) -> AsyncIterator[Any]:
    """Wrap an SSE generator so it runs under the *current* billing context.

    Must be called on the request path, before returning the
    ``StreamingResponse`` — the snapshot is captured here, synchronously,
    while the auth dependency's ContextVars are still bound.
    """
    snapshot = capture_billing_context()

    async def _rebound() -> AsyncIterator[Any]:
        with use_billing_context(snapshot):
            async for chunk in agen:
                yield chunk

    return _rebound()


__all__ = [
    "E2B_MICRO_USD_PER_SECOND",
    "LLM_CALLS_BILLING_MIGRATIONS",
    "billing_attribution",
    "bind_billing_stream",
    "meter_llm_call",
    "meter_sandbox",
    "meter_upload",
    "record_usage",
    "register_usage_events_table",
    "sandbox_cost_micro_usd",
    "start_writer",
    "stop_writer",
    "wire_usage_store",
    "writer_loop",
]
