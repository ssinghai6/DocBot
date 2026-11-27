"""Recover a hybrid answer whose stream broke off in the browser — DOCBOT-2636.

Three pieces:

* ``detach(agen)`` runs an SSE generator to completion in its own task, so a
  client disconnect no longer cancels the turn before its answer is persisted
  (``log_trace`` at the end of ``hybrid_chat``).
* ``mark_running`` / ``mark_done``: in-process registry of turns still running,
  so the recovery route can tell "not persisted yet" (pending) from "missing".
* ``recover_answer``: owner-checked read of the persisted answer (agent_traces
  ``final_answer`` + ``answer_lineage``). A run the caller cannot access is
  indistinguishable from a missing one (``None`` -> 404).

shortcut: the registry and detached tasks are per process (one uvicorn worker
today); with several workers, route recovery reads to the producing worker or
move the registry to the database.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional

from sqlalchemy import Table, select

from api.answer_models import AnswerCitation, RecoveredAnswer
from api.lineage_service import get_lineage
from api.lineage_service import wait_pending_writes as wait_lineage_writes
from api.resource_scope_service import ids_accessible, run_row_accessible
from api.trace_service import wait_pending_writes as wait_trace_writes
from api.utils.caller_scope import CallerScope

logger = logging.getLogger(__name__)

_running: dict[str, tuple[Optional[str], Optional[str]]] = {}
_detached: set[asyncio.Task[None]] = set()


def mark_running(run_id: str, session_id: Optional[str], connection_id: Optional[str]) -> None:
    _running[run_id] = (session_id, connection_id)


def mark_done(run_id: str) -> None:
    _running.pop(run_id, None)


class _End:
    def __init__(self, error: Optional[Exception]) -> None:
        self.error = error


def detach(agen: AsyncIterator[str]) -> AsyncIterator[str]:
    """Relay ``agen`` while a pump task drives it to the end, client or not."""
    queue: asyncio.Queue[str | _End] = asyncio.Queue()

    async def pump() -> None:
        error: Optional[Exception] = None
        try:
            async for item in agen:
                queue.put_nowait(item)
        except Exception as exc:  # surfaced to the relay below
            error = exc
        finally:
            queue.put_nowait(_End(error))

    async def relay() -> AsyncIterator[str]:
        task = asyncio.create_task(pump())
        _detached.add(task)
        task.add_done_callback(_detached.discard)
        while True:
            item = await queue.get()
            if isinstance(item, _End):
                if item.error is not None:
                    raise item.error
                return
            yield item

    return relay()


async def recover_answer(
    run_id: str,
    scope: CallerScope,
    traces_table: Table,
    sessions_table: Table,
    connections_table: Table,
    async_session_factory: Any,
) -> Optional[RecoveredAnswer]:
    """The caller's persisted hybrid answer, ``pending`` while it runs, else ``None``."""

    def accessible() -> Any:
        return run_row_accessible(
            traces_table, run_id, scope, sessions_table, connections_table, async_session_factory
        )

    if not await accessible():
        running = _running.get(run_id)
        if running is not None:
            owned = await ids_accessible(
                running[0], running[1], scope, sessions_table, connections_table, async_session_factory
            )
            return RecoveredAnswer(run_id=run_id, status="pending") if owned else None
        await wait_trace_writes()  # the turn just ended; its trace insert may be in flight
        if not await accessible():
            return None

    async with async_session_factory() as session:
        row = (
            await session.execute(
                select(traces_table.c.id, traces_table.c.final_answer, traces_table.c.retrieved_refs_json)
                .where(traces_table.c.run_id == run_id, traces_table.c.pipeline == "hybrid")
                .order_by(traces_table.c.created_at.desc())
                .limit(1)
            )
        ).first()
    if row is None or not row.final_answer:
        return None

    citations: list[AnswerCitation] = []
    try:
        refs = json.loads(row.retrieved_refs_json or "[]")
        citations = [AnswerCitation.model_validate(r) for r in refs if isinstance(r, dict)]
    except (ValueError, TypeError) as exc:
        logger.warning("answer_recovery: bad citations for run %s: %s", run_id, exc)

    await wait_lineage_writes()
    return RecoveredAnswer(
        run_id=run_id,
        status="ready",
        answer=row.final_answer,
        citations=citations,
        trace_id=row.id,
        lineage=await get_lineage(run_id),
    )
