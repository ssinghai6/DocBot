"""DOCBOT-1510: answer lineage persistence + SSE emission.

Public API
----------
register_lineage_table(metadata) -> Table
wire_lineage_store(table, async_session_factory) -> None
emit_lineage(collector, session_id=None) -> str
    Build the lineage, persist it fire-and-forget, return the SSE line.
get_lineage(run_id) -> Lineage | None
    Read the persisted record and join llm_calls by run_id.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from sqlalchemy import Column, DateTime, String, Table, Text, func, select
from sqlalchemy import insert as sa_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert

from api.utils.lineage import Lineage, LineageCollector

logger = logging.getLogger(__name__)

_lineage_table: Optional[Table] = None
_async_session_factory: Any = None
# Strong refs so fire-and-forget writes are not garbage collected mid-flight.
_pending_writes: set[asyncio.Task] = set()


def register_lineage_table(metadata) -> Table:
    """Define answer_lineage on shared metadata (called from api/index.py)."""
    return Table(
        "answer_lineage",
        metadata,
        Column("run_id", String, primary_key=True),
        Column("session_id", String, index=True),
        Column("mode", String, nullable=False),
        Column("data_json", Text, nullable=False),
        Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    )


def wire_lineage_store(table: Table, async_session_factory: Any) -> None:
    global _lineage_table, _async_session_factory
    _lineage_table = table
    _async_session_factory = async_session_factory


async def _persist(lineage: Lineage, session_id: Optional[str]) -> None:
    if _lineage_table is None or _async_session_factory is None:
        return
    values = {
        "run_id": lineage.run_id,
        "session_id": session_id,
        "mode": lineage.mode,
        "data_json": lineage.model_dump_json(),
    }
    try:
        async with _async_session_factory() as session:
            async with session.begin():
                dialect = session.bind.dialect.name if session.bind is not None else ""
                if dialect == "postgresql":
                    stmt = pg_insert(_lineage_table).values(**values)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["run_id"],
                        set_={"data_json": stmt.excluded.data_json, "mode": stmt.excluded.mode},
                    )
                    await session.execute(stmt)
                else:
                    await session.execute(
                        _lineage_table.delete().where(_lineage_table.c.run_id == lineage.run_id)
                    )
                    await session.execute(sa_insert(_lineage_table).values(**values))
    except Exception as exc:  # observability must never break a request
        logger.warning("lineage_service: persist failed for %s: %s", lineage.run_id, exc)


def emit_lineage(collector: LineageCollector, session_id: Optional[str] = None) -> str:
    """Finalize ``collector``, persist in the background, return the SSE line.

    Never raises; on failure returns an empty string so callers can
    ``yield`` the result unconditionally only when it is truthy.
    """
    try:
        lineage = collector.build()
        try:
            task = asyncio.get_running_loop().create_task(_persist(lineage, session_id))
            _pending_writes.add(task)
            task.add_done_callback(_pending_writes.discard)
        except RuntimeError:
            logger.debug("lineage_service: no running loop, skipping persist")
        payload = {"type": "lineage", **lineage.model_dump(mode="json")}
        return f"data: {json.dumps(payload)}\n\n"
    except Exception as exc:
        logger.warning("lineage_service: emit failed: %s", exc)
        return ""


async def get_lineage(run_id: str) -> Optional[Lineage]:
    """Load a persisted lineage and attach llm_calls rows for the run."""
    if _lineage_table is None or _async_session_factory is None:
        return None
    async with _async_session_factory() as session:
        result = await session.execute(
            select(_lineage_table.c.data_json).where(_lineage_table.c.run_id == run_id)
        )
        row = result.first()
    if row is None:
        return None
    try:
        lineage = Lineage.model_validate_json(row.data_json)
    except ValueError as exc:
        logger.warning("lineage_service: corrupt lineage %s: %s", run_id, exc)
        return None

    from api.llm_trace_service import get_calls_by_run
    from api.utils.lineage import LineageModelCall

    calls = await get_calls_by_run(run_id)
    lineage.model_calls = [LineageModelCall(**c) for c in calls]
    return lineage
