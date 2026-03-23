"""
DOCBOT-501: Session Artifact Store

Persists DataFrames and charts produced by the E2B sandbox across turns so the
Analytical Autopilot (DOCBOT-405) can reference prior results without re-running
queries.

Public API
----------
save_artifact(session_id, turn_id, artifact_type, name, *, df=None, chart_b64=None,
              session_artifacts_table, async_session_factory) -> str
list_artifacts(session_id, session_artifacts_table, async_session_factory)
    -> list[ArtifactSummary]
get_artifact(artifact_id, session_artifacts_table, async_session_factory)
    -> ArtifactDetail | None
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from pydantic import BaseModel
from sqlalchemy import Table, select
from sqlalchemy.ext.asyncio import async_sessionmaker

logger = logging.getLogger(__name__)

_MAX_ROWS = 500  # cap serialized DataFrame rows to keep storage bounded


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ArtifactSummary(BaseModel):
    """Lightweight artifact record returned by list_artifacts()."""
    id: str
    session_id: str
    turn_id: int
    artifact_type: str   # 'dataframe' | 'chart' | 'sql_result'
    name: str
    row_count: Optional[int]
    columns: Optional[list[str]]
    has_chart: bool
    created_at: str


class ArtifactDetail(ArtifactSummary):
    """Full artifact record including serialized data and chart."""
    data_json: Optional[str]   # JSON (records orient), None for chart-only artifacts
    chart_b64: Optional[str]   # base64 PNG, None for data-only artifacts


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

async def save_artifact(
    session_id: str,
    turn_id: int,
    artifact_type: str,
    name: str,
    *,
    result_dicts: Optional[list[dict]] = None,
    chart_b64: Optional[str] = None,
    session_artifacts_table: Table,
    async_session_factory: async_sessionmaker,
) -> str:
    """Persist a DataFrame snapshot and/or chart PNG from an E2B execution.

    Parameters
    ----------
    session_id      : DocBot session this artifact belongs to
    turn_id         : 1-based conversation turn number (used for ordering)
    artifact_type   : 'dataframe' | 'chart' | 'sql_result'
    name            : Human-readable name (e.g. "Q3 Revenue by Region")
    result_dicts    : Raw query result rows as list[dict] (truncated to 500)
    chart_b64       : Base64-encoded PNG string (no data-URI prefix)
    session_artifacts_table : SQLAlchemy Table object
    async_session_factory   : Async session factory

    Returns
    -------
    artifact_id : str  UUID identifying the newly saved artifact
    """
    artifact_id = str(uuid.uuid4())

    # Serialize DataFrame rows (capped at _MAX_ROWS)
    data_json: Optional[str] = None
    columns: Optional[list[str]] = None
    row_count: Optional[int] = None

    if result_dicts:
        truncated = result_dicts[:_MAX_ROWS]
        row_count = len(result_dicts)  # report original row count, not truncated
        data_json = json.dumps(truncated)
        if truncated:
            columns = list(truncated[0].keys())

    columns_json = json.dumps(columns) if columns else None

    try:
        async with async_session_factory() as session:
            async with session.begin():
                await session.execute(
                    session_artifacts_table.insert().values(
                        id=artifact_id,
                        session_id=session_id,
                        turn_id=turn_id,
                        artifact_type=artifact_type,
                        name=name,
                        data_json=data_json,
                        chart_b64=chart_b64,
                        row_count=row_count,
                        columns=columns_json,
                    )
                )
        logger.info(
            "save_artifact: session=%s turn=%d type=%s rows=%s has_chart=%s id=%s",
            session_id, turn_id, artifact_type, row_count, chart_b64 is not None, artifact_id,
        )
    except Exception as exc:
        logger.warning("save_artifact failed (non-fatal): %s", exc)
        # Return the ID anyway so the caller can still reference it in the SSE stream
    return artifact_id


async def list_artifacts(
    session_id: str,
    session_artifacts_table: Table,
    async_session_factory: async_sessionmaker,
) -> list[ArtifactSummary]:
    """Return all artifacts for a session, ordered by turn (oldest first)."""
    try:
        async with async_session_factory() as session:
            result = await session.execute(
                select(
                    session_artifacts_table.c.id,
                    session_artifacts_table.c.session_id,
                    session_artifacts_table.c.turn_id,
                    session_artifacts_table.c.artifact_type,
                    session_artifacts_table.c.name,
                    session_artifacts_table.c.row_count,
                    session_artifacts_table.c.columns,
                    session_artifacts_table.c.chart_b64,
                    session_artifacts_table.c.created_at,
                )
                .where(session_artifacts_table.c.session_id == session_id)
                .order_by(session_artifacts_table.c.turn_id.asc())
            )
            rows = result.fetchall()

        summaries: list[ArtifactSummary] = []
        for row in rows:
            cols = json.loads(row.columns) if row.columns else None
            summaries.append(ArtifactSummary(
                id=row.id,
                session_id=row.session_id,
                turn_id=row.turn_id,
                artifact_type=row.artifact_type,
                name=row.name,
                row_count=row.row_count,
                columns=cols,
                has_chart=row.chart_b64 is not None,
                created_at=str(row.created_at),
            ))
        return summaries
    except Exception as exc:
        logger.warning("list_artifacts failed: %s", exc)
        return []


async def get_artifact(
    artifact_id: str,
    session_artifacts_table: Table,
    async_session_factory: async_sessionmaker,
) -> Optional[ArtifactDetail]:
    """Fetch a single artifact by ID, including full data and chart."""
    try:
        async with async_session_factory() as session:
            result = await session.execute(
                select(session_artifacts_table)
                .where(session_artifacts_table.c.id == artifact_id)
            )
            row = result.fetchone()

        if row is None:
            return None

        cols = json.loads(row.columns) if row.columns else None
        return ArtifactDetail(
            id=row.id,
            session_id=row.session_id,
            turn_id=row.turn_id,
            artifact_type=row.artifact_type,
            name=row.name,
            row_count=row.row_count,
            columns=cols,
            has_chart=row.chart_b64 is not None,
            created_at=str(row.created_at),
            data_json=row.data_json,
            chart_b64=row.chart_b64,
        )
    except Exception as exc:
        logger.warning("get_artifact failed: %s", exc)
        return None
