"""Admin Metrics Service — Investor Readiness Sprint.

Provides aggregate platform metrics for the admin dashboard:
  - Total sessions
  - Total queries by type (doc / db / hybrid / csv)
  - Total documents uploaded
  - Active DB connections
  - Average response time (from audit log metadata)
  - Uptime

All queries run against the existing PostgreSQL tables (sessions,
audit_log, db_connections) using raw SQL for portability.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

# Module-level start time for uptime tracking
_START_TIME = time.monotonic()
_START_DATETIME = datetime.now(timezone.utc)


async def get_platform_metrics(
    async_session_factory: Any,
) -> dict[str, Any]:
    """Compute and return aggregate platform metrics.

    Parameters
    ----------
    async_session_factory : async_sessionmaker
        The async session factory bound to the PostgreSQL database.

    Returns
    -------
    dict
        Metrics dictionary.
    """
    async with async_session_factory() as db:
        # Total sessions
        result = await db.execute(text("SELECT count(*) FROM sessions"))
        total_sessions = result.scalar() or 0

        # Total queries (audit events with event_type = 'query')
        result = await db.execute(
            text("SELECT count(*) FROM audit_log WHERE event_type = 'query'")
        )
        total_queries = result.scalar() or 0

        # Break down queries by type from metadata_json
        queries_by_type = await _count_queries_by_type(db)

        # Total documents uploaded
        result = await db.execute(
            text("SELECT count(*) FROM audit_log WHERE event_type = 'upload'")
        )
        total_uploads = result.scalar() or 0

        # Active DB connections (non-CSV)
        result = await db.execute(
            text("SELECT count(*) FROM db_connections WHERE dialect != 'csv'")
        )
        active_db_connections = result.scalar() or 0

        # Average response time
        avg_response_time = await _compute_avg_response_time(db)

    uptime_seconds = round(time.monotonic() - _START_TIME, 1)

    return {
        "total_sessions": total_sessions,
        "total_queries": total_queries,
        "queries_by_type": queries_by_type,
        "total_documents_uploaded": total_uploads,
        "active_db_connections": active_db_connections,
        "avg_response_time_ms": avg_response_time,
        "uptime_seconds": uptime_seconds,
        "uptime_since": _START_DATETIME.isoformat(),
    }


async def _count_queries_by_type(db: Any) -> dict[str, int]:
    """Count query events broken down by query type from metadata_json."""
    result = await db.execute(
        text("SELECT metadata_json FROM audit_log WHERE event_type = 'query'")
    )
    rows = result.fetchall()

    counts: dict[str, int] = {"doc": 0, "db": 0, "hybrid": 0, "csv": 0, "unknown": 0}

    for row in rows:
        metadata_str = row[0] if row else None
        if not metadata_str:
            counts["unknown"] += 1
            continue
        try:
            meta = json.loads(metadata_str)
            query_type = meta.get("query_type", meta.get("type", "unknown"))
            if query_type in counts:
                counts[query_type] += 1
            else:
                counts["unknown"] += 1
        except (json.JSONDecodeError, TypeError):
            counts["unknown"] += 1

    return counts


async def _compute_avg_response_time(db: Any) -> Optional[float]:
    """Compute average response time in ms from audit log metadata.

    Looks for 'response_time_ms' or 'elapsed_ms' in metadata_json.
    Returns None if no response time data is available.
    """
    result = await db.execute(
        text(
            "SELECT metadata_json FROM audit_log "
            "WHERE event_type = 'query' LIMIT 1000"
        )
    )
    rows = result.fetchall()

    times: list[float] = []
    for row in rows:
        metadata_str = row[0] if row else None
        if not metadata_str:
            continue
        try:
            meta = json.loads(metadata_str)
            rt = meta.get("response_time_ms") or meta.get("elapsed_ms")
            if rt is not None:
                times.append(float(rt))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    if not times:
        return None

    return round(sum(times) / len(times), 1)
