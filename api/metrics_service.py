"""Admin Metrics Service — Investor Readiness Sprint.

Provides aggregate platform metrics for the admin dashboard:
  - Total sessions
  - Total queries by type (doc / db / hybrid / csv)
  - Total documents uploaded
  - Active DB connections
  - Average response time (from audit log metadata)
  - Uptime
  - LLM cost/latency/token spend — DOCBOT-1502 (sourced from DOCBOT-1501's
    persisted llm_calls log; see get_llm_cost_metrics)

All queries run against the existing PostgreSQL tables (sessions,
audit_log, db_connections) using raw SQL for portability.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from pydantic import BaseModel
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Module-level start time for uptime tracking
_START_TIME = time.monotonic()
_START_DATETIME = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# DOCBOT-1502: LLM cost/latency dashboard
# ---------------------------------------------------------------------------


class DailyLlmSpend(BaseModel):
    date: str  # YYYY-MM-DD (UTC)
    calls: int
    total_tokens: int
    cost_usd: float


class CallerLlmSpend(BaseModel):
    """Per-`caller` breakdown — the tag each llm_provider call site passes
    (e.g. "sql_gen", "autopilot_planner", "hybrid_synthesis"). The ticket's
    acceptance criteria asked for a "per-persona" breakdown, but persona
    isn't part of the persisted call-log schema (see api/llm_trace_service.py)
    — the SQL/CSV/hybrid pipelines only tag calls with a caller string, not
    the active persona, and threading persona through every chat_completion
    call site was out of scope for this ticket's size. `caller` is the
    closest available dimension and is arguably more useful for engineering
    (it maps 1:1 to a code path), so it's used here instead.
    """
    caller: str
    calls: int
    cost_usd: float
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None


class LlmCostMetrics(BaseModel):
    window_days: int
    total_calls: int
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    success_rate: Optional[float] = None
    fallback_rate: Optional[float] = None
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    by_day: list[DailyLlmSpend] = []
    by_caller: list[CallerLlmSpend] = []


def _percentile(values: list[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile — no numpy dependency for a handful of floats."""
    if not values:
        return None
    ordered = sorted(values)
    idx = min(int(round(pct / 100 * (len(ordered) - 1))), len(ordered) - 1)
    return round(ordered[idx], 1)


async def get_llm_cost_metrics(days: int = 7) -> LlmCostMetrics:
    """Aggregate DOCBOT-1501's persisted llm_calls rows into cost/latency
    metrics for the admin dashboard.

    Reads the raw rows via api.llm_trace_service.get_call_stats() and
    aggregates in Python — row volume is small enough for a solo-deploy
    container (see that module's docstring); no window-function SQL needed
    yet. Returns an all-zero/empty LlmCostMetrics if the trace store isn't
    wired (e.g. in a test context) rather than raising.
    """
    from api.llm_trace_service import get_call_stats

    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = await get_call_stats(since=since)

    if not rows:
        return LlmCostMetrics(
            window_days=days, total_calls=0, total_cost_usd=0.0,
            total_input_tokens=0, total_output_tokens=0,
        )

    total_cost = 0.0
    total_in_tok = 0
    total_out_tok = 0
    success_count = 0
    fallback_count = 0
    all_latencies: list[float] = []

    by_day: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "total_tokens": 0, "cost_usd": 0.0}
    )
    by_caller_latencies: dict[str, list[float]] = defaultdict(list)
    by_caller_cost: dict[str, float] = defaultdict(float)
    by_caller_calls: dict[str, int] = defaultdict(int)

    for row in rows:
        cost = row.get("estimated_cost_usd") or 0.0
        in_tok = row.get("input_tokens") or 0
        out_tok = row.get("output_tokens") or 0
        latency = row.get("latency_ms")
        caller = row.get("caller") or "unknown"

        total_cost += cost
        total_in_tok += in_tok
        total_out_tok += out_tok
        if row.get("success"):
            success_count += 1
        if row.get("fallback_triggered"):
            fallback_count += 1
        if latency is not None:
            all_latencies.append(latency)
            by_caller_latencies[caller].append(latency)

        created_at = row.get("created_at")
        day_key = (created_at or "")[:10] or "unknown"
        by_day[day_key]["calls"] += 1
        by_day[day_key]["total_tokens"] += in_tok + out_tok
        by_day[day_key]["cost_usd"] += cost

        by_caller_cost[caller] += cost
        by_caller_calls[caller] += 1

    total_calls = len(rows)

    return LlmCostMetrics(
        window_days=days,
        total_calls=total_calls,
        total_cost_usd=round(total_cost, 6),
        total_input_tokens=total_in_tok,
        total_output_tokens=total_out_tok,
        success_rate=round(success_count / total_calls, 4) if total_calls else None,
        fallback_rate=round(fallback_count / total_calls, 4) if total_calls else None,
        p50_latency_ms=_percentile(all_latencies, 50),
        p95_latency_ms=_percentile(all_latencies, 95),
        by_day=[
            DailyLlmSpend(
                date=day,
                calls=int(v["calls"]),
                total_tokens=int(v["total_tokens"]),
                cost_usd=round(v["cost_usd"], 6),
            )
            for day, v in sorted(by_day.items())
        ],
        by_caller=[
            CallerLlmSpend(
                caller=caller,
                calls=by_caller_calls[caller],
                cost_usd=round(by_caller_cost[caller], 6),
                p50_latency_ms=_percentile(by_caller_latencies[caller], 50),
                p95_latency_ms=_percentile(by_caller_latencies[caller], 95),
            )
            for caller in sorted(by_caller_calls)
        ],
    )


async def get_platform_metrics(
    async_session_factory: Any,
    llm_days: int = 7,
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

    # DOCBOT-1502: LLM cost/latency/token spend, sourced from DOCBOT-1501's
    # persisted call log. Never let a trace-store hiccup break the whole
    # metrics endpoint — degrade to an empty block instead.
    try:
        llm_metrics = (await get_llm_cost_metrics(days=llm_days)).model_dump()
    except Exception as exc:
        logger.warning("get_llm_cost_metrics failed (non-fatal): %s", exc)
        llm_metrics = None

    # DOCBOT-1506: exact-match LLM response cache hit/miss counters.
    # In-process counters (see api/utils/llm_cache.py) — never let a
    # hiccup there break the whole metrics endpoint.
    try:
        from api.utils.llm_cache import get_cache_metrics
        llm_cache_metrics = get_cache_metrics()
    except Exception as exc:
        logger.warning("get_cache_metrics failed (non-fatal): %s", exc)
        llm_cache_metrics = None

    return {
        "total_sessions": total_sessions,
        "total_queries": total_queries,
        "queries_by_type": queries_by_type,
        "total_documents_uploaded": total_uploads,
        "active_db_connections": active_db_connections,
        "avg_response_time_ms": avg_response_time,
        "uptime_seconds": uptime_seconds,
        "uptime_since": _START_DATETIME.isoformat(),
        "llm_metrics": llm_metrics,
        "llm_cache_metrics": llm_cache_metrics,
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
