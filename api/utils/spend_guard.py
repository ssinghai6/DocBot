"""Platform-wide daily spend kill switch — DOCBOT-1615.

``PLATFORM_DAILY_USD_CAP`` (default 25, ``0`` disables) bounds what the whole
platform spends per UTC day on metered work: LLM calls (``llm_calls``
``estimated_cost_usd``, BYOK rows excluded) plus E2B sandbox time
(``usage_events`` kind="sandbox", ``cost_micro_usd``). The total comes from one
aggregate query, cached in-process for ``_CACHE_TTL_SEC``.

Once the cap is reached, free-tier callers (anonymous, demo routes, orgs that
never purchased credits) get HTTP 503 ``platform_capacity`` on expensive
routes; paid orgs and the owner keep working. The first breach of each day is
logged at ERROR and sent to Sentry (when initialised) once per process.

Fails OPEN: a cost-query error or an unwired store never blocks a request.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import date, datetime, timezone
from typing import Any, Optional

from fastapi import HTTPException, Request
from sqlalchemy import func, or_, select
from sqlalchemy.exc import SQLAlchemyError

from api.utils import request_guard

logger = logging.getLogger(__name__)

_CACHE_TTL_SEC = 60.0
_QUERY_TIMEOUT_SEC = 3.0

CAPACITY_ERROR = {
    "error": "platform_capacity",
    "message": "DocBot is at capacity for today — please try again tomorrow.",
}

_llm_calls: Any = None
_usage_events: Any = None
_session_factory: Any = None
# (utc_day, spent_usd or None on error, monotonic fetched_at)
_cache: Optional[tuple[date, Optional[float], float]] = None
_alerted_day: Optional[date] = None


def wire_spend_guard(llm_calls_table: Any, usage_events_table: Any, async_session_factory: Any) -> None:
    """Inject tables + session factory. Called once from the index.py lifespan."""
    global _llm_calls, _usage_events, _session_factory, _cache, _alerted_day
    _llm_calls, _usage_events, _session_factory = llm_calls_table, usage_events_table, async_session_factory
    _cache, _alerted_day = None, None


def daily_cap_usd() -> float:
    try:
        return float(os.getenv("PLATFORM_DAILY_USD_CAP", "25"))
    except ValueError:
        return 25.0


async def _query_spend(day: date) -> float:
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    llm, ue = _llm_calls, _usage_events
    llm_sum = (
        select(func.coalesce(func.sum(llm.c.estimated_cost_usd), 0.0))
        .where(llm.c.created_at >= start, or_(llm.c.key_source.is_(None), llm.c.key_source != "byok"))
        .scalar_subquery()
    )
    sandbox_sum = (
        select(func.coalesce(func.sum(ue.c.cost_micro_usd), 0))
        .where(ue.c.kind == "sandbox", ue.c.created_at >= start)
        .scalar_subquery()
    )
    async with _session_factory() as session:
        row = (await session.execute(select(llm_sum, sandbox_sum))).one()
    return float(row[0] or 0.0) + int(row[1] or 0) / 1_000_000


async def todays_spend_usd() -> Optional[float]:
    """Today's (UTC) platform spend in USD, cached ~60s. None when unknown (fail open)."""
    global _cache
    day = datetime.now(timezone.utc).date()
    now = time.monotonic()
    if _cache is not None and _cache[0] == day and now - _cache[2] < _CACHE_TTL_SEC:
        return _cache[1]
    spent: Optional[float]
    try:
        spent = await asyncio.wait_for(_query_spend(day), _QUERY_TIMEOUT_SEC)
    except (SQLAlchemyError, OSError, asyncio.TimeoutError) as exc:
        logger.warning("spend_guard: cost query failed (%s) — failing open", type(exc).__name__)
        spent = None
    _cache = (day, spent, now)
    return spent


def _alert_once(spent: float, cap: float) -> None:
    global _alerted_day
    day = datetime.now(timezone.utc).date()
    if _alerted_day == day:
        return
    _alerted_day = day
    msg = f"spend_guard: platform daily cap reached — spent ${spent:.2f} of ${cap:.2f} (UTC {day})"
    logger.error(msg)
    try:
        import sentry_sdk

        if sentry_sdk.is_initialized():
            sentry_sdk.capture_message(msg, level="error")
    except ImportError:
        pass


async def _is_paid_caller(request: Request) -> bool:
    if request.url.path.startswith("/api/demo/"):
        return False  # demo traffic is free tier by definition
    from api import credit_service
    from api.utils.caller_scope import resolve_caller_scope

    try:
        # DOCBOT-2330: /v1 API-key routes: require_api_key (resolved first) set the key's org.
        key_org = getattr(request.state, "org_id", None)
        if key_org:
            org: Optional[str] = key_org
        else:
            scope = await resolve_caller_scope(request)
            org = credit_service.billing_org_for(scope.user_id, scope.org_id)
        return org is not None and await credit_service.is_paid_org(org)
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("spend_guard: paid-tier lookup failed (%s) — failing open", type(exc).__name__)
        return True


async def check_platform_capacity(request: Request) -> None:
    """FastAPI dependency: 503 for free-tier callers once today's cap is spent."""
    cap = daily_cap_usd()
    if cap <= 0 or _session_factory is None or request_guard.is_owner(request):
        return
    spent = await todays_spend_usd()
    if spent is None or spent < cap:
        return
    _alert_once(spent, cap)
    if await _is_paid_caller(request):
        return
    raise HTTPException(status_code=503, detail=CAPACITY_ERROR)
