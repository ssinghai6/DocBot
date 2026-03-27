"""Background sync jobs for marketplace connectors — DOCBOT-704.

Uses APScheduler ``AsyncIOScheduler`` running inside the FastAPI lifespan.
Each active connector gets interval jobs for orders (15 min) and
financials (4 hours).  Sync is incremental via ``sync_cursor``.

Error handling:
- ``ConnectorRateLimitError`` → exponential backoff (retry_after * 1.5)
- ``ConnectorAuthError`` → disable schedule, mark ``sync_status='auth_failed'``
- Generic exceptions → log + mark ``sync_status='error'``, keep schedule
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from api.connectors.base import ConnectorAuthError, ConnectorRateLimitError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level references — set by wire_sync_scheduler()
# ---------------------------------------------------------------------------

_scheduler: AsyncIOScheduler | None = None
_app_state: Any = None  # FastAPI app.state — holds .connectors dict


def wire_sync_scheduler(scheduler: AsyncIOScheduler, app_state: Any) -> None:
    """Inject scheduler + app state refs.  Called from lifespan."""
    global _scheduler, _app_state
    _scheduler = scheduler
    _app_state = app_state


# ---------------------------------------------------------------------------
# Sync intervals (seconds)
# ---------------------------------------------------------------------------

ORDERS_INTERVAL_SECONDS = 15 * 60      # 15 minutes
FINANCIALS_INTERVAL_SECONDS = 4 * 3600  # 4 hours

# Default lookback when no sync_cursor exists
DEFAULT_LOOKBACK_DAYS = 30


# ---------------------------------------------------------------------------
# Core sync function
# ---------------------------------------------------------------------------

async def sync_connector_incremental(
    connector_id: str,
    sync_type: str = "orders",
) -> dict[str, Any] | None:
    """Run an incremental sync for a single connector.

    ``sync_type`` is ``"orders"`` or ``"financials"``.
    Returns a summary dict on success, or None on skip/failure.
    """
    from api.connector_store import (
        get_sync_cursor,
        update_last_sync,
        update_sync_status,
    )
    from api.commerce_service import persist_orders, persist_financials

    connectors = getattr(_app_state, "connectors", {})
    connector = connectors.get(connector_id)
    if connector is None:
        logger.warning("sync_connector_incremental: connector %s not in memory", connector_id)
        return None

    # Determine date range
    cursor = await get_sync_cursor(connector_id)
    now = datetime.now(timezone.utc)
    if cursor:
        start_date = cursor
    else:
        start_date = (now - timedelta(days=DEFAULT_LOOKBACK_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_date = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    await update_sync_status(connector_id, "syncing")

    try:
        if sync_type == "orders":
            records = await connector.fetch_orders(start_date, end_date)
            written = await persist_orders(connector_id, connector.connector_type, records)
        else:
            records = await connector.fetch_financials(start_date, end_date)
            written = await persist_financials(connector_id, connector.connector_type, records)

        # Update cursor to now
        await update_last_sync(connector_id, sync_cursor=end_date)

        logger.info(
            "Background sync %s for %s: %d fetched, %d written",
            sync_type, connector_id, len(records), written,
        )
        return {
            "connector_id": connector_id,
            "sync_type": sync_type,
            "fetched": len(records),
            "written": written,
        }

    except ConnectorRateLimitError as exc:
        await update_sync_status(connector_id, "idle")
        _schedule_backoff_retry(connector_id, sync_type, exc.retry_after)
        logger.warning(
            "Rate limited on %s sync for %s — retry in %.1fs",
            sync_type, connector_id, exc.retry_after * 1.5,
        )
        return None

    except ConnectorAuthError as exc:
        await update_sync_status(connector_id, "auth_failed")
        _deregister_sync_schedules(connector_id)
        logger.error(
            "Auth failed on %s sync for %s — schedules disabled: %s",
            sync_type, connector_id, exc,
        )
        return None

    except Exception as exc:
        await update_sync_status(connector_id, "error")
        logger.error(
            "Background %s sync error for %s: %s",
            sync_type, connector_id, exc,
        )
        return None


# ---------------------------------------------------------------------------
# Schedule management
# ---------------------------------------------------------------------------

def _job_id(connector_id: str, sync_type: str) -> str:
    return f"sync_{sync_type}_{connector_id}"


def register_sync_schedules(connector_id: str) -> None:
    """Register interval jobs for a connector's orders and financials."""
    if _scheduler is None:
        logger.warning("Scheduler not initialized — cannot register sync jobs")
        return

    # Orders — every 15 min
    orders_id = _job_id(connector_id, "orders")
    if _scheduler.get_job(orders_id) is None:
        _scheduler.add_job(
            sync_connector_incremental,
            "interval",
            seconds=ORDERS_INTERVAL_SECONDS,
            id=orders_id,
            args=[connector_id, "orders"],
            replace_existing=True,
            max_instances=1,
        )

    # Financials — every 4 hours
    fin_id = _job_id(connector_id, "financials")
    if _scheduler.get_job(fin_id) is None:
        _scheduler.add_job(
            sync_connector_incremental,
            "interval",
            seconds=FINANCIALS_INTERVAL_SECONDS,
            id=fin_id,
            args=[connector_id, "financials"],
            replace_existing=True,
            max_instances=1,
        )

    logger.info("Registered sync schedules for connector %s", connector_id)


def deregister_sync_schedules(connector_id: str) -> None:
    """Remove all sync jobs for a connector."""
    _deregister_sync_schedules(connector_id)


def _deregister_sync_schedules(connector_id: str) -> None:
    """Internal: remove all sync jobs for a connector."""
    if _scheduler is None:
        return

    for sync_type in ("orders", "financials"):
        job_id = _job_id(connector_id, sync_type)
        job = _scheduler.get_job(job_id)
        if job is not None:
            job.remove()
            logger.info("Removed sync job %s", job_id)


def _schedule_backoff_retry(
    connector_id: str,
    sync_type: str,
    retry_after: float,
) -> None:
    """Schedule a one-shot retry with exponential backoff."""
    if _scheduler is None:
        return

    backoff_seconds = retry_after * 1.5
    retry_id = f"retry_{sync_type}_{connector_id}"

    # Remove existing retry job if any
    existing = _scheduler.get_job(retry_id)
    if existing is not None:
        existing.remove()

    run_at = datetime.now(timezone.utc) + timedelta(seconds=backoff_seconds)
    _scheduler.add_job(
        sync_connector_incremental,
        "date",
        run_date=run_at,
        id=retry_id,
        args=[connector_id, sync_type],
        max_instances=1,
    )
    logger.info("Scheduled backoff retry %s in %.1fs", retry_id, backoff_seconds)


def register_all_active_connectors() -> int:
    """Register sync schedules for all connectors in memory.

    Called on startup after connectors are restored from DB.
    Returns the count of connectors registered.
    """
    connectors = getattr(_app_state, "connectors", {})
    count = 0
    for connector_id in connectors:
        register_sync_schedules(connector_id)
        count += 1
    return count
