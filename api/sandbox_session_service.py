"""Sandbox sessions for test API keys — DOCBOT-2597.

A ``dk_test_`` key may only touch sandbox sessions it created through
``POST /v1/sandbox/sessions`` (demo data from ``api/demo_service.py``), never
debits the org balance, and draws from a per-user daily test allowance
instead. Sandbox sessions are purged after 24h.

Hook points for the ``/v1`` handlers (kept as two calls so they survive the
DOCBOT-3901 move of those handlers into ``api/routes/v1_routes.py``):

* ``enforce_test_key_scope(key, session_id, connection_id)`` before any
  resource access: 404 for a test key on a non-sandbox resource.
* ``begin_test_key_billing(route, key)`` in place of org billing when
  ``is_test_key(key)``: 429 ``test_allowance_exhausted`` when spent.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException
from sqlalchemy import delete, select

from api.credit_service import BillingRun, cost_to_credits
from api.utils import request_guard
from api.utils.billing_context import Principal

logger = logging.getLogger(__name__)

SANDBOX_DATASETS: tuple[str, ...] = ("quickbite", "fuel", "tieout")
SANDBOX_TTL = timedelta(hours=24)
_HOUR = 3600
_DAY = 86400
_CREATE_BUCKET = "v1_sandbox_create"
_ALLOWANCE_BUCKET = "test_key_credits"

SANDBOX_COLUMN_MIGRATIONS: list[str] = [
    "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS is_sandbox BOOLEAN NOT NULL DEFAULT false",
]

_NOT_FOUND = {"code": "not_found", "message": "Session or connection not found."}

_state: dict[str, Any] = {}


def wire_sandbox_store(
    sessions_table: Any,
    db_connections_table: Any,
    schema_cache_table: Any,
    async_session_factory: Any,
    vector_stores: Any,
    get_embeddings_fn: Any,
) -> None:
    """Bind shared tables, the session factory and the in-memory vector stores."""
    _state.update(
        sessions=sessions_table,
        connections=db_connections_table,
        schema_cache=schema_cache_table,
        factory=async_session_factory,
        vector_stores=vector_stores,
        get_embeddings=get_embeddings_fn,
    )


def sandbox_creates_per_hour() -> int:
    return int(os.getenv("SANDBOX_CREATES_PER_HOUR", "5"))


def daily_test_credits() -> int:
    return int(os.getenv("TEST_KEY_DAILY_CREDITS", "300"))


def is_test_key(key: Any) -> bool:
    return getattr(key, "mode", "live") == "test"


def _owner_key(key: Any) -> str:
    # Per-user windows; a service-account key (no created_by) gets its own.
    return getattr(key, "created_by", None) or f"key:{key.id}"


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


async def create_sandbox_session(key: Any, dataset: str) -> dict[str, Any]:
    """Build a demo session owned by the key's user + org and marked ``is_sandbox``."""
    if dataset not in SANDBOX_DATASETS:
        raise HTTPException(
            status_code=400,
            detail={"code": "unsupported_dataset", "message": f"dataset must be one of {list(SANDBOX_DATASETS)}."},
        )
    retry = (await request_guard.charge_window(_CREATE_BUCKET, _owner_key(key), _HOUR, sandbox_creates_per_hour()))[0]
    if retry is not None:
        raise HTTPException(
            status_code=429,
            detail={"code": "sandbox_rate_limited", "message": f"Sandbox limit reached. Retry in {retry}s."},
            headers={"Retry-After": str(retry)},
        )

    from api.demo_service import init_demo_session

    result = await init_demo_session(
        vector_stores=_state["vector_stores"],
        get_embeddings_fn=_state["get_embeddings"],
        sessions_table=_state["sessions"],
        db_connections_table=_state["connections"],
        schema_cache_table=_state["schema_cache"],
        async_session_factory=_state["factory"],
        dataset=dataset,
        # is_sandbox lands on the session row only (demo_service filters
        # connection owner keys to existing columns).
        owner={"user_id": key.created_by, "org_id": key.org_id, "is_sandbox": True},
    )
    return {
        "session_id": result["session_id"],
        "connection_id": result["connection_id"],
        "dataset": dataset,
        "expires_at": datetime.now(timezone.utc) + SANDBOX_TTL,
    }


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


async def _is_own_sandbox_session(session: Any, key: Any, session_id: str) -> bool:
    t = _state["sessions"]
    result = await session.execute(
        select(t.c.session_id).where(
            t.c.session_id == session_id,
            t.c.is_sandbox.is_(True),
            t.c.user_id == key.created_by,
            t.c.org_id == key.org_id,
        )
    )
    return result.first() is not None


async def enforce_test_key_scope(key: Any, session_id: Optional[str], connection_id: Optional[str]) -> None:
    """Test keys only reach their own sandbox sessions/connections; anything else is 404.

    No-op for live keys. Same 404 as cross-org, so no existence oracle.
    """
    if not is_test_key(key):
        return
    if not key.created_by:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    conns = _state["connections"]
    async with _state["factory"]() as session:
        if session_id and not await _is_own_sandbox_session(session, key, session_id):
            raise HTTPException(status_code=404, detail=_NOT_FOUND)
        if connection_id:
            row = (await session.execute(
                select(conns.c.session_id).where(conns.c.id == connection_id)
            )).first()
            if row is None or not row.session_id or not await _is_own_sandbox_session(session, key, row.session_id):
                raise HTTPException(status_code=404, detail=_NOT_FOUND)


async def enforce_test_key_run_scope(key: Any, runs_table: Any, run_id: str) -> None:
    """Lineage/trace rows: a test key only sees runs on its own sandbox resources."""
    if not is_test_key(key):
        return
    async with _state["factory"]() as session:
        row = (await session.execute(
            select(runs_table.c.session_id, runs_table.c.connection_id).where(runs_table.c.run_id == run_id)
        )).first()
    if row is None or not (row.session_id or row.connection_id):
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    await enforce_test_key_scope(key, row.session_id, row.connection_id)


# ---------------------------------------------------------------------------
# Daily test allowance (never the org balance)
# ---------------------------------------------------------------------------


@dataclass
class TestAllowanceRun(BillingRun):
    """A BillingRun that never touches the org ledger: settle charges the
    metered credits to the key owner's daily test counter instead."""

    __test__ = False  # not a pytest test class
    allowance_key: str = ""

    async def settle(self) -> None:
        if self.settled:
            return None
        self.settled = True
        credits = cost_to_credits(self.meter.snapshot().cost_micro_usd)
        if credits > 0:
            await request_guard.window_add(_ALLOWANCE_BUCKET, self.allowance_key, _DAY, credits)
        return None


async def begin_test_key_billing(route: str, key: Any) -> TestAllowanceRun:
    """429 ``test_allowance_exhausted`` once today's allowance is spent; else a ledger-free run.

    shortcut: the check is before the run and the charge after, so the last
    call of the day can overshoot by one run's cost. Fine at 300 credits/day.
    """
    owner = _owner_key(key)
    used = await request_guard.window_add(_ALLOWANCE_BUCKET, owner, _DAY, 0)
    if used is not None and used >= daily_test_credits():
        raise HTTPException(
            status_code=429,
            detail={
                "code": "test_allowance_exhausted",
                "message": "Today's free test allowance is used up. It resets at 00:00 UTC.",
            },
        )
    return TestAllowanceRun(
        route=route,
        run_id=f"test-{uuid.uuid4()}",
        org_id=None,
        principal=Principal(
            user_id=key.created_by, org_id=None, auth_method="api_key", is_demo=True, api_key_id=key.id,
        ),
        exempt=True,
        allowance_key=owner,
    )


# ---------------------------------------------------------------------------
# 24h purge
# ---------------------------------------------------------------------------


async def purge_expired_sandbox_sessions(now: Optional[datetime] = None) -> int:
    """Delete sandbox sessions older than 24h: session row, vector store, SQLite file, connection."""
    from api.utils.vector_store import delete_store

    sessions_t, conns_t, cache_t = _state["sessions"], _state["connections"], _state["schema_cache"]
    cutoff = (now or datetime.now(timezone.utc)) - SANDBOX_TTL
    async with _state["factory"]() as session:
        result = await session.execute(
            select(sessions_t.c.session_id).where(
                sessions_t.c.is_sandbox.is_(True), sessions_t.c.created_at < cutoff
            )
        )
        session_ids = [r.session_id for r in result.all()]

    for sid in session_ids:
        _state["vector_stores"].pop(sid, None)
        delete_store(sid)
        async with _state["factory"]() as session:
            async with session.begin():
                conns = (await session.execute(
                    select(conns_t.c.id, conns_t.c.host, conns_t.c.dbname).where(conns_t.c.session_id == sid)
                )).all()
                for conn in conns:
                    if conn.host == "__local_file__" and conn.dbname:
                        try:
                            Path(conn.dbname).unlink(missing_ok=True)
                        except OSError as exc:
                            logger.warning("sandbox purge: could not delete %s: %s", conn.dbname, exc)
                    await session.execute(delete(cache_t).where(cache_t.c.connection_id == conn.id))
                await session.execute(delete(conns_t).where(conns_t.c.session_id == sid))
                # shortcut: messages rows for the session are left to the general retention sweep.
                await session.execute(delete(sessions_t).where(sessions_t.c.session_id == sid))
    if session_ids:
        logger.info("sandbox purge: removed %d session(s)", len(session_ids))
    return len(session_ids)


async def _run_purge_job() -> None:
    try:
        await purge_expired_sandbox_sessions()
    except Exception as exc:  # noqa: BLE001 — a failed sweep must not kill the scheduler
        logger.warning("sandbox purge failed (non-fatal): %s", type(exc).__name__)


def register_sandbox_purge_job(scheduler: Any) -> None:
    """Hourly sweep on the shared APScheduler (DOCBOT-704 instance)."""
    scheduler.add_job(
        _run_purge_job, "interval", hours=1,
        id="sandbox_session_purge", replace_existing=True, max_instances=1,
    )
