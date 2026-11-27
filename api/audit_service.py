"""Audit logging service — DOCBOT-602.

Provides an append-only audit log for security and compliance.
All writes are fire-and-forget to avoid slowing down request handlers.

Event types:
    query           — SQL/NL query executed against a live DB connection
    db_connect      — New database connection created
    db_disconnect   — Database connection removed
    upload          — File uploaded (PDF, CSV, SQLite)
    login           — SSO login (SAML ACS success)
    logout          — Session logout

The PostgreSQL immutability trigger is installed by init_db() in index.py.
It blocks UPDATE and DELETE on the audit_log table at the DB level.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


def get_client_ip(request: Any) -> Optional[str]:
    """Extract the real client IP from a FastAPI Request object.

    DOCBOT-4018: delegates to ``request_guard.client_ip`` — the right-most
    TRUSTED_PROXY_HOPS X-Forwarded-For entry, never the client-controlled
    left-most one — falling back to request.client.host. Returns None if the
    request object is None or has no client info.
    """
    if request is None:
        return None
    from api.utils.request_guard import client_ip  # lazy: keeps audit_service import-light
    try:
        ip = client_ip(request)
    except Exception:
        return None
    return None if ip == "unknown" else ip


def audit_ids(scope: Any) -> dict:
    """``user_id``/``org_id`` kwargs for ``log_event`` from a CallerScope (DOCBOT-4016).

    /admin/audit-log filters on org_id, so browser events must carry it.
    Anonymous callers (or no scope) yield None for both.
    """
    if scope is None:
        return {"user_id": None, "org_id": None}
    return {"user_id": scope.user_id, "org_id": scope.org_id}


async def audit_ids_for_user(user_id: str, users_table: Any, async_session_factory: Any) -> dict:
    """``user_id``/``org_id`` kwargs for a login event, reading ``users.org_id`` (DOCBOT-4016).

    A lookup failure keeps user_id and drops org_id — auditing never fails a login.
    """
    from sqlalchemy import select
    org_id = None
    try:
        async with async_session_factory() as session:
            org_id = (await session.execute(
                select(users_table.c.org_id).where(users_table.c.id == user_id)
            )).scalar_one_or_none()
    except Exception as exc:  # any DB/driver error; auditing must not fail the login
        logger.warning("audit org lookup failed (non-fatal): %s", type(exc).__name__)
    return {"user_id": str(user_id), "org_id": str(org_id) if org_id else None}


class AuditEventType(str, Enum):
    query = "query"
    db_connect = "db_connect"
    db_disconnect = "db_disconnect"
    upload = "upload"
    login = "login"
    logout = "logout"
    api_key_use = "api_key_use"  # DOCBOT-1530: developer API key used to call a /v1/* route
    api_key_create = "api_key_create"  # DOCBOT-2907: key minted
    api_key_revoke = "api_key_revoke"  # DOCBOT-2907: key revoked
    api_key_expired = "api_key_expired"  # DOCBOT-2907: expired key presented


async def _write_event(
    event_type: AuditEventType,
    session_id: Optional[str],
    user_id: Optional[str],
    detail: str,
    metadata_json: Optional[str],
    audit_log_table: Any,
    async_session_factory: Any,
    org_id: Optional[str] = None,
) -> None:
    """Persist one audit event. Called via asyncio.create_task — never awaited directly.

    DOCBOT-1524: org_id is set only at INSERT time (an append, so it never
    touches the audit_log_no_mutate immutability trigger). It defaults to
    None — existing callers that don't know the acting user's org_id are
    unaffected; such rows are picked up by `org_service.backfill_default_organization`
    only if they land in a table other than audit_log (audit_log rows that
    were written without org_id stay that way — see org_service.py for why).
    """
    from sqlalchemy import insert as sa_insert
    try:
        async with async_session_factory() as session:
            async with session.begin():
                await session.execute(
                    sa_insert(audit_log_table).values(
                        id=str(uuid.uuid4()),
                        event_type=event_type.value,
                        session_id=session_id,
                        user_id=user_id,
                        detail=detail,
                        metadata_json=metadata_json,
                        occurred_at=datetime.now(timezone.utc),
                        org_id=org_id,
                    )
                )
    except Exception as exc:
        # Audit failures must never crash the request handler.
        logger.warning("audit log write failed (non-fatal): %s — %s", type(exc).__name__, exc)


def log_event(
    event_type: AuditEventType,
    audit_log_table: Any,
    async_session_factory: Any,
    *,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    detail: str = "",
    metadata: Optional[dict] = None,
    org_id: Optional[str] = None,
) -> None:
    """Fire-and-forget audit log write.

    Call from any async route handler — returns immediately without blocking.
    Uses asyncio.create_task so the event is written in the background.

    Parameters
    ----------
    event_type:
        One of the AuditEventType values.
    audit_log_table:
        SQLAlchemy Table object for audit_log.
    async_session_factory:
        The app's async_sessionmaker instance.
    session_id:
        DocBot session ID (doc session or DB session_id), if applicable.
    user_id:
        SSO user ID, if the user is authenticated.
    detail:
        Short human-readable description (e.g. email, connection host, filename).
    metadata:
        Optional dict of extra structured data (dialect, row_count, pii_detected, etc.).
        Stored as JSON string.
    org_id:
        DOCBOT-1524: the acting user's org_id, if known. Optional and
        defaults to None — existing callers are unaffected.
    """
    import json
    from api.utils.pii_masking import mask_pii
    detail = mask_pii(detail) if detail else detail
    metadata_json = json.dumps(metadata) if metadata else None

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(
            _write_event(
                event_type, session_id, user_id, detail,
                metadata_json, audit_log_table, async_session_factory,
                org_id=org_id,
            )
        )
    except RuntimeError:
        # No running event loop (e.g., in tests calling sync code).
        logger.debug("audit log_event called outside event loop — skipped")


# ---------------------------------------------------------------------------
# Immutability trigger DDL
# ---------------------------------------------------------------------------
# Kept as a list of individual statements — asyncpg rejects multi-statement
# strings, so each DO block must be executed in its own conn.execute() call.

IMMUTABILITY_TRIGGER_STATEMENTS: list[str] = [
    # Step 1 — create the trigger function if it doesn't already exist
    """
    DO $outer$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_proc p
            JOIN pg_namespace n ON n.oid = p.pronamespace
            WHERE p.proname = 'audit_log_immutable'
              AND n.nspname = 'public'
        ) THEN
            EXECUTE $body$
                CREATE FUNCTION public.audit_log_immutable()
                RETURNS TRIGGER LANGUAGE plpgsql AS $$
                BEGIN
                    RAISE EXCEPTION 'audit_log rows are immutable';
                END;
                $$
            $body$;
        END IF;
    END
    $outer$
    """,
    # Step 2 — attach the trigger to audit_log if not already attached
    """
    DO $outer$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_trigger t
            JOIN pg_class c ON c.oid = t.tgrelid
            WHERE t.tgname = 'audit_log_no_mutate'
              AND c.relname = 'audit_log'
        ) THEN
            EXECUTE $body$
                CREATE TRIGGER audit_log_no_mutate
                BEFORE UPDATE OR DELETE ON audit_log
                FOR EACH ROW EXECUTE FUNCTION public.audit_log_immutable()
            $body$;
        END IF;
    END
    $outer$
    """,
]

# Kept for backwards-compat with any code that imported the old name
IMMUTABILITY_TRIGGER_DDL = "\n".join(IMMUTABILITY_TRIGGER_STATEMENTS)
