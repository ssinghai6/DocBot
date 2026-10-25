"""Developer-API (/v1/*) resource ownership checks — DOCBOT-1603 (SEC-06).

`/v1/*` callers authenticate with an org-scoped API key and pass
``session_id`` / ``connection_id`` in the request body. Both must belong to
the key's org; anything else is reported as not found (404) so a caller
cannot probe for the existence of another org's ids.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import Table, select

from api.db_service import get_connection_row
from api.utils.caller_scope import CallerScope


async def session_in_org(
    session_id: str,
    org_id: Optional[str],
    sessions_table: Table,
    async_session_factory,
) -> bool:
    """True iff the session row exists and is stamped with ``org_id``."""
    if not org_id or not session_id:
        return False
    async with async_session_factory() as session:
        result = await session.execute(
            select(sessions_table.c.session_id).where(
                sessions_table.c.session_id == session_id,
                sessions_table.c.org_id == org_id,
            )
        )
        return result.fetchone() is not None


async def v1_resources_in_org(
    org_id: Optional[str],
    session_id: str,
    connection_id: Optional[str],
    sessions_table: Table,
    db_connections_table: Table,
    async_session_factory,
) -> bool:
    """True iff the session (and the connection, when given) belong to ``org_id``."""
    if not org_id:
        return False
    if not await session_in_org(session_id, org_id, sessions_table, async_session_factory):
        return False
    if connection_id:
        row = await get_connection_row(
            connection_id,
            db_connections_table,
            async_session_factory,
            CallerScope(org_id=org_id),
        )
        if row is None:
            return False
    return True
