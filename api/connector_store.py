"""Connector persistence — DOCBOT-706.

Stores marketplace connector registrations in PostgreSQL with
Fernet-encrypted credentials.  Connectors survive Railway restarts.

Table:
    marketplace_connections — connector_id, type, encrypted creds, sync metadata

Every function that touches credentials encrypts/decrypts via
``api.utils.encryption``.  Credentials never appear in logs or
query results from ``list_active_connectors`` / ``get_connector_info``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    String,
    Table,
    Text,
    func,
    select,
    update,
)
from sqlalchemy import insert as sa_insert

from api.utils.encryption import decrypt_credentials, encrypt_credentials

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Table definition
# ---------------------------------------------------------------------------

def register_connector_tables(metadata):
    """Define the marketplace_connections table on shared metadata.

    Called once at import time from ``api/index.py``.
    Returns the Table object.
    """
    marketplace_connections = Table(
        "marketplace_connections",
        metadata,
        Column("id", String, primary_key=True),
        Column("connector_type", String, nullable=False),
        Column("encrypted_credentials", Text, nullable=False),
        Column("is_active", Boolean, server_default="true", nullable=False),
        Column("last_sync_at", DateTime(timezone=True)),
        Column(
            "created_at",
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
        ),
        Column(
            "updated_at",
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
        ),
    )
    return marketplace_connections


# ---------------------------------------------------------------------------
# Module-level references — set by wire_connector_store()
# ---------------------------------------------------------------------------

_connections_table: Optional[Table] = None
_async_session_factory: Any = None


def wire_connector_store(connections_table, async_session_factory):
    """Inject table + session references.  Called once from index.py lifespan."""
    global _connections_table, _async_session_factory
    _connections_table = connections_table
    _async_session_factory = async_session_factory


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

async def save_connector(
    connector_id: str,
    connector_type: str,
    credentials: dict[str, str],
) -> None:
    """Encrypt credentials and upsert a connector row."""
    if _connections_table is None or _async_session_factory is None:
        logger.warning("connector_store not wired — skipping save")
        return

    encrypted = encrypt_credentials(credentials)
    now = datetime.now(timezone.utc)

    async with _async_session_factory() as session:
        async with session.begin():
            # Try PostgreSQL upsert first
            try:
                from sqlalchemy.dialects.postgresql import insert as pg_insert

                stmt = pg_insert(_connections_table).values(
                    id=connector_id,
                    connector_type=connector_type,
                    encrypted_credentials=encrypted,
                    is_active=True,
                    updated_at=now,
                ).on_conflict_do_update(
                    index_elements=["id"],
                    set_={
                        "connector_type": connector_type,
                        "encrypted_credentials": encrypted,
                        "is_active": True,
                        "updated_at": now,
                    },
                )
                await session.execute(stmt)
            except Exception:
                # Fallback for SQLite (tests)
                try:
                    await session.execute(
                        sa_insert(_connections_table).values(
                            id=connector_id,
                            connector_type=connector_type,
                            encrypted_credentials=encrypted,
                            is_active=True,
                            updated_at=now,
                        )
                    )
                except Exception:
                    # Row already exists — update it
                    await session.execute(
                        update(_connections_table)
                        .where(_connections_table.c.id == connector_id)
                        .values(
                            connector_type=connector_type,
                            encrypted_credentials=encrypted,
                            is_active=True,
                            updated_at=now,
                        )
                    )

    logger.info("Saved connector %s (type=%s)", connector_id, connector_type)


async def load_all_active_connectors() -> list[dict[str, Any]]:
    """Load all active connectors with decrypted credentials.

    Returns list of ``{"connector_id", "connector_type", "credentials"}``.
    """
    if _connections_table is None or _async_session_factory is None:
        return []

    stmt = (
        select(
            _connections_table.c.id,
            _connections_table.c.connector_type,
            _connections_table.c.encrypted_credentials,
        )
        .where(_connections_table.c.is_active == True)  # noqa: E712
    )

    async with _async_session_factory() as session:
        result = await session.execute(stmt)
        rows = result.all()

    connectors: list[dict[str, Any]] = []
    for row in rows:
        try:
            creds = decrypt_credentials(row.encrypted_credentials)
            connectors.append({
                "connector_id": row.id,
                "connector_type": row.connector_type,
                "credentials": creds,
            })
        except Exception as exc:
            logger.warning(
                "Failed to decrypt credentials for connector %s: %s",
                row.id,
                exc,
            )
    return connectors


async def delete_connector(connector_id: str) -> bool:
    """Soft-delete a connector (set is_active=False).

    Returns True if a row was affected, False otherwise.
    """
    if _connections_table is None or _async_session_factory is None:
        return False

    async with _async_session_factory() as session:
        async with session.begin():
            result = await session.execute(
                update(_connections_table)
                .where(_connections_table.c.id == connector_id)
                .values(
                    is_active=False,
                    updated_at=datetime.now(timezone.utc),
                )
            )
    return result.rowcount > 0


async def list_active_connectors() -> list[dict[str, Any]]:
    """Return metadata for all active connectors (no credentials)."""
    if _connections_table is None or _async_session_factory is None:
        return []

    stmt = (
        select(
            _connections_table.c.id,
            _connections_table.c.connector_type,
            _connections_table.c.is_active,
            _connections_table.c.last_sync_at,
            _connections_table.c.created_at,
            _connections_table.c.updated_at,
        )
        .where(_connections_table.c.is_active == True)  # noqa: E712
    )

    async with _async_session_factory() as session:
        result = await session.execute(stmt)
        rows = result.all()

    return [
        {
            "connector_id": r.id,
            "connector_type": r.connector_type,
            "is_active": r.is_active,
            "last_sync_at": r.last_sync_at.isoformat() if r.last_sync_at else None,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }
        for r in rows
    ]


async def get_connector_info(connector_id: str) -> Optional[dict[str, Any]]:
    """Return metadata for a single connector (no credentials)."""
    if _connections_table is None or _async_session_factory is None:
        return None

    stmt = (
        select(
            _connections_table.c.id,
            _connections_table.c.connector_type,
            _connections_table.c.is_active,
            _connections_table.c.last_sync_at,
            _connections_table.c.created_at,
            _connections_table.c.updated_at,
        )
        .where(_connections_table.c.id == connector_id)
    )

    async with _async_session_factory() as session:
        result = await session.execute(stmt)
        row = result.first()

    if row is None:
        return None

    return {
        "connector_id": row.id,
        "connector_type": row.connector_type,
        "is_active": row.is_active,
        "last_sync_at": row.last_sync_at.isoformat() if row.last_sync_at else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


async def update_last_sync(connector_id: str) -> None:
    """Set last_sync_at to now for a connector."""
    if _connections_table is None or _async_session_factory is None:
        return

    async with _async_session_factory() as session:
        async with session.begin():
            await session.execute(
                update(_connections_table)
                .where(_connections_table.c.id == connector_id)
                .values(
                    last_sync_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
            )
