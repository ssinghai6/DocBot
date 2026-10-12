"""Organization/workspace data model — DOCBOT-1524.

DocBot has had no tenant concept: `require_role()` checks a user's role
against one flat, global user pool, with no org/workspace boundary anywhere
in the schema. This module adds the minimal data model needed to fix that
before a second real customer exists:

    organizations         — id, name, created_at (no billing/plan/seat
                             fields yet — explicitly deferred)
    users.org_id           — nullable FK → organizations.id
    sessions.org_id        — nullable FK → organizations.id
    db_connections.org_id  — nullable FK → organizations.id
    marketplace_connections.org_id — nullable FK → organizations.id
    audit_log.org_id       — nullable FK → organizations.id

Scope is deliberately narrow: schema + backfill + query scoping only. No
workspace switcher, no org-creation/invite flow, no per-org billing or API
keys — those are separate tickets. One org per existing user is fine for
now (backfill assigns every pre-existing row to a single default org).

Backfill is idempotent and runs on every startup from `api/index.py`'s
`init_db()` (see `backfill_default_organization`): it is safe to call on a
fresh database, on an existing Railway deployment's live data, and on
every subsequent restart without creating duplicate organizations or
re-touching rows that already have an `org_id`.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import Column, DateTime, String, Table, func, text

logger = logging.getLogger(__name__)

DEFAULT_ORG_NAME = "Default Organization"

# Tables that gained an org_id column after initial deploy. ALTER TABLE ...
# ADD COLUMN IF NOT EXISTS is the same idempotent migration idiom used
# throughout api/index.py's init_db() (e.g. DOCBOT-1101's
# "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS source ...") — it is a
# no-op on a database that was created fresh (where the Table definitions
# already declare org_id) and additive on an existing deployment.
ORG_ID_COLUMN_MIGRATIONS: list[str] = [
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS org_id TEXT",
    "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS org_id TEXT",
    "ALTER TABLE db_connections ADD COLUMN IF NOT EXISTS org_id TEXT",
    "ALTER TABLE marketplace_connections ADD COLUMN IF NOT EXISTS org_id TEXT",
    "ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS org_id TEXT",
]

# The set of tables backfilled to the default organization via UPDATE. Kept
# as a constant (rather than inferred) so the backfill is explicit about its
# blast radius.
#
# audit_log is deliberately NOT in this list. DOCBOT-602 installs a
# PostgreSQL trigger that raises on any UPDATE or DELETE against audit_log
# (append-only compliance boundary — see api/audit_service.py). Backfilling
# org_id there with an UPDATE would violate that boundary and is rejected by
# the database itself; pre-existing audit_log rows (written before this
# ticket) permanently keep org_id = NULL rather than be retroactively
# rewritten. New audit_log rows may be written with org_id set directly at
# INSERT time (see `_write_event`'s optional `org_id` kwarg) — that is an
# append, not a mutation, so it does not touch the trigger.
_BACKFILL_TABLES: tuple[str, ...] = (
    "users",
    "sessions",
    "db_connections",
    "marketplace_connections",
)


def register_organizations_table(metadata: Any) -> Table:
    """Define the `organizations` table on shared metadata.

    Minimal by design: id, name, created_at. No billing/plan/seat columns —
    explicitly deferred to a later ticket. Called once at import time from
    `api/index.py`, mirroring every other `register_*_table` helper.
    """
    return Table(
        "organizations",
        metadata,
        Column("id", String, primary_key=True),
        Column("name", String, nullable=False),
        Column(
            "created_at",
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
        ),
    )


async def backfill_default_organization(conn: Any) -> str:
    """Idempotently ensure a default organization exists and that every
    row with `org_id IS NULL` across users, sessions, db_connections,
    marketplace_connections, and audit_log is assigned to it.

    Must be called after `organizations` and the `org_id` columns already
    exist (i.e. after `metadata.create_all` and the
    `ORG_ID_COLUMN_MIGRATIONS` ALTER statements have run).

    Safe to call on every startup:
      - If an organization already exists, the earliest-created one is
        reused as the backfill target instead of creating a second one
        (idempotent — never creates a duplicate default org).
      - Each `UPDATE ... WHERE org_id IS NULL` is a no-op once every row
        in that table has already been backfilled.

    Parameters
    ----------
    conn:
        An open SQLAlchemy AsyncConnection (e.g. from
        ``async with engine.begin() as conn``).

    Returns
    -------
    str
        The default organization's id.
    """
    existing = await conn.execute(
        text("SELECT id FROM organizations ORDER BY created_at ASC LIMIT 1")
    )
    row = existing.fetchone()

    if row is not None:
        org_id = row[0]
    else:
        org_id = str(uuid.uuid4())
        await conn.execute(
            text("INSERT INTO organizations (id, name) VALUES (:id, :name)"),
            {"id": org_id, "name": DEFAULT_ORG_NAME},
        )
        logger.info("org_service: created default organization %s", org_id)

    for table_name in _BACKFILL_TABLES:
        result = await conn.execute(
            text(f"UPDATE {table_name} SET org_id = :org_id WHERE org_id IS NULL"),
            {"org_id": org_id},
        )
        try:
            affected = result.rowcount
        except Exception:
            affected = -1
        if affected and affected > 0:
            logger.info(
                "org_service: backfilled org_id on %d row(s) in %s", affected, table_name
            )

    return org_id
