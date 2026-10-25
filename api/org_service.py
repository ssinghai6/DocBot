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

DOCBOT-1702 replaces "one shared org for everyone": every NEW signup gets
its own personal organization (creator = admin) atomically inside
`api.auth_service.jit_provision_user` via `create_personal_org()`. The
default org now only holds legacy (pre-1702, NULL org_id) rows, and
`remediate_personal_orgs()` (driven by scripts/remediate_personal_orgs.py)
moves the post-1524 signups that were wrongly swept into it.

Backfill is idempotent and runs on every startup from `api/index.py`'s
`init_db()` (see `backfill_default_organization`): it is safe to call on a
fresh database, on an existing Railway deployment's live data, and on
every subsequent restart without creating duplicate organizations or
re-touching rows that already have an `org_id`.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from sqlalchemy import (
    Column,
    DateTime,
    String,
    Table,
    and_,
    column,
    func,
    literal,
    or_,
    select,
    text,
    update,
)
from sqlalchemy import table as table_clause
from sqlalchemy.sql import ColumnElement, Executable
from sqlalchemy.sql.expression import TableClause

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


# ---------------------------------------------------------------------------
# DOCBOT-1702: personal organization per signup
# ---------------------------------------------------------------------------

# Role the creator of a personal org is provisioned with. Each new user is
# the only member of their own org, so "admin" means "admin of my own
# workspace". Consequence: EVERY admin route must be org-scoped (PATCH
# /admin/users/{id}/role and /admin/metrics in api/index.py are not yet —
# owned by Lane S, tracked as a DOCBOT-1702 merge prerequisite).
PERSONAL_ORG_CREATOR_ROLE = "admin"

# Personal org names always end with this suffix so they can never collide
# with DEFAULT_ORG_NAME (which is how the legacy default org is located).
PERSONAL_ORG_NAME_SUFFIX = "'s workspace"
_PERSONAL_ORG_FALLBACK_NAME = "Personal workspace"
_PERSONAL_ORG_NAME_MAX = 80

# Tables whose rows carry a `user_id` owner column. NULL-org rows in these
# tables inherit their owner's org_id on backfill instead of being dumped
# into the default org (otherwise a personal-org user's session written
# with a NULL org_id would be re-homed into the shared default org on the
# next restart).
_USER_OWNED_TABLES: tuple[str, ...] = ("sessions", "db_connections")


def personal_org_name(display_name: Optional[str]) -> str:
    """Return the organization name for a user's personal workspace."""
    base = (display_name or "").strip()[:_PERSONAL_ORG_NAME_MAX]
    if not base:
        return _PERSONAL_ORG_FALLBACK_NAME
    return f"{base}{PERSONAL_ORG_NAME_SUFFIX}"


async def create_personal_org(conn: Any, display_name: Optional[str]) -> str:
    """Insert a new personal organization and return its id.

    Runs on the caller's connection/session so it joins the caller's
    transaction: `jit_provision_user` inserts the org and then the user row
    (carrying this org_id) inside one `session.begin()` block, so a failure
    on either insert rolls both back — no user can exist without an org and
    no orphan org survives a failed user insert.

    ``conn`` may be an SQLAlchemy ``AsyncConnection`` or ``AsyncSession`` —
    only ``.execute()`` is used.
    """
    org_id = str(uuid.uuid4())
    await conn.execute(
        text("INSERT INTO organizations (id, name) VALUES (:id, :name)"),
        {"id": org_id, "name": personal_org_name(display_name)},
    )
    logger.info("org_service: created personal organization %s", org_id)
    return org_id


async def get_default_org_id(conn: Any) -> Optional[str]:
    """Return the legacy default organization's id, or None if absent.

    Located by its fixed name (earliest on ties) rather than "the earliest
    org overall", so a personal org can never be mistaken for it.
    """
    result = await conn.execute(
        text(
            "SELECT id FROM organizations WHERE name = :name "
            "ORDER BY created_at ASC LIMIT 1"
        ),
        {"name": DEFAULT_ORG_NAME},
    )
    row = result.fetchone()
    return row[0] if row is not None else None


async def get_user_org(conn: Any, user_id: str) -> Optional[str]:
    """Return the org_id a user belongs to (None for unknown user / no org).

    Helper for org-scoped admin routes: a handler acting on a target user
    should 404 unless ``await get_user_org(conn, target_id)`` equals the
    requesting user's org_id (§4 gate #2: org B → 404 on org A ids).
    """
    result = await conn.execute(
        text("SELECT org_id FROM users WHERE id = :id"), {"id": user_id}
    )
    row = result.fetchone()
    return row[0] if row is not None else None


def _org_table(name: str, *extra_columns: str) -> TableClause:
    """Lightweight Core construct for an org-scoped table (DOCBOT-1633).

    Table names come only from this module's fixed allowlists
    (``_BACKFILL_TABLES`` / ``_USER_OWNED_TABLES`` / literals); building the
    statements with SQLAlchemy Core instead of f-string SQL keeps identifiers
    out of string interpolation and every value a bound parameter.
    """
    return table_clause(name, column("org_id"), *(column(c) for c in extra_columns))


def _rowcount(result: Any) -> int:
    try:
        return int(result.rowcount or 0)
    except (AttributeError, TypeError, ValueError):
        return -1


async def backfill_default_organization(conn: Any) -> str:
    """Idempotently ensure the default organization exists and home legacy
    rows (``org_id IS NULL``). Never re-homes a row that already has an org.

    Must be called after `organizations` and the `org_id` columns already
    exist (i.e. after `metadata.create_all` and the
    `ORG_ID_COLUMN_MIGRATIONS` ALTER statements have run).

    DOCBOT-1702 semantics:
      - ``users``: only rows with ``org_id IS NULL`` (legacy, pre-1702
        users) are assigned to the default org. New signups always receive
        a personal org atomically in `jit_provision_user`, so they are never
        touched here.
      - ``sessions`` / ``db_connections`` (user-owned): a NULL-org row whose
        owner has an org inherits the OWNER's org — a personal-org user's
        rows are never pulled into the shared default org.
      - anything still NULL afterwards (guest/anonymous rows, legacy
        marketplace connections) goes to the default org, as before.
      - audit_log is never UPDATEd (append-only trigger — see above).

    Safe on every startup: the default org is located by name and created
    only when missing; every UPDATE is guarded by ``org_id IS NULL`` so a
    second run is a no-op.

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
    org_id = await get_default_org_id(conn)
    if org_id is None:
        org_id = str(uuid.uuid4())
        await conn.execute(
            text("INSERT INTO organizations (id, name) VALUES (:id, :name)"),
            {"id": org_id, "name": DEFAULT_ORG_NAME},
        )
        logger.info("org_service: created default organization %s", org_id)

    # 1) Legacy users only (NULL org_id).
    affected = _rowcount(await conn.execute(
        text("UPDATE users SET org_id = :org_id WHERE org_id IS NULL"),
        {"org_id": org_id},
    ))
    if affected > 0:
        logger.info("org_service: backfilled org_id on %d legacy user(s)", affected)

    # 2) User-owned rows inherit their owner's org.
    owner = _org_table("users", "id").alias("u")
    for table_name in _USER_OWNED_TABLES:
        target = _org_table(table_name, "user_id")
        owner_org = (
            select(owner.c.org_id)
            .where(owner.c.id == target.c.user_id)
            .scalar_subquery()
        )
        owner_has_org = (
            select(literal(1))
            .where(owner.c.id == target.c.user_id, owner.c.org_id.is_not(None))
            .exists()
        )
        affected = _rowcount(await conn.execute(
            update(target)
            .values(org_id=owner_org)
            .where(
                target.c.org_id.is_(None),
                target.c.user_id.is_not(None),
                owner_has_org,
            )
        ))
        if affected > 0:
            logger.info(
                "org_service: assigned owner org_id on %d row(s) in %s", affected, table_name
            )

    # 3) Everything still unowned → default org.
    for table_name in _BACKFILL_TABLES:
        target = _org_table(table_name)
        affected = _rowcount(await conn.execute(
            update(target).values(org_id=org_id).where(target.c.org_id.is_(None))
        ))
        if affected > 0:
            logger.info(
                "org_service: backfilled org_id on %d row(s) in %s", affected, table_name
            )

    return org_id


# ---------------------------------------------------------------------------
# DOCBOT-1702: one-time remediation of post-DOCBOT-1524 signups
# ---------------------------------------------------------------------------
#
# Identification heuristic
# ------------------------
# DOCBOT-1524 (merged 2026-10-12, commit ffc2bcd) creates the default org in
# `init_db()` on the first startup after deploy — BEFORE the app serves any
# request. Every user who existed at that moment was backfilled into it
# (legitimate legacy members). Every user created afterwards was inserted
# by `jit_provision_user` with org_id = NULL and swept into the default org
# by the next restart's backfill (or is still NULL if no restart happened
# since). Both timestamps come from the same database clock
# (`server_default=now()`), so:
#
#     post-1524 signup  <=>  users.created_at > default_org.created_at
#                            AND users.org_id IN (default_org, NULL)
#
# `since=` overrides the cutoff (e.g. with the exact deploy time) if the
# default org's created_at is ever in doubt.
#
# Users with role 'admin' are SKIPPED by default: signups default to
# 'analyst', so a post-1524 admin was promoted by hand — almost certainly
# the founder/operator account that owns the legacy data, API keys and DEK.
# `include_admins=True` moves them anyway.


@dataclass
class UserMove:
    user_id: str
    new_org_id: Optional[str]
    sessions: int = 0
    db_connections: int = 0
    connections_on_default_dek: int = 0


@dataclass
class RemediationReport:
    applied: bool
    default_org_id: Optional[str]
    cutoff: Optional[datetime]
    moves: list[UserMove] = field(default_factory=list)
    skipped_admins: list[str] = field(default_factory=list)
    skipped_excluded: list[str] = field(default_factory=list)
    unattributable_marketplace_connections: int = 0
    default_org_api_keys: int = 0

    def summary_lines(self) -> list[str]:
        """Human-readable summary. Contains user/org ids only — no PII."""
        mode = "APPLIED" if self.applied else "DRY RUN (no changes written)"
        lines = [
            f"DOCBOT-1702 personal-org remediation: {mode}",
            f"default org: {self.default_org_id}",
            f"cutoff (users created after): {self.cutoff.isoformat() if self.cutoff else None}",
            f"users to move: {len(self.moves)}",
        ]
        for m in self.moves:
            lines.append(
                f"  user {m.user_id} -> org {m.new_org_id or '<new personal org>'} "
                f"(sessions={m.sessions}, db_connections={m.db_connections}, "
                f"still_on_default_org_dek={m.connections_on_default_dek})"
            )
        if self.skipped_admins:
            lines.append(
                f"skipped admins ({len(self.skipped_admins)}; pass --include-admins to move): "
                + ", ".join(self.skipped_admins)
            )
        if self.skipped_excluded:
            lines.append(
                f"skipped by --exclude-user-id ({len(self.skipped_excluded)}): "
                + ", ".join(self.skipped_excluded)
            )
        lines.append(
            "left in default org (no owner column, review manually): "
            f"marketplace_connections={self.unattributable_marketplace_connections}, "
            f"api_keys={self.default_org_api_keys}"
        )
        return lines


def _as_utc(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


async def _count(conn: Any, stmt: Executable) -> int:
    """Execute a ``SELECT COUNT(*)`` Core statement and return the integer."""
    row = (await conn.execute(stmt)).fetchone()
    return int(row[0]) if row is not None and row[0] is not None else 0


def _count_where(target: TableClause, *conditions: ColumnElement[bool]) -> Executable:
    return select(func.count()).select_from(target).where(*conditions)


def _owned_by(
    target: TableClause, user_id: str, default_org_id: str
) -> ColumnElement[bool]:
    """Rows owned by ``user_id`` that sit in the default org (or no org)."""
    return and_(
        target.c.user_id == user_id,
        or_(target.c.org_id == default_org_id, target.c.org_id.is_(None)),
    )


async def _count_optional_table(conn: Any, table_name: str, org_id: str) -> int:
    """COUNT(*) rows for ``org_id`` in a table that may not exist yet."""
    from sqlalchemy.exc import DBAPIError

    target = _org_table(table_name)
    try:
        async with conn.begin_nested():
            return await _count(conn, _count_where(target, target.c.org_id == org_id))
    except DBAPIError:
        return 0


async def remediate_personal_orgs(
    conn: Any,
    *,
    apply: bool = False,
    since: Optional[datetime] = None,
    include_admins: bool = False,
    exclude_user_ids: Iterable[str] = (),
    promote_to_admin: bool = True,
) -> RemediationReport:
    """Move post-DOCBOT-1524 signups out of the default org into personal orgs.

    With ``apply=False`` (default) this only reads and returns the plan.
    With ``apply=True`` it writes on ``conn``; the caller owns the
    transaction (``async with engine.begin() as conn``) so the whole
    remediation commits or rolls back as one unit.

    For each selected user: create a personal org, set ``users.org_id``
    (and ``role = 'admin'`` unless ``promote_to_admin=False``, matching what
    new signups get), and move that user's ``sessions`` / ``db_connections``
    rows that sit in the default org (or NULL) into the new org. audit_log
    is append-only and never touched; marketplace_connections and api_keys
    have no owner column and stay in the default org (counted for manual
    review).

    Envelope-encrypted credential blobs embed the org whose DEK encrypted
    them (``ENV1:<org_id>:...``), so moved db_connections keep decrypting
    with the default org's DEK. They are counted, NOT batch re-encrypted
    (api/utils/envelope_encryption.py forbids batch re-encryption).
    Re-keying them is a follow-up.

    Idempotent: once moved, a user is no longer in the default org, so a
    second run selects nobody.
    """
    excluded = set(exclude_user_ids)
    default_org_id = await get_default_org_id(conn)
    report = RemediationReport(applied=apply, default_org_id=default_org_id, cutoff=None)
    if default_org_id is None:
        return report

    org_row = (await conn.execute(
        text("SELECT created_at FROM organizations WHERE id = :id"),
        {"id": default_org_id},
    )).fetchone()
    cutoff = _as_utc(since) if since is not None else _as_utc(org_row[0])
    report.cutoff = cutoff

    rows = (await conn.execute(
        text(
            "SELECT id, name, role, created_at FROM users "
            "WHERE org_id = :org OR org_id IS NULL ORDER BY created_at ASC"
        ),
        {"org": default_org_id},
    )).fetchall()

    selected: list[Any] = []
    for row in rows:
        created = _as_utc(row.created_at)
        if cutoff is None or created is None or created <= cutoff:
            continue
        if row.id in excluded:
            report.skipped_excluded.append(row.id)
            continue
        if row.role == "admin" and not include_admins:
            report.skipped_admins.append(row.id)
            continue
        selected.append(row)

    env_prefix = f"ENV1:{default_org_id}:%"
    sessions_t = _org_table("sessions", "user_id")
    connections_t = _org_table("db_connections", "user_id", "credentials_blob")
    users_t = _org_table("users", "id", "role")
    for row in selected:
        move = UserMove(
            user_id=row.id,
            new_org_id=None,
            sessions=await _count(conn, _count_where(
                sessions_t, _owned_by(sessions_t, row.id, default_org_id)
            )),
            db_connections=await _count(conn, _count_where(
                connections_t, _owned_by(connections_t, row.id, default_org_id)
            )),
            connections_on_default_dek=await _count(conn, _count_where(
                connections_t,
                _owned_by(connections_t, row.id, default_org_id),
                connections_t.c.credentials_blob.like(env_prefix),
            )),
        )
        if apply:
            new_org_id = await create_personal_org(conn, row.name)
            move.new_org_id = new_org_id
            user_vals: dict[str, Any] = {"org_id": new_org_id}
            if promote_to_admin:
                user_vals["role"] = PERSONAL_ORG_CREATOR_ROLE
            await conn.execute(
                update(users_t).values(**user_vals).where(users_t.c.id == row.id)
            )
            for table_name in _USER_OWNED_TABLES:
                target = _org_table(table_name, "user_id")
                await conn.execute(
                    update(target)
                    .values(org_id=new_org_id)
                    .where(_owned_by(target, row.id, default_org_id))
                )
        report.moves.append(move)

    marketplace_t = _org_table("marketplace_connections")
    report.unattributable_marketplace_connections = await _count(
        conn, _count_where(marketplace_t, marketplace_t.c.org_id == default_org_id)
    )
    report.default_org_api_keys = await _count_optional_table(conn, "api_keys", default_org_id)
    return report
