#!/usr/bin/env python3
"""DOCBOT-1702 — one-time remediation: move post-DOCBOT-1524 signups out of
the shared "Default Organization" into their own personal orgs.

Default is a DRY RUN: the plan is computed inside a transaction that is
always rolled back. Nothing is written unless ``--apply`` is passed, and in
apply mode the whole remediation runs in ONE transaction (all-or-nothing).

Identification heuristic (see api/org_service.py, "one-time remediation"):
    post-1524 signup  <=>  users.created_at > default_org.created_at
                           AND users.org_id IN (default_org, NULL)
The default org was created by init_db() on the first startup after the
DOCBOT-1524 deploy (2026-10-12), before any request was served, from the
same DB clock as users.created_at. Use ``--since`` to override the cutoff.

Admins are skipped unless ``--include-admins`` (a post-1524 admin was
promoted by hand: most likely the operator account owning legacy data).

Output contains user/org ids and counts only — no emails or names.

Usage (reads DATABASE_URL from the environment; never pass it as an arg):
    python scripts/remediate_personal_orgs.py                # dry run
    python scripts/remediate_personal_orgs.py --apply        # write
Options:
    --since 2026-10-12T11:07:30+00:00   override the cutoff timestamp
    --include-admins                     also move admin users
    --exclude-user-id <id>               never move this user (repeatable)
    --keep-roles                         don't promote moved users to admin
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine  # noqa: E402

from api.org_service import RemediationReport, remediate_personal_orgs  # noqa: E402


class _DryRunRollback(Exception):
    """Raised inside the transaction to force a rollback in dry-run mode."""


def _async_url(raw: str) -> str:
    return (
        raw.replace("postgresql://", "postgresql+asyncpg://", 1)
        .replace("postgres://", "postgresql+asyncpg://", 1)
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    p.add_argument("--since", type=datetime.fromisoformat, default=None,
                   help="ISO-8601 cutoff; users created after it are moved")
    p.add_argument("--include-admins", action="store_true")
    p.add_argument("--exclude-user-id", action="append", default=[])
    p.add_argument("--keep-roles", action="store_true")
    return p


async def run(engine: AsyncEngine, args: argparse.Namespace) -> RemediationReport:
    """Run the remediation on ``engine``. Dry run always rolls back."""
    holder: dict[str, RemediationReport] = {}
    try:
        async with engine.begin() as conn:
            holder["report"] = await remediate_personal_orgs(
                conn,
                apply=args.apply,
                since=args.since,
                include_admins=args.include_admins,
                exclude_user_ids=args.exclude_user_id,
                promote_to_admin=not args.keep_roles,
            )
            if not args.apply:
                raise _DryRunRollback()
    except _DryRunRollback:
        pass
    return holder["report"]


async def _main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    raw = os.getenv("DATABASE_URL")
    if not raw:
        print("DATABASE_URL is not set.", file=sys.stderr)
        return 2
    engine = create_async_engine(_async_url(raw))
    try:
        report = await run(engine, args)
    finally:
        await engine.dispose()
    for line in report.summary_lines():
        print(line)
    if not args.apply and report.moves:
        print("\nRe-run with --apply to write these changes.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(sys.argv[1:])))
