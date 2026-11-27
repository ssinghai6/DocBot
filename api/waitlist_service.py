"""Waitlist storage + optional founder notification — DOCBOT-2581.

One row per lowercased email; a re-submit updates the row instead of
duplicating it. The notification email reuses Resend's plain HTTP API (same
shape as ``email_verification_service``) and is a no-op unless RESEND_API_KEY,
EMAIL_FROM and WAITLIST_NOTIFY_EMAIL are all set. It never blocks the 202.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Optional

import httpx
from sqlalchemy import Column, DateTime, String, Table, Text, func, insert, select, update
from sqlalchemy.exc import IntegrityError

from api.email_verification_service import RESEND_API_URL
from api.waitlist_models import WaitlistRequest

logger = logging.getLogger(__name__)


def register_waitlist_table(metadata: Any) -> Table:
    return Table(
        "waitlist",
        metadata,
        Column("id", String, primary_key=True),
        Column("email", String, nullable=False, unique=True),
        Column("name", String, nullable=False),
        Column("company", String, nullable=False),
        Column("role", String),
        Column("note", Text),
        Column("source", String, nullable=False),
        Column("user_id", String),
        Column("org_id", String),
        Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False, index=True),
        Column("updated_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    )


async def upsert_entry(
    table: Table,
    session_factory: Any,
    body: WaitlistRequest,
    user_id: Optional[str],
    org_id: Optional[str],
) -> None:
    """Insert, or update the existing row for this email (never duplicates)."""
    values = {
        "name": body.name.strip(),
        "company": body.company.strip(),
        "role": (body.role or "").strip() or None,
        "note": (body.note or "").strip() or None,
        "source": body.source,
        "user_id": user_id,
        "org_id": org_id,
    }

    async def _update(session: Any) -> None:
        await session.execute(
            update(table).where(table.c.email == body.email).values(**values, updated_at=func.now())
        )

    async with session_factory() as session:
        exists = (await session.execute(select(table.c.id).where(table.c.email == body.email))).first()
        try:
            if exists:
                await _update(session)
            else:
                await session.execute(insert(table).values(id=str(uuid.uuid4()), email=body.email, **values))
            await session.commit()
        except IntegrityError:
            # Lost a race with a concurrent first submit for the same email.
            await session.rollback()
            await _update(session)
            await session.commit()


async def list_entries(table: Table, session_factory: Any, limit: int = 1000) -> list[dict[str, Any]]:
    async with session_factory() as session:
        rows = (
            await session.execute(select(table).order_by(table.c.created_at.desc()).limit(limit))
        ).mappings().all()
    return [dict(r) for r in rows]


def notification_configured() -> bool:
    return all((os.getenv(k) or "").strip() for k in ("RESEND_API_KEY", "EMAIL_FROM", "WAITLIST_NOTIFY_EMAIL"))


async def notify_founder(body: WaitlistRequest) -> None:
    """Email the founder a short heads-up. No-op when unconfigured; never raises."""
    if not notification_configured():
        return
    text = (
        f"New DocBot waitlist signup ({body.source}).\n"
        f"{body.name} <{body.email}>, {body.company}\n"
        "Read the full entry (and note) at GET /api/admin/waitlist.\n"
    )
    payload = {
        "from": os.environ["EMAIL_FROM"].strip(),
        "to": [os.environ["WAITLIST_NOTIFY_EMAIL"].strip()],
        "subject": "DocBot waitlist signup",
        "text": text,
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                RESEND_API_URL,
                headers={"Authorization": f"Bearer {os.environ['RESEND_API_KEY'].strip()}"},
                json=payload,
            )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        # Type only: the payload carries the submitter's PII.
        logger.warning("waitlist notification failed: %s", type(exc).__name__)
