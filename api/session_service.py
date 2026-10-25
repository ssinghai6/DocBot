"""Chat-session history: owner-scoped list / get / delete / export — DOCBOT-1601.

Every query here carries ``scope_filter(sessions_table, scope)`` so a caller
can only see or remove sessions they own (see ``api/utils/caller_scope.py``).
A session that exists but belongs to someone else is indistinguishable from
a missing one: both raise ``SessionNotFound`` (HTTP 404 at the route).

Tables and the session factory are injected by the caller (``api/index.py``)
like the other services, so tests can run against SQLite.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

from sqlalchemy import and_, delete, select

from api.utils.caller_scope import CallerScope, scope_filter

SESSION_LIST_LIMIT = 50
EXPORT_FORMATS = ("txt", "json", "markdown")


class SessionNotFound(Exception):
    """The session does not exist or is not owned by the caller."""


@dataclass(frozen=True)
class ExportFile:
    """A rendered text export."""

    body: bytes
    media_type: str
    filename: str


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _files_info(raw: Optional[str]) -> list[dict[str, Any]]:
    return json.loads(raw) if raw else []


def _summary(row: Any) -> dict[str, Any]:
    return {
        "session_id": row.session_id,
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
        "persona": row.persona,
        "file_count": row.file_count,
        "files_info": _files_info(row.files_info),
    }


async def list_sessions(
    scope: CallerScope, sessions_table: Any, async_session_factory: Any
) -> list[dict[str, Any]]:
    """Most recently updated sessions owned by the caller (max 50)."""
    async with async_session_factory() as db:
        result = await db.execute(
            select(sessions_table)
            .where(scope_filter(sessions_table, scope))
            .order_by(sessions_table.c.updated_at.desc())
            .limit(SESSION_LIST_LIMIT)
        )
        return [_summary(row) for row in result.fetchall()]


async def _owned_session_row(db: Any, session_id: str, scope: CallerScope, sessions_table: Any) -> Any:
    result = await db.execute(
        select(sessions_table).where(
            and_(sessions_table.c.session_id == session_id, scope_filter(sessions_table, scope))
        )
    )
    row = result.fetchone()
    if row is None:
        raise SessionNotFound(session_id)
    return row


async def is_session_owned(
    session_id: str, scope: CallerScope, sessions_table: Any, async_session_factory: Any
) -> bool:
    """True when ``session_id`` exists and is visible to ``scope``."""
    async with async_session_factory() as db:
        try:
            await _owned_session_row(db, session_id, scope, sessions_table)
        except SessionNotFound:
            return False
    return True


async def get_session_detail(
    session_id: str,
    scope: CallerScope,
    sessions_table: Any,
    messages_table: Any,
    async_session_factory: Any,
) -> dict[str, Any]:
    """Session summary + ordered messages. Raises ``SessionNotFound``."""
    async with async_session_factory() as db:
        row = await _owned_session_row(db, session_id, scope, sessions_table)
        msg_result = await db.execute(
            select(messages_table)
            .where(messages_table.c.session_id == session_id)
            .order_by(messages_table.c.timestamp.asc())
        )
        message_rows = msg_result.fetchall()
    messages = [
        {
            "role": m.role,
            "content": m.content,
            "sources": json.loads(m.sources) if m.sources else [],
            "timestamp": _iso(m.timestamp),
        }
        for m in message_rows
    ]
    return {"session": _summary(row), "messages": messages}


async def delete_owned_session(
    session_id: str,
    scope: CallerScope,
    sessions_table: Any,
    messages_table: Any,
    async_session_factory: Any,
    on_deleted: Optional[Callable[[str], None]] = None,
) -> None:
    """Delete a session the caller owns, plus its messages.

    The ownership check runs first, inside the same transaction as the
    DELETEs, and the session DELETE repeats the scope predicate (defence in
    depth). ``on_deleted`` (vector-store cleanup) runs only after the rows
    are gone, so a foreign id never touches another user's index.
    Raises ``SessionNotFound`` without deleting anything.
    """
    async with async_session_factory() as db:
        async with db.begin():
            await _owned_session_row(db, session_id, scope, sessions_table)
            await db.execute(delete(messages_table).where(messages_table.c.session_id == session_id))
            await db.execute(
                delete(sessions_table).where(
                    and_(sessions_table.c.session_id == session_id, scope_filter(sessions_table, scope))
                )
            )
    if on_deleted is not None:
        on_deleted(session_id)


def _filenames(files_info: list[dict[str, Any]]) -> str:
    return ", ".join(str(f.get("filename", "")) for f in files_info)


def render_export(detail: dict[str, Any], fmt: str, now: Optional[datetime] = None) -> ExportFile:
    """Render a txt / markdown export from ``get_session_detail`` output."""
    session = detail["session"]
    messages = detail["messages"]
    stamp = (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    short_id = session["session_id"][:8]
    files = _filenames(session["files_info"])

    if fmt == "txt":
        out = (
            "DocBot Conversation Export\n"
            "=============================\n"
            f"Session ID: {session['session_id']}\n"
            f"Persona: {session['persona']}\n"
            f"Files: {files}\n"
            f"Exported: {stamp}\n\n"
        )
        for msg in messages:
            role = "You" if msg["role"] == "user" else "DocBot"
            out += f"\n--- {role} ---\n{msg['content']}"
            if msg["sources"]:
                out += "\nSources:"
                for src in msg["sources"]:
                    out += f"\n  - {src.get('source')}, Page {src.get('page')}"
            out += "\n"
        return ExportFile(out.encode("utf-8"), "text/plain", f"docbot_{short_id}.txt")

    if fmt == "markdown":
        out = (
            "# DocBot Conversation Export\n\n"
            f"**Session ID:** {session['session_id']}\n"
            f"**Persona:** {session['persona']}\n"
            f"**Files:** {files}\n"
            f"**Exported:** {stamp}\n\n---\n"
        )
        for msg in messages:
            role = "**You**" if msg["role"] == "user" else "**DocBot**"
            out += f"\n### {role}\n\n{msg['content']}\n"
            if msg["sources"]:
                out += "\n**Sources:**\n"
                for src in msg["sources"]:
                    out += f"- {src.get('source')}, Page {src.get('page')}\n"
            out += "\n---\n"
        return ExportFile(out.encode("utf-8"), "text/markdown", f"docbot_{short_id}.md")

    raise ValueError(f"Unsupported export format: {fmt}")
