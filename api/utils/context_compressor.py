"""
DOCBOT-502: Context Compression

Keeps LLM context costs bounded for long sessions by summarising older turns
into a rolling bullet-point summary. The summary is stored in the `sessions`
table (`context_summary` column) and prepended to subsequent LLM calls instead
of the full message history.

Policy
------
- Compression fires when a session has accumulated >= COMPRESSION_THRESHOLD new
  messages since the last compression (or since the session start).
- After compression, the LLM receives:  [summary] + last RECENT_WINDOW messages
- The user's full conversation history in the UI is NEVER truncated.

Public API
----------
should_compress(session_id, sessions_table, async_session_factory) -> bool
compress_session(session_id, groq_api_key, messages_table, sessions_table,
                 async_session_factory) -> str   # returns new summary
build_compressed_context(session_id, messages_table, sessions_table,
                         async_session_factory) -> list[dict]
    Returns the message list to pass to the LLM, already compressed when needed.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import Table, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

logger = logging.getLogger(__name__)

COMPRESSION_THRESHOLD = 20   # compress after this many new messages
RECENT_WINDOW = 10           # keep this many raw messages after compression


async def should_compress(
    session_id: str,
    messages_table: Table,
    sessions_table: Table,
    async_session_factory: async_sessionmaker,
) -> bool:
    """Return True if this session needs compression.

    Fires when total message count minus the count at last compression >= threshold.
    """
    try:
        async with async_session_factory() as session:
            # Count total messages for this session
            total_result = await session.execute(
                select(messages_table.c.id)
                .where(messages_table.c.session_id == session_id)
            )
            total = len(total_result.fetchall())

            # Get message_count_at_compression from sessions table
            sess_result = await session.execute(
                select(sessions_table.c.message_count_at_compression)
                .where(sessions_table.c.session_id == session_id)
            )
            row = sess_result.fetchone()
            count_at_last = row.message_count_at_compression if row and row.message_count_at_compression else 0

        return (total - count_at_last) >= COMPRESSION_THRESHOLD
    except Exception as exc:
        logger.warning("should_compress check failed: %s", exc)
        return False


async def compress_session(
    session_id: str,
    groq_api_key: str,
    messages_table: Table,
    sessions_table: Table,
    async_session_factory: async_sessionmaker,
) -> str:
    """Summarise conversation history and persist the summary.

    Fetches all but the last RECENT_WINDOW messages, calls Groq Llama to
    produce a 3–5 bullet-point summary, and writes it back to `sessions`.

    Returns the new summary string, or empty string on failure.
    Never raises — compression failures are non-fatal.
    """
    try:
        from groq import Groq

        async with async_session_factory() as session:
            result = await session.execute(
                select(messages_table.c.role, messages_table.c.content)
                .where(messages_table.c.session_id == session_id)
                .order_by(messages_table.c.timestamp.asc())
            )
            all_messages = result.fetchall()

        total = len(all_messages)
        if total <= RECENT_WINDOW:
            return ""  # nothing to compress

        messages_to_compress = all_messages[: total - RECENT_WINDOW]

        # Build a plain-text history for the LLM
        history_text = "\n".join(
            f"{row.role.upper()}: {row.content[:500]}"
            for row in messages_to_compress
        )

        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise summariser. Summarise the conversation below "
                        "into 3–5 short bullet points. Focus on: what data sources are "
                        "connected, what questions were asked, and what key findings emerged. "
                        "Be factual and brief."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Conversation to summarise:\n\n{history_text[:8000]}",
                },
            ],
            temperature=0.3,
            max_tokens=300,
        )
        summary = response.choices[0].message.content.strip()

        # Persist summary and update compression checkpoint
        async with async_session_factory() as session:
            async with session.begin():
                await session.execute(
                    update(sessions_table)
                    .where(sessions_table.c.session_id == session_id)
                    .values(
                        context_summary=summary,
                        message_count_at_compression=total,
                    )
                )

        logger.info(
            "compress_session: session=%s compressed %d messages into %d-char summary",
            session_id,
            len(messages_to_compress),
            len(summary),
        )
        return summary

    except Exception as exc:
        logger.warning("compress_session failed (non-fatal): %s", exc)
        return ""


async def build_compressed_context(
    session_id: str,
    messages_table: Table,
    sessions_table: Table,
    async_session_factory: async_sessionmaker,
) -> list[dict]:
    """Return the LLM message list with compression applied when a summary exists.

    If a context_summary exists:
        [{"role": "system", "content": "<summary>"}] + last RECENT_WINDOW messages
    Otherwise:
        All messages in chronological order (existing behaviour).
    """
    try:
        async with async_session_factory() as session:
            # Fetch session summary
            sess_result = await session.execute(
                select(sessions_table.c.context_summary)
                .where(sessions_table.c.session_id == session_id)
            )
            sess_row = sess_result.fetchone()
            summary: Optional[str] = sess_row.context_summary if sess_row else None

            # Fetch messages
            msg_result = await session.execute(
                select(messages_table.c.role, messages_table.c.content)
                .where(messages_table.c.session_id == session_id)
                .order_by(messages_table.c.timestamp.asc())
            )
            all_messages = [{"role": r.role, "content": r.content} for r in msg_result.fetchall()]

        if summary:
            recent = all_messages[-RECENT_WINDOW:]
            context_intro = {
                "role": "system",
                "content": f"Earlier conversation summary:\n{summary}",
            }
            return [context_intro] + recent

        return all_messages

    except Exception as exc:
        logger.warning("build_compressed_context failed: %s", exc)
        return []
