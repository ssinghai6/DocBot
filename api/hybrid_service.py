"""
DocBot Hybrid Service — DOCBOT-401

Query Intent Classifier: determines whether a question should be answered from
the document store ("doc"), the connected database ("sql"), or both ("hybrid").

Fallback rules apply when only one source is available (no LLM call needed).
When both sources are present the decision is delegated to the LLM and the
result is logged to query_history for observability.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import insert

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_INTENTS = {"sql", "doc", "hybrid"}
_MODEL = "llama-3.3-70b-versatile"
_SYSTEM_PROMPT = (
    "You are a query routing assistant. "
    "Classify the user's question into exactly one of three categories:\n"
    "  sql    — the answer requires querying a live database\n"
    "  doc    — the answer can be found in uploaded documents\n"
    "  hybrid — the answer requires both database data and document context\n\n"
    "Respond with one word only: sql, doc, or hybrid. No punctuation, no explanation."
)


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------


class IntentClassification(BaseModel):
    intent: Literal["sql", "doc", "hybrid"]
    fallback_applied: bool
    question_hash: str


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _hash_question(question: str) -> str:
    """Return a short, non-reversible fingerprint of the question."""
    return hashlib.sha256(question.encode()).hexdigest()[:16]


async def _log_classification(
    session_id: str,
    question_hash: str,
    result: str,
    query_history_table,
    async_session_factory,
) -> None:
    """
    Persist the classification result to query_history.

    This is fire-and-forget: any database error is swallowed so that a logging
    failure never propagates to the caller.
    """
    try:
        import uuid

        row = {
            "id": str(uuid.uuid4()),
            "connection_id": session_id,
            "nl_question": question_hash,
            "sql_query": "",
            "result_summary": f"intent_classification:{result}",
        }
        async with async_session_factory() as session:
            await session.execute(insert(query_history_table).values(**row))
            await session.commit()
    except Exception as exc:  # noqa: BLE001 — intentionally broad; logging must not raise
        logger.warning("classify_intent: failed to log classification — %s", exc)


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------


async def classify_intent(
    question: str,
    has_db: bool,
    has_docs: bool,
    session_id: str,
    groq_client,
    async_session_factory,
    query_history_table=None,
) -> str:
    """
    Classify a question as "sql", "doc", or "hybrid".

    Parameters
    ----------
    question:
        The natural-language question to classify.
    has_db:
        True when the session has an active database connection.
    has_docs:
        True when the session has at least one uploaded document.
    session_id:
        Used for logging; never sent to the LLM.
    groq_client:
        An initialised groq.AsyncGroq (or compatible) instance.
    async_session_factory:
        SQLAlchemy async_sessionmaker for writing to query_history.
    query_history_table:
        The SQLAlchemy Table object for query_history.  When None, logging is
        skipped (useful in tests that don't wire up a full DB).

    Returns
    -------
    str
        One of "sql", "doc", or "hybrid".
    """
    question_hash = _hash_question(question)

    # ── Fallback rules (no LLM required) ──────────────────────────────────
    if not has_db:
        result = "doc"
        if query_history_table is not None:
            await _log_classification(
                session_id, question_hash, result, query_history_table, async_session_factory
            )
        return result

    if not has_docs:
        result = "sql"
        if query_history_table is not None:
            await _log_classification(
                session_id, question_hash, result, query_history_table, async_session_factory
            )
        return result

    # ── LLM classification (both sources present) ─────────────────────────
    response = await groq_client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        max_tokens=10,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip().lower()
    result = raw if raw in _VALID_INTENTS else "hybrid"

    if result != raw:
        logger.warning(
            "classify_intent: LLM returned unexpected value %r — defaulting to 'hybrid'", raw
        )

    if query_history_table is not None:
        await _log_classification(
            session_id, question_hash, result, query_history_table, async_session_factory
        )

    return result


# ---------------------------------------------------------------------------
# Safe wrapper
# ---------------------------------------------------------------------------


async def classify_intent_safe(
    question: str,
    has_db: bool,
    has_docs: bool,
    session_id: str,
    groq_client,
    async_session_factory,
    query_history_table=None,
) -> IntentClassification:
    """
    Thin wrapper around classify_intent that never raises.

    On any error, returns an IntentClassification with intent="hybrid".
    This ensures the hybrid pipeline always has a route even when the
    classifier is unavailable.
    """
    question_hash = _hash_question(question)
    fallback_applied = not has_db or not has_docs

    try:
        intent = await classify_intent(
            question=question,
            has_db=has_db,
            has_docs=has_docs,
            session_id=session_id,
            groq_client=groq_client,
            async_session_factory=async_session_factory,
            query_history_table=query_history_table,
        )
        return IntentClassification(
            intent=intent,
            fallback_applied=fallback_applied,
            question_hash=question_hash,
        )
    except Exception as exc:
        logger.error(
            "classify_intent_safe: classifier raised unexpectedly — defaulting to 'hybrid'. "
            "Error: %s",
            exc,
        )
        return IntentClassification(
            intent="hybrid",
            fallback_applied=fallback_applied,
            question_hash=question_hash,
        )
