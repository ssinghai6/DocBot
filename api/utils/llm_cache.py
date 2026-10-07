"""Exact-match LLM response cache — DOCBOT-1506.

Deterministic (temperature=0) LLM calls on hot paths — SQL generation and
intent classification — re-pay the full Groq/Gemini cost every time the same
question (or an identical generated prompt) is asked again. This module adds
a small Postgres-backed exact-match cache keyed on ``(prompt_hash, model)``
so an identical prompt within its TTL window is served without a network
call at all.

Design notes
------------
Mirrors ``api/llm_trace_service.py``'s table-registration / wiring pattern:
a table is registered on the shared ``metadata`` at import time in
``api/index.py``, and the table + async session factory are injected once
at startup via :func:`wire_llm_cache`. No new dependency (Redis, etc.) is
introduced — this reuses the existing Railway PostgreSQL instance, per
CLAUDE.md's "no new dependency unless already present and trivial" rule.

Scope is deliberately narrow: only call sites that already run at
``temperature=0`` (SQL generation, intent classification) read/write this
cache. Higher-temperature creative calls (hybrid synthesis, autopilot
planning, persona-formatted answers) are intentionally NOT cached here —
caching a sampling call would make repeated identical questions always
return the exact same phrasing, which is a product regression for those
call sites, not a win.

Safety / correctness
---------------------
* Keyed on a SHA-256 hash of the *fully rendered* prompt text (including
  schema, few-shot examples, system prompt, etc.) plus the model name — a
  cache hit is only possible when the entire prompt is byte-identical, so
  stale/irrelevant hits are structurally impossible.
* TTL'd (``expires_at`` column) — a hit query filters on
  ``expires_at > now()``, so expired rows are simply never returned. A
  lazy sweep isn't implemented (row volume is small for a solo-deploy
  container, matching the precedent set by ``llm_trace_service`` and
  ``metrics_service``); expired rows are overwritten on next write to the
  same key via the delete-then-insert upsert in :func:`set_cached_response`.
* Every lookup/write is wrapped in try/except — a cache outage must never
  break a real LLM call. On any DB error, a lookup degrades to a miss and a
  write is silently dropped.
* Never caches empty/falsy responses (an empty string is almost certainly a
  malformed/failed generation, not a legitimate answer worth replaying).
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import (
    Column,
    DateTime,
    String,
    Table,
    Text,
    UniqueConstraint,
    delete,
    func,
    select,
)
from sqlalchemy import insert as sa_insert

logger = logging.getLogger(__name__)

# Default TTL for a cached response. Deliberately short-ish: these are
# exact-match hits on fully-rendered prompts (schema text, few-shot
# examples, the literal question), so a long TTL buys little extra hit
# rate once the underlying schema/data drifts, while unnecessarily
# resurfacing possibly-stale SQL past that point does carry some risk.
DEFAULT_TTL_SECONDS = 6 * 60 * 60  # 6 hours


# ---------------------------------------------------------------------------
# Table definition
# ---------------------------------------------------------------------------


def register_llm_cache_table(metadata) -> Table:
    """Define the llm_response_cache table on shared metadata.

    Called once at import time from ``api/index.py``, mirroring
    ``llm_trace_service.register_llm_calls_table``.
    """
    llm_response_cache = Table(
        "llm_response_cache",
        metadata,
        Column("id", String, primary_key=True),  # UUID
        Column("prompt_hash", String, nullable=False, index=True),
        Column("model", String, nullable=False, index=True),
        Column("response", Text, nullable=False),
        Column(
            "created_at",
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
        ),
        Column("expires_at", DateTime(timezone=True), nullable=False, index=True),
        UniqueConstraint("prompt_hash", "model", name="uq_llm_cache_prompt_model"),
    )
    return llm_response_cache


# ---------------------------------------------------------------------------
# Module-level wiring
# ---------------------------------------------------------------------------

_llm_cache_table: Optional[Table] = None
_async_session_factory: Any = None

# In-process counters — DOCBOT-1506 acceptance criterion #3. Deliberately
# the same lightweight in-memory-counter approach as
# hybrid_service._intent_telemetry rather than a new persisted table; a
# process restart resetting these is acceptable for this ticket's scope.
_hits = 0
_misses = 0
_sets = 0
_errors = 0


def wire_llm_cache(llm_cache_table: Table, async_session_factory: Any) -> None:
    """Inject table + session references. Called once from index.py lifespan."""
    global _llm_cache_table, _async_session_factory
    _llm_cache_table = llm_cache_table
    _async_session_factory = async_session_factory


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def hash_prompt(prompt_text: str) -> str:
    """Return a stable SHA-256 hex digest of the fully-rendered prompt text."""
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------


async def get_cached_response(prompt_hash: str, model: str) -> Optional[str]:
    """Return the cached response for (prompt_hash, model) if present and
    unexpired, else None.

    Never raises — a lookup failure (DB unavailable, not wired, etc.)
    degrades to a miss, counted the same as a genuine miss.
    """
    global _hits, _misses, _errors

    if _llm_cache_table is None or _async_session_factory is None:
        _misses += 1
        return None

    try:
        now = datetime.now(timezone.utc)
        async with _async_session_factory() as session:
            result = await session.execute(
                select(_llm_cache_table.c.response).where(
                    _llm_cache_table.c.prompt_hash == prompt_hash,
                    _llm_cache_table.c.model == model,
                    _llm_cache_table.c.expires_at > now,
                )
            )
            row = result.first()
    except Exception as exc:  # cache must never break the real call path
        _errors += 1
        logger.warning("llm_cache: get_cached_response failed (%s)", exc)
        _misses += 1
        return None

    if row is None:
        _misses += 1
        return None

    _hits += 1
    return row[0]


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------


async def set_cached_response(
    prompt_hash: str,
    model: str,
    response: str,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> None:
    """Upsert a cache row for (prompt_hash, model).

    Implemented as delete-then-insert within one transaction rather than a
    dialect-specific ON CONFLICT upsert — this module is intentionally
    dialect-agnostic (DocBot's test suite exercises it against SQLite;
    production runs Postgres) and the row volume here never justifies the
    extra complexity of a native upsert.

    Never raises — a write failure is logged and dropped; losing a cache
    entry only costs a future cache miss, never correctness.
    """
    global _sets, _errors

    if not response:
        return
    if _llm_cache_table is None or _async_session_factory is None:
        return

    try:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=ttl_seconds)
        async with _async_session_factory() as session:
            async with session.begin():
                await session.execute(
                    delete(_llm_cache_table).where(
                        _llm_cache_table.c.prompt_hash == prompt_hash,
                        _llm_cache_table.c.model == model,
                    )
                )
                await session.execute(
                    sa_insert(_llm_cache_table).values(
                        id=str(uuid.uuid4()),
                        prompt_hash=prompt_hash,
                        model=model,
                        response=response,
                        expires_at=expires_at,
                    )
                )
        _sets += 1
    except Exception as exc:  # cache must never break the real call path
        _errors += 1
        logger.warning("llm_cache: set_cached_response failed (%s)", exc)


# ---------------------------------------------------------------------------
# Metrics — DOCBOT-1506 acceptance criterion #3
# ---------------------------------------------------------------------------


def get_cache_metrics() -> dict[str, Any]:
    """Snapshot of in-process cache hit/miss/set/error counters.

    Consumed by ``api/metrics_service.py`` for the ``/admin/metrics``
    endpoint (added as a minimal ``llm_cache_metrics`` block alongside the
    existing ``llm_metrics`` cost/latency block).
    """
    total = _hits + _misses
    return {
        "hits": _hits,
        "misses": _misses,
        "sets": _sets,
        "errors": _errors,
        "hit_rate": round(_hits / total, 4) if total else None,
    }


def reset_metrics_for_tests() -> None:
    """Reset in-process counters. Test-only helper."""
    global _hits, _misses, _sets, _errors
    _hits = 0
    _misses = 0
    _sets = 0
    _errors = 0
