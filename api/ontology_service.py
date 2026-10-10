"""Data Ontology Service — DOCBOT-1517.

Persistent "data ontology" layer for connected databases: precomputed table
embeddings + compact per-table "data cards" + naming-convention join hints,
so the SQL pipeline's table selector (``api/db_service.py`` step 2) and the
Autopilot planner (``api/autopilot_service.py``'s ``_planner_node``) can do a
cheap nearest-neighbor lookup *before* spending an LLM call, instead of
re-deriving table relevance from a flattened schema dump on every question.

This is strictly additive. ``ontology_lookup()`` returns ``None`` when no
ontology has been built yet for a connection — callers MUST treat that as an
explicit "not built" signal and fall through to their existing behavior
unchanged. The ontology is never a verdict; it only narrows/biases a
candidate set that the existing semantic/LLM table selector still runs on.

Design notes
------------
This mirrors the ``register_X_table`` / ``wire_X_store`` pattern already used
by ``api/trace_service.py`` (itself modeled on ``api/llm_trace_service.py``
and ``api/lineage_service.py``):

  * ``register_ontology_table(metadata)`` defines the table once at import
    time from ``api/index.py``.
  * ``wire_ontology_store(table, db_connections_table, schema_cache_table,
    async_session_factory)`` injects the live table + table refs + session
    factory from the FastAPI lifespan.
  * ``build_ontology(connection_id)`` (re)computes and upserts this
    connection's ontology rows. Called at the same points schema is already
    introspected/cached (``connect_database``, the schema-drift retry path
    inside ``run_sql_pipeline``, and the manual
    ``POST /api/db/refresh-schema/{connection_id}`` endpoint) — never on a
    separate trigger of its own. Always fire-and-forget from callers; never
    blocks a request on this computation.
  * ``ontology_lookup(connection_id, question, top_k)`` is the cheap
    nearest-neighbor read path.

Vector storage convention
--------------------------
This codebase has **no pgvector usage anywhere** (checked
``api/utils/vector_store.py`` — that wraps Chroma for document RAG, not SQL
rows) and no pgvector-backed column type elsewhere in ``api/index.py``'s
table definitions. The one existing "stored embeddings for nearest-neighbor
lookup over SQL schema objects" table — ``table_embeddings`` (DOCBOT-503,
see ``api/utils/table_selector.py``) — stores each embedding as a JSON
float-array string in a plain ``Text`` column and ranks candidates with
``cosine_similarity()`` in Python (``api/utils/embeddings.py``, also reused
by few-shot query retrieval). ``data_ontology.embedding`` follows that exact
convention rather than introducing pgvector.

For the upsert itself, this module intentionally does NOT copy
``table_embeddings``'s PostgreSQL-only ``INSERT ... ON CONFLICT`` — it
instead mirrors the explicit "row exists? UPDATE : INSERT" check already
used for ``schema_cache_table`` in ``api/db_service.py::get_schema()``. That
keeps ``build_ontology()`` dialect-agnostic (testable against an in-memory
SQLite engine like every other service module's unit tests, not only a live
PostgreSQL), while still being race-safe within the single
``session.begin()`` transaction used here.

Deferred scope (explicitly, per DOCBOT-1517 ticket text)
---------------------------------------------------------
* **FK-based join hints**: the existing schema introspection
  (``api/db_service.py::_introspect_schema_from_url``) does not collect
  ``inspector.get_foreign_keys()`` — only column name/type. Adding that query
  is out of scope here (it would be new introspection work, not reuse), so
  ``join_hints_json`` is populated from a cheap *naming-convention* heuristic
  only (``customer_id`` -> likely references a ``customers``/``customer``
  table), computed purely from already-cached schema — no new DB queries.
* **agent_traces co-occurrence mining**: nontrivial (would need aggregating
  ``plan_steps_json``/``retrieved_refs_json`` across rows, no existing index
  for it) — explicitly deferred, not attempted in this ticket.
* **row_count / cardinality / example rows**: the existing introspection
  path does not fetch row counts or sample data (the one exception,
  ``_sort_tables_by_row_estimate_pg``, is a PostgreSQL-only, >50-table-only
  ordering helper not exposed as reusable data) — adding a new query per
  table to populate these would violate "don't add new expensive queries".
  ``data_card_json`` therefore only carries cheaply-available structural
  facts (column count, is_view) with ``row_count``/``example_rows`` left
  null/empty and documented as deferred.
"""

from __future__ import annotations

import json
import logging
import uuid as _uuid
from typing import Any, Optional

from sqlalchemy import Column, DateTime, String, Table, Text, UniqueConstraint, func, select
from sqlalchemy import insert as sa_insert
from sqlalchemy import update as sa_update

from api.utils.embeddings import cosine_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Semantic-tag heuristics (name/type based — no new queries)
# ---------------------------------------------------------------------------

_DATE_NAME_HINTS = ("date", "time", "created", "updated", "_at", "timestamp")
_DATE_TYPE_HINTS = ("date", "timestamp")
_AMOUNT_NAME_HINTS = (
    "amount", "price", "cost", "total", "revenue", "qty", "quantity",
    "balance", "fee", "sum", "spend", "value",
)
_AMOUNT_TYPE_HINTS = ("int", "float", "numeric", "decimal", "double", "real", "money")
_CATEGORY_NAME_HINTS = ("type", "category", "status", "name", "label", "kind", "region", "segment")

_MAX_COLS_PER_TABLE = 20  # mirrors api/utils/table_selector.py's cap


def _infer_semantic_tag(col_name: str, col_type: str) -> str:
    """Heuristic semantic tag for one column, from its name + type only."""
    name_l = (col_name or "").lower()
    type_l = (col_type or "").lower()

    if name_l == "id" or name_l.endswith("_id") or name_l.endswith("id"):
        return "entity"
    if any(h in name_l for h in _DATE_NAME_HINTS) or any(h in type_l for h in _DATE_TYPE_HINTS):
        return "date"
    if any(h in name_l for h in _AMOUNT_NAME_HINTS) and any(h in type_l for h in _AMOUNT_TYPE_HINTS):
        return "amount"
    if any(h in name_l for h in _CATEGORY_NAME_HINTS):
        return "category"
    return "other"


def _build_column_metadata(table: dict) -> list[dict]:
    cols = (table.get("columns") or [])[:_MAX_COLS_PER_TABLE]
    return [
        {
            "name": c.get("name", ""),
            "type": c.get("type", "unknown"),
            "semantic_tag": _infer_semantic_tag(c.get("name", ""), c.get("type", "")),
            # null_pct: not cheaply available from existing introspection — deferred.
            "null_pct": None,
        }
        for c in cols
    ]


def _build_data_card(table: dict) -> dict:
    """Cheap, structural-only data card. See module docstring for deferrals."""
    return {
        "column_count": len(table.get("columns") or []),
        "is_view": bool(table.get("is_view", False)),
        "row_count": None,  # deferred — would require a new per-table query
        "example_rows": [],  # deferred — would require a new per-table query
    }


def _infer_join_hints(table_name: str, column_metadata: list[dict], all_table_names: list[str]) -> list[dict]:
    """Naming-convention-only join hints (FK introspection/co-occurrence deferred).

    Matches columns like ``customer_id`` against known table names
    (``customer``, ``customers``) already present in this connection's
    schema. Purely string matching over already-cached schema — zero new
    DB queries.
    """
    hints: list[dict] = []
    table_name_set = {t.lower() for t in all_table_names}
    self_name_l = (table_name or "").lower()

    for col in column_metadata:
        name_l = (col.get("name") or "").lower()
        if name_l in ("id",) or not name_l.endswith("_id"):
            continue
        base = name_l[: -len("_id")]
        if not base:
            continue
        for candidate in (base, f"{base}s", f"{base}es"):
            if candidate in table_name_set and candidate != self_name_l:
                hints.append({
                    "column": col.get("name"),
                    "likely_references_table": candidate,
                    "basis": "naming_convention",
                })
                break
    return hints


def _build_embedding_text(table_name: str, column_metadata: list[dict]) -> str:
    """Richer-than-table_selector text repr (includes inferred semantic tags)."""
    col_parts = ", ".join(
        f"{c['name']} ({c['type']}, {c['semantic_tag']})" for c in column_metadata
    )
    return f"{table_name}: {col_parts}" if col_parts else table_name


# ---------------------------------------------------------------------------
# Table definition
# ---------------------------------------------------------------------------


def register_ontology_table(metadata: Any) -> Table:
    """Define the data_ontology table on shared metadata.

    Called once at import time from ``api/index.py``, mirroring
    ``register_agent_traces_table`` / ``register_llm_calls_table``.
    """
    return Table(
        "data_ontology",
        metadata,
        Column("id", String, primary_key=True),  # UUID, minted by build_ontology()
        Column("connection_id", String, nullable=False, index=True),
        Column("table_name", String, nullable=False),
        Column("column_metadata_json", Text, nullable=False),  # per-column name/type/semantic_tag/null_pct
        Column("embedding", Text, nullable=False),  # JSON float array — same convention as table_embeddings
        Column("data_card_json", Text),  # row_count/cardinality/example_rows (structural-only, see module docstring)
        Column("join_hints_json", Text),  # naming-convention-inferred FK-like relationships
        Column(
            "updated_at",
            DateTime(timezone=True),
            server_default=func.now(),
            onupdate=func.now(),
            nullable=False,
        ),
        UniqueConstraint("connection_id", "table_name", name="uq_data_ontology_connection_table"),
    )


# ---------------------------------------------------------------------------
# Module-level wiring
# ---------------------------------------------------------------------------

_ontology_table: Optional[Table] = None
_db_connections_table: Optional[Table] = None
_schema_cache_table: Optional[Table] = None
_async_session_factory: Any = None


def wire_ontology_store(
    table: Table,
    db_connections_table: Table,
    schema_cache_table: Table,
    async_session_factory: Any,
) -> None:
    """Inject table + table refs + session factory. Called once from index.py lifespan."""
    global _ontology_table, _db_connections_table, _schema_cache_table, _async_session_factory
    _ontology_table = table
    _db_connections_table = db_connections_table
    _schema_cache_table = schema_cache_table
    _async_session_factory = async_session_factory


def is_wired() -> bool:
    return _ontology_table is not None and _async_session_factory is not None


# ---------------------------------------------------------------------------
# Build / refresh
# ---------------------------------------------------------------------------


async def build_ontology(connection_id: str) -> None:
    """(Re)compute and upsert the data ontology for one connection.

    Safe to call repeatedly (upserts on connection_id+table_name). Never
    raises — any failure is logged and swallowed so this can always be
    scheduled fire-and-forget from the same lifecycle points schema is
    already introspected/cached, without ever blocking or breaking that
    request.
    """
    if not is_wired():
        logger.debug("ontology_service: not wired, skipping build for %s", connection_id)
        return

    import asyncio

    try:
        # Local import — avoids a module-level circular import with
        # api.db_service (which calls into this module for the step-2
        # pre-filter). get_schema() reuses the existing schema_cache, so
        # this is a cache hit in the common case, not a fresh introspection.
        from api.db_service import get_schema
        from api.utils.embeddings_provider import get_embeddings

        schema = await get_schema(
            connection_id, _db_connections_table, _schema_cache_table, _async_session_factory
        )
        if not schema:
            return

        all_table_names = [t.get("name", "") for t in schema]
        embeddings_model = get_embeddings()

        per_table: list[dict] = []
        texts: list[str] = []
        for table in schema:
            table_name = table.get("name", "")
            if not table_name:
                continue
            column_metadata = _build_column_metadata(table)
            per_table.append({
                "table_name": table_name,
                "column_metadata": column_metadata,
                "data_card": _build_data_card(table),
                "join_hints": _infer_join_hints(table_name, column_metadata, all_table_names),
            })
            texts.append(_build_embedding_text(table_name, column_metadata))

        if not per_table:
            return

        def _batch_embed() -> list[list[float]]:
            return embeddings_model.embed_documents(texts)

        loop = asyncio.get_running_loop()
        vectors: list[list[float]] = await loop.run_in_executor(None, _batch_embed)

        # Upsert pattern: explicit "row exists? update : insert" check, same
        # convention already used for schema_cache_table in
        # api/db_service.py::get_schema() — kept dialect-agnostic (works
        # against SQLite for unit tests, not just PostgreSQL) rather than
        # the pg-only ON CONFLICT used by the sibling table_embeddings table.
        async with _async_session_factory() as session:
            async with session.begin():
                existing_result = await session.execute(
                    select(_ontology_table.c.table_name).where(
                        _ontology_table.c.connection_id == connection_id
                    )
                )
                existing_names = {row.table_name for row in existing_result.fetchall()}

                for row, vec in zip(per_table, vectors):
                    values = {
                        "column_metadata_json": json.dumps(row["column_metadata"]),
                        "embedding": json.dumps(vec),
                        "data_card_json": json.dumps(row["data_card"]),
                        "join_hints_json": json.dumps(row["join_hints"]),
                    }
                    if row["table_name"] in existing_names:
                        await session.execute(
                            sa_update(_ontology_table)
                            .where(
                                _ontology_table.c.connection_id == connection_id,
                                _ontology_table.c.table_name == row["table_name"],
                            )
                            .values(**values, updated_at=func.now())
                        )
                    else:
                        await session.execute(
                            sa_insert(_ontology_table).values(
                                id=str(_uuid.uuid4()),
                                connection_id=connection_id,
                                table_name=row["table_name"],
                                **values,
                            )
                        )

        logger.info(
            "ontology_service: built ontology for connection=%s tables=%d",
            connection_id, len(per_table),
        )
    except Exception as exc:
        logger.warning("ontology_service: build_ontology failed (non-fatal) for %s: %s", connection_id, exc)


# ---------------------------------------------------------------------------
# Lookup — cheap nearest-neighbor pre-filter
# ---------------------------------------------------------------------------


async def ontology_lookup(connection_id: str, question: str, top_k: int = 8) -> Optional[list[dict]]:
    """Return the top-k most relevant tables for *question*, or ``None``.

    ``None`` is the explicit "no ontology built yet for this connection"
    signal — callers MUST fall back to their existing behavior unchanged
    when they see it. A non-``None`` result is a cheap *pre-filter*, not a
    verdict: callers should narrow/bias a candidate set, not treat this as
    the final answer.

    Never raises — any failure (DB error, embeddings error, etc.) is logged
    and treated the same as "not built yet" (returns ``None``), so a lookup
    failure can never break the caller's existing pipeline.
    """
    if not is_wired():
        return None

    try:
        async with _async_session_factory() as session:
            result = await session.execute(
                select(
                    _ontology_table.c.table_name,
                    _ontology_table.c.column_metadata_json,
                    _ontology_table.c.embedding,
                    _ontology_table.c.data_card_json,
                    _ontology_table.c.join_hints_json,
                ).where(_ontology_table.c.connection_id == connection_id)
            )
            rows = result.fetchall()

        if not rows:
            return None  # no ontology built yet — explicit fallback signal

        import asyncio

        from api.utils.embeddings_provider import get_embeddings

        embeddings_model = get_embeddings()

        def _embed_q() -> list[float]:
            return embeddings_model.embed_query(question)

        loop = asyncio.get_running_loop()
        q_vec: list[float] = await loop.run_in_executor(None, _embed_q)

        scored = []
        for row in rows:
            vec = json.loads(row.embedding) if isinstance(row.embedding, str) else row.embedding
            score = cosine_similarity(q_vec, vec)
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        return [
            {
                "table_name": row.table_name,
                "score": score,
                "column_metadata": json.loads(row.column_metadata_json) if row.column_metadata_json else [],
                "data_card": json.loads(row.data_card_json) if row.data_card_json else {},
                "join_hints": json.loads(row.join_hints_json) if row.join_hints_json else [],
            }
            for score, row in top
        ]
    except Exception as exc:
        logger.warning("ontology_service: ontology_lookup failed (non-fatal) for %s: %s", connection_id, exc)
        return None
