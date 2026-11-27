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
import re
import uuid as _uuid
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import Column, DateTime, String, Table, Text, UniqueConstraint, func, select
from sqlalchemy import insert as sa_insert
from sqlalchemy import update as sa_update

from api.utils.embeddings import cosine_similarity
from api.utils.value_fingerprint import MAX_FINGERPRINTS, fingerprint_values

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
# DOCBOT-2911: SQL value fingerprints only for join-key-like columns, bounded per connection.
_FINGERPRINT_TAGS = ("entity", "category")
_MAX_FINGERPRINT_COLS = 30


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
            "value_hashes": [],  # DOCBOT-2911: filled for id/category columns by _attach_sql_value_hashes
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

    try:
        conn_row = await _load_connection_row(connection_id)
        if conn_row is not None and conn_row.dialect == "csv":
            # CSV ontology is built once at upload from the raw bytes
            # (build_csv_ontology); the bytes never change, so a schema
            # refresh must not overwrite it with the TEXT-only schema cache.
            return

        # Local import — avoids a module-level circular import with
        # api.db_service (which calls into this module for the step-2
        # pre-filter). get_schema() reuses the existing schema_cache, so
        # this is a cache hit in the common case, not a fresh introspection.
        from api.db_service import get_schema

        schema = await get_schema(
            connection_id, _db_connections_table, _schema_cache_table, _async_session_factory
        )
        if not schema:
            return

        all_table_names = [t.get("name", "") for t in schema]

        per_table: list[dict] = []
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

        if not per_table:
            return

        if conn_row is not None:
            await _attach_sql_value_hashes(connection_id, conn_row, per_table)
        await _embed_and_upsert(connection_id, per_table)

        logger.info(
            "ontology_service: built ontology for connection=%s tables=%d",
            connection_id, len(per_table),
        )
    except Exception as exc:
        logger.warning("ontology_service: build_ontology failed (non-fatal) for %s: %s", connection_id, exc)


async def _load_connection_row(connection_id: str) -> Optional[Any]:
    """The db_connections row, or None if unavailable (never raises)."""
    try:
        async with _async_session_factory() as session:
            result = await session.execute(
                select(_db_connections_table).where(_db_connections_table.c.id == connection_id)
            )
            return result.fetchone()
    except Exception as exc:
        logger.debug("ontology_service: connection row unavailable for %s: %s", connection_id, exc)
        return None


async def _attach_sql_value_hashes(connection_id: str, conn_row: Any, per_table: list[dict]) -> None:
    """Fill ``value_hashes`` for id/category columns via bounded DISTINCT reads.

    Read-only enforcement is the pipeline's own: sqlglot AST validation, then
    ``_execute_query`` (read-only transaction, 15s timeout, row cap). At most
    ``_MAX_FINGERPRINT_COLS`` columns per connection; any failure skips silently.
    """
    import asyncio

    from sqlglot import exp

    from api.db_service import (
        ExecutionTimeoutError,
        _decrypt_conn_credentials_lazy,
        _execute_query,
        _resolve_connection,
    )
    from api.utils.sql_validator import validate_and_sanitize_sql

    dialect = conn_row.dialect
    sg_dialect = {"postgresql": "postgres", "azure_sql": "tsql"}.get(dialect, dialect)
    try:
        creds = await _decrypt_conn_credentials_lazy(
            conn_row, connection_id, _db_connections_table, _async_session_factory
        )
        sync_url, entra_args = await asyncio.get_running_loop().run_in_executor(
            None, _resolve_connection, creds
        )
    except Exception as exc:
        logger.debug("ontology_service: value hashes skipped for %s: %s", connection_id, type(exc).__name__)
        return

    budget = _MAX_FINGERPRINT_COLS
    for row in per_table:
        for col in row["column_metadata"]:
            if col["semantic_tag"] not in _FINGERPRINT_TAGS:
                continue
            if budget <= 0:
                return
            budget -= 1
            sql = (
                exp.select(exp.column(col["name"], quoted=True))
                .distinct()
                .from_(exp.table_(row["table_name"], quoted=True))
                .limit(MAX_FINGERPRINTS)
                .sql(dialect=sg_dialect)
            )
            try:
                safe_sql = validate_and_sanitize_sql(sql, dialect=dialect)
                rows, _ = await _execute_query(
                    safe_sql, sync_url, dialect, entra_args, row_cap=MAX_FINGERPRINTS
                )
            except ExecutionTimeoutError:
                return  # a slow source: don't spend 15s on each remaining column
            except Exception as exc:
                logger.debug("ontology_service: value hashes skipped for a column: %s", type(exc).__name__)
                continue
            col["value_hashes"] = fingerprint_values(r[0] for r in rows)


async def _embed_and_upsert(connection_id: str, per_table: list[dict]) -> None:
    """Embed each table's text repr and upsert its row (raises on failure)."""
    import asyncio

    from api.utils.embeddings_provider import get_embeddings

    embeddings_model = get_embeddings()
    texts = [_build_embedding_text(r["table_name"], r["column_metadata"]) for r in per_table]

    def _batch_embed() -> list[list[float]]:
        return embeddings_model.embed_documents(texts)

    vectors: list[list[float]] = await asyncio.get_running_loop().run_in_executor(None, _batch_embed)

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


# ---------------------------------------------------------------------------
# CSV uploads (DOCBOT-2911) — built from the raw bytes + cached DataProfile,
# never through the SQL pipeline
# ---------------------------------------------------------------------------


def _read_csv_raw(file_bytes: bytes) -> Any:
    """Headerless str frame, read exactly like ``split_csv_sections`` so its
    section row/col indices line up."""
    import io

    import pandas as pd

    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            raw = pd.read_csv(io.BytesIO(file_bytes), header=None, dtype=str, on_bad_lines="skip", encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
        return raw.dropna(how="all").reset_index(drop=True).dropna(axis=1, how="all")
    return None


def _csv_tables(
    file_bytes: bytes, table_name: str, sections: list[dict], data_profile: Optional[dict]
) -> list[dict]:
    """One ontology row per CSV section. Raw values only ever leave as hashes."""
    from api.utils.csv_preprocessor import dicts_to_sections

    raw = _read_csv_raw(file_bytes)
    if raw is None or raw.empty:
        return []
    dtypes = (data_profile or {}).get("dtypes") or {}
    parsed = dicts_to_sections(sections)

    out: list[dict] = []
    for sec in parsed:
        name = table_name
        if len(parsed) > 1:
            slug = re.sub(r"[^a-z0-9_]", "_", sec.name.strip().lower())[:40]
            name = f"{table_name}__{sec.index}_{slug}"
        data = raw.iloc[sec.data_start_idx:sec.data_end_idx]
        column_metadata = []
        for i, col in enumerate(sec.columns[:_MAX_COLS_PER_TABLE]):
            if not col or i >= raw.shape[1]:
                continue
            values = data.iloc[:, i]
            col_type = dtypes.get(col, "TEXT") if len(parsed) == 1 else "TEXT"
            column_metadata.append({
                "name": col,
                "type": col_type,
                "semantic_tag": _infer_semantic_tag(col, col_type),
                "null_pct": round(float(values.isna().mean()) * 100, 1) if len(values) else None,
                "value_hashes": fingerprint_values(values.tolist()),
            })
        out.append({
            "table_name": name,
            "column_metadata": column_metadata,
            "data_card": {
                "column_count": len(sec.columns),
                "is_view": False,
                "row_count": sec.row_count,
                "example_rows": [],  # never raw values: this table is plaintext
            },
            "join_hints": [],
        })
    all_names = [t["table_name"] for t in out]
    for t in out:
        t["join_hints"] = _infer_join_hints(t["table_name"], t["column_metadata"], all_names)
    return out


async def build_csv_ontology(
    connection_id: str,
    file_bytes: bytes,
    table_name: str,
    sections: list[dict],
    data_profile: Optional[dict] = None,
) -> None:
    """Build the ontology for a CSV upload. Never raises (fire-and-forget)."""
    if not is_wired():
        return
    import asyncio

    try:
        per_table = await asyncio.get_running_loop().run_in_executor(
            None, _csv_tables, file_bytes, table_name, sections, data_profile
        )
        if per_table:
            await _embed_and_upsert(connection_id, per_table)
            logger.info(
                "ontology_service: built CSV ontology for connection=%s tables=%d",
                connection_id, len(per_table),
            )
    except Exception as exc:
        logger.warning("ontology_service: build_csv_ontology failed (non-fatal) for %s: %s", connection_id, exc)


# ---------------------------------------------------------------------------
# Org-wide read (DOCBOT-2911) — cross-source matching across an org's sources
# ---------------------------------------------------------------------------


async def list_org_ontology(org_id: str) -> list[dict]:
    """Every ontology table of every connection in *org_id*.

    Org scope comes from the join to ``db_connections.org_id`` (no copy on
    the ontology table). Only user-owned rows count: the DOCBOT-1524 backfill
    stamps the default org on anonymous rows, so ``org_id`` alone would leak
    anonymous/demo connections into the default org (same rule as
    ``db_service._api_key_org_clause``). ``org_id`` None/empty returns ``[]``.
    Never raises.
    """
    if not org_id or not is_wired() or _db_connections_table is None:
        return []
    o, c = _ontology_table, _db_connections_table
    try:
        async with _async_session_factory() as session:
            result = await session.execute(
                select(
                    o.c.connection_id, o.c.table_name, o.c.column_metadata_json, o.c.embedding,
                    c.c.dialect, c.c.dbname, c.c.host,
                )
                .select_from(o.join(c, o.c.connection_id == c.c.id))
                .where(c.c.org_id == org_id, c.c.user_id.is_not(None))
                .order_by(o.c.connection_id, o.c.table_name)
            )
            rows = result.fetchall()
    except Exception as exc:
        logger.warning("ontology_service: list_org_ontology failed (non-fatal): %s", exc)
        return []

    out: list[dict] = []
    for r in rows:
        cols = json.loads(r.column_metadata_json) if r.column_metadata_json else []
        out.append({
            "connection_id": r.connection_id,
            "dialect": r.dialect,
            # uploaded sqlite dbname is a temp path; never expose the directory
            "connection_name": Path(r.dbname).name if r.host == "__local_file__" else r.dbname,
            "table_name": r.table_name,
            "columns": [
                {
                    "name": col.get("name"),
                    "type": col.get("type"),
                    "semantic_tag": col.get("semantic_tag"),
                    "null_pct": col.get("null_pct"),
                    "value_hashes": col.get("value_hashes") or [],
                }
                for col in cols
            ],
            "embedding": json.loads(r.embedding) if isinstance(r.embedding, str) else r.embedding,
        })
    return out


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
