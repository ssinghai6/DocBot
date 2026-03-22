"""
DocBot Database Service — DOCBOT-201, 202, 203, 204, 205

Handles all database connectivity logic:
  - SSRF-safe connection validation (DOCBOT-201)
  - Schema introspection + 24-hour TTL cache (DOCBOT-202)
  - 7-step bounded SQL generation pipeline (DOCBOT-203)
  - MySQL connector support via pymysql (DOCBOT-205)

Routes are thin wrappers in api/index.py; all business logic lives here.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel, field_validator
from sqlalchemy import (
    Column, DateTime, Integer, MetaData, String, Table, Text,
    func, insert, select, text, update, delete,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from api.utils.ssrf_validator import validate_ssrf
from api.utils.encryption import encrypt_credentials, decrypt_credentials
from api.utils.sql_validator import validate_and_sanitize_sql, QueryValidationError
from api.utils.embeddings import find_similar_queries, embedding_to_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported dialects
# ---------------------------------------------------------------------------

SUPPORTED_DIALECTS = {"postgresql", "mysql", "sqlite"}

_DIALECT_DRIVER = {
    "postgresql": "postgresql+psycopg2",
    "mysql": "mysql+pymysql",
    "sqlite": "sqlite",
}

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ConnectionNotFoundError(Exception):
    """Raised when a connection_id does not exist or belongs to a different session."""


class ExecutionTimeoutError(Exception):
    """Raised when the SQL executor exceeds its time budget."""


# Re-export for routes
__all__ = [
    "DBConnectionRequest",
    "DBChatRequest",
    "DBChatResponse",
    "QueryValidationError",
    "ConnectionNotFoundError",
    "ExecutionTimeoutError",
    "connect_database",
    "disconnect_database",
    "get_schema",
    "run_sql_pipeline",
]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class DBConnectionRequest(BaseModel):
    session_id: str
    dialect: str
    host: str
    port: int
    dbname: str
    user: str
    password: str

    @field_validator("dialect")
    @classmethod
    def dialect_must_be_supported(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in SUPPORTED_DIALECTS:
            raise ValueError(
                f"Unsupported dialect '{v}'. Supported: {', '.join(sorted(SUPPORTED_DIALECTS))}"
            )
        return v

    @field_validator("host")
    @classmethod
    def host_must_pass_ssrf(cls, v: str) -> str:
        validate_ssrf(v)
        return v


class DBChatRequest(BaseModel):
    connection_id: str
    question: str
    persona: str = "Generalist"
    session_id: str


class DBChatResponse(BaseModel):
    answer: str
    sql_query: str
    explanation: str
    result_preview: List[Dict[str, Any]]
    row_count: int
    sources: List[str]
    execution_time_ms: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_connection_url(
    dialect: str, host: str, port: int, dbname: str, user: str, password: str
) -> str:
    driver = _DIALECT_DRIVER[dialect]
    if dialect == "sqlite":
        return f"sqlite:///{dbname}"
    return f"{driver}://{user}:{password}@{host}:{port}/{dbname}"


def _get_groq_client():
    from groq import Groq
    api_key = os.getenv("groq_api_key")
    if not api_key:
        raise RuntimeError("groq_api_key environment variable is not set.")
    return Groq(api_key=api_key)


def _get_embeddings_model():
    """Reuse the same HuggingFace embeddings singleton pattern used by the PDF pipeline."""
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
    hf_token = os.getenv("huggingface_api_key") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=hf_token,
    )


# ---------------------------------------------------------------------------
# DOCBOT-201: Connect / Disconnect
# ---------------------------------------------------------------------------


async def connect_database(
    req: DBConnectionRequest,
    db_connections_table: Table,
    schema_cache_table: Table,
    async_session_factory,
) -> Dict[str, Any]:
    """
    Validate credentials, encrypt, store, and return connection_id + schema summary.

    Steps:
      1. SSRF already validated by Pydantic field_validator on host.
      2. Test connection with a lightweight SELECT 1.
      3. Fernet-encrypt credentials.
      4. Insert into db_connections.
      5. Introspect schema → populate schema_cache with 24h TTL.
      6. Return connection_id + schema summary.
    """
    connection_id = str(uuid.uuid4())

    sync_url = _build_connection_url(
        req.dialect, req.host, req.port, req.dbname, req.user, req.password
    )
    await _test_connection(sync_url, req.dialect)

    creds_blob = encrypt_credentials({
        "dialect": req.dialect,
        "host": req.host,
        "port": req.port,
        "dbname": req.dbname,
        "user": req.user,
        "password": req.password,
    })

    async with async_session_factory() as session:
        async with session.begin():
            await session.execute(
                insert(db_connections_table).values(
                    id=connection_id,
                    session_id=req.session_id,
                    dialect=req.dialect,
                    host=req.host,
                    port=req.port,
                    dbname=req.dbname,
                    credentials_blob=creds_blob,
                )
            )

    schema = await _introspect_schema_from_url(sync_url, req.dialect)
    schema_json = json.dumps(schema)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=24)

    async with async_session_factory() as session:
        async with session.begin():
            await session.execute(
                insert(schema_cache_table).values(
                    connection_id=connection_id,
                    schema_json=schema_json,
                    expires_at=expires_at,
                )
            )

    table_names = [t["name"] for t in schema]
    return {
        "connection_id": connection_id,
        "schema_summary": {
            "table_count": len(table_names),
            "tables": table_names[:10],
        },
    }


async def disconnect_database(
    connection_id: str,
    session_id: str,
    db_connections_table: Table,
    schema_cache_table: Table,
    async_session_factory,
) -> None:
    """Remove the connection record and invalidate its schema cache."""
    async with async_session_factory() as session:
        async with session.begin():
            result = await session.execute(
                select(db_connections_table).where(
                    db_connections_table.c.id == connection_id,
                    db_connections_table.c.session_id == session_id,
                )
            )
            row = result.fetchone()
            if not row:
                raise ConnectionNotFoundError(f"Connection '{connection_id}' not found.")

            await session.execute(
                delete(schema_cache_table).where(
                    schema_cache_table.c.connection_id == connection_id
                )
            )
            await session.execute(
                delete(db_connections_table).where(
                    db_connections_table.c.id == connection_id
                )
            )


# ---------------------------------------------------------------------------
# DOCBOT-202: Schema introspection + cache
# ---------------------------------------------------------------------------


async def get_schema(
    connection_id: str,
    db_connections_table: Table,
    schema_cache_table: Table,
    async_session_factory,
) -> List[Dict[str, Any]]:
    """
    Return cached schema or re-introspect if cache has expired (24h TTL).
    """
    async with async_session_factory() as session:
        cache_result = await session.execute(
            select(schema_cache_table).where(
                schema_cache_table.c.connection_id == connection_id
            )
        )
        cache_row = cache_result.fetchone()

        now = datetime.now(timezone.utc)
        if cache_row:
            expires = cache_row.expires_at
            # Make timezone-aware if naive
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if expires > now:
                return json.loads(cache_row.schema_json)

        conn_result = await session.execute(
            select(db_connections_table).where(
                db_connections_table.c.id == connection_id
            )
        )
        conn_row = conn_result.fetchone()

    if not conn_row:
        raise ConnectionNotFoundError(f"Connection '{connection_id}' not found.")

    creds = decrypt_credentials(conn_row.credentials_blob)
    sync_url = _build_connection_url(
        creds["dialect"], creds["host"], creds["port"],
        creds["dbname"], creds["user"], creds["password"],
    )
    schema = await _introspect_schema_from_url(sync_url, creds["dialect"])
    schema_json = json.dumps(schema)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=24)

    async with async_session_factory() as session:
        async with session.begin():
            if cache_row:
                await session.execute(
                    update(schema_cache_table)
                    .where(schema_cache_table.c.connection_id == connection_id)
                    .values(schema_json=schema_json, expires_at=expires_at)
                )
            else:
                await session.execute(
                    insert(schema_cache_table).values(
                        connection_id=connection_id,
                        schema_json=schema_json,
                        expires_at=expires_at,
                    )
                )

    return schema


async def _introspect_schema_from_url(sync_url: str, dialect: str) -> List[Dict[str, Any]]:
    """
    Use SQLAlchemy Inspector to return:
      [{"name": "table", "columns": [{"name": "col", "type": "VARCHAR"}]}]

    Capped at 50 tables. PostgreSQL sorts by row estimate.
    """
    import asyncio
    from sqlalchemy import create_engine, inspect as sa_inspect

    def _sync_introspect() -> List[Dict[str, Any]]:
        connect_args: Dict[str, Any] = {}
        if dialect != "sqlite":
            connect_args["connect_timeout"] = 10

        engine = create_engine(sync_url, connect_args=connect_args)
        try:
            inspector = sa_inspect(engine)
            table_names = inspector.get_table_names()

            if dialect == "postgresql" and len(table_names) > 50:
                table_names = _sort_tables_by_row_estimate_pg(engine, table_names)

            table_names = table_names[:50]
            schema = []
            for tname in table_names:
                columns = inspector.get_columns(tname)
                schema.append({
                    "name": tname,
                    "columns": [
                        {"name": col["name"], "type": str(col["type"])}
                        for col in columns
                    ],
                })
            return schema
        finally:
            engine.dispose()

    return await asyncio.get_event_loop().run_in_executor(None, _sync_introspect)


def _sort_tables_by_row_estimate_pg(engine, table_names: List[str]) -> List[str]:
    """Return table names sorted descending by PostgreSQL row estimate."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT relname, reltuples::bigint AS estimate "
                "FROM pg_class WHERE relkind = 'r' ORDER BY reltuples DESC"
            ))
            name_set = set(table_names)
            ordered = [row[0] for row in result if row[0] in name_set]
            remaining = [t for t in table_names if t not in set(ordered)]
            return ordered + remaining
    except Exception:
        return table_names


async def _test_connection(sync_url: str, dialect: str) -> None:
    """Run SELECT 1 to verify credentials. Sanitises any exception before re-raising."""
    import asyncio
    from sqlalchemy import create_engine

    def _sync_test() -> None:
        connect_args: Dict[str, Any] = {}
        if dialect != "sqlite":
            connect_args["connect_timeout"] = 10

        engine = create_engine(sync_url, connect_args=connect_args)
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        finally:
            engine.dispose()

    try:
        await asyncio.get_event_loop().run_in_executor(None, _sync_test)
    except Exception as exc:
        raise ValueError(
            "Database connection test failed. Please verify your credentials and host."
        ) from exc


# ---------------------------------------------------------------------------
# DOCBOT-203: 7-step SQL generation pipeline
# ---------------------------------------------------------------------------


async def run_sql_pipeline(
    connection_id: str,
    question: str,
    persona: str,
    db_connections_table: Table,
    schema_cache_table: Table,
    query_history_table: Table,
    query_embeddings_table: Table,
    async_session_factory,
    expert_personas: Dict[str, Any],
) -> AsyncGenerator[str, None]:
    """
    Execute the 7-step bounded SQL pipeline and yield SSE-formatted strings.

    Yield order:
      1. One metadata chunk (JSON) with sql, explanation, preview, row_count, execution_time_ms
      2. N token chunks (streaming answer)
      3. One done chunk
    """
    start_ms = int(time.time() * 1000)

    # ── Step 1: Schema retrieval ──────────────────────────────────────────
    schema = await get_schema(
        connection_id, db_connections_table, schema_cache_table, async_session_factory
    )

    # ── Step 2: Table selector (LLM call #1) ─────────────────────────────
    selected_tables = await _select_relevant_tables(question, schema)
    schema_subset = [t for t in schema if t["name"] in set(selected_tables)]
    if not schema_subset:
        schema_subset = schema[:10]

    # ── Step 3: Few-shot retrieval ────────────────────────────────────────
    embeddings_model = _get_embeddings_model()
    q_embedding: List[float] = await _get_embedding(question, embeddings_model)
    few_shot_examples = await _retrieve_few_shot(
        connection_id, q_embedding,
        query_history_table, query_embeddings_table, async_session_factory
    )

    # ── Step 4: SQL generation (LLM call #2) ─────────────────────────────
    raw_sql = await _generate_sql(question, schema_subset, few_shot_examples)

    # ── Step 5: SQL validation (deterministic, no LLM) ───────────────────
    async with async_session_factory() as session:
        conn_result = await session.execute(
            select(db_connections_table).where(db_connections_table.c.id == connection_id)
        )
        conn_row = conn_result.fetchone()

    if not conn_row:
        raise ConnectionNotFoundError(f"Connection '{connection_id}' not found.")

    dialect = conn_row.dialect
    validated_sql = validate_and_sanitize_sql(raw_sql, dialect=dialect)

    # ── Step 6: Execute ───────────────────────────────────────────────────
    creds = decrypt_credentials(conn_row.credentials_blob)
    sync_url = _build_connection_url(
        creds["dialect"], creds["host"], creds["port"],
        creds["dbname"], creds["user"], creds["password"],
    )
    rows, column_names = await _execute_query(validated_sql, sync_url, dialect)
    execution_time_ms = int(time.time() * 1000) - start_ms

    result_dicts = [dict(zip(column_names, row)) for row in rows]

    # ── Step 7: Answer generation (LLM call #3, streaming) ───────────────
    persona_def = (
        expert_personas.get(persona, expert_personas.get("Generalist", {}))
        .get("persona_def", "You are a helpful data analyst.")
    )
    explanation = _build_explanation(validated_sql, schema_subset)

    # First chunk: metadata
    meta_chunk = {
        "type": "metadata",
        "sql_query": validated_sql,
        "explanation": explanation,
        "result_preview": result_dicts[:10],
        "row_count": len(rows),
        "execution_time_ms": execution_time_ms,
        "sources": [t["name"] for t in schema_subset],
    }
    yield f"data: {json.dumps(meta_chunk)}\n\n"

    # Persist query + embedding (non-blocking; failure is only a warning)
    query_id = str(uuid.uuid4())
    await _store_query_history(
        query_id, connection_id, question, validated_sql,
        f"{len(rows)} rows", q_embedding,
        query_history_table, query_embeddings_table, async_session_factory
    )

    # Stream answer tokens
    async for token in _stream_answer(question, validated_sql, result_dicts, persona_def):
        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# ---------------------------------------------------------------------------
# Step 2 helper — table selector
# ---------------------------------------------------------------------------


async def _select_relevant_tables(question: str, schema: List[Dict[str, Any]]) -> List[str]:
    """LLM call #1: pick the relevant tables for this question."""
    table_list = "\n".join(
        f"- {t['name']}: {', '.join(c['name'] for c in t['columns'][:10])}"
        for t in schema
    )
    prompt = (
        "You are a SQL expert. Given the following database schema and a user question, "
        "return ONLY a JSON array of the table names needed to answer the question. "
        "Return at most 5 tables. No explanation.\n\n"
        f"Schema:\n{table_list}\n\n"
        f"Question: {question}\n\n"
        "Response (JSON array only):"
    )
    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
    except Exception as exc:
        logger.warning("Table selector LLM call failed: %s", exc)

    return [t["name"] for t in schema[:10]]


# ---------------------------------------------------------------------------
# Step 3 helper — few-shot retrieval
# ---------------------------------------------------------------------------


async def _retrieve_few_shot(
    connection_id: str,
    query_embedding: List[float],
    query_history_table: Table,
    query_embeddings_table: Table,
    async_session_factory,
) -> List[Dict[str, Any]]:
    """Return up to 3 similar past queries for this connection."""
    try:
        async with async_session_factory() as session:
            result = await session.execute(
                select(
                    query_history_table.c.nl_question,
                    query_history_table.c.sql_query,
                    query_embeddings_table.c.embedding_json,
                )
                .join(
                    query_embeddings_table,
                    query_history_table.c.id == query_embeddings_table.c.query_id,
                )
                .where(query_history_table.c.connection_id == connection_id)
                .limit(100)
            )
            rows = result.fetchall()

        stored = [
            {
                "nl_question": row.nl_question,
                "sql_query": row.sql_query,
                "embedding_json": row.embedding_json,
            }
            for row in rows
        ]
        return find_similar_queries(query_embedding, stored, top_k=3)
    except Exception as exc:
        logger.warning("Few-shot retrieval failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Step 4 helper — SQL generation
# ---------------------------------------------------------------------------


async def _generate_sql(
    question: str,
    schema_subset: List[Dict[str, Any]],
    few_shot_examples: List[Dict[str, Any]],
) -> str:
    """LLM call #2: generate a SELECT query given schema + few-shot examples."""
    def _fmt_table(t: Dict[str, Any]) -> str:
        cols = ", ".join(f"{c['name']} ({c['type']})" for c in t["columns"])
        return f"Table: {t['name']}\nColumns: {cols}"

    schema_text = "\n\n".join(_fmt_table(t) for t in schema_subset)

    few_shot_text = ""
    if few_shot_examples:
        examples_str = "\n\n".join(
            f"Q: {ex['nl_question']}\nSQL: {ex['sql_query']}"
            for ex in few_shot_examples
        )
        few_shot_text = f"\n\nSimilar past queries for reference:\n{examples_str}\n"

    prompt = (
        "You are an expert SQL generator. Generate a single valid SELECT query to answer the question.\n"
        "RULES:\n"
        "1. Read-only queries ONLY — no INSERT, UPDATE, DELETE, DROP, CREATE, or any DDL/DML.\n"
        "2. Use only the tables and columns listed in the schema.\n"
        "3. Include a LIMIT clause (max 500 rows).\n"
        "4. Return ONLY the SQL query — no explanation, no markdown fences.\n\n"
        f"Schema:\n{schema_text}"
        f"{few_shot_text}\n\n"
        f"Question: {question}\n\n"
        "SQL:"
    )

    client = _get_groq_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
    )
    sql = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if sql.startswith("```"):
        lines = sql.split("\n")
        sql = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    return sql


# ---------------------------------------------------------------------------
# Step 6 helper — query executor (DOCBOT-205: MySQL read-only enforcement)
# ---------------------------------------------------------------------------


async def _execute_query(
    sql: str, sync_url: str, dialect: str
):
    """
    Execute *sql* with read-only enforcement, 15s timeout, 500-row cap.

    Layer 3 of the 3-layer read-only enforcement:
      - PostgreSQL: SET TRANSACTION READ ONLY + statement_timeout = 15000ms
      - MySQL:      SET SESSION TRANSACTION READ ONLY  (DOCBOT-205)
      - SQLite:     no write ops possible via SELECT-only validated SQL

    Returns (rows, column_names).
    """
    import asyncio
    from sqlalchemy import create_engine

    def _sync_execute():
        connect_args: Dict[str, Any] = {}
        if dialect != "sqlite":
            connect_args["connect_timeout"] = 15

        engine = create_engine(sync_url, connect_args=connect_args)
        try:
            with engine.connect() as conn:
                if dialect == "postgresql":
                    conn.execute(text("SET TRANSACTION READ ONLY"))
                    conn.execute(text("SET LOCAL statement_timeout = '15000'"))
                elif dialect == "mysql":
                    conn.execute(text("SET SESSION TRANSACTION READ ONLY"))

                result = conn.execute(text(sql))
                rows = result.fetchmany(500)
                column_names = list(result.keys())
                return rows, column_names
        finally:
            engine.dispose()

    try:
        return await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _sync_execute),
            timeout=15.0,
        )
    except asyncio.TimeoutError as exc:
        raise ExecutionTimeoutError("Query exceeded the 15-second time limit.") from exc


# ---------------------------------------------------------------------------
# Step 7 helper — streaming answer generator
# ---------------------------------------------------------------------------


async def _stream_answer(
    question: str,
    sql: str,
    result_dicts: List[Dict[str, Any]],
    persona_def: str,
) -> AsyncGenerator[str, None]:
    """LLM call #3: stream a plain-English answer in persona voice."""
    preview = json.dumps(result_dicts[:20], default=str, indent=2)
    prompt = (
        f"{persona_def}\n\n"
        "A user asked a question about their database. Answer using ONLY the query result below.\n"
        "Rules:\n"
        "- Do not reveal the raw SQL unless explicitly asked.\n"
        "- Cite the data source as [DB: <table names>] at the end.\n"
        "- Be concise and use the persona's voice.\n\n"
        f"Question: {question}\n\n"
        f"Query result ({len(result_dicts)} rows):\n{preview}\n\n"
        "Answer:"
    )

    client = _get_groq_client()
    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=800,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------


async def _get_embedding(text_input: str, embeddings_model) -> List[float]:
    """Embed a single string into a float list via the HuggingFace endpoint."""
    import asyncio

    def _sync_embed() -> List[float]:
        return embeddings_model.embed_query(text_input)

    return await asyncio.get_event_loop().run_in_executor(None, _sync_embed)


# ---------------------------------------------------------------------------
# Query history persistence
# ---------------------------------------------------------------------------


async def _store_query_history(
    query_id: str,
    connection_id: str,
    nl_question: str,
    sql_query: str,
    result_summary: str,
    embedding: List[float],
    query_history_table: Table,
    query_embeddings_table: Table,
    async_session_factory,
) -> None:
    """Persist a successful query and its NL embedding for future few-shot retrieval."""
    try:
        async with async_session_factory() as session:
            async with session.begin():
                await session.execute(
                    insert(query_history_table).values(
                        id=query_id,
                        connection_id=connection_id,
                        nl_question=nl_question,
                        sql_query=sql_query,
                        result_summary=result_summary,
                    )
                )
                await session.execute(
                    insert(query_embeddings_table).values(
                        query_id=query_id,
                        embedding_json=embedding_to_json(embedding),
                    )
                )
    except Exception as exc:
        logger.warning("Failed to store query history: %s", exc)


# ---------------------------------------------------------------------------
# Explanation builder (deterministic, no LLM)
# ---------------------------------------------------------------------------


def _build_explanation(sql: str, schema_subset: List[Dict[str, Any]]) -> str:
    """Build a brief plain-English description of what the SQL does."""
    table_names = [t["name"] for t in schema_subset]
    tables_str = ", ".join(table_names) if table_names else "the database"
    sql_upper = sql.upper()

    parts = []
    if "GROUP BY" in sql_upper:
        parts.append("groups results")
    if "ORDER BY" in sql_upper:
        parts.append("orders by a column")
    if "JOIN" in sql_upper:
        parts.append("joins multiple tables")
    if "WHERE" in sql_upper:
        parts.append("filters rows by a condition")
    if any(agg in sql_upper for agg in ("COUNT(", "SUM(", "AVG(", "MAX(", "MIN(")):
        parts.append("computes aggregates")

    action = (", ".join(parts) + " from") if parts else "queries"
    return f"This query {action} {tables_str}."
