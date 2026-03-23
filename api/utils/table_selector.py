"""
DOCBOT-503: Schema-Aware Semantic Table Selection

Embeds each table's name + column list once (on schema introspection) and stores
the vectors in the `table_embeddings` PostgreSQL table.  At query time, the user
question is embedded and the top-k most similar tables are returned — replacing
the naive "send all tables to the LLM" approach.

Public API
----------
build_schema_summary(table: dict) -> str
    Build a compact text representation of one schema table.

upsert_table_embeddings(connection_id, schema, embeddings_model,
                         table_embeddings_table, async_session_factory) -> None
    Embed every table and upsert into the DB.

select_relevant_tables(question, connection_id, top_k,
                        embeddings_model, table_embeddings_table,
                        async_session_factory) -> list[str]
    Return top-k table names most similar to the question.
    Falls back to [] so the caller can use the LLM path as a fallback.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from sqlalchemy import Table, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import async_sessionmaker

from api.utils.embeddings import cosine_similarity

logger = logging.getLogger(__name__)


def build_schema_summary(table: dict) -> str:
    """Compact text representation of a single schema table for embedding.

    Format: "table_name: col1 (type), col2 (type), ..."
    Capped at 20 columns to keep the embedding input focused.
    """
    cols = table.get("columns", [])[:20]
    col_parts = ", ".join(
        f"{c['name']} ({c.get('type', 'unknown')})" for c in cols
    )
    return f"{table['name']}: {col_parts}" if col_parts else table["name"]


async def upsert_table_embeddings(
    connection_id: str,
    schema: List[dict],
    embeddings_model,
    table_embeddings_table: Table,
    async_session_factory: async_sessionmaker,
) -> None:
    """Embed each table's schema summary and upsert into table_embeddings.

    Called after successful schema introspection. Safe to call repeatedly —
    uses INSERT ... ON CONFLICT DO UPDATE so stale embeddings are refreshed
    when the schema changes.

    Silently no-ops on any error so schema introspection is never blocked.
    """
    if not schema:
        return

    import asyncio

    try:
        # Build summaries and embed in a single batch call for efficiency
        summaries = [build_schema_summary(t) for t in schema]
        table_names = [t["name"] for t in schema]

        def _batch_embed() -> List[List[float]]:
            return embeddings_model.embed_documents(summaries)

        loop = asyncio.get_event_loop()
        vectors: List[List[float]] = await loop.run_in_executor(None, _batch_embed)

        async with async_session_factory() as session:
            async with session.begin():
                for name, summary, vec in zip(table_names, summaries, vectors):
                    stmt = (
                        pg_insert(table_embeddings_table)
                        .values(
                            connection_id=connection_id,
                            table_name=name,
                            embedding=json.dumps(vec),
                            schema_summary=summary,
                        )
                        .on_conflict_do_update(
                            index_elements=["connection_id", "table_name"],
                            set_={
                                "embedding": json.dumps(vec),
                                "schema_summary": summary,
                            },
                        )
                    )
                    await session.execute(stmt)

        logger.info(
            "upsert_table_embeddings: connection=%s tables=%d",
            connection_id,
            len(schema),
        )
    except Exception as exc:
        logger.warning("upsert_table_embeddings failed (non-fatal): %s", exc)


async def select_relevant_tables(
    question: str,
    connection_id: str,
    embeddings_model,
    table_embeddings_table: Table,
    async_session_factory: async_sessionmaker,
    top_k: int = 5,
) -> List[str]:
    """Return the top-k table names most semantically similar to the question.

    Returns an empty list if no embeddings are stored yet (callers should fall
    back to the LLM-based table selection path in that case).

    Parameters
    ----------
    question               : Natural language question from the user
    connection_id          : DB connection whose embeddings to search
    embeddings_model       : HuggingFace embeddings singleton
    table_embeddings_table : SQLAlchemy Table object
    async_session_factory  : Async session factory
    top_k                  : Number of tables to return (default 5)
    """
    import asyncio

    try:
        # Step 1: Fetch stored embeddings for this connection
        async with async_session_factory() as session:
            result = await session.execute(
                select(
                    table_embeddings_table.c.table_name,
                    table_embeddings_table.c.embedding,
                ).where(table_embeddings_table.c.connection_id == connection_id)
            )
            rows = result.fetchall()

        if not rows:
            return []  # no embeddings stored yet — let caller use LLM path

        # Step 2: Embed the question
        def _embed_q() -> List[float]:
            return embeddings_model.embed_query(question)

        loop = asyncio.get_event_loop()
        q_vec: List[float] = await loop.run_in_executor(None, _embed_q)

        # Step 3: Rank by cosine similarity
        scored = []
        for row in rows:
            vec = json.loads(row.embedding) if isinstance(row.embedding, str) else row.embedding
            score = cosine_similarity(q_vec, vec)
            scored.append((score, row.table_name))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [name for _, name in scored[:top_k]]

    except Exception as exc:
        logger.warning("select_relevant_tables failed (non-fatal): %s", exc)
        return []
