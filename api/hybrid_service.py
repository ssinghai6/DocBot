"""
DocBot Hybrid Service — DOCBOT-401

Query Intent Classifier: determines whether a question should be answered from
the document store ("doc"), the connected database ("sql"), or both ("hybrid").

Fallback rules apply when only one source is available (no LLM call needed).
When both sources are present the decision is delegated to the LLM and the
result is logged to query_history for observability.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime
from typing import AsyncGenerator, Literal

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


# ---------------------------------------------------------------------------
# DOCBOT-402: RAG retrieval helper
# ---------------------------------------------------------------------------


async def rag_retrieve(
    question: str,
    session_id: str,
    vector_stores: dict,
) -> tuple[str, list[dict]]:
    """Retrieve document context for a question via vector similarity search.

    Uses multi-query expansion to improve recall for short or ambiguous
    questions (e.g. "His position or title?" will also search for
    "job title", "SOC occupation title", etc. so structured form labels
    are found even when the user's phrasing is semantically distant).

    Returns (context_text, citations_list).
    Returns ("", []) when the session has no documents or on any error.
    """
    if session_id not in vector_stores:
        return ("", [])

    try:
        from api.utils.query_expansion import expand_query, deduplicate_docs

        db = vector_stores[session_id]
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8},
        )

        # Expand the question into synonym variants and retrieve for each;
        # merge results so each unique chunk appears at most once.
        expanded_queries = expand_query(question)
        all_result_lists = [retriever.invoke(q) for q in expanded_queries]
        docs = deduplicate_docs(all_result_lists)

        context = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}, "
            f"Page {doc.metadata.get('page', 0)}\n{doc.page_content}"
            for doc in docs
        )

        seen: set[str] = set()
        citations: list[dict] = []
        for doc in docs:
            key = f"{doc.metadata.get('source', 'Unknown')}_{doc.metadata.get('page', 0)}"
            if key not in seen:
                seen.add(key)
                citations.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", 0),
                })

        return (context, citations)

    except Exception as exc:
        logger.warning("rag_retrieve failed: %s", exc)
        return ("", [])


# ---------------------------------------------------------------------------
# DOCBOT-402: SQL result collector (non-streaming, for parallel gather)
# ---------------------------------------------------------------------------


async def _collect_sql_result(
    connection_id: str,
    question: str,
    persona: str,
    db_connections_table,
    schema_cache_table,
    query_history_table,
    query_embeddings_table,
    async_session_factory,
    expert_personas: dict,
) -> dict | None:
    """Run the SQL pipeline and return the metadata dict, or None on failure.

    Drains run_sql_pipeline() until the first metadata chunk, then returns
    the parsed dict so hybrid_chat() can incorporate results into synthesis.
    """
    from api.db_service import run_sql_pipeline

    try:
        metadata_result: dict | None = None
        answer_tokens: list[str] = []

        async for raw_chunk in run_sql_pipeline(
            connection_id=connection_id,
            question=question,
            persona=persona,
            db_connections_table=db_connections_table,
            schema_cache_table=schema_cache_table,
            query_history_table=query_history_table,
            query_embeddings_table=query_embeddings_table,
            async_session_factory=async_session_factory,
            expert_personas=expert_personas,
        ):
            if not raw_chunk.startswith("data: "):
                continue
            data = json.loads(raw_chunk[6:].strip())
            if data.get("type") == "metadata":
                metadata_result = data
            elif data.get("type") == "token":
                answer_tokens.append(data.get("content", ""))

        # For CSV connections, metadata has empty result_preview.
        # Inject the streamed answer tokens as the result text so
        # hybrid synthesis has actual data to work with.
        if metadata_result is not None and answer_tokens:
            answer_text = "".join(answer_tokens).strip()
            if not metadata_result.get("result_preview") and answer_text:
                metadata_result["csv_answer"] = answer_text[:3000]

        return metadata_result
    except Exception as exc:
        logger.warning("_collect_sql_result failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# DOCBOT-402: Hybrid chat async generator
# ---------------------------------------------------------------------------


async def hybrid_chat(
    question: str,
    session_id: str,
    connection_id: str | None,
    persona: str,
    has_docs: bool,
    messages_table,
    sessions_table,
    db_connections_table,
    schema_cache_table,
    query_history_table,
    query_embeddings_table,
    async_session_factory,
    expert_personas: dict,
    vector_stores: dict,
    extracted_fields: list | None = None,
    deep_research: bool = False,
) -> AsyncGenerator[str, None]:
    """Hybrid chat pipeline: intent classification → SQL + RAG → synthesis.

    Yields SSE-formatted strings with the same chunk contract as /api/db/chat:
      {"type": "metadata", "intent": ..., "has_sql": ..., "has_docs": ...}
      {"type": "token",    "content": "..."}  (N times)
      {"type": "done"}
    """
    has_db = connection_id is not None

    # ── Step 1: classify intent ───────────────────────────────────────────
    groq_api_key = os.getenv("groq_api_key")
    if not groq_api_key:
        yield f"data: {json.dumps({'type': 'error', 'detail': 'groq_api_key not configured'})}\n\n"
        return

    import groq as groq_module
    groq_client = groq_module.Groq(api_key=groq_api_key)

    classification = await classify_intent_safe(
        question=question,
        has_db=has_db,
        has_docs=has_docs,
        session_id=session_id,
        groq_client=groq_client,
        async_session_factory=async_session_factory,
        query_history_table=query_history_table,
    )
    intent = classification.intent

    yield f"data: {json.dumps({'type': 'metadata', 'intent': intent, 'has_sql': has_db, 'has_docs': has_docs})}\n\n"

    # ── Step 2: gather context based on intent ────────────────────────────
    doc_context = ""
    doc_citations: list[dict] = []
    sql_metadata: dict | None = None

    if intent == "doc":
        doc_context, doc_citations = await rag_retrieve(question, session_id, vector_stores)

    elif intent == "sql" and has_db and connection_id:
        sql_metadata = await _collect_sql_result(
            connection_id=connection_id,
            question=question,
            persona=persona,
            db_connections_table=db_connections_table,
            schema_cache_table=schema_cache_table,
            query_history_table=query_history_table,
            query_embeddings_table=query_embeddings_table,
            async_session_factory=async_session_factory,
            expert_personas=expert_personas,
        )

    else:  # hybrid — parallel gather
        rag_task = asyncio.create_task(
            rag_retrieve(question, session_id, vector_stores)
        )
        sql_task = asyncio.create_task(
            _collect_sql_result(
                connection_id=connection_id,
                question=question,
                persona=persona,
                db_connections_table=db_connections_table,
                schema_cache_table=schema_cache_table,
                query_history_table=query_history_table,
                query_embeddings_table=query_embeddings_table,
                async_session_factory=async_session_factory,
                expert_personas=expert_personas,
            )
        ) if has_db and connection_id else None

        doc_context, doc_citations = await rag_task

        if sql_task is not None:
            sql_metadata = await sql_task

    # ── Step 3: build synthesis prompt ───────────────────────────────────
    persona_def = (
        expert_personas.get(persona, expert_personas.get("Generalist", {}))
        .get("persona_def", "You are a helpful data analyst.")
    )

    if deep_research:
        from api.index import DEEP_RESEARCH_ADDON
        persona_def = persona_def + DEEP_RESEARCH_ADDON

    doc_note = ""
    sql_note = ""

    if intent == "hybrid":
        if not doc_context:
            doc_note = "\n*(No document context available — answering from database only)*"
        if sql_metadata is None:
            sql_note = "\n*(SQL query failed — supplementing with document context only)*"

    sql_section = ""
    if sql_metadata:
        # CSV connections provide answer text in csv_answer; SQL connections
        # provide structured result_preview rows.
        csv_answer = sql_metadata.get("csv_answer")
        if csv_answer:
            sources = sql_metadata.get("sources", [])
            source_str = ", ".join(sources) if sources else "CSV file"
            sql_section = (
                f"\n\nData analysis results from [{source_str}]:\n{csv_answer}\n"
                "Cite data results as [DB: csv_file]."
            )
        else:
            preview = json.dumps(sql_metadata.get("result_preview", [])[:20], default=str, indent=2)
            row_count = sql_metadata.get("row_count", 0)
            sources = sql_metadata.get("sources", [])
            source_str = ", ".join(sources) if sources else "database"
            sql_section = (
                f"\n\nDatabase results ({row_count} rows) from [{source_str}]:\n{preview}\n"
                "Cite database results as [DB: table_name]."
            )

    doc_section = ""
    if doc_context:
        doc_section = (
            f"\n\nDocument context:\n{doc_context}\n"
            "Cite document sources as [Source: filename, Page X]."
        )

    # DOCBOT-406: include span-verified financial values when available
    extracted_section = ""
    if extracted_fields:
        from api.document_extractor import format_extracted_fields_for_prompt
        formatted = format_extracted_fields_for_prompt(extracted_fields)
        if formatted:
            extracted_section = f"\n\n{formatted}"

    # DOCBOT-403: discrepancy detection instructions (only in hybrid mode)
    discrepancy_instruction = ""
    if intent == "hybrid" and doc_context and sql_metadata:
        discrepancy_instruction = (
            "\n\nDISCREPANCY DETECTION: When the document and database results contain "
            "conflicting numeric values for the same metric:\n"
            "- Flag it explicitly with a [DISCREPANCY] marker\n"
            "- Format: \"[DISCREPANCY] Doc says {doc_value} [Source: ...]. "
            "DB shows {db_value} [DB: table]. Delta: {delta} ({pct}%)\"\n"
            "- Also flag non-numeric inconsistencies (e.g. different time periods, "
            "different categories)\n"
            "- Do NOT say \"no discrepancy found\" unless the user explicitly asks\n"
            "- If sources agree, simply synthesize without mentioning discrepancy"
        )

    prompt = (
        f"{persona_def}\n\n"
        f"Answer the following question using the context below. "
        f"Be concise and accurate.{doc_note}{sql_note}"
        f"{discrepancy_instruction}\n\n"
        "RETRIEVAL ACCURACY RULES:\n"
        "- Read EVERY chunk in the document context carefully before concluding any "
        "field is absent.\n"
        "- Structured forms (government forms, legal filings) store fields as labelled "
        "rows such as 'Job Title: X' or 'SOC Code: Y'. If ANY chunk contains a relevant "
        "field value you MUST report it — do not say the value is missing.\n"
        "- If the question asks about a person's role, title, position, or occupation, "
        "look for any of: Job Title, Position, Role, Designation, SOC Occupation Title, "
        "Occupation Title.\n"
        "- Only say information is absent when you have examined all chunks and confirmed "
        "it does not appear anywhere.\n\n"
        f"Question: {question}"
        f"{extracted_section}"
        f"\n\n[DOCUMENT CONTEXT]\n{doc_section}"
        f"\n\n[DATABASE RESULTS]\n{sql_section}\n\n"
        "Answer:"
    )

    # ── Step 4: stream Groq synthesis ────────────────────────────────────
    try:
        stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield f"data: {json.dumps({'type': 'token', 'content': delta.content})}\n\n"
    except Exception as exc:
        logger.error("hybrid_chat synthesis failed: %s", exc)
        yield f"data: {json.dumps({'type': 'error', 'detail': 'Synthesis failed. Please try again.'})}\n\n"
        return

    yield f"data: {json.dumps({'type': 'done', 'citations': doc_citations})}\n\n"
