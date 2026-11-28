"""
DocBot Hybrid Service — DOCBOT-401

Query Intent Classifier: determines whether a question should be answered from
the document store ("doc"), the connected database ("sql"), or both ("hybrid").

Fallback rules apply when only one source is available (no LLM call needed).
When both sources are present the decision is delegated to the LLM and the
result is logged to query_history for observability.
"""

from __future__ import annotations

from api.utils.doc_refs import chunk_header, chunk_key, chunk_label, chunk_page  # DOCBOT-2637
from api.utils.yoy_facts import YOY_RULE, yoy_contradictions, yoy_facts, yoy_prompt_block  # DOCBOT-2637
import asyncio
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import AsyncGenerator, Literal

from pydantic import BaseModel
from sqlalchemy import insert

from api.utils.llm_provider import GROQ_MODEL as _MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_INTENTS = {"sql", "doc", "hybrid"}
# _MODEL imported from llm_provider.GROQ_MODEL at top of file — DOCBOT-1403:
# this used to be a hardcoded duplicate literal, and only one of the two
# copies got updated when Groq decommissioned "llama-3.3-70b-versatile",
# leaving classify_intent() silently 404ing while the rest of the app was
# already fixed.
_SYSTEM_PROMPT = (
    "You are a query routing assistant. "
    "Classify the user's question into exactly one of three categories:\n"
    "  sql    — the answer requires querying a live database\n"
    "  doc    — the answer can be found in uploaded documents\n"
    "  hybrid — the answer requires both database data and document context\n\n"
    "Respond with one word only: sql, doc, or hybrid. No punctuation, no explanation."
)
# DOCBOT-1507: bump manually ("v1" -> "v2") whenever _SYSTEM_PROMPT's wording
# changes — see PROMPT_VERSION convention in api/utils/llm_provider.py.
PROMPT_VERSION_INTENT_CLASSIFICATION = "v1"

# DOCBOT-1507: hybrid_chat's synthesis prompt is built dynamically (persona +
# doc/sql context), so there's no single string constant to tag — this
# version applies to the prompt-assembly logic in hybrid_chat() below.
PROMPT_VERSION_HYBRID_SYNTHESIS = "v3"  # DOCBOT-2723: empty books result rule (v2: 2637 scope + YoY)

# DOCBOT-2542: a question that asks whether the document and the books agree
# needs both lanes, or the discrepancy detector never runs. The LLM router
# (gpt-oss, a reasoning model: temperature 0 is not deterministic) can send
# "Does segment revenue in Note 12 agree with the books?" to "doc" (Note 12)
# or "sql" (the books), and a 6h exact-match cache then pins that miss, so
# route these deterministically.
_TIEOUT_QUESTION_RE = re.compile(
    r"\b(?:(?:dis)?agree[sd]?|tie[sd]?|tie-?outs?|match(?:es|ing)?|mismatch\w*|reconcil\w*|"
    r"discrepan\w*|versus|vs)\b",
    re.IGNORECASE,
)

# DOCBOT-2335: hard cap on the DB lane (table select + SQL gen + execute) so a
# hung LLM call degrades to a doc-only answer with a notice instead of a ~180s stall.
SQL_LANE_TIMEOUT_S = 45.0
SQL_TIMEOUT_NOTICE = (
    f"> **Note:** The database query did not finish within {int(SQL_LANE_TIMEOUT_S)}s, "
    "so this answer does not include database results. Please try again.\n\n"
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

    if _TIEOUT_QUESTION_RE.search(question):
        result = "hybrid"
        if query_history_table is not None:
            await _log_classification(
                session_id, question_hash, result, query_history_table, async_session_factory
            )
        return result

    # ── LLM classification (both sources present) ─────────────────────────
    from api.utils.llm_provider import achat_completion
    from api.utils import llm_cache

    # DOCBOT-1506: exact-match cache — this call runs at temperature=0 with
    # a fixed system prompt, so (system prompt + question) fully determines
    # the response; a byte-identical prompt is a legitimate re-ask.
    cache_prompt_hash = llm_cache.hash_prompt(f"{_SYSTEM_PROMPT}\n{question}")
    raw = await llm_cache.get_cached_response(cache_prompt_hash, _MODEL)

    if raw is None:
        # DOCBOT-2307: async provider path (Groq → Gemini, off the event loop).
        # max_tokens 200: gpt-oss spends tokens reasoning; 10 could return empty text.
        raw = (await achat_completion(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            model=_MODEL,
            max_tokens=1000,
            temperature=0,
            caller="intent_classification",
            prompt_version=PROMPT_VERSION_INTENT_CLASSIFICATION,
        )).strip().lower()
        await llm_cache.set_cached_response(cache_prompt_hash, _MODEL, raw)
    else:
        raw = raw.strip().lower()

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


# PR5 telemetry: when both sources are connected, count how often the LLM
# classifier diverges from the rule-of-thumb "default to hybrid". If the
# divergence rate stays low after a sprint of real traffic, we can replace
# the LLM call with a deterministic rule (docs+db → hybrid; else single
# source) and save one LLM call per hybrid query.
_intent_telemetry: dict[str, int] = {
    "both_connected_calls": 0,
    "both_connected_hybrid": 0,
    "both_connected_diverged": 0,
    "fallback_applied": 0,
}


def get_intent_telemetry() -> dict[str, int]:
    """Return a snapshot of the in-process intent-classifier counters."""
    return dict(_intent_telemetry)


async def classify_intent_safe(
    question: str,
    has_db: bool,
    has_docs: bool,
    session_id: str,
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
    both_connected = has_db and has_docs

    if fallback_applied:
        _intent_telemetry["fallback_applied"] += 1

    try:
        intent = await classify_intent(
            question=question,
            has_db=has_db,
            has_docs=has_docs,
            session_id=session_id,
            async_session_factory=async_session_factory,
            query_history_table=query_history_table,
        )
        if both_connected:
            _intent_telemetry["both_connected_calls"] += 1
            if intent == "hybrid":
                _intent_telemetry["both_connected_hybrid"] += 1
            else:
                _intent_telemetry["both_connected_diverged"] += 1
                logger.info(
                    "classify_intent_safe: both sources connected but classifier "
                    "chose intent=%r (not 'hybrid') — counted for telemetry",
                    intent,
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
    lineage=None,
    k: int = 8,
) -> tuple[str, list[dict]]:
    """Retrieve document context for a question via vector similarity search.

    ``lineage`` (optional ``LineageCollector``, DOCBOT-1510) records the
    expanded queries and each returned chunk with its rerank score.

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
            search_kwargs={"k": k},
        )

        # Expand the question into synonym variants and retrieve for each;
        # merge results so each unique chunk appears at most once.
        expanded_queries = expand_query(question)
        if lineage is not None:
            lineage.set_expanded_queries(list(expanded_queries))
        all_result_lists = [retriever.invoke(q) for q in expanded_queries]
        docs = deduplicate_docs(all_result_lists)

        # DOCBOT-1002: re-rank with cross-encoder when HF key is available
        from api.utils.reranker import rerank_scored
        hf_key = os.getenv("huggingface_api_key", "")  # only used when RERANKER_PROVIDER=hf
        rerank_scores: dict[int, float | None] = {}
        if docs:
            scored = rerank_scored(question, docs, hf_key, top_k=5)
            docs = [d for d, _ in scored]
            rerank_scores = {id(d): sc for d, sc in scored}
        if lineage is not None:
            for doc in docs:
                lineage.add_source(
                    kind="pdf",
                    label=str(doc.metadata.get("source", "Unknown")),
                    page=chunk_page(doc.metadata),
                    chunk=doc.metadata.get("chunk_index"),
                    snippet=doc.page_content,
                    rerank_score=rerank_scores.get(id(doc)),
                )

        # DOCBOT-1527: heuristic injection-pattern scan over retrieved chunks
        # before they enter the prompt. Flag-and-continue — see
        # api/utils/prompt_guard.py docstring for why this never drops content.
        from api.utils.prompt_guard import scan_chunks

        scan_chunks(
            [
                (chunk_label(doc.metadata), doc.page_content)
                for doc in docs
            ],
            session_id=session_id,
        )

        context = "\n\n".join(
            f"{chunk_header(doc.metadata)}\n{doc.page_content}"
            for doc in docs
        )

        from api.utils.lineage import safe_snippet
        seen: set[str] = set()
        citations: list[dict] = []
        for doc in docs:
            key = chunk_key(doc.metadata)
            if key not in seen:
                seen.add(key)
                citations.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": chunk_page(doc.metadata),
                    "text": safe_snippet(doc.page_content),
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
    chart_type: str = "auto",
) -> dict | None:
    """Run the SQL pipeline and return the metadata dict, or None on failure.

    Drains run_sql_pipeline() until the first metadata chunk, then returns
    the parsed dict so hybrid_chat() can incorporate results into synthesis.
    """
    from api.db_service import ConnectionNotFoundError, run_sql_pipeline

    try:
        metadata_result: dict | None = None
        answer_tokens: list[str] = []
        chart_events: list[dict] = []
        analysis_code_event: dict | None = None

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
            chart_type=chart_type,
            include_tieout_rows=True,
        ):
            if not raw_chunk.startswith("data: "):
                continue
            data = json.loads(raw_chunk[6:].strip())
            evt_type = data.get("type")
            if evt_type == "metadata":
                metadata_result = data
            elif evt_type == "token":
                answer_tokens.append(data.get("content", ""))
            elif evt_type == "chart":
                chart_events.append(data)
            elif evt_type == "analysis_code":
                analysis_code_event = data

        # For CSV connections, metadata has empty result_preview.
        # Inject the streamed answer tokens as the result text so
        # hybrid synthesis has actual data to work with.
        if metadata_result is not None and answer_tokens:
            answer_text = "".join(answer_tokens).strip()
            if not metadata_result.get("result_preview") and answer_text:
                metadata_result["csv_answer"] = answer_text[:3000]

        # BUG #3 FIX: forward chart/analysis_code events back to hybrid_chat()
        # so visualizations aren't silently dropped in hybrid mode.
        if metadata_result is None and (chart_events or analysis_code_event):
            metadata_result = {}
        if metadata_result is not None:
            metadata_result["chart_events"] = chart_events
            metadata_result["analysis_code_event"] = analysis_code_event

        return metadata_result
    except ConnectionNotFoundError:
        raise  # DOCBOT-2733: "please re-upload" must reach the user, not a books-less answer
    except Exception as exc:
        logger.warning("_collect_sql_result failed: %s", exc)
        return None


BOOK_GUIDED_MAX_CHUNKS = 3
# DOCBOT-2639: a number-dense statement table embeds poorly (Chipotle's income
# statement ranks ~17th), but the reranker scores it first; search deeper so it reaches it.
BOOK_GUIDED_SEARCH_K = 40


async def _book_guided_retrieval(
    doc_context: str,
    doc_citations: list[dict],
    sql_metadata: dict | None,
    session_id: str,
    vector_stores: dict,
    lineage,
) -> tuple[str, list[dict]]:
    """DOCBOT-2637: fetch the filing pages for books lines retrieval missed.

    Retrieval runs on the question alone, so a phrasing like "how did each
    expense line change vs FY2024" can return a stock-comp table instead of
    the income statement, and the tie-out silently compares nothing. When the
    books returned lines that no retrieved filing figure could be compared
    with, search again by those line names and add up to
    BOOK_GUIDED_MAX_CHUNKS new chunks. Filings only (retrieved context holds
    year-header statement tables); no LLM call.
    """
    if not sql_metadata or not doc_context or sql_metadata.get("csv_answer"):
        return doc_context, doc_citations
    if not sql_metadata.get("result_preview") or not yoy_prompt_block(doc_context):
        return doc_context, doc_citations
    from api.utils.discrepancy_detector import detect_discrepancies

    missing = detect_discrepancies(doc_context, sql_metadata).unpaired_db_lines
    if not missing:
        return doc_context, doc_citations
    started = time.perf_counter()
    # DOCBOT-2638: the bare line names match MD&A prose; naming the statement
    # puts the income statement table first.
    query = "Consolidated statements of operations " + " ".join(missing[:8])
    extra, extra_citations = await rag_retrieve(
        query, session_id, vector_stores, k=BOOK_GUIDED_SEARCH_K
    )
    added = [
        chunk for chunk in re.split(r"\n\n(?=Source: )", extra)
        if chunk.strip() and chunk.split("\n", 1)[-1].strip() not in doc_context
    ][:BOOK_GUIDED_MAX_CHUNKS]
    if not added:
        return doc_context, doc_citations
    keys = {(c.get("source"), c.get("page")) for c in doc_citations}
    added_heads = {a.split("\n", 1)[0] for a in added}
    for c in extra_citations:
        key = (c.get("source"), c.get("page"))
        if key not in keys and chunk_header({"source": key[0], "page": key[1]}) in added_heads:
            keys.add(key)
            doc_citations = doc_citations + [c]
            lineage.add_source(kind="pdf", label=str(c.get("source")), page=c.get("page"),
                               snippet=c.get("text"))
    lineage.add_step(
        "book_guided_retrieval", tool="vector_search+rerank", lane="docs",
        latency_ms=(time.perf_counter() - started) * 1000,
        detail=f"{len(added)} chunk(s) for books lines: {', '.join(missing[:8])}",
    )
    return doc_context + "\n\n" + "\n\n".join(added), doc_citations


async def _complete_tieout_lines(
    sql_metadata: dict | None,
    doc_context: str,
    connection_id: str | None,
    db_connections_table,
    schema_cache_table,
    async_session_factory,
    lineage,
) -> None:
    """DOCBOT-2637: re-run the books query with filing lines it skipped.

    Without this, a line the SQL step left out (Etsy's "Asset impairment
    charges") reached synthesis as "not in the books" although the books hold
    it. Updates ``sql_metadata`` in place; on any failure the first query's
    result stands.
    """
    if not sql_metadata or not doc_context or not connection_id:
        return
    if not sql_metadata.get("sql_query") or sql_metadata.get("csv_answer"):
        return
    from api.db_service import complete_tieout_lines

    started = time.perf_counter()
    try:
        done = await complete_tieout_lines(
            connection_id, sql_metadata["sql_query"], doc_context,
            list(sql_metadata.get("sources") or []),
            db_connections_table, schema_cache_table, async_session_factory,
        )
    except Exception as exc:  # noqa: BLE001 -- best effort; the original rows stand
        logger.warning("hybrid_chat: skipped-line completion failed: %s", exc)
        return
    if done is None:
        return
    sql, rows, added = done
    sql_metadata.update(sql_query=sql, result_preview=rows[:20], tieout_rows=rows, row_count=len(rows))
    lineage.add_step(
        "complete_tieout_lines", tool="sqlglot_ast", lane="db",
        latency_ms=(time.perf_counter() - started) * 1000,
        detail=f"added skipped lines: {', '.join(added)}",
    )


async def _correct_yoy_claims(
    answer: str, wrong: list[str], persona_def: str, format_block: str,
    yoy_section: str, question: str, run_id: str,
) -> str | None:
    """DOCBOT-2637: one rewrite fixing statements that contradict computed changes."""
    from api.utils.llm_provider import achat_completion

    prompt = (
        f"{persona_def}\n\n"
        "Some statements in your previous answer contradict the year-over-year changes "
        "computed from the filing. Rewrite the FULL answer, keeping everything else "
        "exactly, and correct each listed statement so it matches the computed change. "
        "Keep every [[discrepancy:...]] marker exactly as written.\n"
        f"{format_block}\n"
        "STATEMENTS TO CORRECT:\n" + "\n".join(f"- {w}" for w in wrong) + "\n"
        f"{yoy_section}\n"
        f"Question: {question}\n\n"
        f"Previous answer:\n{answer}\n\n"
        "Rewrite the full answer:"
    )
    try:
        return await achat_completion(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2500,
            caller="hybrid_yoy_correction",
            run_id=run_id,
            prompt_version=PROMPT_VERSION_HYBRID_SYNTHESIS,
        )
    except Exception as exc:  # noqa: BLE001 -- the original answer stands
        logger.warning("hybrid_chat: year-over-year correction failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# DOCBOT-1518: active discrepancy gate helper
# ---------------------------------------------------------------------------


def _answer_addresses_discrepancies(answer: str) -> bool:
    """Cheap, deterministic check for whether an answer surfaced a discrepancy.

    No LLM call — hybrid_chat's synthesis prompt pre-computes each discrepancy
    with an inline marker token ``[[discrepancy:<id>]]`` (see
    DiscrepancyReport.to_prompt_block()), so the word "discrepan[cy/t]" —
    present in every marker and in any plain-language flag — is a reliable,
    cheap signal that the model actually surfaced it rather than silently
    dropping it. This intentionally does not try to match discrepancies
    one-by-one — the goal is a fast yes/no gate, not another LLM call.
    """
    if not answer:
        return False
    return "discrepan" in answer.lower()


# ---------------------------------------------------------------------------
# DOCBOT-402: Hybrid chat async generator
# ---------------------------------------------------------------------------


def _synthesis_sources(has_doc_context: bool, sql_metadata: dict | None) -> set[str]:
    """Source keys (for ``build_format_block``) present in a hybrid synthesis prompt.

    CSV connections report their pandas answer under ``csv_answer``; SQL
    connections report ``result_preview`` rows.
    """
    from api.utils.response_format import SOURCE_CSV, SOURCE_DB, SOURCE_DOC

    sources: set[str] = set()
    if has_doc_context:
        sources.add(SOURCE_DOC)
    if sql_metadata:
        sources.add(SOURCE_CSV if sql_metadata.get("csv_answer") else SOURCE_DB)
    return sources


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
    chat_history: list[dict[str, str]] | None = None,
    chart_type: str = "auto",
) -> AsyncGenerator[str, None]:
    """Hybrid chat pipeline: intent classification → SQL + RAG → synthesis.

    Yields SSE-formatted strings with the same chunk contract as /api/db/chat:
      {"type": "metadata", "intent": ..., "has_sql": ..., "has_docs": ...}
      {"type": "token",    "content": "..."}  (N times)
      {"type": "done"}
    """
    has_db = connection_id is not None
    # DOCBOT-1512: wall-clock timer for this turn's agent trace row
    _hybrid_start = time.perf_counter()

    # DOCBOT-1501: bind one run_id for every LLM call made in this hybrid
    # turn (intent classification, SQL pipeline, RAG synthesis). The CM
    # instance is held in _trace_cm (not a bare `run_trace(...).__enter__()`
    # temporary) — a temporary with no held reference gets garbage
    # collected immediately, which triggers the generator's implicit
    # close()/GeneratorExit and resets the ContextVar before the next line
    # runs. See run_trace() docs in api/utils/llm_provider.py.
    from api.utils.llm_provider import current_run_id, new_run_id, run_trace
    run_id = current_run_id() or new_run_id()
    _trace_cm = run_trace(run_id)
    _trace_cm.__enter__()
    # DOCBOT-2636: lets GET /api/hybrid/answer/{run_id} answer "pending" until done.
    from api.answer_recovery_service import mark_done, mark_running
    mark_running(run_id, session_id, connection_id)

    # DOCBOT-1510: per-answer lineage collector
    from api.utils.lineage import LineageCollector
    lineage = LineageCollector(run_id, "hybrid", question=question)
    lineage.set_persona(persona)
    try:
        # ── DOCBOT-2311: never answer silently without the session's docs ─────
        # Lazy-load the session's vectors from CHROMA_DIR on a cache miss. If
        # they are gone (redeploy on ephemeral disk, retention cleanup), tell
        # the user explicitly; with a DB attached, continue DB-only.
        if has_docs:
            from api.utils.vector_store import (
                DOCUMENTS_EXPIRED_MESSAGE,
                documents_expired_sse,
                get_or_load_store,
            )

            store = await asyncio.to_thread(get_or_load_store, session_id, vector_stores)
            if store is None:
                logger.info("hybrid_chat: documents expired for session %s", session_id)
                yield documents_expired_sse(session_id)
                has_docs = False
                if not has_db:
                    yield f"data: {json.dumps({'type': 'token', 'content': DOCUMENTS_EXPIRED_MESSAGE})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'citations': [], 'run_id': run_id})}\n\n"
                    return
                notice = f"> **Note:** {DOCUMENTS_EXPIRED_MESSAGE} This answer uses only your connected data.\n\n"
                yield f"data: {json.dumps({'type': 'token', 'content': notice})}\n\n"

        # Rephrase follow-up questions using chat history
        if chat_history:
            from api.db_service import _rephrase_with_history
            try:
                question = await _rephrase_with_history(question, chat_history)
                lineage.set_standalone_query(question)
            except Exception as _exc:
                logger.warning("Hybrid rephrase failed, using original: %s", _exc)

        # ── Step 1: classify intent ───────────────────────────────────────────
        classification = await classify_intent_safe(
            question=question,
            has_db=has_db,
            has_docs=has_docs,
            session_id=session_id,
            async_session_factory=async_session_factory,
            query_history_table=query_history_table,
        )
        intent = classification.intent
        lineage.set_intent(intent)
        lineage.add_step(
            "classify_intent",
            tool="intent_classifier",
            status="retried" if classification.fallback_applied else "ok",
            detail="fallback to default route" if classification.fallback_applied else intent,
        )

        yield f"data: {json.dumps({'type': 'metadata', 'intent': intent, 'has_sql': has_db, 'has_docs': has_docs, 'run_id': run_id})}\n\n"

        # ── Step 2: gather context based on intent ────────────────────────────
        doc_context = ""
        doc_citations: list[dict] = []
        sql_metadata: dict | None = None
        sql_timed_out = False

        async def _bounded_sql() -> dict | None:
            nonlocal sql_timed_out
            try:
                return await asyncio.wait_for(
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
                        chart_type=chart_type,
                    ),
                    SQL_LANE_TIMEOUT_S,
                )
            except TimeoutError:
                logger.warning("hybrid_chat: SQL lane timed out after %.0fs", SQL_LANE_TIMEOUT_S)
                sql_timed_out = True
                return None

        if intent == "doc":
            with lineage.step("retrieve_docs", tool="vector_search+rerank", lane="docs"):
                doc_context, doc_citations = await rag_retrieve(
                    question, session_id, vector_stores, lineage=lineage
                )

        elif intent == "sql" and has_db and connection_id:
            with lineage.step("run_sql_pipeline", tool="sql_pipeline", lane="db"):
                sql_metadata = await _bounded_sql()

        else:  # hybrid — parallel gather
            async def _timed_rag():
                with lineage.step("retrieve_docs", tool="vector_search+rerank", lane="docs"):
                    return await rag_retrieve(
                        question, session_id, vector_stores, lineage=lineage
                    )

            async def _timed_sql():
                with lineage.step("run_sql_pipeline", tool="sql_pipeline", lane="db"):
                    return await _bounded_sql()

            rag_task = asyncio.create_task(_timed_rag())
            sql_task = (
                asyncio.create_task(_timed_sql()) if has_db and connection_id else None
            )

            doc_context, doc_citations = await rag_task

            if sql_task is not None:
                sql_metadata = await sql_task

            doc_context, doc_citations = await _book_guided_retrieval(
                doc_context, doc_citations, sql_metadata, session_id, vector_stores, lineage,
            )
            await _complete_tieout_lines(
                sql_metadata, doc_context, connection_id, db_connections_table,
                schema_cache_table, async_session_factory, lineage,
            )

        if has_db and intent != "doc" and sql_metadata is None:
            lineage.mark_step(
                "run_sql_pipeline", "error",
                "SQL query timed out" if sql_timed_out else "SQL query failed; answered without database results",
            )
        if sql_timed_out:
            yield f"data: {json.dumps({'type': 'token', 'content': SQL_TIMEOUT_NOTICE})}\n\n"

        # BUG #3 FIX: forward any chart/analysis_code events captured by
        # _collect_sql_result so the frontend hybrid chart handler receives them.
        # Without this, plotting requests via hybrid_chat() silently drop all charts.
        if sql_metadata:
            # DOCBOT-1510: forward the SQL result summary (previously swallowed)
            # so the Inspector Query tab works for hybrid answers. Result rows
            # are already PII-masked by the SQL pipeline.
            if sql_metadata.get("sql_query"):
                fwd = {
                    "type": "metadata",
                    "sql_query": sql_metadata.get("sql_query"),
                    "explanation": sql_metadata.get("explanation"),
                    "row_count": sql_metadata.get("row_count"),
                    "execution_time_ms": sql_metadata.get("execution_time_ms"),
                    "sources": sql_metadata.get("sources", []),
                }
                yield f"data: {json.dumps(fwd, default=str)}\n\n"
            lineage.set_sql(
                sql=sql_metadata.get("sql_query"),
                tables_selected=list(sql_metadata.get("sources") or []),
                row_count=sql_metadata.get("row_count"),
                execution_time_ms=sql_metadata.get("execution_time_ms"),
                result_preview=sql_metadata.get("result_preview") or [],
            )
            analysis_code_event = sql_metadata.get("analysis_code_event")
            if analysis_code_event:
                yield f"data: {json.dumps(analysis_code_event)}\n\n"
            for chart_evt in sql_metadata.get("chart_events", []) or []:
                yield f"data: {json.dumps(chart_evt)}\n\n"

        # ── Step 3: build synthesis prompt ───────────────────────────────────
        persona_def = (
            expert_personas.get(persona, expert_personas.get("Generalist", {}))
            .get("persona_def", "You are a helpful data analyst.")
        )

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
                )
            else:
                preview = json.dumps(sql_metadata.get("result_preview", [])[:20], default=str, indent=2)
                row_count = sql_metadata.get("row_count", 0)
                sources = sql_metadata.get("sources", [])
                source_str = ", ".join(sources) if sources else "database"
                sql_section = (
                    f"\n\nDatabase results ({row_count} rows) from [{source_str}]:\n{preview}\n"
                )
                if sql_metadata.get("empty_result"):
                    from api.db_service import EMPTY_RESULT_RULE
                    sql_section += EMPTY_RESULT_RULE + "\n"

        doc_section = ""
        # DOCBOT-2637: year-over-year changes computed in code, never by the model
        yoy_section = yoy_prompt_block(doc_context) if doc_context else ""
        if doc_context:
            from api.utils.prompt_guard import wrap_untrusted

            doc_section = (
                f"\n\nDocument context (untrusted — see PROMPT INJECTION DEFENSE "
                f"above):\n{wrap_untrusted(doc_context)}\n"
            )

        # DOCBOT-406: include span-verified financial values when available
        extracted_section = ""
        if extracted_fields:
            from api.document_extractor import format_extracted_fields_for_prompt
            formatted = format_extracted_fields_for_prompt(extracted_fields)
            if formatted:
                extracted_section = f"\n\n{formatted}"

        # DOCBOT-403: real discrepancy detection — extract numeric values from both
        # sources, compute deltas in code, inject pre-computed facts into the prompt.
        # The LLM receives confirmed discrepancies with exact numbers; it does NOT
        # compute deltas itself (which caused hallucination in the prompt-only stub).
        discrepancy_instruction = ""
        # DOCBOT-1518: kept in the outer scope (not just inside the `if`
        # below) so the active discrepancy gate after streaming can check
        # `report.has_discrepancies` regardless of which branch set it.
        discrepancy_report = None

        # DOCBOT-2911: org-scoped cross-source column mappings. Non-fatal —
        # returns ("", {}) on any failure or when the org has < 2 sources.
        from api.cross_source_service import canonicalize_sql_metadata, cross_source_context

        xsource_section = ""
        xsource_terms: dict[str, str] = {}
        if has_db:
            from api.utils.billing_context import get_principal

            _xs_principal = get_principal()
            xsource_section, xsource_terms = await cross_source_context(
                _xs_principal.org_id if _xs_principal else None, question, connection_id
            )

        if intent == "hybrid" and doc_context and sql_metadata:
            from api.utils.discrepancy_detector import detect_discrepancies
            report = detect_discrepancies(
                doc_context, canonicalize_sql_metadata(sql_metadata, xsource_terms)
            )
            discrepancy_report = report
            for item in report.discrepancies:
                lineage.add_discrepancy(item)
            discrepancy_instruction = report.to_prompt_block()
            if not discrepancy_instruction:
                # No numeric discrepancies found by code — tell LLM to flag only
                # non-numeric inconsistencies it observes (time periods,
                # categories, etc.) in plain language. There are no structured
                # tie-out items here, so no marker tokens exist — and the
                # literal "[DISCREPANCY]" tag is reserved for the detector's
                # own prompt block (DOCBOT-2002) and must never be taught.
                discrepancy_instruction = (
                    "\n\nDISCREPANCY CHECK: No numeric value conflicts were detected between "
                    "the document and database results. If you observe non-numeric "
                    "inconsistencies (e.g. different time periods, mismatched categories), "
                    "flag them clearly in your own words. Otherwise synthesize normally."
                )

        from api.utils.prompt_guard import PROMPT_INJECTION_DEFENSE_NOTICE
        from api.utils.response_format import (
            build_format_block,
            normalize_markdown,
            unclosed_fence_suffix,
        )

        # DOCBOT-2001: one shared format contract; citation rules only for the
        # sources that actually made it into this prompt.
        format_block = build_format_block(
            _synthesis_sources(bool(doc_context), sql_metadata)
        )

        prompt = (
            f"{persona_def}\n\n"
            f"Answer the following question using the context below. "
            f"Be accurate and thorough.{doc_note}{sql_note}\n\n"
            "COMPUTATION RULE: When the user provides explicit numerical parameters, "
            "assumptions, or scenarios, you MUST perform the requested calculations step by "
            "step. Show your arithmetic. Never say data is insufficient when the user or "
            "context provides the inputs needed.\n\n"
            f"{format_block}\n\n"
            f"{PROMPT_INJECTION_DEFENSE_NOTICE}"
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
            "it does not appear anywhere.\n"
            # DOCBOT-2637: the database results are one query's rows, not the whole books.
            "- The database results hold only the rows this query returned. If a document "
            "line has no matching database row, say it was not part of the query result; "
            "never say it is missing from the books or the database, and never mark it as "
            "a mismatch.\n"
            f"{YOY_RULE if yoy_section else ''}\n"
            f"Question: {question}"
            f"{extracted_section}"
            f"\n\n[DOCUMENT CONTEXT]\n{doc_section}{yoy_section}"
            f"\n\n[DATABASE RESULTS]\n{sql_section}{xsource_section}\n\n"
            "Answer:"
        )

        # ── Step 4: stream synthesis (Groq → Gemini fallback) ───────────────
        try:
            from api.utils.llm_provider import achat_completion_stream
            from api.utils.pii_masking import StreamingPIIMasker, mask_pii

            lineage.set_pii(True)
            _synth_start = time.perf_counter()
            _answer_parts: list[str] = []
            _masker = StreamingPIIMasker()  # DOCBOT-2003: line-buffered masking
            async for token in achat_completion_stream(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2000,
                caller="hybrid_synthesis",
                run_id=run_id,
                prompt_version=PROMPT_VERSION_HYBRID_SYNTHESIS,
            ):
                masked_token = _masker.feed(token)
                if masked_token:
                    _answer_parts.append(masked_token)
                    yield f"data: {json.dumps({'type': 'token', 'content': masked_token})}\n\n"
            masked_token = _masker.flush()
            if masked_token:
                _answer_parts.append(masked_token)
                yield f"data: {json.dumps({'type': 'token', 'content': masked_token})}\n\n"
            # DOCBOT-2001: append-only close of a dangling code fence.
            fence_tail = unclosed_fence_suffix("".join(_answer_parts))
            if fence_tail:
                _answer_parts.append(fence_tail)
                yield f"data: {json.dumps({'type': 'token', 'content': fence_tail})}\n\n"
        except Exception as exc:
            logger.error("hybrid_chat synthesis failed: %s", exc)
            from api.utils.exceptions import StreamInterruptedError
            detail = str(exc) if isinstance(exc, StreamInterruptedError) else "Synthesis failed. Please try again."
            yield f"data: {json.dumps({'type': 'error', 'detail': detail})}\n\n"
            # DOCBOT-2583: always close the stream so the client never waits forever.
            yield f"data: {json.dumps({'type': 'done', 'citations': [], 'run_id': run_id})}\n\n"
            return

        # DOCBOT-1518: active discrepancy gate. detect_discrepancies() above
        # only computes deltas and *instructs* the LLM to flag them — nothing
        # previously checked whether the streamed answer actually did. If a
        # meaningful discrepancy was detected but the answer text never
        # surfaced it, trigger exactly ONE corrective re-synthesis (same
        # one-shot-retry principle as the sandbox codegen retry in
        # api/sandbox_service.py) with an explicit instruction to address it.
        # The cheap substring check below never makes an LLM call, so the
        # common "no discrepancy" / "already addressed" cases add no cost.
        if discrepancy_report is not None and discrepancy_report.has_discrepancies:
            _synth_answer_text = "".join(_answer_parts)
            if not _answer_addresses_discrepancies(_synth_answer_text):
                try:
                    from api.utils.llm_provider import achat_completion

                    correction_prompt = (
                        f"{persona_def}\n\n"
                        "Your previous answer below did not explicitly surface a known "
                        "discrepancy between the document and database values. Rewrite "
                        "the FULL answer, keeping everything that was already correct, "
                        "and for EACH discrepancy below add one short sentence naming "
                        "the metric and ending with its marker token copied exactly. "
                        "The interface renders a full tie-out card from the marker; do "
                        "not repeat the breakdown and NEVER write the literal text "
                        "[DISCREPANCY].\n"
                        f"{format_block}\n"
                        f"{discrepancy_report.to_prompt_block()}\n\n"
                        f"Question: {question}\n\n"
                        f"Previous answer:\n{_synth_answer_text}\n\n"
                        "Rewrite the full answer including the discrepancy note:"
                    )
                    corrected = await achat_completion(
                        [{"role": "user", "content": correction_prompt}],
                        temperature=0.2,
                        max_tokens=2000,
                        caller="hybrid_discrepancy_correction",
                        run_id=run_id,
                        prompt_version=PROMPT_VERSION_HYBRID_SYNTHESIS,
                    )
                except Exception as exc:
                    logger.warning(
                        "hybrid_chat discrepancy correction failed (%s) — "
                        "keeping original answer unchanged", exc,
                    )
                    corrected = None

                if corrected and corrected.strip():
                    masked_corrected = normalize_markdown(mask_pii(corrected))
                    # DOCBOT-2004: the rewrite is a FULL answer, so swap it in
                    # via a "replace" event instead of appending (which showed
                    # the answer twice). Clients that ignore "replace" keep the
                    # original streamed answer. _answer_parts is replaced so
                    # lineage/log_trace persist the corrected text.
                    _answer_parts[:] = [masked_corrected]
                    yield f"data: {json.dumps({'type': 'replace', 'content': masked_corrected})}\n\n"
                    lineage.add_step(
                        "verify_discrepancy_addressed",
                        tool="llm",
                        status="retried",
                        detail="corrective re-synthesis triggered",
                    )
                else:
                    lineage.add_step(
                        "verify_discrepancy_addressed",
                        tool="llm",
                        status="error",
                        detail="correction failed — original answer kept",
                    )
            else:
                lineage.add_step("verify_discrepancy_addressed", tool="none", detail="already addressed")

        # DOCBOT-2637: year-over-year wording must match the computed changes.
        # Deterministic check, no LLM call unless a statement contradicts a
        # computed change ("higher tax provision" when it fell 22%); then one
        # rewrite, kept only if it fixes something and keeps the markers.
        if yoy_section:
            _yoy_facts = yoy_facts(doc_context)
            _yoy_answer = "".join(_answer_parts)
            _yoy_wrong = yoy_contradictions(_yoy_answer, _yoy_facts)
            if _yoy_wrong:
                rewritten = await _correct_yoy_claims(
                    _yoy_answer, _yoy_wrong, persona_def, format_block, yoy_section, question, run_id,
                )
                keeps_markers = not (
                    discrepancy_report is not None and discrepancy_report.has_discrepancies
                    and not _answer_addresses_discrepancies(rewritten or "")
                )
                if rewritten and keeps_markers and len(yoy_contradictions(rewritten, _yoy_facts)) < len(_yoy_wrong):
                    masked_rewrite = normalize_markdown(mask_pii(rewritten))
                    _answer_parts[:] = [masked_rewrite]
                    yield f"data: {json.dumps({'type': 'replace', 'content': masked_rewrite})}\n\n"
                    lineage.add_step("verify_yoy_claims", tool="llm", status="retried",
                                     detail=f"rewrote {len(_yoy_wrong)} year-over-year statement(s)")
                else:
                    lineage.add_step("verify_yoy_claims", tool="llm", status="error",
                                     detail="rewrite did not fix the statements; original kept")
            else:
                lineage.add_step("verify_yoy_claims", tool="none", detail="matches computed changes")

        # DOCBOT-2002: marker hygiene before the lineage/trace persist. The
        # model was told to place ``[[discrepancy:<id>]]`` markers and never
        # write the literal "[DISCREPANCY]" tag — but it sometimes does both
        # wrong. Rewrite legacy tags/echoed blocks into markers (or a neutral
        # label when nothing matches), then append a one-line tail badge for
        # every card the answer never referenced so each TieOutCard has a
        # badge in the text. Only the appended tail streams to the client
        # (as a token, after any DOCBOT-2004 "replace"); the rewritten text
        # is what lineage and the agent trace persist, and the renderer also
        # rewrites at display time for already-streamed chunks.
        _tieout_items = list(discrepancy_report.discrepancies) if discrepancy_report else []
        if _tieout_items:
            from api.utils.tieout import missing_marker_tail, replace_discrepancy_tags

            _final_text = replace_discrepancy_tags("".join(_answer_parts), _tieout_items)
            _tail = missing_marker_tail(_final_text, _tieout_items)
            if _tail:
                _final_text += _tail
                yield f"data: {json.dumps({'type': 'token', 'content': _tail})}\n\n"
            # log_trace below persists the rewritten text too.
            _answer_parts.clear()
            _answer_parts.append(_final_text)

        lineage.add_step(
            "synthesize",
            tool="llm_stream",
            latency_ms=(time.perf_counter() - _synth_start) * 1000,
        )
        from api.lineage_service import emit_lineage
        lineage_event = emit_lineage(lineage, session_id, connection_id)
        if lineage_event:
            yield lineage_event

        # DOCBOT-1706: first tie-out = a hybrid answer whose lineage carries a discrepancy.
        if discrepancy_report is not None and discrepancy_report.has_discrepancies:
            from api.funnel_service import record_funnel_event
            from api.utils.billing_context import get_principal

            _principal = get_principal()
            if _principal is not None:
                record_funnel_event(
                    "first_tieout", user_id=_principal.user_id, org_id=_principal.org_id,
                    props={"discrepancies": len(discrepancy_report.discrepancies)},
                )

        # DOCBOT-1512: fire-and-forget agent trace row for this turn
        from api.trace_service import log_trace
        _tool_chosen = (
            "sql+rag" if intent == "hybrid" else ("sql" if intent == "sql" else "rag")
        )
        trace_id = await log_trace(
            run_id=run_id,
            session_id=session_id,
            connection_id=connection_id,
            pipeline="hybrid",
            question=question,
            intent_classified=intent,
            tool_chosen=_tool_chosen,
            retrieved_refs=doc_citations,
            final_answer=normalize_markdown("".join(_answer_parts)),
            latency_ms=(time.perf_counter() - _hybrid_start) * 1000,
        )

        yield f"data: {json.dumps({'type': 'done', 'citations': doc_citations, 'run_id': run_id, 'trace_id': trace_id})}\n\n"
    finally:
        mark_done(run_id)
        _trace_cm.__exit__(None, None, None)
