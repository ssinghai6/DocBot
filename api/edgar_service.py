"""EDGAR ingestion service — download SEC filings into the RAG pipeline.

Orchestrates:  EDGAR connector → text extraction → chunking → embedding → session

Reuses the existing document pipeline (chunker, vector_store, document_extractor)
so EDGAR filings get the same semantic chunking + financial field extraction as
manually uploaded PDFs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from typing import Any

from sqlalchemy import insert

logger = logging.getLogger(__name__)


async def _insert_session_row(
    *,
    session_id: str,
    source_label: str,
    pages: int,
    size: int,
    owner_cols: dict[str, Any],
    sessions_table: Any,
    async_session_factory: Any,
) -> None:
    """Insert the ``sessions`` row for an EDGAR filing, stamped with its owner."""
    files_info = json.dumps([{
        "filename": f"{source_label}.htm",
        "pages": pages,
        "size": size,
        "source": "edgar",
    }])
    async with async_session_factory() as session:
        async with session.begin():
            await session.execute(
                insert(sessions_table).values(
                    session_id=session_id,
                    persona="Finance Expert",
                    file_count=1,
                    files_info=files_info,
                    source="edgar",
                    # DOCBOT-1601: owner columns from caller_scope.owner_values()
                    **owner_cols,
                )
            )


async def _session_from_cache(
    *,
    cached_session_id: str,
    source_label: str,
    owner_cols: dict[str, Any],
    sessions_table: Any,
    async_session_factory: Any,
    vector_stores: dict,
    get_embeddings_fn: Any,
    extracted_fields_store: dict,
) -> str | None:
    """DOCBOT-1611: new caller-owned session cloned from a cached filing's vectors.

    Returns the new session id, or None when the cached vectors are gone.
    """
    from api.utils.vector_store import clone_store

    session_id = str(uuid.uuid4())
    try:
        cloned = clone_store(cached_session_id, session_id, get_embeddings_fn())
    except Exception as exc:  # chromadb raises assorted error types
        logger.warning("edgar: could not clone cached vectors (%s)", type(exc).__name__)
        return None
    if cloned is None:
        return None
    store, chunk_count = cloned
    vector_stores[session_id] = store
    await _insert_session_row(
        session_id=session_id,
        source_label=source_label,
        pages=chunk_count,
        size=0,
        owner_cols=owner_cols,
        sessions_table=sessions_table,
        async_session_factory=async_session_factory,
    )
    fields = extracted_fields_store.get(cached_session_id)
    if fields:
        extracted_fields_store[session_id] = list(fields)
    return session_id


async def ingest_edgar_filing(
    *,
    cik: str,
    accession_number: str,
    primary_document: str,
    filing_type: str,
    filing_date: str,
    company_name: str,
    ticker: str,
    user_id: str | None = None,
    owner: dict[str, Any] | None = None,
    # Injected dependencies (from index.py route handler)
    sessions_table: Any,
    async_session_factory: Any,
    vector_stores: dict,
    get_embeddings_fn: Any,
    extracted_fields_store: dict,
) -> dict[str, Any]:
    """Download a single SEC filing and ingest into the RAG pipeline.

    Returns {"session_id", "chunks_created", "cached", "filing_date"}.
    """
    from api.connector_store import (
        get_cached_filing,
        save_filing_cache,
        update_filing_cache_session,
    )
    from api.connectors.edgar_connector import EdgarConnector
    from api.document_extractor import (
        extract_document_fields,
        is_extractable_document,
    )
    from api.utils.chunker import chunk_document, detect_doc_type
    from api.utils.vector_store import create_store

    source_label = f"{ticker}-{filing_type}-{filing_date}"
    owner_cols = owner if owner is not None else {"user_id": user_id}

    # ── 1. Check cache ───────────────────────────────────────────────────
    # DOCBOT-1611: the cached session belongs to whoever first ingested the
    # filing. Never hand that id to another caller: clone its vectors into a
    # fresh session owned by THIS caller instead.
    cached = await get_cached_filing(accession_number)
    refresh_cache = False
    if cached and cached.get("session_id"):
        hit = await _session_from_cache(
            cached_session_id=cached["session_id"],
            source_label=source_label,
            owner_cols=owner_cols,
            sessions_table=sessions_table,
            async_session_factory=async_session_factory,
            vector_stores=vector_stores,
            get_embeddings_fn=get_embeddings_fn,
            extracted_fields_store=extracted_fields_store,
        )
        if hit is not None:
            logger.info("edgar: cache hit for %s → new session %s", accession_number, hit)
            return {
                "session_id": hit,
                "chunks_created": 0,
                "cached": True,
                "filing_date": filing_date,
            }
        # Source store was deleted or reaped: re-ingest and re-point the cache.
        logger.info("edgar: cache entry for %s has no vectors; re-ingesting", accession_number)
        refresh_cache = True

    # ── 2. Download filing text ──────────────────────────────────────────
    connector = EdgarConnector()
    text = await connector.fetch_filing_text(cik, accession_number, primary_document)

    if not text or len(text.strip()) < 100:
        raise ValueError(f"Filing {accession_number} yielded no usable text")

    text_hash = hashlib.sha256(text.encode()).hexdigest()

    # ── 3. Chunk the document ────────────────────────────────────────────
    doc_type = detect_doc_type(text)
    chunks = chunk_document(text, doc_type=doc_type)

    # Tag every chunk with filing metadata
    for chunk in chunks:
        chunk.metadata.update({
            "source": source_label,
            "filing_type": filing_type,
            "cik": cik,
            "ticker": ticker,
            "filing_date": filing_date,
        })

    # ── 4. Embed into ChromaDB ───────────────────────────────────────────
    session_id = str(uuid.uuid4())
    embeddings = get_embeddings_fn()
    store = create_store(session_id, chunks, embeddings)
    vector_stores[session_id] = store

    # ── 5. Create session row ────────────────────────────────────────────
    await _insert_session_row(
        session_id=session_id,
        source_label=source_label,
        pages=len(chunks),
        size=len(text),
        owner_cols=owner_cols,
        sessions_table=sessions_table,
        async_session_factory=async_session_factory,
    )

    # ── 6. Cache the filing ──────────────────────────────────────────────
    if refresh_cache:
        await update_filing_cache_session(accession_number, session_id)
    else:
        await save_filing_cache(
            filing_id=str(uuid.uuid4()),
            cik=cik,
            accession_number=accession_number,
            filing_type=filing_type,
            filing_date=filing_date,
            company_name=company_name,
            ticker=ticker,
            session_id=session_id,
            text_hash=text_hash,
        )

    # ── 7. Run structured extraction (async, non-blocking) ───────────────
    gemini_key = os.getenv("GEMINI_API_KEY") or ""
    if is_extractable_document(text):  # extractor returns [] when no model is configured
        try:
            fields = await extract_document_fields(text, session_id, gemini_key)
            if fields:
                extracted_fields_store[session_id] = fields
                logger.info("edgar: extracted %d fields for %s", len(fields), accession_number)
        except Exception as exc:
            logger.warning("edgar: field extraction failed for %s: %s", accession_number, exc)

    logger.info(
        "edgar: ingested %s → session %s (%d chunks)",
        accession_number,
        session_id,
        len(chunks),
    )

    return {
        "session_id": session_id,
        "chunks_created": len(chunks),
        "cached": False,
        "filing_date": filing_date,
    }


async def ingest_multiple_filings(
    *,
    cik: str,
    ticker: str,
    company_name: str,
    filing_type: str = "10-K",
    count: int = 5,
    user_id: str | None = None,
    owner: dict[str, Any] | None = None,
    # Injected dependencies
    sessions_table: Any,
    async_session_factory: Any,
    vector_stores: dict,
    get_embeddings_fn: Any,
    extracted_fields_store: dict,
) -> list[dict[str, Any]]:
    """Ingest the last N filings of a given type for a company.

    Returns a list of ingest results (one per filing).
    """
    from api.connectors.edgar_connector import EdgarConnector

    connector = EdgarConnector()
    filings = await connector.list_filings(cik, filing_type, count)

    results: list[dict[str, Any]] = []
    for filing in filings:
        try:
            result = await ingest_edgar_filing(
                cik=cik,
                accession_number=filing["accession_number"],
                primary_document=filing["primary_document"],
                filing_type=filing["filing_type"],
                filing_date=filing["filing_date"],
                company_name=company_name,
                ticker=ticker,
                user_id=user_id,
                owner=owner,
                sessions_table=sessions_table,
                async_session_factory=async_session_factory,
                vector_stores=vector_stores,
                get_embeddings_fn=get_embeddings_fn,
                extracted_fields_store=extracted_fields_store,
            )
            results.append(result)
        except Exception as exc:
            logger.warning(
                "edgar: failed to ingest %s: %s",
                filing.get("accession_number"),
                exc,
            )
            results.append({
                "accession_number": filing.get("accession_number"),
                "filing_date": filing.get("filing_date"),
                "error": str(exc),
            })

    return results
