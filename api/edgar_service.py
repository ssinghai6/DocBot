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
    from api.connector_store import get_cached_filing, save_filing_cache
    from api.connectors.edgar_connector import EdgarConnector
    from api.document_extractor import (
        extract_document_fields,
        is_extractable_document,
    )
    from api.utils.chunker import chunk_document, detect_doc_type
    from api.utils.vector_store import create_store

    # ── 1. Check cache ───────────────────────────────────────────────────
    cached = await get_cached_filing(accession_number)
    if cached and cached.get("session_id"):
        logger.info("edgar: cache hit for %s → session %s", accession_number, cached["session_id"])
        return {
            "session_id": cached["session_id"],
            "chunks_created": 0,
            "cached": True,
            "filing_date": filing_date,
        }

    # ── 2. Download filing text ──────────────────────────────────────────
    connector = EdgarConnector()
    text = await connector.fetch_filing_text(cik, accession_number, primary_document)

    if not text or len(text.strip()) < 100:
        raise ValueError(f"Filing {accession_number} yielded no usable text")

    text_hash = hashlib.sha256(text.encode()).hexdigest()

    # ── 3. Chunk the document ────────────────────────────────────────────
    doc_type = detect_doc_type(text)
    hf_api_key = os.getenv("huggingface_api_key") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or ""
    chunks = chunk_document(text, hf_api_key, doc_type)

    # Tag every chunk with filing metadata
    source_label = f"{ticker}-{filing_type}-{filing_date}"
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
    files_info = json.dumps([{
        "filename": f"{source_label}.htm",
        "pages": len(chunks),
        "size": len(text),
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
                    user_id=user_id,
                    source="edgar",
                )
            )

    # ── 6. Cache the filing ──────────────────────────────────────────────
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
    if gemini_key and is_extractable_document(text):
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
