"""Semantic chunker for financial and legal documents.

Uses LangChain SemanticChunker with HuggingFace embeddings.
Falls back to RecursiveCharacterTextSplitter if embeddings unavailable.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

try:
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_huggingface import HuggingFaceEmbeddings
    _SEMANTIC_AVAILABLE = True
except ImportError:
    SemanticChunker = None  # type: ignore[assignment,misc]
    HuggingFaceEmbeddings = None  # type: ignore[assignment,misc]
    _SEMANTIC_AVAILABLE = False

# Document types that benefit from semantic chunking (boundary-aware splitting
# rather than fixed character windows).
_SEMANTIC_DOC_TYPES = {"financial", "legal", "annual_report", "10k", "contract"}

# Keywords used for lightweight doc-type detection from the first 500 characters
# of a document when the caller cannot provide an explicit type.
_FINANCIAL_KEYWORDS = {"annual report", "10-k", "financial statements", "balance sheet", "income statement"}
_LEGAL_KEYWORDS = {"agreement", "contract", "whereas", "hereinafter", "indemnification"}


def detect_doc_type(text: str) -> str:
    """Return a doc_type string from the first 500 characters of ``text``.

    Returns one of ``"financial"``, ``"legal"``, or ``"general"``.
    """
    sample = text[:500].lower()
    for kw in _FINANCIAL_KEYWORDS:
        if kw in sample:
            return "financial"
    for kw in _LEGAL_KEYWORDS:
        if kw in sample:
            return "legal"
    return "general"


def _fallback_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )


def chunk_document(
    text: str,
    hf_api_key: str,
    doc_type: str = "general",
) -> list[Document]:
    """Split ``text`` into LangChain Document chunks.

    Parameters
    ----------
    text:
        Full document text to split.
    hf_api_key:
        HuggingFace API key used to initialise the embeddings model for
        SemanticChunker.  When empty the function falls back to
        RecursiveCharacterTextSplitter.
    doc_type:
        One of the known financial/legal types or ``"general"``.
        Financial/legal types use SemanticChunker; others use
        RecursiveCharacterTextSplitter.

    Returns
    -------
    list[Document]
        List of LangChain Document objects with ``page_content`` set.
    """
    if doc_type in _SEMANTIC_DOC_TYPES and hf_api_key and SemanticChunker is not None:
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
            )
            chunker = SemanticChunker(
                embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=85,
            )
            chunks = chunker.create_documents([text])
            logger.info(
                "chunker: SemanticChunker produced %d chunks for doc_type=%s",
                len(chunks),
                doc_type,
            )
            return chunks
        except Exception as exc:
            logger.warning(
                "chunker: SemanticChunker failed (%s) — "
                "falling back to RecursiveCharacterTextSplitter",
                exc,
            )

    splitter = _fallback_splitter()
    chunks = splitter.create_documents([text])
    logger.debug(
        "chunker: RecursiveCharacterTextSplitter produced %d chunks for doc_type=%s",
        len(chunks),
        doc_type,
    )
    return chunks
