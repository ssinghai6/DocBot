"""Semantic chunker for financial and legal documents.

Uses LangChain SemanticChunker with the shared embeddings model
(``api.utils.embeddings_provider``: local ONNX by default, DOCBOT-1511, so no
API key is needed). Falls back to RecursiveCharacterTextSplitter if the
embeddings or SemanticChunker are unavailable.

PR5 (DOCBOT-1200): switched from ``HuggingFaceEmbeddings`` (which downloads
sentence-transformers locally — ~90MB on Railway cold start, plus torch) to
``HuggingFaceEndpointEmbeddings`` which calls the HuggingFace Inference API
the rest of the codebase already uses. Keeps the SemanticChunker semantics
intact and aligns with the "HuggingFace API only" claim in CLAUDE.md.
"""

from __future__ import annotations

import bisect
import logging
import os
import re
import time

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

try:
    from langchain_experimental.text_splitter import SemanticChunker
    _SEMANTIC_AVAILABLE = True
except ImportError:
    SemanticChunker = None  # type: ignore[assignment,misc]
    _SEMANTIC_AVAILABLE = False

from api.utils.embeddings_provider import get_embeddings

# Document types that benefit from semantic chunking (boundary-aware splitting
# rather than fixed character windows).
_SEMANTIC_DOC_TYPES = {"financial", "legal", "annual_report", "10k", "sec_filing", "contract"}

# SemanticChunker embeds every sentence, so cost grows linearly with length.
# Above this many characters, fall back to the recursive splitter (instant).
# Override with SEMANTIC_CHUNK_MAX_CHARS; 0 or negative disables the cap.
_SEMANTIC_MAX_CHARS = int(os.getenv("SEMANTIC_CHUNK_MAX_CHARS", "60000"))

# Keywords used for lightweight doc-type detection from the first 500 characters
# of a document when the caller cannot provide an explicit type.
_FINANCIAL_KEYWORDS = {
    "annual report", "10-k", "10-q", "financial statements", "balance sheet",
    "income statement", "form 10-k", "form 10-q", "form 8-k",
    "securities and exchange commission", "registrant",
    "consolidated statements", "notes to financial statements",
    "stockholders' equity", "cash flows",
}
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
    hf_api_key: str = "",
    doc_type: str = "general",
) -> list[Document]:
    """Split ``text`` into LangChain Document chunks.

    Parameters
    ----------
    text:
        Full document text to split.
    hf_api_key:
        Deprecated and ignored (DOCBOT-1511): embeddings no longer need an
        API key. Kept so existing positional callers keep working.
    doc_type:
        One of the known financial/legal types or ``"general"``.
        Financial/legal types use SemanticChunker; others use
        RecursiveCharacterTextSplitter.

    Returns
    -------
    list[Document]
        List of LangChain Document objects with ``page_content`` set.
    """
    _too_long = 0 < _SEMANTIC_MAX_CHARS < len(text)
    if _too_long and doc_type in _SEMANTIC_DOC_TYPES:
        logger.info(
            "chunker: %d chars exceeds semantic cap %d — using recursive splitter",
            len(text), _SEMANTIC_MAX_CHARS,
        )
    if doc_type in _SEMANTIC_DOC_TYPES and SemanticChunker is not None and not _too_long:
        _start = time.monotonic()
        try:
            embeddings = get_embeddings()
            chunker = SemanticChunker(
                embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=85,
            )
            chunks = chunker.create_documents([text])
            logger.info(
                "chunker: SemanticChunker produced %d chunks for doc_type=%s in %.1fs",
                len(chunks),
                doc_type,
                time.monotonic() - _start,
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


# ---------------------------------------------------------------------------
# DOCBOT-2634: SEC filing chunker (EDGAR path only; PDFs use chunk_document)
# ---------------------------------------------------------------------------

# Part / Item headings of a 10-K or 10-Q, used as the chunk's section locator.
_SECTION_RE = re.compile(r"^(?:PART\s+[IVX]+\.?$|ITEM\s+\d+[A-Z]?\.)", re.IGNORECASE)
_YEAR_RE = re.compile(r"^(?:19|20)\d\d$")
FILING_TABLE_MAX_CHARS = 1000
_CAPTION_LINE_MAX = 120


def _section_heading(line: str) -> str | None:
    line = line.strip()
    # table rows are excluded so the table of contents does not set sections
    if len(line) <= _CAPTION_LINE_MAX and " | " not in line and _SECTION_RE.match(line):
        return line
    return None


def _is_table_block(block: str) -> bool:
    return any(" | " in line for line in block.splitlines())


def _caption(prose: str) -> str:
    """Up to 3 short lines right before a table (company, title, units line)."""
    out: list[str] = []
    for line in reversed(prose.strip().splitlines()):
        line = line.strip()
        if not line or len(line) > _CAPTION_LINE_MAX or len(out) == 3:
            break
        if line.isdigit() or line.lower() == "table of contents":
            break  # page footer, not a caption
        out.append(line)
    return "\n".join(reversed(out))


def _header_rows(rows: list[str]) -> list[str]:
    """Leading rows before the first one holding a non-year number."""
    header: list[str] = []
    for row in rows[:4]:
        cells = [c.strip() for c in row.split(" | ")]
        if any(re.search(r"\d", c) and not _YEAR_RE.match(c) for c in cells):
            break
        header.append(row)
    return header if len(header) < len(rows) else []


def _split_table(block: str, caption: str, max_chars: int) -> list[str]:
    """The whole table as one piece when it fits, else row-boundary pieces
    that each repeat the caption and header rows."""
    whole = f"{caption}\n{block}" if caption else block
    if len(whole) <= max_chars:
        return [whole]
    rows = block.splitlines()
    header = _header_rows(rows)
    prefix = [p for p in (caption, *header) if p]
    pieces: list[str] = []
    cur: list[str] = []
    budget = max_chars - sum(len(p) + 1 for p in prefix)
    for row in rows[len(header):]:
        if cur and sum(len(r) + 1 for r in cur) + len(row) > budget:
            pieces.append("\n".join(prefix + cur))
            cur = []
        cur.append(row)
    if cur:
        pieces.append("\n".join(prefix + cur))
    return pieces


# A page footer is a line holding only the page number. Kept only when the
# numbers run 1, 2, 3... so a stray number line is never read as a page.
_PAGE_FOOTER_RE = re.compile(r"^[ \t]*(\d{1,3})[ \t]*$", re.MULTILINE)
_MIN_PAGE_FOOTERS = 3


def _printed_pages(text: str):
    """Return ``page_at(offset)``: the printed page at a text offset, or None.

    Text after footer ``n`` is on page ``n + 1``; text before the first footer
    is on that footer's page. Filings without a footer sequence get None.
    """
    footers: list[tuple[int, int]] = []
    for m in _PAGE_FOOTER_RE.finditer(text):
        n = int(m.group(1))
        # a skipped footer (lost in HTML stripping) still continues the run
        if footers and 0 < n - footers[-1][1] <= 3 or not footers and n <= 3:
            footers.append((m.end(), n))
    if len(footers) < _MIN_PAGE_FOOTERS:
        return lambda offset: None
    ends = [end for end, _ in footers]

    def page_at(offset: int) -> int:
        i = bisect.bisect_right(ends, offset)
        return footers[0][1] if i == 0 else footers[i - 1][1] + 1

    return page_at


def _piece_start(text: str, piece: str, cursor: int) -> int:
    """Offset of a chunk in the filing text, searching forward from cursor.

    Table pieces repeat a caption/header, so the first row that is not
    repeated anchors the search; a miss keeps the previous offset.
    """
    for line in reversed(piece.splitlines()):
        line = line.strip()
        if len(line) >= 8:
            found = text.find(line, cursor)
            if found >= 0:
                # back up to the piece's own first line when it is nearby
                first = piece.strip().splitlines()[0].strip()
                head = text.rfind(first, cursor, found + 1) if first else -1
                return head if head >= 0 else found
    return cursor


def chunk_filing(text: str, table_max_chars: int = FILING_TABLE_MAX_CHARS) -> list[Document]:
    """Chunk an SEC filing rendered by ``edgar_connector.html_to_text``.

    Prose goes through the same recursive splitter long documents already use.
    Tables (blank-line-separated blocks of ``a | b | c`` rows) are never split
    mid-row and carry their caption. Every chunk gets a locator: ``page`` is
    the printed page the chunk starts on (DOCBOT-2637, from the filing's page
    footers; None when the filing has none) and ``section`` is the enclosing
    Part / Item heading.
    """
    splitter = _fallback_splitter()
    pieces: list[str] = []
    prose: list[str] = []

    def flush_prose() -> None:
        if prose:
            pieces.extend(splitter.split_text("\n\n".join(prose)))
            prose.clear()

    for block in text.split("\n\n"):
        if not block.strip():
            continue
        if _is_table_block(block):
            caption = _caption(prose[-1]) if prose else ""
            flush_prose()
            pieces.extend(_split_table(block.strip(), caption, table_max_chars))
        else:
            prose.append(block)
    flush_prose()

    page_at = _printed_pages(text)
    docs: list[Document] = []
    section = ""
    cursor = 0
    for i, piece in enumerate(pieces):
        cursor = _piece_start(text, piece, cursor)
        headings = [h for h in map(_section_heading, piece.splitlines()) if h]
        starts_with_heading = bool(headings) and piece.lstrip().startswith(headings[0])
        docs.append(Document(
            page_content=piece,
            metadata={
                "page": page_at(cursor),
                "chunk_index": i,
                "section": headings[0] if starts_with_heading else section,
            },
        ))
        if headings:
            section = headings[-1]
    return docs
