"""Where a retrieved chunk came from, as shown in prompts and citations (DOCBOT-2637).

PDF chunks always carry a page. SEC filing chunks carry the printed page when
the filing has page footers and ``page=None`` otherwise; those are located by
their Part / Item section instead, never by a made-up page number.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional


def chunk_page(meta: Mapping[str, Any]) -> Optional[int]:
    """The chunk's page number, or None when it has no real one."""
    page = meta.get("page")
    return page if isinstance(page, int) and page > 0 else None


def chunk_header(meta: Mapping[str, Any]) -> str:
    """``Source: X, Page N`` (or ``Source: X, Section S``) chunk header line.

    discrepancy_detector._SOURCE_HEADER_RE parses this exact format.
    """
    source = meta.get("source", "Unknown")
    page = chunk_page(meta)
    if page is not None:
        return f"Source: {source}, Page {page}"
    section = str(meta.get("section") or "").strip()
    return f"Source: {source}, Section {section}" if section else f"Source: {source}"


def chunk_key(meta: Mapping[str, Any]) -> str:
    """Dedupe key for citations: one per page, or per section without pages."""
    page = chunk_page(meta)
    return f"{meta.get('source', 'Unknown')}_{page if page is not None else meta.get('section', '')}"


def chunk_label(meta: Mapping[str, Any]) -> str:
    """Short ``X p12`` label for logs and injection-scan reports."""
    page = chunk_page(meta)
    return f"{meta.get('source', 'Unknown')}" + (f" p{page}" if page is not None else "")
