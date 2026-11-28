"""In-memory PDF opening helper (DOCBOT-1633).

Uploaded PDFs used to be written to ``/tmp/{session_id}_{filename}`` and
never deleted: a predictable, user-influenced host path that left every
upload on disk. PyMuPDF parses directly from bytes, so no temporary file is
needed at all.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from api.utils.pdf_layout import layout_page_text

PASSWORD_PROTECTED_DETAIL = (
    "This PDF is password-protected. Remove the password and upload it again."
)


class PdfPasswordProtectedError(ValueError):
    """Raised when a PDF needs a user password to open (DOCBOT-2306)."""


class PdfTooManyPagesError(ValueError):
    """Raised when a PDF exceeds the page cap (DOCBOT-1616)."""


def open_pdf_bytes(content: bytes) -> Any:
    """Open PDF ``content`` with PyMuPDF without writing it to disk.

    Returns a ``fitz.Document``; the caller is responsible for ``close()``.
    Raises the same PyMuPDF errors as ``fitz.open(path)`` for invalid input,
    and ``PdfPasswordProtectedError`` when the PDF needs a password.
    """
    import fitz

    doc = fitz.open(stream=content, filetype="pdf")
    if doc.needs_pass:
        doc.close()
        raise PdfPasswordProtectedError(PASSWORD_PROTECTED_DETAIL)
    return doc


def extract_pdf_pages(
    path: str,
    max_pages: int,
    form_fields_fn: Optional[Callable[[Any], str]] = None,
) -> tuple[list[str], int]:
    """Open the PDF at ``path`` and return ``(page_texts, page_count)``.

    Blocking (DOCBOT-1616): callers run it off the event loop. The page cap is
    checked before any text extraction; the document is always closed here so
    no PyMuPDF object escapes the worker thread. When ``form_fields_fn``
    returns text it is prepended to page 1 (AcroForm widget values).
    """
    import fitz

    doc = fitz.open(path, filetype="pdf")
    try:
        if doc.needs_pass:
            raise PdfPasswordProtectedError(PASSWORD_PROTECTED_DETAIL)
        page_count = len(doc)
        if page_count > max_pages:
            raise PdfTooManyPagesError(
                f"This PDF has {page_count} pages; the limit is {max_pages} pages per file."
            )
        form_text = form_fields_fn(doc) if form_fields_fn else ""
        texts: list[str] = []
        for page_num in range(page_count):
            page = doc[page_num]
            # DOCBOT-2731: tables rendered as "label | v1 | v2" rows
            text = layout_page_text(page.get_text("words"), page.get_text("text"))
            if page_num == 0 and form_text:
                text = form_text + "\n\n" + (text or "")
            texts.append(text)
        return texts, page_count
    finally:
        doc.close()
