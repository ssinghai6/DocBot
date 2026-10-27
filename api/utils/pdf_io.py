"""In-memory PDF opening helper (DOCBOT-1633).

Uploaded PDFs used to be written to ``/tmp/{session_id}_{filename}`` and
never deleted: a predictable, user-influenced host path that left every
upload on disk. PyMuPDF parses directly from bytes, so no temporary file is
needed at all.
"""

from __future__ import annotations

from typing import Any

PASSWORD_PROTECTED_DETAIL = (
    "This PDF is password-protected. Remove the password and upload it again."
)


class PdfPasswordProtectedError(ValueError):
    """Raised when a PDF needs a user password to open (DOCBOT-2306)."""


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
