"""In-memory PDF opening helper (DOCBOT-1633).

Uploaded PDFs used to be written to ``/tmp/{session_id}_{filename}`` and
never deleted: a predictable, user-influenced host path that left every
upload on disk. PyMuPDF parses directly from bytes, so no temporary file is
needed at all.
"""

from __future__ import annotations

from typing import Any


def open_pdf_bytes(content: bytes) -> Any:
    """Open PDF ``content`` with PyMuPDF without writing it to disk.

    Returns a ``fitz.Document``; the caller is responsible for ``close()``.
    Raises the same PyMuPDF errors as ``fitz.open(path)`` for invalid input.
    """
    import fitz

    return fitz.open(stream=content, filetype="pdf")
