"""DOCBOT-2306: POST /api/upload crash chain (encrypted PDFs, page count, audit key)."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet

if not os.environ.get("DB_ENCRYPTION_KEY"):
    os.environ["DB_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql://fake:fake@localhost:5432/testdb"

fitz = pytest.importorskip("fitz")

from fastapi.testclient import TestClient  # noqa: E402

import api.index as idx  # noqa: E402
from api.utils.pdf_io import PdfPasswordProtectedError, open_pdf_bytes  # noqa: E402

_TEXT = "Quarterly revenue grew strongly across every segment this year. "


def _pdf(pages: int = 1, password: str | None = None) -> bytes:
    src = fitz.open()
    for _ in range(pages):
        src.new_page().insert_text((72, 72), _TEXT, fontsize=8)
    if password:
        data = src.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, user_pw=password, owner_pw=password)
    else:
        data = src.tobytes()
    src.close()
    return data


def test_open_pdf_bytes_rejects_password_protected() -> None:
    with pytest.raises(PdfPasswordProtectedError, match="password-protected"):
        open_pdf_bytes(_pdf(password="secret"))


def test_upload_encrypted_pdf_returns_400() -> None:
    client = TestClient(idx.app)
    resp = client.post(
        "/api/upload",
        files={"files": ("locked.pdf", _pdf(password="secret"), "application/pdf")},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"].startswith("This PDF is password-protected")


@asynccontextmanager
async def _fake_begin() -> AsyncIterator[Any]:
    conn = MagicMock()
    conn.execute = AsyncMock()
    yield conn


def test_upload_records_page_count_and_audits_filename() -> None:
    fake_engine = MagicMock()
    fake_engine.begin = _fake_begin
    with (
        patch.object(idx, "engine", fake_engine),
        patch.object(idx, "get_embeddings", return_value=MagicMock()),
        patch.object(idx, "resolve_caller_scope", AsyncMock(return_value=None)),
        patch.object(idx, "owner_values", return_value={}),
        patch.object(idx, "meter_upload") as meter,
        patch("api.utils.vector_store.create_store", return_value=MagicMock()),
        patch("api.document_extractor.is_extractable_document", return_value=False),
        patch("api.audit_service.log_event") as log_event,
    ):
        client = TestClient(idx.app)
        resp = client.post(
            "/api/upload",
            files={"files": ("report.pdf", _pdf(pages=3), "application/pdf")},
        )
        idx.VECTOR_STORES.pop(resp.json().get("session_id", ""), None)

    assert resp.status_code == 200, resp.text
    info = resp.json()["files_info"]
    assert info == [{"filename": "report.pdf", "pages": 3, "size": info[0]["size"]}]
    assert meter.call_args.kwargs["pages"] == 3
    assert log_event.call_args.kwargs["detail"] == "report.pdf"
