"""DOCBOT-1616: streamed upload caps, magic-byte checks, temp-file cleanup, PDF parse timeout."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
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

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api.index as idx  # noqa: E402
from api.utils import upload_limits  # noqa: E402

_TEXT = "Quarterly revenue grew strongly across every segment this year. "


def _pdf(pages: int = 1) -> bytes:
    src = fitz.open()
    for _ in range(pages):
        src.new_page().insert_text((72, 72), _TEXT, fontsize=8)
    data = src.tobytes()
    src.close()
    return data


class _CountingUpload:
    """Minimal UploadFile stand-in that serves ``total`` bytes and counts reads."""

    filename = "big.bin"

    def __init__(self, total: int) -> None:
        self.remaining = total
        self.reads = 0
        self.consumed = 0

    async def read(self, size: int = -1) -> bytes:
        self.reads += 1
        n = self.remaining if size < 0 else min(size, self.remaining)
        self.remaining -= n
        self.consumed += n
        return b"x" * n


@pytest.fixture
def temp_paths(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every temp file the upload helpers create."""
    created: list[str] = []
    real = tempfile.mkstemp

    def _rec(*a: Any, **kw: Any) -> tuple[int, str]:
        fd, path = real(*a, **kw)
        created.append(path)
        return fd, path

    monkeypatch.setattr(upload_limits.tempfile, "mkstemp", _rec)
    return created


def _assert_removed(paths: list[str]) -> None:
    assert paths, "expected at least one temp file"
    assert not [p for p in paths if os.path.exists(p)]


# ── streaming helper ─────────────────────────────────────────────────────────


def test_oversize_stream_aborts_early(temp_paths: list[str]) -> None:
    mb = 1024 * 1024
    fake: Any = _CountingUpload(total=50 * mb)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(upload_limits.spool_upload(fake, 2 * mb))
    assert exc.value.status_code == 413
    assert "2 MB" in exc.value.detail
    assert fake.reads == 3  # 1 MB chunks: stops on the first chunk past 2 MB
    assert fake.consumed < 50 * mb
    _assert_removed(temp_paths)


def test_read_upload_capped_returns_bytes_and_removes_temp(temp_paths: list[str]) -> None:
    small: Any = _CountingUpload(total=10)
    data = asyncio.run(upload_limits.read_upload_capped(small, 100))
    assert data == b"x" * 10
    _assert_removed(temp_paths)


# ── POST /api/upload ─────────────────────────────────────────────────────────


def test_too_many_files_413(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(upload_limits, "MAX_FILES", 2)
    files = [("files", (f"f{i}.pdf", b"%PDF-1.4", "application/pdf")) for i in range(3)]
    resp = TestClient(idx.app).post("/api/upload", files=files)
    assert resp.status_code == 413
    assert "at most 2" in resp.json()["detail"]


def test_oversize_pdf_413(monkeypatch: pytest.MonkeyPatch, temp_paths: list[str]) -> None:
    monkeypatch.setattr(upload_limits, "MAX_FILE_BYTES", 1024)
    resp = TestClient(idx.app).post(
        "/api/upload", files={"files": ("big.pdf", b"%PDF-" + b"0" * 4096, "application/pdf")}
    )
    assert resp.status_code == 413
    _assert_removed(temp_paths)


def test_non_pdf_rejected_415(temp_paths: list[str]) -> None:
    resp = TestClient(idx.app).post(
        "/api/upload", files={"files": ("evil.pdf", b"MZ\x90\x00 not a pdf", "application/pdf")}
    )
    assert resp.status_code == 415
    assert "not a PDF" in resp.json()["detail"]
    _assert_removed(temp_paths)


def test_page_cap_413(monkeypatch: pytest.MonkeyPatch, temp_paths: list[str]) -> None:
    monkeypatch.setattr(upload_limits, "PDF_MAX_PAGES", 2)
    with patch.object(idx, "_extract_form_fields") as form_fields:
        resp = TestClient(idx.app).post(
            "/api/upload", files={"files": ("long.pdf", _pdf(pages=3), "application/pdf")}
        )
    assert resp.status_code == 413
    assert "3 pages" in resp.json()["detail"]
    form_fields.assert_not_called()  # cap checked before any extraction
    _assert_removed(temp_paths)


def test_parse_timeout_422(monkeypatch: pytest.MonkeyPatch, temp_paths: list[str]) -> None:
    monkeypatch.setattr(upload_limits, "PDF_PARSE_TIMEOUT_SECONDS", 0.05)

    def _slow(*_a: Any, **_kw: Any) -> tuple[list[str], int]:
        time.sleep(0.5)
        return [], 0

    with patch.object(idx, "extract_pdf_pages", _slow):
        resp = TestClient(idx.app).post(
            "/api/upload", files={"files": ("slow.pdf", _pdf(), "application/pdf")}
        )
    assert resp.status_code == 422
    assert resp.json()["detail"] == "This PDF took too long to process."
    _assert_removed(temp_paths)


@asynccontextmanager
async def _fake_begin() -> AsyncIterator[Any]:
    conn = MagicMock()
    conn.execute = AsyncMock()
    yield conn


def test_success_removes_temp_files(temp_paths: list[str]) -> None:
    fake_engine = MagicMock()
    fake_engine.begin = _fake_begin
    with (
        patch.object(idx, "engine", fake_engine),
        patch.object(idx, "get_embeddings", return_value=MagicMock()),
        patch.object(idx, "resolve_caller_scope", AsyncMock(return_value=None)),
        patch.object(idx, "owner_values", return_value={}),
        patch.object(idx, "meter_upload"),
        patch("api.utils.vector_store.create_store", return_value=MagicMock()),
        patch("api.document_extractor.is_extractable_document", return_value=False),
        patch("api.audit_service.log_event"),
    ):
        pdf = _pdf(pages=2)
        resp = TestClient(idx.app).post(
            "/api/upload",
            files=[
                ("files", ("a.pdf", pdf, "application/pdf")),
                ("files", ("b.pdf", pdf, "application/pdf")),
            ],
        )
        idx.VECTOR_STORES.pop(resp.json().get("session_id", ""), None)
    assert resp.status_code == 200, resp.text
    assert resp.json()["files_info"] == [
        {"filename": "a.pdf", "pages": 2, "size": len(pdf)},
        {"filename": "b.pdf", "pages": 2, "size": len(pdf)},
    ]
    assert len(temp_paths) == 2
    _assert_removed(temp_paths)


# ── POST /api/db/upload (SQLite) and /api/db/upload/csv ─────────────────────


def test_fake_sqlite_rejected_400(temp_paths: list[str]) -> None:
    resp = TestClient(idx.app).post(
        "/api/db/upload",
        files={"file": ("x.sqlite", b"definitely not sqlite" * 10, "application/octet-stream")},
        data={"session_id": "s1"},
    )
    assert resp.status_code == 400
    assert "not a valid SQLite" in resp.json()["detail"]
    _assert_removed(temp_paths)


def test_sqlite_oversize_413(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(upload_limits, "SQLITE_MAX_BYTES", 16)
    resp = TestClient(idx.app).post(
        "/api/db/upload",
        files={"file": ("x.sqlite", b"SQLite format 3\x00" + b"0" * 64, "application/octet-stream")},
        data={"session_id": "s1"},
    )
    assert resp.status_code == 413


def test_csv_with_nul_rejected_400(temp_paths: list[str]) -> None:
    resp = TestClient(idx.app).post(
        "/api/db/upload/csv",
        files={"file": ("x.csv", b"a,b\n1,\x002\n", "text/csv")},
        data={"session_id": "s1"},
    )
    assert resp.status_code == 400
    assert "NUL" in resp.json()["detail"]
    _assert_removed(temp_paths)


def test_csv_oversize_413(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(upload_limits, "CSV_MAX_BYTES", 8)
    resp = TestClient(idx.app).post(
        "/api/db/upload/csv",
        files={"file": ("x.csv", b"a,b\n" + b"1,2\n" * 50, "text/csv")},
        data={"session_id": "s1"},
    )
    assert resp.status_code == 413
