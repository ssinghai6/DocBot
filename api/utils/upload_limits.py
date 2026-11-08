"""Upload size/count limits and streamed spooling (DOCBOT-1616).

Every upload route reads the ``UploadFile`` in fixed-size chunks into a
``mkstemp`` file and aborts with 413 the moment the per-file cap is crossed,
so an oversized body is never pulled into memory in full.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import HTTPException, UploadFile

_MB = 1024 * 1024
_CHUNK_BYTES = _MB


def _env_num(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


MAX_FILE_BYTES = int(_env_num("UPLOAD_MAX_FILE_MB", 25) * _MB)
MAX_FILES = int(_env_num("UPLOAD_MAX_FILES", 10))
PDF_MAX_PAGES = int(_env_num("UPLOAD_PDF_MAX_PAGES", 500))
PDF_PARSE_TIMEOUT_SECONDS = _env_num("UPLOAD_PDF_PARSE_TIMEOUT_SECONDS", 60)
SQLITE_MAX_BYTES = int(_env_num("UPLOAD_SQLITE_MAX_MB", 100) * _MB)
CSV_MAX_BYTES = int(_env_num("UPLOAD_CSV_MAX_MB", 50) * _MB)

PDF_MAGIC = b"%PDF-"


def _limit_label(max_bytes: int) -> str:
    return f"{max_bytes / _MB:g} MB"


def remove_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


async def spool_upload(file: UploadFile, max_bytes: int) -> tuple[str, int]:
    """Stream ``file`` into a private temp file, enforcing ``max_bytes``.

    Returns ``(path, size)``; the caller must ``remove_quietly(path)``.
    Raises 413 as soon as the running total exceeds the cap (temp file removed).
    """
    fd, path = tempfile.mkstemp(prefix="docbot_upload_")
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := await file.read(_CHUNK_BYTES):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {_limit_label(max_bytes)} upload limit.",
                    )
                out.write(chunk)
    except BaseException:
        remove_quietly(path)
        raise
    return path, size


async def read_upload_capped(file: UploadFile, max_bytes: int) -> bytes:
    """Spool ``file`` with the streamed cap, return its bytes, remove the temp file."""
    path, _ = await spool_upload(file, max_bytes)
    try:
        return Path(path).read_bytes()
    finally:
        remove_quietly(path)


def has_magic(path: str, magic: bytes) -> bool:
    with open(path, "rb") as f:
        return f.read(len(magic)) == magic


def check_file_count(count: int) -> None:
    if count > MAX_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files: upload at most {MAX_FILES} per request.",
        )
