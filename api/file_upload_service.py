"""
DocBot File Upload Service — DOCBOT-206, DOCBOT-207

Handles credential-free data source uploads:
  - DOCBOT-206: SQLite file upload → registered as a db_connection (sqlite dialect)
  - DOCBOT-207: CSV file upload → stored as raw bytes → registered as a db_connection
                (csv dialect) → queries run via E2B pandas sandbox, no SQL pipeline
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import Table, delete, insert, select

from api.utils.encryption import encrypt_credentials

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TMP_DIR = Path("/tmp/docbot_uploads")
_SQLITE_MAX_BYTES = 100 * 1024 * 1024   # 100 MB
_CSV_MAX_BYTES = 50 * 1024 * 1024        # 50 MB
_FILE_TTL_SECONDS = 2 * 60 * 60         # 2 hours


def _ensure_tmp_dir() -> None:
    _TMP_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# DOCBOT-206: SQLite file upload
# ---------------------------------------------------------------------------


async def upload_sqlite(
    file_bytes: bytes,
    original_filename: str,
    session_id: str,
    db_connections_table: Table,
    schema_cache_table: Table,
    async_session_factory,
) -> Dict[str, Any]:
    """
    Save an uploaded SQLite file to /tmp, register it as a db_connection
    (dialect=sqlite), introspect schema, and return connection_id + schema summary.

    Raises:
        ValueError: if the file exceeds the size limit or is not a valid SQLite file.
    """
    if len(file_bytes) > _SQLITE_MAX_BYTES:
        raise ValueError(
            f"SQLite file exceeds the 100 MB limit ({len(file_bytes) // (1024*1024)} MB uploaded)."
        )

    _ensure_tmp_dir()
    file_id = str(uuid.uuid4())
    file_path = _TMP_DIR / f"{file_id}.sqlite"

    file_path.write_bytes(file_bytes)

    # Validate it's a real SQLite file (magic bytes: "SQLite format 3\x00")
    _validate_sqlite_magic(file_path)

    connection_id = await _register_file_as_connection(
        file_path=file_path,
        dialect="sqlite",
        session_id=session_id,
        original_filename=original_filename,
        db_connections_table=db_connections_table,
        schema_cache_table=schema_cache_table,
        async_session_factory=async_session_factory,
    )

    # Return schema summary
    from api.db_service import get_schema
    schema = await get_schema(
        connection_id, db_connections_table, schema_cache_table, async_session_factory
    )

    return {
        "connection_id": connection_id,
        "source_type": "sqlite_upload",
        "original_filename": original_filename,
        "schema_summary": {
            "table_count": len(schema),
            "tables": [t["name"] for t in schema[:10]],
        },
    }


def _validate_sqlite_magic(file_path: Path) -> None:
    """Check the SQLite magic bytes header."""
    magic = b"SQLite format 3\x00"
    with open(file_path, "rb") as f:
        header = f.read(len(magic))
    if header != magic:
        file_path.unlink(missing_ok=True)
        raise ValueError(
            "Uploaded file is not a valid SQLite database. "
            "Please ensure the file has a .sqlite or .db extension and is not corrupted."
        )


# ---------------------------------------------------------------------------
# DOCBOT-207: CSV file upload
# ---------------------------------------------------------------------------


async def upload_csv(
    file_bytes: bytes,
    original_filename: str,
    session_id: str,
    db_connections_table: Table,
    schema_cache_table: Table,
    async_session_factory,
) -> Dict[str, Any]:
    """
    Parse a CSV file with pandas (schema detection only), store raw bytes in
    the credentials blob as dialect="csv", and register it as a db_connection.

    Queries against this connection bypass the SQL pipeline entirely and run
    pandas code on an E2B sandbox instead.

    Raises:
        ValueError: if the file exceeds the size limit or cannot be parsed.
    """
    if len(file_bytes) > _CSV_MAX_BYTES:
        raise ValueError(
            f"CSV file exceeds the 50 MB limit ({len(file_bytes) // (1024*1024)} MB uploaded)."
        )

    # Parse CSV in executor to get schema info (no SQLite write needed)
    table_name, row_count, columns = await asyncio.get_event_loop().run_in_executor(
        None,
        _parse_csv_metadata,
        file_bytes,
        original_filename,
    )

    connection_id = await _register_csv_connection(
        file_bytes=file_bytes,
        table_name=table_name,
        row_count=row_count,
        columns=columns,
        original_filename=original_filename,
        session_id=session_id,
        db_connections_table=db_connections_table,
        schema_cache_table=schema_cache_table,
        async_session_factory=async_session_factory,
    )

    return {
        "connection_id": connection_id,
        "source_type": "csv_upload",
        "original_filename": original_filename,
        "table_name": table_name,
        "schema_summary": {
            "table_count": 1,
            "tables": [table_name],
            "columns": columns,
            "row_count": row_count,
        },
    }


def _parse_csv_metadata(
    file_bytes: bytes,
    original_filename: str,
) -> Tuple[str, int, List[str]]:
    """
    Parse CSV bytes with pandas and return (table_name, row_count, column_names).

    Only reads the CSV to extract schema info — no SQLite write.
    """
    import io
    import re

    import pandas as pd

    df = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(
                io.BytesIO(file_bytes),
                encoding=encoding,
                dtype=str,
                on_bad_lines="skip",
            )
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    if df is None or df.empty:
        raise ValueError("CSV file is empty or could not be parsed. Check encoding (UTF-8 or Latin-1 expected).")

    # Clean column names: lowercase, replace non-alphanum with _
    df.columns = [
        re.sub(r"[^a-z0-9_]", "_", col.strip().lower().replace(" ", "_"))
        for col in df.columns
    ]

    raw_name = Path(original_filename).stem
    table_name = re.sub(r"[^a-z0-9_]", "_", raw_name.strip().lower())[:50] or "uploaded_data"

    return table_name, len(df), list(df.columns)


async def _register_csv_connection(
    file_bytes: bytes,
    table_name: str,
    row_count: int,
    columns: List[str],
    original_filename: str,
    session_id: str,
    db_connections_table: Table,
    schema_cache_table: Table,
    async_session_factory,
) -> str:
    """
    Register a CSV upload as a db_connection with dialect="csv".

    The raw CSV bytes are base64-encoded and stored in the credentials blob so
    they can be uploaded to an E2B sandbox at query time.  No /tmp file is
    created — the bytes live in the encrypted DB record.
    """
    connection_id = str(uuid.uuid4())
    csv_b64 = base64.b64encode(file_bytes).decode()

    creds_blob = encrypt_credentials({
        "dialect": "csv",
        "host": "__local_file__",
        "port": 0,
        "dbname": original_filename,
        "user": "",
        "password": "",
        "original_filename": original_filename,
        "table_name": table_name,
        "columns": columns,
        "row_count": row_count,
        "csv_content": csv_b64,
        "ttl_expires_at": (
            datetime.now(timezone.utc) + timedelta(seconds=_FILE_TTL_SECONDS)
        ).isoformat(),
    })

    async with async_session_factory() as session:
        async with session.begin():
            await session.execute(
                insert(db_connections_table).values(
                    id=connection_id,
                    session_id=session_id,
                    dialect="csv",
                    host="__local_file__",
                    port=0,
                    dbname=original_filename,
                    credentials_blob=creds_blob,
                )
            )

    # Cache schema (columns as TEXT — E2B pandas handles actual types)
    schema = [{
        "name": table_name,
        "columns": [{"name": col, "type": "TEXT"} for col in columns],
    }]
    schema_json = json.dumps(schema)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=24)

    async with async_session_factory() as session:
        async with session.begin():
            await session.execute(
                insert(schema_cache_table).values(
                    connection_id=connection_id,
                    schema_json=schema_json,
                    expires_at=expires_at,
                )
            )

    return connection_id


# ---------------------------------------------------------------------------
# Shared: register a local file as a db_connection
# ---------------------------------------------------------------------------


async def _register_file_as_connection(
    file_path: Path,
    dialect: str,
    session_id: str,
    original_filename: str,
    db_connections_table: Table,
    schema_cache_table: Table,
    async_session_factory,
) -> str:
    """
    Insert a row into db_connections for a local file-based source.

    The 'host' is stored as '__local_file__' to distinguish uploads from
    network connections. The actual path lives in the encrypted credentials blob.
    """
    from api.db_service import _introspect_schema_from_url

    connection_id = str(uuid.uuid4())
    sqlite_url = f"sqlite:///{file_path}"

    # Encrypt the file path as credentials (no real password — path is the secret)
    creds_blob = encrypt_credentials({
        "dialect": dialect,
        "host": "__local_file__",
        "port": 0,
        "dbname": str(file_path),
        "user": "",
        "password": "",
        "original_filename": original_filename,
        "ttl_expires_at": (
            datetime.now(timezone.utc) + timedelta(seconds=_FILE_TTL_SECONDS)
        ).isoformat(),
    })

    async with async_session_factory() as session:
        async with session.begin():
            await session.execute(
                insert(db_connections_table).values(
                    id=connection_id,
                    session_id=session_id,
                    dialect=dialect,
                    host="__local_file__",
                    port=0,
                    dbname=str(file_path),
                    credentials_blob=creds_blob,
                )
            )

    # Introspect + cache schema
    schema = await _introspect_schema_from_url(sqlite_url, dialect)
    schema_json = json.dumps(schema)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=24)

    async with async_session_factory() as session:
        async with session.begin():
            await session.execute(
                insert(schema_cache_table).values(
                    connection_id=connection_id,
                    schema_json=schema_json,
                    expires_at=expires_at,
                )
            )

    return connection_id


# ---------------------------------------------------------------------------
# TTL cleanup — removes expired upload files + their connection records
# ---------------------------------------------------------------------------


async def cleanup_expired_uploads(
    db_connections_table: Table,
    schema_cache_table: Table,
    async_session_factory,
) -> int:
    """
    Remove /tmp SQLite files and their DB records where the 2-hour TTL has elapsed.
    Returns the number of connections cleaned up.

    Called on startup and can be scheduled periodically.
    """
    from api.utils.encryption import decrypt_credentials
    from cryptography.fernet import InvalidToken

    removed = 0
    now = datetime.now(timezone.utc)

    async with async_session_factory() as session:
        result = await session.execute(
            select(db_connections_table).where(
                db_connections_table.c.host == "__local_file__"
            )
        )
        rows = result.fetchall()

    for row in rows:
        try:
            creds = decrypt_credentials(row.credentials_blob)
        except (InvalidToken, Exception):
            continue

        ttl_str = creds.get("ttl_expires_at")
        if not ttl_str:
            continue

        ttl_expires = datetime.fromisoformat(ttl_str)
        if ttl_expires.tzinfo is None:
            ttl_expires = ttl_expires.replace(tzinfo=timezone.utc)

        if now < ttl_expires:
            continue  # Still valid

        # Expired — delete /tmp file (SQLite uploads only; CSV bytes live in the DB)
        if creds.get("dialect") != "csv":
            file_path = Path(creds.get("dbname", ""))
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info("Cleaned up expired upload file: %s", file_path.name)
                except OSError as exc:
                    logger.warning("Could not delete file %s: %s", file_path, exc)

        async with async_session_factory() as session:
            async with session.begin():
                await session.execute(
                    delete(schema_cache_table).where(
                        schema_cache_table.c.connection_id == row.id
                    )
                )
                await session.execute(
                    delete(db_connections_table).where(
                        db_connections_table.c.id == row.id
                    )
                )
        removed += 1

    if removed:
        logger.info("TTL cleanup: removed %d expired file upload connection(s).", removed)
    return removed
