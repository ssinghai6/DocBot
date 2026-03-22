"""
DocBot File Upload Service — DOCBOT-206, DOCBOT-207

Handles credential-free data source uploads:
  - DOCBOT-206: SQLite file upload → registered as a db_connection
  - DOCBOT-207: CSV file upload → converted to SQLite → registered as a db_connection

Both feed into the same 7-step SQL pipeline with zero separate code paths.
"""

from __future__ import annotations

import asyncio
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
    Parse a CSV file with pandas, infer column types, write to a temp SQLite file,
    and register it as a db_connection so the same 7-step pipeline applies.

    Raises:
        ValueError: if the file exceeds the size limit or cannot be parsed.
    """
    if len(file_bytes) > _CSV_MAX_BYTES:
        raise ValueError(
            f"CSV file exceeds the 50 MB limit ({len(file_bytes) // (1024*1024)} MB uploaded)."
        )

    _ensure_tmp_dir()
    file_id = str(uuid.uuid4())
    sqlite_path = _TMP_DIR / f"{file_id}.sqlite"

    # Convert CSV → SQLite in executor (pandas is sync)
    table_name, row_count, columns = await asyncio.get_event_loop().run_in_executor(
        None,
        _csv_bytes_to_sqlite,
        file_bytes,
        original_filename,
        sqlite_path,
    )

    connection_id = await _register_file_as_connection(
        file_path=sqlite_path,
        dialect="sqlite",
        session_id=session_id,
        original_filename=original_filename,
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


def _csv_bytes_to_sqlite(
    file_bytes: bytes,
    original_filename: str,
    sqlite_path: Path,
) -> Tuple[str, int, List[str]]:
    """
    Parse CSV bytes → pandas DataFrame → SQLite table.
    Returns (table_name, row_count, column_names).

    Handles:
      - BOM encoding (UTF-8-SIG)
      - Quoted commas
      - Mixed-type columns (fallback to TEXT)
      - Date string detection and casting
    """
    import io
    import re
    import sqlite3

    import pandas as pd

    # Detect encoding — try UTF-8 with BOM first, then plain UTF-8, then latin-1
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(
                io.BytesIO(file_bytes),
                encoding=encoding,
                dtype=str,          # Read everything as string first
                on_bad_lines="skip",
            )
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    else:
        raise ValueError("CSV file could not be parsed. Check encoding (UTF-8 or Latin-1 expected).")

    if df.empty:
        raise ValueError("CSV file is empty or contains only a header row.")

    # Clean column names: lowercase, strip spaces, replace non-alphanum with _
    df.columns = [
        re.sub(r"[^a-z0-9_]", "_", col.strip().lower().replace(" ", "_"))
        for col in df.columns
    ]

    # Type inference pass — try to coerce each column
    df = _infer_and_cast_columns(df)

    # Derive safe table name from filename (strip extension, sanitise)
    raw_name = Path(original_filename).stem
    table_name = re.sub(r"[^a-z0-9_]", "_", raw_name.strip().lower())[:50] or "uploaded_data"

    # Write to SQLite
    conn = sqlite3.connect(str(sqlite_path))
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()

    return table_name, len(df), list(df.columns)


def _infer_and_cast_columns(df):
    """
    Attempt to cast each column to a more specific type:
      - INTEGER if all non-null values are whole numbers
      - FLOAT   if all non-null values are numeric
      - DATE    if values match common date patterns
      - TEXT    fallback (no data loss)
    """
    import re
    import pandas as pd

    _DATE_PATTERNS = [
        r"^\d{4}-\d{2}-\d{2}$",
        r"^\d{2}/\d{2}/\d{4}$",
        r"^\d{2}-\d{2}-\d{4}$",
    ]

    for col in df.columns:
        series = df[col].dropna().str.strip()
        if series.empty:
            continue

        # Try integer
        try:
            as_float = pd.to_numeric(series, errors="raise")
            if (as_float == as_float.astype(int)).all():
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                continue
        except (ValueError, TypeError):
            pass

        # Try float
        try:
            pd.to_numeric(series, errors="raise")
            df[col] = pd.to_numeric(df[col], errors="coerce")
            continue
        except (ValueError, TypeError):
            pass

        # Try date
        if any(re.match(pat, str(series.iloc[0])) for pat in _DATE_PATTERNS):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                continue
            except Exception:
                pass

        # Fallback: keep as string
    return df


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

        # Expired — delete file and DB records
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
