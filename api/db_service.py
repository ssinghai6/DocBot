"""
DocBot Database Service — DOCBOT-201, 202, 203, 204, 205

Handles all database connectivity logic:
  - SSRF-safe connection validation (DOCBOT-201)
  - Schema introspection + 24-hour TTL cache (DOCBOT-202)
  - 7-step bounded SQL generation pipeline (DOCBOT-203)
  - MySQL connector support via pymysql (DOCBOT-205)

Routes are thin wrappers in api/index.py; all business logic lives here.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel, field_validator, model_validator
from sqlalchemy import (
    Column, DateTime, Integer, MetaData, String, Table, Text,
    func, insert, select, text, update, delete,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from api.utils.ssrf_validator import validate_ssrf
from api.utils.encryption import encrypt_credentials, decrypt_credentials
from api.utils.sql_validator import validate_and_sanitize_sql, QueryValidationError
from api.utils.embeddings import find_similar_queries, embedding_to_json
from api.utils.exceptions import TokenExpiredError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection engine pool — LRU cache avoids create/dispose per query
# ---------------------------------------------------------------------------

import threading
from collections import OrderedDict

_engine_cache: OrderedDict[str, Any] = OrderedDict()
_engine_cache_lock = threading.Lock()
_ENGINE_CACHE_MAX = 20


def _get_or_create_engine(sync_url: str, connect_args: Optional[Dict[str, Any]] = None):
    """Return a cached SQLAlchemy engine or create one with connection pooling.

    Engines are keyed by URL and cached in an LRU with max 20 entries.
    Evicted engines are disposed properly.
    """
    from sqlalchemy import create_engine

    with _engine_cache_lock:
        if sync_url in _engine_cache:
            _engine_cache.move_to_end(sync_url)
            return _engine_cache[sync_url]

    # Create outside lock to avoid blocking
    engine = create_engine(
        sync_url,
        connect_args=connect_args or {},
        pool_size=3,
        max_overflow=2,
        pool_timeout=10,
        pool_recycle=1800,  # 30 min — prevent stale connections
        pool_pre_ping=True,  # validate connections before use
    )

    with _engine_cache_lock:
        # Double-check after lock
        if sync_url in _engine_cache:
            engine.dispose()
            _engine_cache.move_to_end(sync_url)
            return _engine_cache[sync_url]

        _engine_cache[sync_url] = engine
        # Evict oldest if over capacity
        while len(_engine_cache) > _ENGINE_CACHE_MAX:
            _, evicted = _engine_cache.popitem(last=False)
            try:
                evicted.dispose()
            except Exception:
                pass

    return engine


def _dispose_engine(sync_url: str) -> None:
    """Remove and dispose a specific engine from the cache."""
    with _engine_cache_lock:
        engine = _engine_cache.pop(sync_url, None)
    if engine:
        try:
            engine.dispose()
        except Exception:
            pass


def _json_dumps(obj: Any) -> str:
    """json.dumps that handles Decimal and other non-serializable DB types."""
    def _default(o: Any) -> Any:
        if isinstance(o, Decimal):
            return float(o)
        return str(o)
    return json.dumps(obj, default=_default)


# ---------------------------------------------------------------------------
# Supported dialects
# ---------------------------------------------------------------------------

SUPPORTED_DIALECTS = {"postgresql", "mysql", "sqlite", "azure_sql"}

_DIALECT_DRIVER = {
    "postgresql": "postgresql+psycopg2",
    "mysql":      "mysql+pymysql",
    "sqlite":     "sqlite",
    "azure_sql":  "mssql+pyodbc",
}

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ConnectionNotFoundError(Exception):
    """Raised when a connection_id does not exist or belongs to a different session."""


class ExecutionTimeoutError(Exception):
    """Raised when the SQL executor exceeds its time budget."""


class SchemaIntrospectionError(Exception):
    """Raised when schema introspection fails (permissions, network, etc.)."""


class ConnectionError(Exception):
    """Raised when a DB connection cannot be established."""


class SQLGenerationError(Exception):
    """Raised when the LLM fails to generate valid SQL."""


def classify_db_error(exc: Exception, dialect: str) -> str:
    """Return a user-friendly error message based on exception type and dialect.

    Maps raw DB driver exceptions to actionable messages so users know
    what went wrong and how to fix it.
    """
    msg = str(exc).lower()

    # Authentication / permission errors
    if any(k in msg for k in ("password authentication failed", "access denied", "login failed")):
        return "Authentication failed. Please check your username and password."
    if any(k in msg for k in ("permission denied", "insufficient privilege")):
        return "Permission denied. Your database user does not have read access to the requested tables."

    # Network / connectivity
    if any(k in msg for k in ("could not connect", "connection refused", "no route to host",
                                "name or service not known", "getaddrinfo failed")):
        return (
            f"Cannot reach the database server. Please verify the host, port, and "
            f"that the server accepts connections from this IP."
        )
    if any(k in msg for k in ("connection timed out", "connect_timeout")):
        return "Connection timed out. The database server did not respond within 10 seconds."
    if "ssl" in msg and ("required" in msg or "unsupported" in msg):
        return "SSL configuration mismatch. Check whether the server requires or rejects SSL connections."

    # Query execution errors
    if any(k in msg for k in ("relation", "does not exist", "unknown table", "no such table")):
        return "Table not found. The schema may have changed — try refreshing the schema."
    if any(k in msg for k in ("column", "does not exist", "unknown column", "no such column")):
        return "Column not found. The table structure may have changed — try refreshing the schema."
    if "syntax error" in msg:
        return "SQL syntax error. The generated query has a syntax problem — try rephrasing your question."
    if "canceling statement due to statement timeout" in msg:
        return "Query exceeded the 15-second time limit. Try a simpler question or add filters."
    if "deadlock" in msg:
        return "Query hit a deadlock. Please try again in a moment."

    # Dialect-specific
    if dialect == "azure_sql" and "snapshot isolation" in msg:
        return "Azure SQL snapshot isolation is not enabled on this database."

    return f"Database error: {str(exc)[:200]}"


# Re-export for routes
__all__ = [
    "DBConnectionRequest",
    "DBChatRequest",
    "DBChatResponse",
    "QueryValidationError",
    "ConnectionNotFoundError",
    "ExecutionTimeoutError",
    "SchemaIntrospectionError",
    "SQLGenerationError",
    "TokenExpiredError",
    "classify_db_error",
    "connect_database",
    "disconnect_database",
    "get_schema",
    "run_sql_pipeline",
]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class DBConnectionRequest(BaseModel):
    session_id: str
    dialect: str
    host: str
    port: int
    dbname: str
    user: Optional[str] = None
    password: Optional[str] = None
    # PII masking — DOCBOT-604
    pii_masking_enabled: bool = False

    # Entra auth (azure_sql only) — "entra_sp" or "entra_interactive"
    auth_type: Optional[str] = None
    # entra_sp fields
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    # entra_interactive field — pre-acquired MSAL access token (never logged)
    access_token: Optional[str] = None

    @field_validator("dialect")
    @classmethod
    def dialect_must_be_supported(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in SUPPORTED_DIALECTS:
            raise ValueError(
                f"Unsupported dialect '{v}'. Supported: {', '.join(sorted(SUPPORTED_DIALECTS))}"
            )
        return v

    @field_validator("host")
    @classmethod
    def host_must_pass_ssrf(cls, v: str) -> str:
        validate_ssrf(v)
        return v

    @model_validator(mode="after")
    def validate_auth_fields(self) -> "DBConnectionRequest":
        if self.dialect == "azure_sql":
            if self.auth_type == "entra_sp":
                missing = [f for f in ("tenant_id", "client_id", "client_secret")
                           if not getattr(self, f)]
                if missing:
                    raise ValueError(
                        f"azure_sql with entra_sp auth requires: {', '.join(missing)}"
                    )
            elif self.auth_type == "entra_interactive":
                if not self.access_token:
                    raise ValueError(
                        "azure_sql with entra_interactive auth requires 'access_token'"
                    )
            else:
                raise ValueError(
                    "azure_sql dialect requires auth_type='entra_sp' or 'entra_interactive'"
                )
        else:
            if self.access_token:
                raise ValueError("access_token is only valid for entra_interactive auth_type")
            # SQLite doesn't require credentials — empty strings are valid
            if self.dialect != "sqlite" and (not self.user or self.password is None):
                raise ValueError(
                    f"dialect '{self.dialect}' requires 'user' and 'password' fields"
                )
        return self


class ChatMessageCompact(BaseModel):
    """Lightweight chat message for conversational history (DB/CSV pipelines)."""
    role: str
    content: str


class DBChatRequest(BaseModel):
    connection_id: str
    question: str
    persona: str = "Generalist"
    session_id: str
    chart_type: str = "auto"   # DOCBOT-305: "auto"|"bar"|"line"|"scatter"|"heatmap"|"box"|"multi"
    history: List[ChatMessageCompact] = []  # last 6 messages for follow-up context


class DBChatResponse(BaseModel):
    answer: str
    sql_query: str
    explanation: str
    result_preview: List[Dict[str, Any]]
    row_count: int
    sources: List[str]
    execution_time_ms: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _rephrase_with_history(
    question: str,
    chat_history: List[Dict[str, str]],
) -> str:
    """Rephrase a follow-up question into a standalone question using chat history.

    Uses a quick LLM call (max_tokens=200, temperature=0) so the downstream
    SQL pipeline gets full context even when the user says things like
    "now filter that to Q1 only".
    """
    import asyncio as _asyncio
    from api.utils.llm_provider import chat_completion

    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in chat_history
    )
    prompt = (
        "Given the conversation history below and the user's latest question, "
        "rephrase the latest question as a standalone question that includes "
        "all necessary context from the conversation. If the question is "
        "already standalone, return it unchanged.\n\n"
        "Return ONLY the rephrased question, nothing else.\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"Latest question: {question}\n\n"
        "Standalone question:"
    )
    try:
        loop = _asyncio.get_running_loop()
        rephrased = await loop.run_in_executor(
            None,
            lambda: chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            ),
        )
        rephrased = rephrased.strip()
        if rephrased:
            logger.debug("Rephrased follow-up: %r → %r", question, rephrased)
            return rephrased
    except Exception as exc:
        logger.warning("Follow-up rephrase failed, using original question: %s", exc)

    return question


def _build_connection_url(
    dialect: str, host: str, port: int, dbname: str,
    user: str = "", password: str = "",
) -> str:
    driver = _DIALECT_DRIVER[dialect]
    if dialect == "sqlite":
        return f"sqlite:///{dbname}"
    if dialect == "azure_sql":
        odbc_str = (
            f"Driver={{ODBC Driver 18 for SQL Server}};"
            f"Server={host},{port};Database={dbname};"
            "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=10;"
        )
        return f"mssql+pyodbc:///?odbc_connect={odbc_str}"
    url = f"{driver}://{user}:{password}@{host}:{port}/{dbname}"
    if dialect == "postgresql":
        url += "?sslmode=prefer"
    return url


def _get_entra_token(tenant_id: str, client_id: str, client_secret: str) -> str:
    """Acquire Azure SQL access token via Service Principal. Sync — run in executor."""
    try:
        from azure.identity import ClientSecretCredential
    except ImportError as exc:
        raise RuntimeError(
            "azure-identity package is not installed. "
            "Add 'azure-identity>=1.17.0' to requirements.txt and rebuild the container."
        ) from exc
    try:
        credential = ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
        )
        token = credential.get_token("https://database.windows.net/.default")
        return token.token
    except Exception as exc:
        logger.error("Entra token acquisition failed (credentials redacted)")
        raise ValueError(
            "Azure Entra authentication failed. "
            "Please verify tenant_id, client_id, and client_secret."
        ) from exc


def _build_entra_connect_args(token: str) -> Dict[str, Any]:
    """Build pyodbc connect_args for Azure SQL token auth (SQL_COPT_SS_ACCESS_TOKEN=1256)."""
    import struct
    token_bytes = token.encode("utf-16-le")
    token_struct = struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)
    return {"attrs_before": {1256: token_struct}}


def _parse_token_expiry(access_token: str) -> datetime:
    """Extract the exp claim from a JWT without verifying the signature.

    The token is validated by Azure SQL at connection time — we only need
    the expiry timestamp to manage our own staleness check.
    """
    import base64
    segments = access_token.split(".")
    if len(segments) != 3:
        raise ValueError("Malformed JWT: expected 3 segments")
    payload_b64 = segments[1]
    # Pad to multiple of 4 for base64 decoding
    padding = 4 - len(payload_b64) % 4
    if padding != 4:
        payload_b64 += "=" * padding
    try:
        raw = base64.urlsafe_b64decode(payload_b64)
        claims = json.loads(raw)
    except Exception as exc:
        raise ValueError(f"Failed to parse token payload: {exc}") from exc
    if "exp" not in claims:
        raise ValueError("JWT has no exp claim")
    return datetime.fromtimestamp(int(claims["exp"]), tz=timezone.utc)


def _resolve_connection(creds: Dict[str, Any]) -> tuple[str, Optional[Dict[str, Any]]]:
    """From decrypted credentials, return (sync_url, entra_connect_args).
    Sync — intended to be called inside run_in_executor for azure_sql."""
    dialect = creds["dialect"]
    if dialect == "azure_sql":
        auth_type = creds.get("auth_type", "entra_sp")
        url = _build_connection_url(dialect, creds["host"], creds["port"], creds["dbname"])
        if auth_type == "entra_interactive":
            expires_at = datetime.fromisoformat(creds["token_expires_at"])
            if datetime.now(timezone.utc) >= expires_at - timedelta(minutes=5):
                raise TokenExpiredError()
            return url, _build_entra_connect_args(creds["access_token"])
        # entra_sp path
        token = _get_entra_token(
            creds["tenant_id"], creds["client_id"], creds["client_secret"]
        )
        return url, _build_entra_connect_args(token)
    url = _build_connection_url(
        dialect, creds["host"], creds["port"],
        creds["dbname"], creds["user"], creds["password"],
    )
    return url, None


def _get_groq_client():
    """Legacy helper — kept for reference. New code uses llm_provider."""
    from groq import Groq
    api_key = os.getenv("groq_api_key")
    if not api_key:
        raise RuntimeError("groq_api_key environment variable is not set.")
    return Groq(api_key=api_key)


def _get_embeddings_model():
    """Reuse the same HuggingFace embeddings singleton pattern used by the PDF pipeline."""
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
    hf_token = os.getenv("huggingface_api_key") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=hf_token,
    )


# ---------------------------------------------------------------------------
# DOCBOT-201: Connect / Disconnect
# ---------------------------------------------------------------------------


async def connect_database(
    req: DBConnectionRequest,
    db_connections_table: Table,
    schema_cache_table: Table,
    async_session_factory,
) -> Dict[str, Any]:
    """
    Validate credentials, encrypt, store, and return connection_id + schema summary.

    Steps:
      1. SSRF already validated by Pydantic field_validator on host.
      2. Test connection with a lightweight SELECT 1.
      3. Fernet-encrypt credentials.
      4. Insert into db_connections.
      5. Introspect schema → populate schema_cache with 24h TTL.
      6. Return connection_id + schema summary.
    """
    import asyncio as _asyncio
    connection_id = str(uuid.uuid4())

    entra_connect_args: Optional[Dict[str, Any]] = None
    if req.dialect == "azure_sql":
        sync_url = _build_connection_url(req.dialect, req.host, req.port, req.dbname)
        if req.auth_type == "entra_interactive":
            try:
                token_expires_at = _parse_token_expiry(req.access_token)  # type: ignore[arg-type]
            except ValueError as exc:
                raise ValueError(f"Invalid access token: {exc}") from exc
            if datetime.now(timezone.utc) >= token_expires_at:
                raise ValueError("Provided access token is already expired")
            entra_connect_args = _build_entra_connect_args(req.access_token)  # type: ignore[arg-type]
        else:
            token = await _asyncio.get_running_loop().run_in_executor(
                None, _get_entra_token, req.tenant_id, req.client_id, req.client_secret
            )
            entra_connect_args = _build_entra_connect_args(token)
    else:
        sync_url = _build_connection_url(
            req.dialect, req.host, req.port, req.dbname, req.user, req.password
        )

    await _test_connection(sync_url, req.dialect, entra_connect_args)

    if req.dialect == "azure_sql":
        if req.auth_type == "entra_interactive":
            creds_blob = encrypt_credentials({
                "dialect": req.dialect, "host": req.host, "port": req.port,
                "dbname": req.dbname, "auth_type": "entra_interactive",
                "access_token": req.access_token,
                "token_expires_at": token_expires_at.isoformat(),  # type: ignore[possibly-undefined]
            })
        else:
            creds_blob = encrypt_credentials({
                "dialect": req.dialect, "host": req.host, "port": req.port,
                "dbname": req.dbname, "auth_type": "entra_sp",
                "tenant_id": req.tenant_id, "client_id": req.client_id,
                "client_secret": req.client_secret,
            })
    else:
        creds_blob = encrypt_credentials({
            "dialect": req.dialect,
            "host": req.host,
            "port": req.port,
            "dbname": req.dbname,
            "user": req.user,
            "password": req.password,
        })

    async with async_session_factory() as session:
        async with session.begin():
            await session.execute(
                insert(db_connections_table).values(
                    id=connection_id,
                    session_id=req.session_id,
                    dialect=req.dialect,
                    host=req.host,
                    port=req.port,
                    dbname=req.dbname,
                    credentials_blob=creds_blob,
                    pii_masking_enabled=req.pii_masking_enabled,
                )
            )

    schema = await _introspect_schema_from_url(sync_url, req.dialect, entra_connect_args)
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

    table_names = [t["name"] for t in schema]
    return {
        "connection_id": connection_id,
        "schema_summary": {
            "table_count": len(table_names),
            "tables": table_names[:10],
        },
    }


async def disconnect_database(
    connection_id: str,
    session_id: str,
    db_connections_table: Table,
    schema_cache_table: Table,
    async_session_factory,
) -> None:
    """Remove the connection record and invalidate its schema cache."""
    async with async_session_factory() as session:
        async with session.begin():
            result = await session.execute(
                select(db_connections_table).where(
                    db_connections_table.c.id == connection_id,
                    db_connections_table.c.session_id == session_id,
                )
            )
            row = result.fetchone()
            if not row:
                raise ConnectionNotFoundError(f"Connection '{connection_id}' not found.")

            await session.execute(
                delete(schema_cache_table).where(
                    schema_cache_table.c.connection_id == connection_id
                )
            )
            await session.execute(
                delete(db_connections_table).where(
                    db_connections_table.c.id == connection_id
                )
            )


# ---------------------------------------------------------------------------
# DOCBOT-202: Schema introspection + cache
# ---------------------------------------------------------------------------


async def get_schema(
    connection_id: str,
    db_connections_table: Table,
    schema_cache_table: Table,
    async_session_factory,
) -> List[Dict[str, Any]]:
    """
    Return cached schema or re-introspect if cache has expired (24h TTL).
    """
    async with async_session_factory() as session:
        cache_result = await session.execute(
            select(schema_cache_table).where(
                schema_cache_table.c.connection_id == connection_id
            )
        )
        cache_row = cache_result.fetchone()

        now = datetime.now(timezone.utc)
        if cache_row:
            expires = cache_row.expires_at
            # Make timezone-aware if naive
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if expires > now:
                return json.loads(cache_row.schema_json)

        conn_result = await session.execute(
            select(db_connections_table).where(
                db_connections_table.c.id == connection_id
            )
        )
        conn_row = conn_result.fetchone()

    if not conn_row:
        raise ConnectionNotFoundError(f"Connection '{connection_id}' not found.")

    creds = decrypt_credentials(conn_row.credentials_blob)

    # CSV uploads go via E2B pandas — schema is always rebuilt from stored column metadata.
    if creds.get("dialect") == "csv":
        columns = creds.get("columns", [])
        table_name = creds.get("table_name", "data")
        schema = [{"name": table_name, "columns": [{"name": col, "type": "TEXT"} for col in columns]}]
        schema_json = json.dumps(schema)
        expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
        async with async_session_factory() as session:
            async with session.begin():
                if cache_row:
                    await session.execute(
                        update(schema_cache_table)
                        .where(schema_cache_table.c.connection_id == connection_id)
                        .values(schema_json=schema_json, expires_at=expires_at)
                    )
                else:
                    await session.execute(
                        insert(schema_cache_table).values(
                            connection_id=connection_id,
                            schema_json=schema_json,
                            expires_at=expires_at,
                        )
                    )
        return schema

    # For SQLite (SQLite file uploads), the file lives in /tmp which is wiped on container restart.
    # Detect this early and give the user an actionable message rather than a cryptic OperationalError.
    if creds.get("dialect") == "sqlite":
        import os as _os
        db_path = creds.get("dbname", "")
        if db_path and not _os.path.exists(db_path):
            raise ConnectionNotFoundError(
                "The uploaded file is no longer available — the server was restarted and "
                "temporary files were cleared. Please re-upload your file to continue."
            )

    import asyncio as _asyncio
    sync_url, entra_connect_args = await _asyncio.get_running_loop().run_in_executor(
        None, _resolve_connection, creds
    )
    schema = await _introspect_schema_from_url(sync_url, creds["dialect"], entra_connect_args)
    schema_json = json.dumps(schema)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=24)

    async with async_session_factory() as session:
        async with session.begin():
            if cache_row:
                await session.execute(
                    update(schema_cache_table)
                    .where(schema_cache_table.c.connection_id == connection_id)
                    .values(schema_json=schema_json, expires_at=expires_at)
                )
            else:
                await session.execute(
                    insert(schema_cache_table).values(
                        connection_id=connection_id,
                        schema_json=schema_json,
                        expires_at=expires_at,
                    )
                )

    return schema


def _get_columns_via_information_schema(engine, table_name: str) -> List[Dict[str, Any]]:
    """Fallback column introspection using information_schema (no pg_collation needed)."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = :t ORDER BY ordinal_position"
            ), {"t": table_name})
            return [{"name": row[0], "type": row[1]} for row in result]
    except Exception:
        return []


async def _introspect_schema_from_url(
    sync_url: str, dialect: str,
    entra_connect_args: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Use SQLAlchemy Inspector to return:
      [{"name": "table", "columns": [{"name": "col", "type": "VARCHAR"}]}]

    Includes both tables and views. Capped at 200 objects total.
    PostgreSQL sorts tables by row estimate for relevance.
    """
    import asyncio
    from sqlalchemy import inspect as sa_inspect

    def _sync_introspect() -> List[Dict[str, Any]]:
        connect_args: Dict[str, Any] = {}
        if dialect == "azure_sql":
            connect_args = entra_connect_args or {}
        elif dialect != "sqlite":
            connect_args["connect_timeout"] = 10

        engine = _get_or_create_engine(sync_url, connect_args)
        inspector = sa_inspect(engine)
        table_names = inspector.get_table_names()

        # Include views alongside tables
        try:
            view_names = inspector.get_view_names()
        except Exception:
            view_names = []

        if dialect == "postgresql" and len(table_names) > 50:
            table_names = _sort_tables_by_row_estimate_pg(engine, table_names)

        table_names = table_names[:200]

        schema = []
        for tname in table_names:
            try:
                columns = inspector.get_columns(tname)
                col_list = [
                    {"name": col["name"], "type": str(col["type"])}
                    for col in columns
                ]
            except Exception:
                col_list = _get_columns_via_information_schema(engine, tname)
            schema.append({"name": tname, "columns": col_list})

        # Append views (tagged with is_view for downstream awareness)
        views_cap = 200 - len(schema)
        for vname in view_names[:views_cap]:
            try:
                columns = inspector.get_columns(vname)
                col_list = [
                    {"name": col["name"], "type": str(col["type"])}
                    for col in columns
                ]
            except Exception:
                col_list = _get_columns_via_information_schema(engine, vname)
            schema.append({"name": vname, "columns": col_list, "is_view": True})

        return schema

    return await asyncio.get_running_loop().run_in_executor(None, _sync_introspect)


def _sort_tables_by_row_estimate_pg(engine, table_names: List[str]) -> List[str]:
    """Return table names sorted descending by PostgreSQL row estimate."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT relname, reltuples::bigint AS estimate "
                "FROM pg_class WHERE relkind = 'r' ORDER BY reltuples DESC"
            ))
            name_set = set(table_names)
            ordered = [row[0] for row in result if row[0] in name_set]
            remaining = [t for t in table_names if t not in set(ordered)]
            return ordered + remaining
    except Exception:
        return table_names


async def _test_connection(
    sync_url: str,
    dialect: str,
    entra_connect_args: Optional[Dict[str, Any]] = None,
) -> None:
    """Run SELECT 1 to verify credentials. Sanitises any exception before re-raising."""
    import asyncio
    from sqlalchemy import create_engine

    def _sync_test() -> None:
        connect_args: Dict[str, Any] = {}
        if dialect == "azure_sql":
            connect_args = entra_connect_args or {}
        elif dialect != "sqlite":
            connect_args["connect_timeout"] = 10

        engine = create_engine(sync_url, connect_args=connect_args)
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        finally:
            engine.dispose()

    try:
        await asyncio.get_running_loop().run_in_executor(None, _sync_test)
    except Exception as exc:
        logger.error("DB connection test failed: %s", exc)
        raise ValueError(
            "Database connection test failed. Please verify your credentials and host."
        ) from exc


# ---------------------------------------------------------------------------
# DOCBOT-203: 7-step SQL generation pipeline
# ---------------------------------------------------------------------------


async def run_sql_pipeline(
    connection_id: str,
    question: str,
    persona: str,
    db_connections_table: Table,
    schema_cache_table: Table,
    query_history_table: Table,
    query_embeddings_table: Table,
    async_session_factory,
    expert_personas: Dict[str, Any],
    session_id: Optional[str] = None,
    session_artifacts_table: Optional[Table] = None,
    table_embeddings_table: Optional[Table] = None,
    chart_type: str = "auto",
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> AsyncGenerator[str, None]:
    """
    Execute the 7-step bounded SQL pipeline and yield SSE-formatted strings.

    Yield order:
      1. One metadata chunk (JSON) with sql, explanation, preview, row_count, execution_time_ms
      2. N token chunks (streaming answer)
      3. One done chunk
    """
    start_ms = int(time.time() * 1000)

    # ── Step 0: Conversational rephrase — resolve follow-ups ─────────────
    # If chat_history is provided, rephrase the question into a standalone
    # query so table selection and SQL generation get full context.
    if chat_history:
        question = await _rephrase_with_history(question, chat_history)

    # ── CSV fast-path: bypass SQL pipeline, run pandas on E2B ────────────
    async with async_session_factory() as _sess:
        _conn_result = await _sess.execute(
            select(db_connections_table).where(db_connections_table.c.id == connection_id)
        )
        _conn_row = _conn_result.fetchone()

    if not _conn_row:
        raise ConnectionNotFoundError(f"Connection '{connection_id}' not found.")

    if _conn_row.dialect == "csv":
        _creds = decrypt_credentials(_conn_row.credentials_blob)
        # Build data profile string from credentials blob (stored during upload)
        _data_profile = _creds.get("data_profile")
        _data_profile_str: Optional[str] = None
        if _data_profile and isinstance(_data_profile, dict):
            _data_profile_str = json.dumps(_data_profile, indent=2, default=str)

        from api.sandbox_service import run_csv_query_on_e2b
        async for _chunk in run_csv_query_on_e2b(
            csv_content_b64=_creds.get("csv_content", ""),
            question=question,
            persona=persona,
            table_name=_creds.get("table_name", "data"),
            column_names=_creds.get("columns", []),
            chart_type=chart_type,
            expert_personas=expert_personas,
            sections=_creds.get("sections"),
            section_manifest=_creds.get("section_manifest"),
            chat_history=chat_history,
            data_profile_str=_data_profile_str,
        ):
            yield _chunk
        return

    # ── Step 1: Schema retrieval ──────────────────────────────────────────
    schema = await get_schema(
        connection_id, db_connections_table, schema_cache_table, async_session_factory
    )

    # DOCBOT-503: Background upsert of table embeddings (non-blocking)
    if table_embeddings_table is not None:
        import asyncio as _asyncio
        embeddings_model_for_tables = _get_embeddings_model()
        try:
            from api.utils.table_selector import upsert_table_embeddings
            _asyncio.ensure_future(upsert_table_embeddings(
                connection_id=connection_id,
                schema=schema,
                embeddings_model=embeddings_model_for_tables,
                table_embeddings_table=table_embeddings_table,
                async_session_factory=async_session_factory,
            ))
        except Exception:
            pass  # never block the pipeline

    # ── Step 2: Table selector — semantic first, LLM fallback ────────────
    selected_tables: List[str] = []

    # DOCBOT-503: Try semantic similarity against stored table embeddings first
    if table_embeddings_table is not None:
        try:
            from api.utils.table_selector import select_relevant_tables
            embeddings_model_for_q = _get_embeddings_model()
            selected_tables = await select_relevant_tables(
                question=question,
                connection_id=connection_id,
                embeddings_model=embeddings_model_for_q,
                table_embeddings_table=table_embeddings_table,
                async_session_factory=async_session_factory,
                top_k=5,
            )
        except Exception as _exc:
            logger.warning("Semantic table selection failed, using LLM: %s", _exc)

    # Fall back to LLM table selector when semantic path returns nothing
    if not selected_tables:
        selected_tables = await _select_relevant_tables(question, schema)

    schema_subset = [t for t in schema if t["name"] in set(selected_tables)]
    if not schema_subset:
        schema_subset = schema[:10]

    # ── Step 3: Few-shot retrieval ────────────────────────────────────────
    embeddings_model = _get_embeddings_model()
    q_embedding: List[float] = await _get_embedding(question, embeddings_model)
    few_shot_examples = await _retrieve_few_shot(
        connection_id, q_embedding,
        query_history_table, query_embeddings_table, async_session_factory
    )

    # ── Step 4: SQL generation (LLM call #2) ─────────────────────────────
    raw_sql = await _generate_sql(question, schema_subset, few_shot_examples)

    # ── Step 5: SQL validation (deterministic, no LLM) ───────────────────
    async with async_session_factory() as session:
        conn_result = await session.execute(
            select(db_connections_table).where(db_connections_table.c.id == connection_id)
        )
        conn_row = conn_result.fetchone()

    if not conn_row:
        raise ConnectionNotFoundError(f"Connection '{connection_id}' not found.")

    dialect = conn_row.dialect
    validated_sql = validate_and_sanitize_sql(raw_sql, dialect=dialect)

    # ── Step 6: Execute (with schema drift retry) ──────────────────────
    creds = decrypt_credentials(conn_row.credentials_blob)
    import asyncio as _asyncio
    sync_url, entra_connect_args = await _asyncio.get_running_loop().run_in_executor(
        None, _resolve_connection, creds
    )
    try:
        rows, column_names = await _execute_query(validated_sql, sync_url, dialect, entra_connect_args)
    except Exception as exec_err:
        err_msg = str(exec_err).lower()
        is_drift = any(k in err_msg for k in (
            "does not exist", "no such table", "no such column",
            "unknown table", "unknown column", "relation",
        ))
        if not is_drift:
            raise

        # Schema drift detected — invalidate cache, re-introspect, regenerate SQL, retry once
        logger.warning("Schema drift detected, refreshing schema and retrying: %s", exec_err)
        async with async_session_factory() as session:
            async with session.begin():
                await session.execute(
                    delete(schema_cache_table).where(
                        schema_cache_table.c.connection_id == connection_id
                    )
                )
        schema = await get_schema(
            connection_id, db_connections_table, schema_cache_table, async_session_factory
        )
        schema_subset = [t for t in schema if t["name"] in set(selected_tables)]
        if not schema_subset:
            schema_subset = schema[:10]
        raw_sql = await _generate_sql(question, schema_subset, few_shot_examples)
        validated_sql = validate_and_sanitize_sql(raw_sql, dialect=dialect)
        rows, column_names = await _execute_query(validated_sql, sync_url, dialect, entra_connect_args)

    execution_time_ms = int(time.time() * 1000) - start_ms

    result_dicts = [dict(zip(column_names, row)) for row in rows]

    # ── Step 6.1: PII masking (DOCBOT-604) ───────────────────────────────
    if conn_row.pii_masking_enabled:
        from api.utils.pii_masking import mask_rows, detect_pii_summary
        pii_found = detect_pii_summary(result_dicts)
        if any(pii_found.values()):
            logger.info("PII detected and masked — %s", pii_found)
        result_dicts = mask_rows(result_dicts)

    # ── Step 6.5: Python code generation + E2B sandbox (DOCBOT-301/302) ──
    # Runs for any non-empty result set; failure never blocks the main pipeline.
    # Small result sets (1–4 rows) are valid for bar/pie charts — the previous
    # >= 5 gate killed legitimate small-result visualisations.
    if len(rows) >= 1:
        try:
            from api.sandbox_service import generate_analysis_code, run_python as run_sandbox

            analysis_persona = (
                expert_personas.get(persona, expert_personas.get("Generalist", {}))
                .get("persona_def", "You are a helpful data analyst.")
            )
            analysis_code = await generate_analysis_code(
                result_dicts=result_dicts[:50],
                question=question,
                persona_def=analysis_persona,
                chart_type=chart_type,
            )
            if analysis_code:
                yield f"data: {_json_dumps({'type': 'analysis_code', 'code': analysis_code})}\n\n"
                sandbox_result = await run_sandbox(analysis_code)
                for idx, chart_b64 in enumerate(sandbox_result.charts):
                    meta = (
                        sandbox_result.chart_metadata[idx]
                        if idx < len(sandbox_result.chart_metadata)
                        else None
                    )
                    yield f"data: {_json_dumps({'type': 'chart', 'base64': chart_b64, 'index': idx, 'metadata': meta.model_dump() if meta else None})}\n\n"

                # DOCBOT-501: Persist artifact (DataFrame + first chart) for session memory
                if session_id and session_artifacts_table is not None:
                    from api.artifact_service import save_artifact
                    # Determine turn_id from row count in query_history for this connection
                    turn_id = 1
                    try:
                        async with async_session_factory() as _s:
                            _r = await _s.execute(
                                select(query_history_table)
                                .where(query_history_table.c.connection_id == connection_id)
                                .order_by(query_history_table.c.created_at.desc())
                                .limit(1)
                            )
                            latest = _r.fetchone()
                            if latest:
                                # Use rowid-based count proxy: just count rows for this connection
                                cnt_r = await _s.execute(
                                    select(query_history_table.c.id)
                                    .where(query_history_table.c.connection_id == connection_id)
                                )
                                turn_id = len(cnt_r.fetchall())
                    except Exception:
                        pass

                    first_chart = sandbox_result.charts[0] if sandbox_result.charts else None
                    await save_artifact(
                        session_id=session_id,
                        turn_id=turn_id,
                        artifact_type="sql_result",
                        name=question[:100],
                        result_dicts=result_dicts,
                        chart_b64=first_chart,
                        session_artifacts_table=session_artifacts_table,
                        async_session_factory=async_session_factory,
                    )
        except Exception as exc:
            logger.warning("Code gen/sandbox step skipped: %s", exc)

    # ── Step 7: Answer generation (LLM call #3, streaming) ───────────────
    persona_def = (
        expert_personas.get(persona, expert_personas.get("Generalist", {}))
        .get("persona_def", "You are a helpful data analyst.")
    )
    explanation = _build_explanation(validated_sql, schema_subset)

    # First chunk: metadata — mask PII in result preview
    from api.utils.pii_masking import mask_rows
    meta_chunk = {
        "type": "metadata",
        "sql_query": validated_sql,
        "explanation": explanation,
        "result_preview": mask_rows(result_dicts[:10]),
        "row_count": len(rows),
        "execution_time_ms": execution_time_ms,
        "sources": [t["name"] for t in schema_subset],
    }
    yield f"data: {_json_dumps(meta_chunk)}\n\n"

    # Persist query + embedding (non-blocking; failure is only a warning)
    query_id = str(uuid.uuid4())
    await _store_query_history(
        query_id, connection_id, question, validated_sql,
        f"{len(rows)} rows", q_embedding,
        query_history_table, query_embeddings_table, async_session_factory
    )

    # Stream answer tokens
    async for token in _stream_answer(question, validated_sql, result_dicts, persona_def):
        yield f"data: {_json_dumps({'type': 'token', 'content': token})}\n\n"

    yield f"data: {_json_dumps({'type': 'done'})}\n\n"


# ---------------------------------------------------------------------------
# Step 2 helper — table selector
# ---------------------------------------------------------------------------


async def _select_relevant_tables(question: str, schema: List[Dict[str, Any]]) -> List[str]:
    """LLM call #1: pick the relevant tables for this question.

    Shows up to 20 columns per table and tags views so the LLM knows
    they're queryable but may not support JOINs the same way.
    """
    table_list = "\n".join(
        f"- {t['name']}{' (VIEW)' if t.get('is_view') else ''}: "
        f"{', '.join(c['name'] for c in t['columns'][:20])}"
        for t in schema
    )
    prompt = (
        "You are a SQL expert. Given the following database schema and a user question, "
        "return ONLY a JSON array of the table names needed to answer the question. "
        "Include views if they contain relevant data. "
        "Return at most 8 tables. No explanation.\n\n"
        f"Schema:\n{table_list}\n\n"
        f"Question: {question}\n\n"
        "Response (JSON array only):"
    )
    try:
        from api.utils.llm_provider import chat_completion
        raw = chat_completion(
            [{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
    except Exception as exc:
        logger.warning("Table selector LLM call failed: %s", exc)

    return [t["name"] for t in schema[:10]]


# ---------------------------------------------------------------------------
# Step 3 helper — few-shot retrieval
# ---------------------------------------------------------------------------


async def _retrieve_few_shot(
    connection_id: str,
    query_embedding: List[float],
    query_history_table: Table,
    query_embeddings_table: Table,
    async_session_factory,
) -> List[Dict[str, Any]]:
    """Return up to 3 similar past queries for this connection."""
    try:
        async with async_session_factory() as session:
            result = await session.execute(
                select(
                    query_history_table.c.nl_question,
                    query_history_table.c.sql_query,
                    query_embeddings_table.c.embedding_json,
                )
                .join(
                    query_embeddings_table,
                    query_history_table.c.id == query_embeddings_table.c.query_id,
                )
                .where(query_history_table.c.connection_id == connection_id)
                .limit(100)
            )
            rows = result.fetchall()

        stored = [
            {
                "nl_question": row.nl_question,
                "sql_query": row.sql_query,
                "embedding_json": row.embedding_json,
            }
            for row in rows
        ]
        return find_similar_queries(query_embedding, stored, top_k=3)
    except Exception as exc:
        logger.warning("Few-shot retrieval failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Step 4 helper — SQL generation
# ---------------------------------------------------------------------------


async def _generate_sql(
    question: str,
    schema_subset: List[Dict[str, Any]],
    few_shot_examples: List[Dict[str, Any]],
) -> str:
    """LLM call #2: generate a SELECT query given schema + few-shot examples."""
    def _fmt_table(t: Dict[str, Any]) -> str:
        cols = ", ".join(f"{c['name']} ({c['type']})" for c in t["columns"])
        return f"Table: {t['name']}\nColumns: {cols}"

    schema_text = "\n\n".join(_fmt_table(t) for t in schema_subset)

    few_shot_text = ""
    if few_shot_examples:
        examples_str = "\n\n".join(
            f"Q: {ex['nl_question']}\nSQL: {ex['sql_query']}"
            for ex in few_shot_examples
        )
        few_shot_text = f"\n\nSimilar past queries for reference:\n{examples_str}\n"

    prompt = (
        "You are an expert SQL generator. Generate a single valid SELECT query to answer the question.\n"
        "RULES:\n"
        "1. Read-only queries ONLY — no INSERT, UPDATE, DELETE, DROP, CREATE, or any DDL/DML.\n"
        "2. Use only the tables and columns listed in the schema.\n"
        "3. Include a LIMIT clause (max 500 rows).\n"
        "4. Return ONLY the SQL query — no explanation, no markdown fences.\n\n"
        f"Schema:\n{schema_text}"
        f"{few_shot_text}\n\n"
        f"Question: {question}\n\n"
        "SQL:"
    )

    from api.utils.llm_provider import chat_completion
    sql = chat_completion(
        [{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
    )
    # Strip markdown code fences if present
    if sql.startswith("```"):
        lines = sql.split("\n")
        sql = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    return sql


# ---------------------------------------------------------------------------
# Step 6 helper — query executor (DOCBOT-205: MySQL read-only enforcement)
# ---------------------------------------------------------------------------


async def _execute_query(
    sql: str, sync_url: str, dialect: str,
    entra_connect_args: Optional[Dict[str, Any]] = None,
):
    """
    Execute *sql* with read-only enforcement, 15s timeout, 500-row cap.

    Layer 3 of the 3-layer read-only enforcement:
      - PostgreSQL: SET TRANSACTION READ ONLY + statement_timeout = 15000ms
      - MySQL:      SET SESSION TRANSACTION READ ONLY  (DOCBOT-205)
      - azure_sql:  SET TRANSACTION ISOLATION LEVEL SNAPSHOT
      - SQLite:     no write ops possible via SELECT-only validated SQL

    Returns (rows, column_names).
    """
    import asyncio

    def _sync_execute():
        connect_args: Dict[str, Any] = {}
        if dialect == "azure_sql":
            connect_args = entra_connect_args or {}
        elif dialect != "sqlite":
            connect_args["connect_timeout"] = 15

        engine = _get_or_create_engine(sync_url, connect_args)
        with engine.connect() as conn:
            if dialect == "postgresql":
                conn.execute(text("SET TRANSACTION READ ONLY"))
                conn.execute(text("SET LOCAL statement_timeout = '15000'"))
            elif dialect == "mysql":
                conn.execute(text("SET SESSION TRANSACTION READ ONLY"))
            elif dialect == "azure_sql":
                conn.execute(text("SET TRANSACTION ISOLATION LEVEL SNAPSHOT"))

            result = conn.execute(text(sql))
            rows = result.fetchmany(500)
            column_names = list(result.keys())
            return rows, column_names

    try:
        return await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(None, _sync_execute),
            timeout=15.0,
        )
    except asyncio.TimeoutError as exc:
        raise ExecutionTimeoutError("Query exceeded the 15-second time limit.") from exc
    except ExecutionTimeoutError:
        raise
    except Exception as exc:
        raise Exception(classify_db_error(exc, dialect)) from exc


# ---------------------------------------------------------------------------
# Step 7 helper — streaming answer generator
# ---------------------------------------------------------------------------


async def _stream_answer(
    question: str,
    sql: str,
    result_dicts: List[Dict[str, Any]],
    persona_def: str,
) -> AsyncGenerator[str, None]:
    """LLM call #3: stream a plain-English answer in persona voice."""
    preview = json.dumps(result_dicts[:20], default=str, indent=2)
    prompt = (
        f"{persona_def}\n\n"
        "A user asked a question about their database. Answer using the query result below as your primary data source.\n"
        "Rules:\n"
        "- Do not reveal the raw SQL unless explicitly asked.\n"
        "- Cite the data source as [DB: <table names>] at the end.\n"
        "- When the user provides explicit parameters or assumptions in their question, "
        "perform the requested calculations step by step using those values combined with the query results.\n"
        "- Use the persona's voice.\n\n"
        f"Question: {question}\n\n"
        f"Query result ({len(result_dicts)} rows):\n{preview}\n\n"
        "Answer:"
    )

    from api.utils.llm_provider import chat_completion_stream
    from api.utils.pii_masking import mask_pii
    for token in chat_completion_stream(
        [{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=800,
    ):
        yield mask_pii(token)


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------


async def _get_embedding(text_input: str, embeddings_model) -> List[float]:
    """Embed a single string into a float list via the HuggingFace endpoint."""
    import asyncio

    def _sync_embed() -> List[float]:
        return embeddings_model.embed_query(text_input)

    return await asyncio.get_running_loop().run_in_executor(None, _sync_embed)


# ---------------------------------------------------------------------------
# Query history persistence
# ---------------------------------------------------------------------------


async def _store_query_history(
    query_id: str,
    connection_id: str,
    nl_question: str,
    sql_query: str,
    result_summary: str,
    embedding: List[float],
    query_history_table: Table,
    query_embeddings_table: Table,
    async_session_factory,
) -> None:
    """Persist a successful query and its NL embedding for future few-shot retrieval."""
    try:
        async with async_session_factory() as session:
            async with session.begin():
                await session.execute(
                    insert(query_history_table).values(
                        id=query_id,
                        connection_id=connection_id,
                        nl_question=nl_question,
                        sql_query=sql_query,
                        result_summary=result_summary,
                    )
                )
                await session.execute(
                    insert(query_embeddings_table).values(
                        query_id=query_id,
                        embedding_json=embedding_to_json(embedding),
                    )
                )
    except Exception as exc:
        logger.warning("Failed to store query history: %s", exc)


# ---------------------------------------------------------------------------
# Explanation builder (deterministic, no LLM)
# ---------------------------------------------------------------------------


def _build_explanation(sql: str, schema_subset: List[Dict[str, Any]]) -> str:
    """Build a brief plain-English description of what the SQL does."""
    table_names = [t["name"] for t in schema_subset]
    tables_str = ", ".join(table_names) if table_names else "the database"
    sql_upper = sql.upper()

    parts = []
    if "GROUP BY" in sql_upper:
        parts.append("groups results")
    if "ORDER BY" in sql_upper:
        parts.append("orders by a column")
    if "JOIN" in sql_upper:
        parts.append("joins multiple tables")
    if "WHERE" in sql_upper:
        parts.append("filters rows by a condition")
    if any(agg in sql_upper for agg in ("COUNT(", "SUM(", "AVG(", "MAX(", "MIN(")):
        parts.append("computes aggregates")

    action = (", ".join(parts) + " from") if parts else "queries"
    return f"This query {action} {tables_str}."
