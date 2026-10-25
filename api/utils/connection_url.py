"""Injection-safe DB connection URL construction — DOCBOT-1608 (SEC-05, SEC-13).

Every URL is built with ``sqlalchemy.engine.URL.create()`` — user input is
never concatenated or f-string-interpolated into a URL. All components are
validated first (``api.utils.ssrf_validator``) so a host / database / username
can never smuggle extra DSN or query parameters.

DNS-rebinding defence — per-dialect pinning
-------------------------------------------
The host is resolved once, every address is checked with ``is_global``, and
the actual TCP connect is pinned to a checked address so a second (attacker
controlled) resolution cannot redirect it to an internal IP:

* **postgresql (psycopg2 / libpq)** — URL keeps the original hostname (used by
  libpq for TLS SNI / ``verify-full`` and ``.pgpass``) and adds the libpq
  ``hostaddr=<ip>`` parameter, which makes libpq connect to that IP with *no*
  DNS lookup. ``hostaddr`` lives in the URL query, so it is part of the engine
  cache key: a changed IP yields a fresh engine.
* **mysql (pymysql)** — pymysql has no ``hostaddr`` equivalent and no TLS
  hostname verification is configured, so the URL host *is* the pinned IP.
* **azure_sql (pyodbc + ODBC Driver 18)** — the Azure SQL gateway routes on
  the server name in the TLS/TDS login, so connecting to a bare IP breaks
  routing. Instead the host is restricted to Microsoft-operated Azure SQL DNS
  suffixes (``*.database.windows.net`` etc.), whose records an attacker cannot
  rebind; resolution is still validated as public, and
  ``Encrypt=yes;TrustServerCertificate=no`` makes the driver verify the
  certificate against that hostname. ODBC values are brace-escaped.
* **sqlite** — no network. The path is validated (no ``?``/``#`` URI params,
  no control chars, no ``file:`` URIs). User-supplied SQLite paths are already
  refused at the ``/api/db/connect`` route; only server-generated paths
  (uploads, demo) reach this builder.
"""

from __future__ import annotations

from typing import Dict, Optional

from sqlalchemy.engine import URL

from api.utils.ssrf_validator import (
    odbc_escape,
    resolve_and_validate,
    validate_db_identifier,
    validate_hostname,
)

DIALECT_DRIVER: Dict[str, str] = {
    "postgresql": "postgresql+psycopg2",
    "mysql": "mysql+pymysql",
    "sqlite": "sqlite",
    "azure_sql": "mssql+pyodbc",
}

# Microsoft-operated Azure SQL / Synapse / Fabric endpoint suffixes (public +
# sovereign clouds).
AZURE_SQL_HOST_SUFFIXES = (
    ".database.windows.net",
    ".database.usgovcloudapi.net",
    ".database.chinacloudapi.cn",
    ".database.cloudapi.de",
    ".sql.azuresynapse.net",
    ".datawarehouse.fabric.microsoft.com",
)


def validate_port(port: int) -> int:
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("Port must be an integer between 1 and 65535.")
    return port


def validate_azure_sql_host(host: str) -> str:
    candidate = validate_hostname(host).lower().rstrip(".")
    if not any(candidate.endswith(sfx) and len(candidate) > len(sfx) for sfx in AZURE_SQL_HOST_SUFFIXES):
        raise ValueError(
            "Azure SQL host must be an Azure SQL endpoint (e.g. <server>.database.windows.net)."
        )
    return candidate


def validate_sqlite_path(path: str) -> str:
    if not isinstance(path, str) or not path:
        raise ValueError("SQLite path is required.")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in path):
        raise ValueError("SQLite path contains control characters.")
    if "?" in path or "#" in path or path.lower().startswith("file:"):
        raise ValueError("SQLite path contains characters that are not allowed.")
    return path


def build_connection_url(
    dialect: str,
    host: str,
    port: int,
    dbname: str,
    user: Optional[str] = "",
    password: Optional[str] = "",
    pinned_ip: Optional[str] = None,
) -> str:
    """Build a validated SQLAlchemy URL string for *dialect*.

    Returns the URL rendered with the password included (it is handed straight
    to ``create_engine`` and used as an in-memory engine-cache key — never log
    it). ``pinned_ip`` must come from ``resolve_and_validate`` (see
    ``pin_address_for``).
    """
    if dialect not in DIALECT_DRIVER:
        raise ValueError("Unsupported dialect.")

    if dialect == "sqlite":
        url = URL.create("sqlite", database=validate_sqlite_path(dbname))
        return url.render_as_string(hide_password=False)

    clean_host = validate_hostname(host)
    clean_port = validate_port(port)
    clean_db = validate_db_identifier(dbname, "Database name")

    if dialect == "azure_sql":
        clean_host = validate_azure_sql_host(clean_host)
        odbc_str = (
            "Driver={ODBC Driver 18 for SQL Server};"
            f"Server={odbc_escape(clean_host + ',' + str(clean_port))};"
            f"Database={odbc_escape(clean_db)};"
            "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=10;"
        )
        url = URL.create("mssql+pyodbc", query={"odbc_connect": odbc_str})
        return url.render_as_string(hide_password=False)

    clean_user = validate_db_identifier(user, "Username")
    if pinned_ip is not None:
        validate_hostname(pinned_ip)

    query: Dict[str, str] = {}
    url_host = clean_host
    if dialect == "postgresql":
        query["sslmode"] = "prefer"
        if pinned_ip:
            query["hostaddr"] = pinned_ip
    elif dialect == "mysql" and pinned_ip:
        url_host = pinned_ip

    url = URL.create(
        DIALECT_DRIVER[dialect],
        username=clean_user,
        password=password or None,
        host=url_host,
        port=clean_port,
        database=clean_db,
        query=query,
    )
    return url.render_as_string(hide_password=False)


def pin_address_for(dialect: str, host: str, port: int) -> Optional[str]:
    """Resolve + SSRF-validate *host*; return the IP to pin to (None if not pinned).

    Raises ValueError if any resolved address is non-public. SQLite does no
    network I/O; Azure SQL is validated but not pinned (see module docstring).
    """
    if dialect == "sqlite":
        return None
    if dialect == "azure_sql":
        resolve_and_validate(validate_azure_sql_host(host), port)
        return None
    return resolve_and_validate(host, port)[0]


def build_pinned_connection_url(
    dialect: str,
    host: str,
    port: int,
    dbname: str,
    user: Optional[str] = "",
    password: Optional[str] = "",
) -> str:
    """Resolve, validate, and build a URL whose connect is pinned to a checked IP."""
    pinned = pin_address_for(dialect, host, port)
    return build_connection_url(dialect, host, port, dbname, user, password, pinned_ip=pinned)
