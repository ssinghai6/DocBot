"""SSRF prevention and connection-component validation — DOCBOT-201, DOCBOT-1608.

Two layers:

1. **Syntactic** (`validate_hostname`, `validate_db_identifier`): reject any
   host / database / username containing characters that could smuggle extra
   URL or DSN parameters (``? ; = / @``, whitespace, control characters), and
   reject non-canonical numeric IPv4 forms (``2130706433``, ``0x7f.1``,
   ``017700000001``) that ``inet_aton`` would silently turn into loopback.

2. **Resolution** (`resolve_and_validate`): resolve the host via
   ``getaddrinfo`` and require that *every* returned address is globally
   routable (``ipaddress.ip_address(x).is_global``), after unwrapping
   IPv4-mapped / IPv4-compatible / 6to4 / Teredo IPv6 forms. Multicast,
   unspecified (0.0.0.0 / ::), CGNAT (100.64/10), link-local (incl. cloud
   metadata 169.254.169.254), ULA (fc00::/7) and every RFC 1918 / loopback
   range fail ``is_global`` or are rejected explicitly.

The resolved, validated addresses are returned so callers can **pin** the
actual TCP connect to an address that was checked, defeating DNS rebinding
(a second resolution at connect time returning 127.0.0.1). The per-dialect
pinning strategy lives in ``api.db_service._build_connection_url``.

Error messages never echo the host or any resolved IP.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from typing import List, Optional, Union

IPAddress = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]

_BLOCKED_MSG = "Connection to private, loopback, or link-local addresses is not allowed."
_UNRESOLVABLE_MSG = "Cannot resolve host. Please check the hostname is correct."

# Characters that must never appear in a host, database name, or username that
# is placed into a URL / DSN: they delimit userinfo, path, query, ODBC pairs.
_FORBIDDEN_CHARS = frozenset("?;=/@{}#&%\\\"'`<>|,")

# A label-based DNS hostname (RFC 1123), optionally with a trailing dot.
_HOSTNAME_RE = re.compile(
    r"^(?=.{1,253}\.?$)(?:[A-Za-z0-9_](?:[A-Za-z0-9_-]{0,61}[A-Za-z0-9])?\.)*"
    r"[A-Za-z0-9_](?:[A-Za-z0-9_-]{0,61}[A-Za-z0-9])?\.?$"
)

# Dotted/undotted numeric forms that inet_aton accepts: decimal, octal (leading
# 0) or hex (0x) parts, 1-4 of them. Anything matching this that is NOT a
# canonical dotted-quad is rejected outright.
_NUMERIC_HOST_RE = re.compile(
    r"^(?:0x[0-9a-f]*|[0-9]+)(?:\.(?:0x[0-9a-f]*|[0-9]+)){0,3}\.?$",
    re.IGNORECASE,
)

_DB_IDENTIFIER_MAX = 128


def _has_control_or_space(value: str) -> bool:
    return any(ch.isspace() or ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value)


def validate_hostname(host: str) -> str:
    """Syntactically validate a host (DNS name or canonical IP literal).

    Returns the host stripped of surrounding IPv6 brackets. Raises ValueError
    for anything that could inject URL/DSN parameters or that is a
    non-canonical numeric IP form.
    """
    if not isinstance(host, str) or not host:
        raise ValueError("Host is required.")
    if _has_control_or_space(host):
        raise ValueError("Host contains whitespace or control characters.")
    candidate = host
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    if any(ch in _FORBIDDEN_CHARS for ch in candidate) or "[" in candidate or "]" in candidate:
        raise ValueError("Host contains characters that are not allowed.")

    # Canonical IP literal (dotted-quad IPv4 or IPv6, no zone id)?
    if "%" not in candidate:
        try:
            ipaddress.ip_address(candidate)
            return candidate
        except ValueError:
            pass

    if ":" in candidate:
        # Something IPv6-looking that ipaddress rejected (zone ids, port
        # suffixes, garbage) — never a valid hostname.
        raise ValueError("Host is not a valid hostname or IP address.")
    if _NUMERIC_HOST_RE.match(candidate):
        raise ValueError("Non-canonical numeric IP address forms are not allowed.")
    if not _HOSTNAME_RE.match(candidate):
        raise ValueError("Host is not a valid hostname or IP address.")
    return candidate


def validate_db_identifier(value: Optional[str], field: str, *, allow_empty: bool = False) -> str:
    """Validate a database name or username destined for a connection URL/DSN.

    Even though URLs are built with ``sqlalchemy.engine.URL.create`` (which
    escapes components), we reject delimiter characters outright so no value
    can ever be re-interpreted as a parameter by a driver that re-parses the
    DSN (libpq, ODBC).
    """
    if value is None or value == "":
        if allow_empty:
            return ""
        raise ValueError(f"{field} is required.")
    if len(value) > _DB_IDENTIFIER_MAX:
        raise ValueError(f"{field} is too long.")
    if _has_control_or_space(value):
        raise ValueError(f"{field} contains whitespace or control characters.")
    if any(ch in _FORBIDDEN_CHARS for ch in value):
        raise ValueError(f"{field} contains characters that are not allowed.")
    return value


def odbc_escape(value: str) -> str:
    """Brace-quote an ODBC connection-string value, doubling any ``}``.

    Per the ODBC spec a braced value may contain ``;`` and ``=`` literally;
    the only character needing escape inside braces is ``}`` (as ``}}``).
    """
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        raise ValueError("ODBC value contains control characters.")
    return "{" + value.replace("}", "}}") + "}"


def _embedded_ipv4(addr: ipaddress.IPv6Address) -> List[ipaddress.IPv4Address]:
    """Return any IPv4 address tunnelled inside an IPv6 address."""
    embedded: List[ipaddress.IPv4Address] = []
    if addr.ipv4_mapped is not None:
        embedded.append(addr.ipv4_mapped)
    if addr.sixtofour is not None:
        embedded.append(addr.sixtofour)
    if addr.teredo is not None:
        embedded.extend(addr.teredo)
    # IPv4-compatible (deprecated) ::a.b.c.d
    packed = addr.packed
    if packed[:12] == b"\x00" * 12 and packed[12:] not in (b"\x00\x00\x00\x00", b"\x00\x00\x00\x01"):
        embedded.append(ipaddress.IPv4Address(packed[12:]))
    # NAT64 well-known prefix 64:ff9b::/96
    if addr in ipaddress.IPv6Network("64:ff9b::/96"):
        embedded.append(ipaddress.IPv4Address(packed[12:]))
    return embedded


def is_public_address(addr: IPAddress) -> bool:
    """True only if *addr* (and any IPv4 tunnelled inside it) is globally routable."""
    if not addr.is_global:
        return False
    if addr.is_multicast or addr.is_unspecified or addr.is_reserved or addr.is_loopback:
        return False
    if isinstance(addr, ipaddress.IPv6Address):
        for inner in _embedded_ipv4(addr):
            if not is_public_address(inner):
                return False
    return True


def _parse_resolved(raw_ip: str) -> Optional[IPAddress]:
    # getaddrinfo may append an IPv6 zone id ("fe80::1%en0").
    raw_ip = raw_ip.split("%", 1)[0]
    try:
        return ipaddress.ip_address(raw_ip)
    except ValueError:
        return None


def resolve_and_validate(host: str, port: Optional[int] = None) -> List[str]:
    """Resolve *host* and return its validated public IP addresses (deduped, in order).

    Raises ValueError if the host is syntactically invalid, unresolvable, or
    if ANY resolved address is non-public (one bad record poisons the host —
    otherwise an attacker could mix a public and a private A record).
    """
    candidate = validate_hostname(host)
    try:
        resolved = socket.getaddrinfo(candidate, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError, OSError):
        raise ValueError(_UNRESOLVABLE_MSG) from None

    addresses: List[str] = []
    for info in resolved:
        sockaddr = info[4]
        if not sockaddr:
            continue
        addr = _parse_resolved(str(sockaddr[0]))
        if addr is None:
            raise ValueError(_BLOCKED_MSG)
        if not is_public_address(addr):
            raise ValueError(_BLOCKED_MSG)
        text = str(addr)
        if text not in addresses:
            addresses.append(text)

    if not addresses:
        raise ValueError(_UNRESOLVABLE_MSG)
    return addresses


def resolve_pinned_ip(host: str, port: Optional[int] = None) -> str:
    """Resolve + validate *host* and return the single address to pin the connect to."""
    return resolve_and_validate(host, port)[0]


def validate_ssrf(host: str) -> List[str]:
    """Back-compat entry point: validate *host*, raise ValueError if unsafe.

    Returns the validated addresses (callers may ignore the return value).
    """
    return resolve_and_validate(host)
