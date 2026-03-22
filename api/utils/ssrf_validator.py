"""SSRF prevention — blocks RFC 1918, loopback, and link-local addresses."""

import ipaddress
import socket


_BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),  # link-local
    ipaddress.ip_network("::1/128"),           # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),          # IPv6 ULA
    ipaddress.ip_network("fe80::/10"),         # IPv6 link-local
]


def validate_ssrf(host: str) -> None:
    """
    Resolve *host* and raise ValueError if any resolved address falls
    within a private, loopback, or link-local range.

    Raises:
        ValueError: with a safe message (no internal IP detail exposed).
    """
    try:
        resolved = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve host '{host}': {exc}") from exc

    for info in resolved:
        raw_ip = info[4][0]
        try:
            addr = ipaddress.ip_address(raw_ip)
        except ValueError:
            continue

        for network in _BLOCKED_NETWORKS:
            if addr in network:
                raise ValueError(
                    "Connection to private, loopback, or link-local addresses is not allowed."
                )
