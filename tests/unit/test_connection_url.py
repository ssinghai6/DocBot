"""DOCBOT-1608 (SEC-05, SEC-13): injection-safe URL building + SSRF / DNS-rebinding.

No network: every resolution goes through a mocked ``socket.getaddrinfo``.
"""

from __future__ import annotations

import socket
from typing import List
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from sqlalchemy.engine import make_url

from api.db_service import DBConnectionRequest, _resolve_connection
from api.utils.connection_url import (
    build_connection_url,
    build_pinned_connection_url,
    pin_address_for,
)
from api.utils.ssrf_validator import (
    is_public_address,
    odbc_escape,
    resolve_and_validate,
    resolve_pinned_ip,
    validate_db_identifier,
    validate_hostname,
)

GAI = "api.utils.ssrf_validator.socket.getaddrinfo"
PUBLIC_V4 = "93.184.216.34"
PUBLIC_V6 = "2606:2800:220:1:248:1893:25c8:1946"


def _gai(*ips: str):
    """Build a fake getaddrinfo returning *ips*."""
    def fake(host, port, *args, **kwargs) -> List[tuple]:
        out = []
        for ip in ips:
            fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
            sockaddr = (ip, port or 0, 0, 0) if fam == socket.AF_INET6 else (ip, port or 0)
            out.append((fam, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr))
        return out
    return fake


# ---------------------------------------------------------------------------
# Syntactic injection cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHostnameInjection:

    @pytest.mark.parametrize("host", [
        "db.example.com?sslmode=disable",
        "db.example.com;Database=master",
        "db.example.com/otherdb",
        "user@db.example.com",
        "db.example.com=1",
        "db.example.com host2",
        "db.example.com\t",
        "db.example.com\n",
        "db.example.com\x00",
        "db.example.com\x7f",
        "db.example.com,1434",
        "db.example.com}",
        "{db.example.com",
        "db.example.com#frag",
        "db.example.com&x=1",
        "db%2eexample.com",
        "db.example.com:5432",
        "fe80::1%eth0",
        "",
    ])
    def test_rejects_injection_chars(self, host):
        with pytest.raises(ValueError):
            validate_hostname(host)

    @pytest.mark.parametrize("host", [
        "2130706433",        # decimal 127.0.0.1
        "0x7f.1",            # hex + short form
        "0x7f000001",        # single hex
        "017700000001",      # octal
        "0177.0.0.1",        # octal dotted
        "127.1",             # short form
        "127.0.1",
        "0x0a.0.0.1",
        "4294967295",
    ])
    def test_rejects_non_canonical_numeric_ip(self, host):
        with pytest.raises(ValueError):
            validate_hostname(host)

    @pytest.mark.parametrize("host,expected", [
        ("db.example.com", "db.example.com"),
        ("my-db_1.us-east-1.rds.amazonaws.com", "my-db_1.us-east-1.rds.amazonaws.com"),
        ("8.8.8.8", "8.8.8.8"),
        ("2001:4860:4860::8888", "2001:4860:4860::8888"),
        ("[2001:4860:4860::8888]", "2001:4860:4860::8888"),
        ("db.example.com.", "db.example.com."),
    ])
    def test_accepts_valid_hosts(self, host, expected):
        assert validate_hostname(host) == expected


@pytest.mark.unit
class TestIdentifierInjection:

    @pytest.mark.parametrize("value", [
        "db?sslmode=disable", "db;drop", "db=1", "a/b", "user@host",
        "has space", "tab\there", "nl\n", "nul\x00", "x{y}", "a,b",
    ])
    def test_rejects(self, value):
        with pytest.raises(ValueError):
            validate_db_identifier(value, "Database name")

    @pytest.mark.parametrize("value", ["mydb", "my-db_2", "Prod.Analytics", "ünïcode"])
    def test_accepts(self, value):
        assert validate_db_identifier(value, "Database name") == value

    def test_empty_rejected_unless_allowed(self):
        with pytest.raises(ValueError):
            validate_db_identifier("", "Username")
        assert validate_db_identifier("", "Username", allow_empty=True) == ""


@pytest.mark.unit
class TestOdbcEscape:

    @pytest.mark.parametrize("raw,escaped", [
        ("mydb", "{mydb}"),
        ("a}b", "{a}}b}"),
        ("x;Database=master", "{x;Database=master}"),
        ("}}", "{}}}}}"),
    ])
    def test_brace_escape(self, raw, escaped):
        assert odbc_escape(raw) == escaped

    def test_rejects_control_chars(self):
        with pytest.raises(ValueError):
            odbc_escape("a\x00b")


# ---------------------------------------------------------------------------
# URL building — happy-path equality + injection
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildConnectionURL:

    @pytest.mark.parametrize("args,expected", [
        (("postgresql", "myhost.com", 5432, "mydb", "user", "pass"),
         "postgresql+psycopg2://user:pass@myhost.com:5432/mydb?sslmode=prefer"),
        (("mysql", "myhost.com", 3306, "mydb", "user", "pass"),
         "mysql+pymysql://user:pass@myhost.com:3306/mydb"),
        (("sqlite", "", 0, "/tmp/test.db", "", ""),
         "sqlite:////tmp/test.db"),
        (("sqlite", "__local_file__", 0, ":memory:", "", ""),
         "sqlite:///:memory:"),
    ])
    def test_happy_path_equality(self, args, expected):
        assert build_connection_url(*args) == expected

    def test_postgres_pinned_uses_hostaddr_keeps_hostname(self):
        url = build_connection_url("postgresql", "myhost.com", 5432, "mydb", "u", "p",
                                   pinned_ip=PUBLIC_V4)
        parsed = make_url(url)
        assert parsed.host == "myhost.com"
        assert parsed.query["hostaddr"] == PUBLIC_V4
        assert parsed.query["sslmode"] == "prefer"

    def test_mysql_pinned_connects_to_ip(self):
        url = build_connection_url("mysql", "myhost.com", 3306, "mydb", "u", "p",
                                   pinned_ip=PUBLIC_V4)
        assert make_url(url).host == PUBLIC_V4

    def test_mysql_pinned_ipv6(self):
        url = build_connection_url("mysql", "myhost.com", 3306, "mydb", "u", "p",
                                   pinned_ip=PUBLIC_V6)
        assert make_url(url).host == PUBLIC_V6

    @pytest.mark.parametrize("password", [
        "p@ss", "p?w=1&x", "a/b:c", "semi;colon", "%41", "sp ace", "#hash",
    ])
    def test_password_special_chars_roundtrip(self, password):
        url = build_connection_url("postgresql", "myhost.com", 5432, "mydb", "u", password)
        parsed = make_url(url)
        assert parsed.password == password
        assert parsed.host == "myhost.com"
        assert parsed.database == "mydb"
        assert dict(parsed.query) == {"sslmode": "prefer"}

    @pytest.mark.parametrize("dialect", ["postgresql", "mysql"])
    @pytest.mark.parametrize("field,value", [
        ("host", "evil.com?sslmode=disable"),
        ("host", "evil.com/db"),
        ("dbname", "mydb?sslmode=disable"),
        ("dbname", "mydb?hostaddr=127.0.0.1"),
        ("dbname", "../etc"),
        ("user", "u@127.0.0.1"),
        ("user", "u?x=1"),
        ("user", "u p"),
    ])
    def test_injection_rejected(self, dialect, field, value):
        kwargs = {"host": "myhost.com", "dbname": "mydb", "user": "u"}
        kwargs[field] = value
        with pytest.raises(ValueError):
            build_connection_url(dialect, kwargs["host"], 5432, kwargs["dbname"], kwargs["user"], "p")

    @pytest.mark.parametrize("port", [0, -1, 65536, True])
    def test_bad_port_rejected(self, port):
        with pytest.raises(ValueError):
            build_connection_url("postgresql", "myhost.com", port, "mydb", "u", "p")

    def test_azure_odbc_values_are_braced(self):
        url = build_connection_url("azure_sql", "srv.database.windows.net", 1433, "mydb")
        odbc = make_url(url).query["odbc_connect"]
        assert odbc == (
            "Driver={ODBC Driver 18 for SQL Server};"
            "Server={srv.database.windows.net,1433};Database={mydb};"
            "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=10;"
        )

    @pytest.mark.parametrize("dbname", ["mydb;Server=evil.com", "my}db", "db=x"])
    def test_azure_dbname_injection_rejected(self, dbname):
        with pytest.raises(ValueError):
            build_connection_url("azure_sql", "srv.database.windows.net", 1433, dbname)

    @pytest.mark.parametrize("host", [
        "evil.com", "database.windows.net", "srv.database.windows.net.evil.com",
        "srv.database.windows.net;Server=x",
    ])
    def test_azure_host_must_be_azure_endpoint(self, host):
        with pytest.raises(ValueError):
            build_connection_url("azure_sql", host, 1433, "mydb")

    @pytest.mark.parametrize("path", ["/tmp/a.db?mode=rw", "file:/etc/passwd", "/tmp/a\x00.db", "/tmp/a#b"])
    def test_sqlite_path_injection_rejected(self, path):
        with pytest.raises(ValueError):
            build_connection_url("sqlite", "", 0, path)


# ---------------------------------------------------------------------------
# SSRF — resolution validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestResolveAndValidate:

    @pytest.mark.parametrize("ip", [
        "10.0.0.1", "172.16.0.1", "192.168.1.1",
        "127.0.0.1", "0.0.0.0",
        "169.254.169.254",          # cloud metadata
        "100.64.0.1", "100.127.255.254",  # CGNAT
        "224.0.0.1",                # multicast
        "240.0.0.1",                # reserved
        "255.255.255.255",
        "::1", "::",
        "fc00::1", "fd12:3456::1",  # ULA
        "fe80::1",                  # link-local
        "::ffff:10.0.0.1",          # IPv4-mapped private
        "::ffff:127.0.0.1",
        "::ffff:169.254.169.254",
        "::127.0.0.1",              # IPv4-compatible
        "2002:7f00:0001::1",        # 6to4 wrapping 127.0.0.1
        "64:ff9b::a00:1",           # NAT64 wrapping 10.0.0.1
        "ff02::1",                  # multicast
    ])
    def test_blocked_resolved_addresses(self, ip):
        with patch(GAI, side_effect=_gai(ip)):
            with pytest.raises(ValueError, match="not allowed"):
                resolve_and_validate("db.example.com")

    def test_one_private_record_poisons_host(self):
        with patch(GAI, side_effect=_gai(PUBLIC_V4, "10.0.0.5")):
            with pytest.raises(ValueError, match="not allowed"):
                resolve_and_validate("db.example.com")

    def test_public_v4_and_v6_allowed(self):
        with patch(GAI, side_effect=_gai(PUBLIC_V4, PUBLIC_V6, PUBLIC_V4)):
            assert resolve_and_validate("db.example.com") == [PUBLIC_V4, PUBLIC_V6]

    def test_pinned_ip_prefers_ipv4(self):
        with patch(GAI, side_effect=_gai(PUBLIC_V6, PUBLIC_V4)):
            assert resolve_pinned_ip("db.example.com") == PUBLIC_V4

    def test_pinned_ip_ipv6_only(self):
        with patch(GAI, side_effect=_gai(PUBLIC_V6)):
            assert resolve_pinned_ip("db.example.com") == PUBLIC_V6

    def test_pinned_ip_private_record_still_poisons(self):
        with patch(GAI, side_effect=_gai(PUBLIC_V6, PUBLIC_V4, "10.0.0.5")):
            with pytest.raises(ValueError, match="not allowed"):
                resolve_pinned_ip("db.example.com")

    def test_zone_id_stripped(self):
        with patch(GAI, side_effect=_gai("fe80::1%en0")):
            with pytest.raises(ValueError, match="not allowed"):
                resolve_and_validate("db.example.com")

    def test_unresolvable(self):
        with patch(GAI, side_effect=socket.gaierror("nope")):
            with pytest.raises(ValueError, match="Cannot resolve"):
                resolve_and_validate("db.example.com")

    def test_empty_result(self):
        with patch(GAI, return_value=[]):
            with pytest.raises(ValueError, match="Cannot resolve"):
                resolve_and_validate("db.example.com")

    @pytest.mark.parametrize("host", ["2130706433", "0x7f.1", "017700000001"])
    def test_numeric_forms_rejected_before_dns(self, host):
        with patch(GAI) as gai:
            with pytest.raises(ValueError):
                resolve_and_validate(host)
            gai.assert_not_called()

    def test_error_messages_do_not_leak_ip(self):
        with patch(GAI, side_effect=_gai("10.1.2.3")):
            with pytest.raises(ValueError) as exc:
                resolve_and_validate("internal.example.com")
        assert "10.1.2.3" not in str(exc.value)
        assert "internal.example.com" not in str(exc.value)

    def test_is_public_address_mapped_public(self):
        import ipaddress
        assert is_public_address(ipaddress.ip_address(PUBLIC_V4))
        assert not is_public_address(ipaddress.ip_address("::ffff:192.168.0.1"))


# ---------------------------------------------------------------------------
# DNS rebinding — pinned connect uses the validated address
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDNSRebinding:

    def test_pinned_url_uses_first_validated_answer(self):
        """Rebinding sim: first lookup public, later lookups loopback. The URL
        produced from the single validated lookup pins to the public IP, so a
        driver never performs its own (rebindable) lookup."""
        answers = iter([_gai(PUBLIC_V4), _gai("127.0.0.1")])

        def rebinding(host, port, *a, **k):
            return next(answers)(host, port, *a, **k)

        with patch(GAI, side_effect=rebinding):
            pg = build_pinned_connection_url("postgresql", "rebind.example.com", 5432, "db", "u", "p")
        assert make_url(pg).query["hostaddr"] == PUBLIC_V4

    def test_rebound_to_private_on_reuse_is_rejected(self):
        """Stored creds re-resolve at query time; a host rebound to an internal
        IP after the initial connect is refused."""
        creds = {"dialect": "postgresql", "host": "rebind.example.com", "port": 5432,
                 "dbname": "db", "user": "u", "password": "p"}
        with patch(GAI, side_effect=_gai(PUBLIC_V4)):
            url, extra = _resolve_connection(creds)
        assert make_url(url).query["hostaddr"] == PUBLIC_V4
        assert extra is None
        with patch(GAI, side_effect=_gai("169.254.169.254")):
            with pytest.raises(ValueError, match="not allowed"):
                _resolve_connection(creds)

    def test_mysql_pin(self):
        with patch(GAI, side_effect=_gai(PUBLIC_V4)):
            url = build_pinned_connection_url("mysql", "db.example.com", 3306, "db", "u", "p")
        assert make_url(url).host == PUBLIC_V4

    def test_sqlite_no_dns(self):
        with patch(GAI) as gai:
            assert pin_address_for("sqlite", "__local_file__", 0) is None
            creds = {"dialect": "sqlite", "host": "__local_file__", "port": 0,
                     "dbname": "/tmp/upload.sqlite", "user": "", "password": ""}
            url, _ = _resolve_connection(creds)
            gai.assert_not_called()
        assert url == "sqlite:////tmp/upload.sqlite"

    def test_azure_validated_not_pinned(self):
        with patch(GAI, side_effect=_gai(PUBLIC_V4)):
            assert pin_address_for("azure_sql", "srv.database.windows.net", 1433) is None
        with patch(GAI, side_effect=_gai("10.0.0.4")):
            with pytest.raises(ValueError, match="not allowed"):
                pin_address_for("azure_sql", "srv.database.windows.net", 1433)


# ---------------------------------------------------------------------------
# Pydantic request model
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDBConnectionRequestValidation:

    def _req(self, **over):
        base = dict(session_id="s", dialect="postgresql", host="db.example.com",
                    port=5432, dbname="mydb", user="u", password="p")
        base.update(over)
        return DBConnectionRequest(**base)

    @pytest.mark.parametrize("over", [
        {"host": "db.example.com?sslmode=disable"},
        {"host": "2130706433"},
        {"dbname": "mydb?hostaddr=10.0.0.1"},
        {"user": "u;x"},
        {"dialect": "azure_sql", "host": "evil.com", "auth_type": "entra_interactive",
         "access_token": "a.b.c", "user": None, "password": None},
    ])
    def test_rejected(self, over):
        with patch(GAI, side_effect=_gai(PUBLIC_V4)):
            with pytest.raises(ValidationError):
                self._req(**over)

    def test_mapped_ipv6_rejected(self):
        with patch(GAI, side_effect=_gai("::ffff:10.0.0.1")):
            with pytest.raises(ValidationError):
                self._req()

    def test_accepted(self):
        with patch(GAI, side_effect=_gai(PUBLIC_V4)):
            assert self._req().host == "db.example.com"
