"""Unit tests for SSRF validator — DOCBOT-201."""

import pytest
from api.utils.ssrf_validator import validate_ssrf


@pytest.mark.unit
class TestSSRFValidator:

    # ── Addresses that must be blocked ──────────────────────────────────

    @pytest.mark.parametrize("host", [
        "10.0.0.1",
        "10.255.255.255",
        "172.16.0.1",
        "172.31.255.255",
        "192.168.0.1",
        "192.168.255.255",
    ])
    def test_blocks_rfc1918_addresses(self, host):
        with pytest.raises(ValueError, match="not allowed"):
            validate_ssrf(host)

    @pytest.mark.parametrize("host", [
        "127.0.0.1",
        "127.0.0.2",
        "localhost",
    ])
    def test_blocks_loopback(self, host):
        with pytest.raises(ValueError):
            validate_ssrf(host)

    def test_blocks_link_local(self):
        with pytest.raises(ValueError):
            validate_ssrf("169.254.1.1")

    # ── Non-resolvable host ──────────────────────────────────────────────

    def test_raises_on_unresolvable_host(self):
        with pytest.raises(ValueError, match="Cannot resolve"):
            validate_ssrf("this.host.does.not.exist.invalid")

    # ── Public addresses that must pass ─────────────────────────────────

    @pytest.mark.parametrize("host", [
        "8.8.8.8",        # Google DNS
        "1.1.1.1",        # Cloudflare
    ])
    def test_allows_public_ip(self, host):
        validate_ssrf(host)  # Must not raise
