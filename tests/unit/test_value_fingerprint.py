"""Unit tests for api.utils.value_fingerprint (DOCBOT-2911)."""

import hashlib

import pytest

from api.utils.value_fingerprint import MAX_FINGERPRINTS, fingerprint_values


@pytest.mark.unit
class TestFingerprintValues:
    def test_numbers_canonicalize(self):
        assert fingerprint_values([1]) == fingerprint_values(["1.0"]) == fingerprint_values([1.0, " 1 "])
        assert fingerprint_values(["2.50"]) == fingerprint_values([2.5])

    def test_strings_strip_and_lowercase(self):
        assert fingerprint_values([" ACME Corp "]) == fingerprint_values(["acme corp"])

    def test_hash_is_keyed_not_plain_sha256(self):
        (h,) = fingerprint_values(["Alice@Example.com"])
        assert len(h) == 16
        assert h != hashlib.sha256(b"alice@example.com").hexdigest()[:16]  # not guessable without the key
        assert "alice" not in h

    def test_key_change_changes_hash(self, monkeypatch):
        (a,) = fingerprint_values(["x"])
        monkeypatch.setenv("DB_ENCRYPTION_KEY", "another-key")
        (b,) = fingerprint_values(["x"])
        assert a != b

    def test_no_key_returns_empty(self, monkeypatch):
        monkeypatch.delenv("DB_ENCRYPTION_KEY", raising=False)
        assert fingerprint_values(["x"]) == []

    def test_nulls_skipped_and_distinct(self):
        assert fingerprint_values([None, float("nan"), "", "NaN", "x", "X"]) == fingerprint_values(["x"])

    def test_capped(self):
        assert len(fingerprint_values(range(1000))) == MAX_FINGERPRINTS

    def test_huge_exponent_does_not_expand(self):
        assert len(fingerprint_values(["1e999999999"])) == 1
