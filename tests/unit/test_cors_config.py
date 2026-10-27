"""DOCBOT-1624: CORS origin normalization + boot guard."""

from __future__ import annotations

import logging

import pytest

import api.index as idx
from api.utils.cors_config import (
    UnsafeCorsConfigError,
    assert_safe_cors_config,
    parse_allowed_origins,
)


class TestParse:
    def test_strips_whitespace_trailing_slash_and_empties(self):
        raw = " https://app.example.com/ ,, https://docbot.vercel.app//,  ,http://localhost:3000"
        assert parse_allowed_origins(raw) == [
            "https://app.example.com",
            "https://docbot.vercel.app",
            "http://localhost:3000",
        ]

    def test_empty_string_gives_no_origins(self):
        assert parse_allowed_origins(" , ") == []

    def test_env_default(self, monkeypatch):
        monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
        assert parse_allowed_origins() == ["http://localhost:3000"]

    def test_reads_env(self, monkeypatch):
        monkeypatch.setenv("ALLOWED_ORIGINS", "https://a.com/, https://b.com")
        assert parse_allowed_origins() == ["https://a.com", "https://b.com"]


class TestBootGuard:
    @pytest.mark.parametrize("raw", ["*", "https://a.com, * ", "*/"])
    def test_refuses_wildcard_with_credentials(self, raw):
        with pytest.raises(UnsafeCorsConfigError):
            assert_safe_cors_config(parse_allowed_origins(raw), allow_credentials=True)

    def test_wildcard_ok_without_credentials(self):
        assert_safe_cors_config(["*"], allow_credentials=False)

    def test_logs_final_list(self, caplog):
        with caplog.at_level(logging.INFO, logger="api.utils.cors_config"):
            assert_safe_cors_config(["https://a.com"], allow_credentials=True)
        assert "['https://a.com']" in caplog.text

    async def test_lifespan_refuses_to_boot(self, monkeypatch):
        monkeypatch.setattr(idx, "ALLOWED_ORIGINS", ["*"])
        init_db = pytest.fail  # must never be reached
        monkeypatch.setattr(idx, "init_db", init_db)
        with pytest.raises(UnsafeCorsConfigError):
            async with idx.lifespan(idx.app):
                pass

    def test_app_uses_normalized_origins(self):
        assert idx.CORS_ALLOW_CREDENTIALS is True
        assert all(o == o.strip().rstrip("/") and o for o in idx.ALLOWED_ORIGINS)
