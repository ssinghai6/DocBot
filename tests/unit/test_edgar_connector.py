"""Unit tests for EdgarConnector — all httpx calls are mocked."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.connectors.edgar_connector import (
    EdgarConnector,
    _pad_cik,
    _accession_to_path,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mock_response(data: dict | str | list, status: int = 200, content_type: str = "application/json") -> MagicMock:
    mock = MagicMock()
    mock.status_code = status
    mock.headers = {"content-type": content_type}
    if isinstance(data, str):
        mock.text = data
        mock.json.side_effect = json.JSONDecodeError("not json", "", 0)
    else:
        mock.text = json.dumps(data)
        mock.json.return_value = data
    return mock


_SAMPLE_TICKERS = {
    "0": {"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": "789019", "ticker": "MSFT", "title": "Microsoft Corp"},
    "2": {"cik_str": "1018724", "ticker": "AMZN", "title": "Amazon.com Inc"},
}

_SAMPLE_SUBMISSIONS = {
    "filings": {
        "recent": {
            "form": ["10-K", "10-Q", "8-K", "10-K", "10-Q"],
            "accessionNumber": [
                "0000320193-23-000106",
                "0000320193-23-000077",
                "0000320193-23-000050",
                "0000320193-22-000108",
                "0000320193-22-000070",
            ],
            "filingDate": ["2023-11-03", "2023-08-04", "2023-05-05", "2022-10-28", "2022-07-29"],
            "primaryDocument": [
                "aapl-20230930.htm",
                "aapl-20230701.htm",
                "aapl-20230401.htm",
                "aapl-20220924.htm",
                "aapl-20220625.htm",
            ],
            "primaryDocDescription": [
                "10-K", "10-Q", "8-K", "10-K", "10-Q",
            ],
        }
    }
}


# ── Utility function tests ───────────────────────────────────────────────────

class TestCikPadding:
    def test_pads_short_cik(self):
        assert _pad_cik("320193") == "0000320193"

    def test_already_padded(self):
        assert _pad_cik("0000320193") == "0000320193"

    def test_strips_leading_zeros_then_pads(self):
        assert _pad_cik("00320193") == "0000320193"


class TestAccessionToPath:
    def test_removes_dashes(self):
        assert _accession_to_path("0000320193-23-000106") == "000032019323000106"


# ── Registry tests ───────────────────────────────────────────────────────────

class TestEdgarRegistry:
    def test_edgar_registered(self):
        from api.connectors.registry import get_connector_class
        cls = get_connector_class("edgar")
        assert cls is EdgarConnector

    def test_connector_type_property(self):
        connector = EdgarConnector()
        assert connector.connector_type == "edgar"


# ── search_company tests ─────────────────────────────────────────────────────

class TestSearchCompany:
    def test_finds_apple_by_ticker(self):
        connector = EdgarConnector()
        # Clear class-level cache for isolated test
        EdgarConnector._ticker_cache = None

        with patch("httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=_mock_response(_SAMPLE_TICKERS))

            async def run():
                results = await connector.search_company("AAPL")
                assert len(results) >= 1
                assert results[0]["ticker"] == "AAPL"
                assert results[0]["cik"] == "320193"
                assert results[0]["name"] == "Apple Inc."

            asyncio.run(run())

    def test_finds_by_partial_name(self):
        connector = EdgarConnector()
        EdgarConnector._ticker_cache = None

        with patch("httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=_mock_response(_SAMPLE_TICKERS))

            async def run():
                results = await connector.search_company("microsoft")
                assert len(results) >= 1
                assert results[0]["ticker"] == "MSFT"

            asyncio.run(run())

    def test_empty_results_for_unknown(self):
        connector = EdgarConnector()
        EdgarConnector._ticker_cache = None

        with patch("httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=_mock_response(_SAMPLE_TICKERS))

            async def run():
                results = await connector.search_company("ZZZZNOTREAL")
                assert results == []

            asyncio.run(run())

    def test_caches_tickers_across_calls(self):
        connector = EdgarConnector()
        EdgarConnector._ticker_cache = None

        with patch("httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=_mock_response(_SAMPLE_TICKERS))

            async def run():
                await connector.search_company("AAPL")
                await connector.search_company("MSFT")
                # Second call should use cache — only 1 HTTP call
                assert instance.get.call_count == 1

            asyncio.run(run())


# ── list_filings tests ───────────────────────────────────────────────────────

class TestListFilings:
    def test_returns_10k_filings(self):
        connector = EdgarConnector()

        with patch("httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=_mock_response(_SAMPLE_SUBMISSIONS))

            async def run():
                filings = await connector.list_filings("320193", "10-K", 10)
                assert len(filings) == 2  # two 10-Ks in sample data
                assert filings[0]["filing_type"] == "10-K"
                assert filings[0]["accession_number"] == "0000320193-23-000106"

            asyncio.run(run())

    def test_respects_count_limit(self):
        connector = EdgarConnector()

        with patch("httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=_mock_response(_SAMPLE_SUBMISSIONS))

            async def run():
                filings = await connector.list_filings("320193", "10-K", 1)
                assert len(filings) == 1

            asyncio.run(run())

    def test_filters_by_filing_type(self):
        connector = EdgarConnector()

        with patch("httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=_mock_response(_SAMPLE_SUBMISSIONS))

            async def run():
                filings = await connector.list_filings("320193", "8-K", 10)
                assert len(filings) == 1
                assert filings[0]["filing_type"] == "8-K"

            asyncio.run(run())


# ── fetch_filing_text tests ──────────────────────────────────────────────────

class TestFetchFilingText:
    def test_strips_html_tags(self):
        connector = EdgarConnector()
        html = "<html><body><h1>Title</h1><p>Revenue was $1B.</p><script>alert(1)</script></body></html>"

        with patch("httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=_mock_response(
                html, content_type="text/html"
            ))

            async def run():
                text = await connector.fetch_filing_text("320193", "0000320193-23-000106", "aapl-20230930.htm")
                assert "Title" in text
                assert "Revenue was $1B." in text
                assert "<h1>" not in text
                assert "alert(1)" not in text  # script tags should be removed

            asyncio.run(run())

    def test_handles_plain_text(self):
        connector = EdgarConnector()
        plain = "This is a plain text filing.\nLine 2."

        with patch("httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=_mock_response(
                plain, content_type="text/plain"
            ))

            async def run():
                text = await connector.fetch_filing_text("320193", "0000320193-23-000106", "filing.txt")
                assert "plain text filing" in text

            asyncio.run(run())

    def test_raises_on_http_error(self):
        connector = EdgarConnector()

        with patch("httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.status_code = 404
            instance.get = AsyncMock(return_value=mock_resp)

            async def run():
                with pytest.raises(RuntimeError, match="EDGAR request failed 404"):
                    await connector.fetch_filing_text("320193", "bad-accession", "missing.htm")

            asyncio.run(run())
