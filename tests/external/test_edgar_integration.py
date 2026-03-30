"""End-to-end integration tests for the SEC EDGAR connector.

These tests hit the real SEC EDGAR API (https://www.sec.gov) and require
internet access. They are marked @pytest.mark.external and are skipped
automatically in CI.

SEC EDGAR has no API key requirement, but enforces a 10 req/s rate limit
and requires a descriptive User-Agent header (handled by the connector).

Note: SEC servers can be slow, especially during market hours.
All tests use generous timeouts to account for this.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from api.connectors.edgar_connector import EdgarConnector

logger = logging.getLogger(__name__)

# Apple Inc — well-known, stable CIK, always has 10-K filings
APPLE_TICKER = "AAPL"
APPLE_CIK = "320193"
APPLE_NAME_SUBSTRING = "Apple"

# Microsoft — used for full-flow test
MSFT_TICKER = "MSFT"
MSFT_CIK = "789019"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def connector() -> EdgarConnector:
    """Fresh EdgarConnector with cleared ticker cache."""
    EdgarConnector._ticker_cache = None
    return EdgarConnector()


@pytest.fixture(scope="module")
def event_loop():
    """Module-scoped event loop so async fixtures/tests share one loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# search_company — live SEC ticker lookup
# ---------------------------------------------------------------------------


@pytest.mark.external
@pytest.mark.asyncio
class TestSearchCompanyLive:
    """Verify company search against the real SEC ticker file."""

    @pytest.mark.timeout(30)
    async def test_search_apple_by_ticker(self, connector: EdgarConnector):
        results = await connector.search_company(APPLE_TICKER)

        assert len(results) >= 1, "Expected at least 1 result for AAPL"
        top = results[0]
        assert top["ticker"] == "AAPL"
        assert top["cik"] == APPLE_CIK
        assert APPLE_NAME_SUBSTRING.lower() in top["name"].lower()

    @pytest.mark.timeout(30)
    async def test_search_apple_by_name(self, connector: EdgarConnector):
        results = await connector.search_company("Apple")

        assert len(results) >= 1, "Expected at least 1 result for 'Apple'"
        tickers = [r["ticker"] for r in results]
        assert "AAPL" in tickers, f"AAPL not found in results: {tickers}"

    @pytest.mark.timeout(30)
    async def test_search_microsoft_by_ticker(self, connector: EdgarConnector):
        results = await connector.search_company(MSFT_TICKER)

        assert len(results) >= 1
        assert results[0]["ticker"] == "MSFT"
        assert results[0]["cik"] == MSFT_CIK

    @pytest.mark.timeout(30)
    async def test_search_returns_max_10(self, connector: EdgarConnector):
        # "A" should match many tickers — verify cap at 10
        results = await connector.search_company("A")
        assert len(results) <= 10

    @pytest.mark.timeout(30)
    async def test_search_unknown_returns_empty(self, connector: EdgarConnector):
        results = await connector.search_company("ZZZNOTREAL999")
        assert results == []


# ---------------------------------------------------------------------------
# list_filings — live SEC submissions endpoint
# ---------------------------------------------------------------------------


@pytest.mark.external
@pytest.mark.asyncio
class TestListFilingsLive:
    """Verify filing listing against real SEC EDGAR submissions data."""

    @pytest.mark.timeout(30)
    async def test_apple_10k_filings_exist(self, connector: EdgarConnector):
        filings = await connector.list_filings(APPLE_CIK, "10-K", count=5)

        assert len(filings) >= 1, "Apple should have at least one 10-K filing"

        first = filings[0]
        assert first["filing_type"] == "10-K"
        assert first["accession_number"], "accession_number must be non-empty"
        assert first["filing_date"], "filing_date must be non-empty"
        assert first["primary_document"], "primary_document must be non-empty"

    @pytest.mark.timeout(30)
    async def test_filing_date_format(self, connector: EdgarConnector):
        filings = await connector.list_filings(APPLE_CIK, "10-K", count=1)
        assert len(filings) >= 1

        date = filings[0]["filing_date"]
        # SEC dates are YYYY-MM-DD
        assert len(date) == 10, f"Unexpected date format: {date}"
        parts = date.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 4  # year

    @pytest.mark.timeout(30)
    async def test_10q_filings(self, connector: EdgarConnector):
        filings = await connector.list_filings(APPLE_CIK, "10-Q", count=3)
        assert len(filings) >= 1, "Apple should have 10-Q filings"
        for f in filings:
            assert f["filing_type"] == "10-Q"

    @pytest.mark.timeout(30)
    async def test_count_limit_respected(self, connector: EdgarConnector):
        filings = await connector.list_filings(APPLE_CIK, "10-K", count=2)
        assert len(filings) <= 2

    @pytest.mark.timeout(30)
    async def test_accession_number_format(self, connector: EdgarConnector):
        filings = await connector.list_filings(APPLE_CIK, "10-K", count=1)
        assert len(filings) >= 1

        accession = filings[0]["accession_number"]
        # Format: XXXXXXXXXX-YY-ZZZZZZ (digits-digits-digits)
        parts = accession.split("-")
        assert len(parts) == 3, f"Unexpected accession format: {accession}"


# ---------------------------------------------------------------------------
# fetch_filing_text — live filing download + HTML stripping
# ---------------------------------------------------------------------------


@pytest.mark.external
@pytest.mark.asyncio
class TestFetchFilingTextLive:
    """Download a real 10-K filing and verify text extraction."""

    @pytest.mark.timeout(120)
    async def test_fetch_apple_10k_returns_substantial_text(self, connector: EdgarConnector):
        # First, get a real filing to download
        filings = await connector.list_filings(APPLE_CIK, "10-K", count=1)
        assert len(filings) >= 1, "Need at least one 10-K to test fetch"

        filing = filings[0]
        text = await connector.fetch_filing_text(
            cik=APPLE_CIK,
            accession_number=filing["accession_number"],
            primary_document=filing["primary_document"],
        )

        assert len(text) > 1000, (
            f"10-K filing text should be substantial, got {len(text)} chars"
        )
        logger.info(
            "Fetched Apple 10-K (%s): %d chars",
            filing["filing_date"],
            len(text),
        )

    @pytest.mark.timeout(120)
    async def test_html_is_stripped(self, connector: EdgarConnector):
        filings = await connector.list_filings(APPLE_CIK, "10-K", count=1)
        assert len(filings) >= 1

        filing = filings[0]
        text = await connector.fetch_filing_text(
            cik=APPLE_CIK,
            accession_number=filing["accession_number"],
            primary_document=filing["primary_document"],
        )

        # HTML tags should be stripped by BeautifulSoup
        assert "<html" not in text.lower(), "Raw HTML tags should be stripped"
        assert "<body" not in text.lower(), "Raw HTML tags should be stripped"
        assert "<script" not in text.lower(), "Script tags should be removed"
        assert "<style" not in text.lower(), "Style tags should be removed"

    @pytest.mark.timeout(120)
    async def test_filing_contains_expected_content(self, connector: EdgarConnector):
        filings = await connector.list_filings(APPLE_CIK, "10-K", count=1)
        assert len(filings) >= 1

        filing = filings[0]
        text = await connector.fetch_filing_text(
            cik=APPLE_CIK,
            accession_number=filing["accession_number"],
            primary_document=filing["primary_document"],
        )

        # A 10-K filing should contain standard SEC sections
        text_lower = text.lower()
        assert any(
            term in text_lower
            for term in ["risk factor", "financial statement", "revenue", "item 1"]
        ), "10-K should contain standard SEC filing terms"


# ---------------------------------------------------------------------------
# Full flow — search -> list -> fetch for Microsoft
# ---------------------------------------------------------------------------


@pytest.mark.external
@pytest.mark.asyncio
class TestFullFlowLive:
    """End-to-end: search company, list filings, fetch text."""

    @pytest.mark.timeout(120)
    async def test_search_list_fetch_microsoft(self, connector: EdgarConnector):
        # Step 1: Search for Microsoft
        companies = await connector.search_company(MSFT_TICKER)
        assert len(companies) >= 1, "Microsoft should appear in search results"

        msft = companies[0]
        assert msft["ticker"] == "MSFT"
        cik = msft["cik"]
        logger.info("Found Microsoft: CIK=%s", cik)

        # Step 2: List 10-K filings
        filings = await connector.list_filings(cik, "10-K", count=2)
        assert len(filings) >= 1, "Microsoft should have 10-K filings"

        filing = filings[0]
        logger.info(
            "Most recent MSFT 10-K: %s (filed %s)",
            filing["accession_number"],
            filing["filing_date"],
        )

        # Step 3: Fetch the filing text
        text = await connector.fetch_filing_text(
            cik=cik,
            accession_number=filing["accession_number"],
            primary_document=filing["primary_document"],
        )

        assert len(text) > 1000, (
            f"Microsoft 10-K should be substantial, got {len(text)} chars"
        )

        # Verify it looks like a Microsoft filing
        text_lower = text.lower()
        assert any(
            term in text_lower
            for term in ["microsoft", "azure", "windows", "office", "cloud"]
        ), "Filing text should mention Microsoft products"

        logger.info(
            "Full flow complete: MSFT 10-K %s — %d chars extracted",
            filing["filing_date"],
            len(text),
        )

    @pytest.mark.timeout(120)
    async def test_search_list_fetch_apple(self, connector: EdgarConnector):
        # Parallel full-flow test with Apple for robustness
        companies = await connector.search_company(APPLE_TICKER)
        assert len(companies) >= 1

        apple = companies[0]
        assert apple["cik"] == APPLE_CIK

        filings = await connector.list_filings(apple["cik"], "10-K", count=1)
        assert len(filings) >= 1

        text = await connector.fetch_filing_text(
            cik=apple["cik"],
            accession_number=filings[0]["accession_number"],
            primary_document=filings[0]["primary_document"],
        )

        assert len(text) > 1000
        assert "<html" not in text.lower()

        text_lower = text.lower()
        assert any(
            term in text_lower
            for term in ["apple", "iphone", "mac", "services", "revenue"]
        ), "Filing text should mention Apple products"
