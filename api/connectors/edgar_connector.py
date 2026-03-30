"""SEC EDGAR connector — fetch company filings from the SEC EDGAR API.

No API key required.  SEC asks for a descriptive User-Agent header and
limits clients to 10 requests/second.

Endpoints used:
  - Company lookup:  https://www.sec.gov/files/company_tickers.json
  - Filing list:     https://data.sec.gov/submissions/CIK{padded}.json
  - Filing download: https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx
from bs4 import BeautifulSoup

from api.connectors.base_document import BaseDocumentConnector
from api.connectors.rate_limiter import RateLimiter
from api.connectors.registry import register

logger = logging.getLogger(__name__)

_USER_AGENT = "DocBot/1.0 support@docbot.app"
_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

# SEC fair-access policy: max 10 requests/second
_rate_limiter = RateLimiter(calls_per_second=10)


def _pad_cik(cik: str) -> str:
    """Pad CIK to 10 digits with leading zeros."""
    return cik.lstrip("0").zfill(10)


def _accession_to_path(accession_number: str) -> str:
    """Convert accession number '0000320193-23-000106' to path-safe '000032019323000106'."""
    return accession_number.replace("-", "")


@register("edgar")
class EdgarConnector(BaseDocumentConnector):
    """SEC EDGAR connector for company filings (10-K, 10-Q, 8-K, etc.)."""

    # Cached ticker lookup table (loaded lazily, shared across instances)
    _ticker_cache: list[dict[str, Any]] | None = None

    @property
    def connector_type(self) -> str:
        return "edgar"

    # ── Internal helpers ─────────────────────────────────────────────────

    async def _edgar_get(self, url: str, *, timeout: float = 30.0) -> httpx.Response:
        """Rate-limited GET with SEC-required User-Agent header."""
        async with _rate_limiter:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    url,
                    headers={
                        "User-Agent": _USER_AGENT,
                        "Accept-Encoding": "gzip, deflate",
                    },
                    timeout=timeout,
                    follow_redirects=True,
                )
        if resp.status_code != 200:
            raise RuntimeError(f"EDGAR request failed {resp.status_code}: {url}")
        return resp

    async def _load_tickers(self) -> list[dict[str, Any]]:
        """Load the SEC company tickers JSON (cached after first call)."""
        if EdgarConnector._ticker_cache is not None:
            return EdgarConnector._ticker_cache

        resp = await self._edgar_get(_TICKER_URL)
        raw = resp.json()
        # SEC returns {"0": {"cik_str": ..., "ticker": ..., "title": ...}, ...}
        entries = []
        for _key, entry in raw.items():
            entries.append({
                "cik": str(entry.get("cik_str", "")),
                "ticker": str(entry.get("ticker", "")).upper(),
                "name": str(entry.get("title", "")),
            })
        EdgarConnector._ticker_cache = entries
        logger.info("edgar: loaded %d company tickers from SEC", len(entries))
        return entries

    # ── Public interface ─────────────────────────────────────────────────

    async def search_company(self, query: str) -> list[dict[str, Any]]:
        """Search for companies by ticker or name.

        Returns up to 10 matches sorted by relevance (exact ticker match first).
        """
        tickers = await self._load_tickers()
        query_upper = query.strip().upper()
        query_lower = query.strip().lower()

        exact_ticker: list[dict[str, Any]] = []
        ticker_prefix: list[dict[str, Any]] = []
        name_matches: list[dict[str, Any]] = []

        for entry in tickers:
            if entry["ticker"] == query_upper:
                exact_ticker.append(entry)
            elif entry["ticker"].startswith(query_upper):
                ticker_prefix.append(entry)
            elif query_lower in entry["name"].lower():
                name_matches.append(entry)

        results = exact_ticker + ticker_prefix + name_matches
        return results[:10]

    async def list_filings(
        self,
        cik: str,
        filing_type: str = "10-K",
        count: int = 5,
    ) -> list[dict[str, Any]]:
        """List recent filings for a CIK, filtered by filing type.

        Returns list of {accession_number, filing_type, filing_date,
        primary_document, description}.
        """
        padded = _pad_cik(cik)
        url = _SUBMISSIONS_URL.format(cik=padded)
        resp = await self._edgar_get(url)
        data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])
        descriptions = recent.get("primaryDocDescription", [])

        filings: list[dict[str, Any]] = []
        filing_type_upper = filing_type.upper()

        for i in range(len(forms)):
            if forms[i].upper() != filing_type_upper:
                continue
            filings.append({
                "accession_number": accessions[i],
                "filing_type": forms[i],
                "filing_date": dates[i],
                "primary_document": primary_docs[i] if i < len(primary_docs) else "",
                "description": descriptions[i] if i < len(descriptions) else "",
            })
            if len(filings) >= count:
                break

        return filings

    async def fetch_filing_text(
        self,
        cik: str,
        accession_number: str,
        primary_document: str,
    ) -> str:
        """Download a filing and extract clean text from HTML.

        Uses BeautifulSoup to strip HTML tags and return readable text.
        """
        padded = _pad_cik(cik)
        accession_path = _accession_to_path(accession_number)
        url = _ARCHIVES_URL.format(
            cik=padded,
            accession=accession_path,
            document=primary_document,
        )

        resp = await self._edgar_get(url, timeout=60.0)
        content_type = resp.headers.get("content-type", "")

        if "html" in content_type or primary_document.endswith((".htm", ".html")):
            soup = BeautifulSoup(resp.text, "lxml")
            # Remove script and style elements
            for tag in soup(["script", "style", "meta", "link"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        else:
            # Plain text filing
            text = resp.text

        # Collapse excessive blank lines (3+ → 2)
        text = re.sub(r"\n{3,}", "\n\n", text)
        logger.info(
            "edgar: fetched filing %s (%d chars)",
            accession_number,
            len(text),
        )
        return text
