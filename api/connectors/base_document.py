"""Base connector interface for document data sources (SEC EDGAR, etc.).

Separate from BaseConnector (commerce) because document sources fetch
filings/reports rather than orders/financials, and many are public APIs
that require no credentials.
"""

from __future__ import annotations

import abc
from typing import Any


class BaseDocumentConnector(abc.ABC):
    """Abstract base for connectors that fetch documents."""

    @property
    @abc.abstractmethod
    def connector_type(self) -> str:
        """Short identifier, e.g. 'edgar'."""
        ...

    @abc.abstractmethod
    async def search_company(self, query: str) -> list[dict[str, Any]]:
        """Search for companies by ticker or name.

        Returns list of {"cik": str, "name": str, "ticker": str}.
        """
        ...

    @abc.abstractmethod
    async def list_filings(
        self,
        cik: str,
        filing_type: str = "10-K",
        count: int = 5,
    ) -> list[dict[str, Any]]:
        """List available filings for a company.

        Returns list of {
            "accession_number": str,
            "filing_type": str,
            "filing_date": str,
            "primary_document": str,
        }.
        """
        ...

    @abc.abstractmethod
    async def fetch_filing_text(
        self,
        cik: str,
        accession_number: str,
        primary_document: str,
    ) -> str:
        """Download and return the full text of a filing."""
        ...
