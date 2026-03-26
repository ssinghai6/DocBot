"""Base connector interface for commerce data sources."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConnectorCredentials:
    """Validated credential bundle passed to every connector."""

    connector_type: str
    credentials: dict[str, str] = field(default_factory=dict)

    def require(self, key: str) -> str:
        """Return credential value or raise ValueError if missing."""
        value = self.credentials.get(key)
        if not value:
            raise ValueError(
                f"Missing required credential '{key}' for connector '{self.connector_type}'"
            )
        return value


class BaseConnector(abc.ABC):
    """Abstract base for all commerce data connectors."""

    def __init__(self, creds: ConnectorCredentials) -> None:
        self._creds = creds

    @property
    @abc.abstractmethod
    def connector_type(self) -> str:
        """Short identifier, e.g. 'amazon', 'shopify'."""
        ...

    @abc.abstractmethod
    async def test_connection(self) -> bool:
        """Return True if the connector can reach the remote API."""
        ...

    @abc.abstractmethod
    async def fetch_orders(self, start_date: str, end_date: str) -> list[dict[str, Any]]:
        """Fetch orders in the unified commerce schema shape."""
        ...

    @abc.abstractmethod
    async def fetch_financials(self, start_date: str, end_date: str) -> list[dict[str, Any]]:
        """Fetch financial events in the unified commerce schema shape."""
        ...
