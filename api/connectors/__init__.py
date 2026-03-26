"""Commerce connectors package — EPIC-07.

Importing this module registers all built-in connectors with the registry.
"""

from api.connectors.base import BaseConnector, ConnectorCredentials
from api.connectors.registry import get_connector_class, list_connector_types, register
from api.connectors.rate_limiter import RateLimiter

# Import concrete connectors so their @register decorators execute on import
from api.connectors import amazon_connector  # noqa: F401

__all__ = [
    "BaseConnector",
    "ConnectorCredentials",
    "RateLimiter",
    "get_connector_class",
    "list_connector_types",
    "register",
]
