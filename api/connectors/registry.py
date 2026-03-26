"""Connector registry — maps string keys to concrete BaseConnector subclasses.

Usage:
    from api.connectors.registry import register, get_connector_class

    @register("amazon")
    class AmazonConnector(BaseConnector):
        ...

    cls = get_connector_class("amazon")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from api.connectors.base import BaseConnector

_REGISTRY: dict[str, Type["BaseConnector"]] = {}


def register(connector_type: str):
    """Class decorator that registers a connector under *connector_type*."""

    def decorator(cls: Type["BaseConnector"]) -> Type["BaseConnector"]:
        _REGISTRY[connector_type] = cls
        return cls

    return decorator


def get_connector_class(connector_type: str) -> Type["BaseConnector"]:
    """Return the registered connector class or raise KeyError."""
    try:
        return _REGISTRY[connector_type]
    except KeyError:
        available = ", ".join(_REGISTRY.keys()) or "<none>"
        raise KeyError(
            f"No connector registered for type '{connector_type}'. "
            f"Available: {available}"
        )


def list_connector_types() -> list[str]:
    """Return all registered connector type identifiers."""
    return list(_REGISTRY.keys())
