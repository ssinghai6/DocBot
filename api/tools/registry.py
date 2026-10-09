"""Tool/capability registry — maps string keys to ``ToolSpec`` descriptors.

Mirrors the decorator/lookup pattern in ``api/connectors/registry.py``, but
registers static descriptors (``ToolSpec``) rather than connector classes,
since a "tool" here may be a backend pipeline, an autopilot tool, a persona,
or a marketplace connector type — none of which share a common base class.

Usage:
    from api.tools.registry import register_tool, get_tool, list_tools, ToolSpec

    register_tool(ToolSpec(
        key="sql_query",
        name="SQL Query",
        description="Query a connected relational database.",
        category="pipeline",
        input_schema={"type": "object", "properties": {"question": {"type": "string"}}},
        output_schema={"type": "object"},
    ))

    spec = get_tool("sql_query")
    all_pipeline_tools = list_tools(category="pipeline")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

_REGISTRY: dict[str, "ToolSpec"] = {}


@dataclass(frozen=True)
class ToolSpec:
    """Static descriptor for a tool/capability exposed to the frontend picker.

    ``category`` is one of ``"pipeline"``, ``"tool"``, ``"persona"``, or ``"connector"``
    (not an enum, to keep this module dependency-free and easy to extend).
    """

    key: str
    name: str
    description: str
    category: str
    input_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)
    cost_estimate: Optional[str] = None
    icon: Optional[str] = None


def register_tool(spec: ToolSpec) -> ToolSpec:
    """Register (or overwrite) a tool descriptor under ``spec.key``."""
    _REGISTRY[spec.key] = spec
    return spec


def get_tool(key: str) -> ToolSpec:
    """Return the registered tool descriptor or raise KeyError."""
    try:
        return _REGISTRY[key]
    except KeyError:
        available = ", ".join(_REGISTRY.keys()) or "<none>"
        raise KeyError(f"No tool registered for key '{key}'. Available: {available}")


def list_tools(category: Optional[str] = None) -> list[ToolSpec]:
    """Return all registered tool descriptors, optionally filtered by category."""
    specs = list(_REGISTRY.values())
    if category is not None:
        specs = [s for s in specs if s.category == category]
    return specs
