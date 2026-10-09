"""Static ``ToolSpec`` registrations for DocBot's built-in capabilities.

Registers, as ``ToolSpec`` entries in ``api.tools.registry``:

- 4 backend pipelines (category ``"pipeline"``): ``autopilot``, ``db_chat``,
  ``hybrid``, ``chat`` (docs).
- 3 autopilot execution tools (category ``"tool"``): ``sql_query``,
  ``doc_search``, ``python_analysis`` — the exact tool name strings used by
  ``api/autopilot_service.py``'s ``_select_tool_heuristic``.
- 6 expert personas (category ``"persona"``), sourced from
  ``api.personas.EXPERT_PERSONAS`` so the registry never drifts from the
  actual persona definitions.
- All registered marketplace connector types (category ``"connector"``),
  sourced from ``api.connectors.registry.list_connector_types()``. This does
  NOT modify ``api/connectors/base.py``'s ``BaseConnector`` ABC — it only
  wraps each already-registered connector type in a descriptor.

Call ``register_builtin_tools()`` once at import/startup time (it is
idempotent — re-registering a key just overwrites its descriptor).
"""

from __future__ import annotations

from api.tools.registry import ToolSpec, register_tool


def _register_pipelines() -> None:
    register_tool(ToolSpec(
        key="autopilot",
        name="Analytical Autopilot",
        description="Multi-step investigation that plans and runs SQL, document search, and Python analysis steps automatically.",
        category="pipeline",
        input_schema={"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
        output_schema={"type": "object"},
        cost_estimate="high",
        icon="wand",
    ))
    register_tool(ToolSpec(
        key="db_chat",
        name="Database Query",
        description="Ask a question directly against a connected database or uploaded CSV/SQLite file.",
        category="pipeline",
        input_schema={"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
        output_schema={"type": "object"},
        cost_estimate="low",
        icon="database",
    ))
    register_tool(ToolSpec(
        key="hybrid",
        name="Hybrid Docs + Database",
        description="Ask a question spanning both an uploaded document and a connected database, with discrepancy detection.",
        category="pipeline",
        input_schema={"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
        output_schema={"type": "object"},
        cost_estimate="medium",
        icon="layers",
    ))
    register_tool(ToolSpec(
        key="chat",
        name="Document Chat",
        description="Ask a question about an uploaded document using retrieval-augmented generation.",
        category="pipeline",
        input_schema={"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
        output_schema={"type": "object"},
        cost_estimate="low",
        icon="file-text",
    ))


def _register_autopilot_tools() -> None:
    # Exact tool name strings returned by
    # api/autopilot_service.py::_select_tool_heuristic.
    register_tool(ToolSpec(
        key="sql_query",
        name="SQL Query",
        description="Run a single SQL query step against a connected database.",
        category="tool",
        input_schema={"type": "object", "properties": {"step": {"type": "string"}}},
        output_schema={"type": "object"},
        icon="database",
    ))
    register_tool(ToolSpec(
        key="doc_search",
        name="Document Search",
        description="Run a single deep-retrieval document search step.",
        category="tool",
        input_schema={"type": "object", "properties": {"step": {"type": "string"}}},
        output_schema={"type": "object"},
        icon="search",
    ))
    register_tool(ToolSpec(
        key="python_analysis",
        name="Python Analysis",
        description="Run a single Python/pandas analysis step in the E2B sandbox (charts, forecasting, CSV data).",
        category="tool",
        input_schema={"type": "object", "properties": {"step": {"type": "string"}}},
        output_schema={"type": "object"},
        icon="chart",
    ))


def _register_personas() -> None:
    from api.personas import EXPERT_PERSONAS

    for name, data in EXPERT_PERSONAS.items():
        register_tool(ToolSpec(
            key=name,
            name=name,
            description=data.get("response_style", ""),
            category="persona",
            input_schema={"type": "object", "properties": {"question": {"type": "string"}}},
            output_schema={"type": "object"},
            icon="sparkles",
        ))


def _register_connectors() -> None:
    # Importing api.connectors runs its @register(...) decorators, which
    # populate api.connectors.registry's in-memory map. We only read that
    # map here — api/connectors/base.py's BaseConnector ABC is untouched.
    import api.connectors  # noqa: F401 — triggers connector registration
    from api.connectors.registry import list_connector_types

    labels = {"amazon": "Amazon SP-API", "shopify": "Shopify", "edgar": "SEC EDGAR"}
    for connector_type in list_connector_types():
        register_tool(ToolSpec(
            key=connector_type,
            name=labels.get(connector_type, connector_type.title()),
            description=f"Connect and sync data from the {labels.get(connector_type, connector_type)} marketplace connector.",
            category="connector",
            input_schema={"type": "object", "properties": {"credentials": {"type": "object"}}},
            output_schema={"type": "object"},
            icon="shopping-cart",
        ))


def register_builtin_tools() -> None:
    """Register every built-in pipeline, autopilot tool, persona, and connector."""
    _register_pipelines()
    _register_autopilot_tools()
    _register_personas()
    _register_connectors()
