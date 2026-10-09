"""Unit tests for the tool/capability registry — DOCBOT-1514."""

import pytest

from api.tools.registry import ToolSpec, register_tool, get_tool, list_tools, _REGISTRY


@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot and restore the module-level registry around each test so
    tests never leak state into each other or into builtin registrations.
    """
    snapshot = dict(_REGISTRY)
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


def _make_spec(key: str = "sql_query", category: str = "tool") -> ToolSpec:
    return ToolSpec(
        key=key,
        name="SQL Query",
        description="Run a SQL query.",
        category=category,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )


@pytest.mark.unit
class TestRegisterTool:

    def test_register_returns_the_spec(self):
        spec = _make_spec()
        result = register_tool(spec)
        assert result is spec

    def test_register_overwrites_existing_key(self):
        register_tool(_make_spec(key="dup"))
        second = _make_spec(key="dup", category="persona")
        register_tool(second)
        assert get_tool("dup").category == "persona"


@pytest.mark.unit
class TestGetTool:

    def test_get_returns_registered_spec(self):
        spec = _make_spec(key="doc_search")
        register_tool(spec)
        assert get_tool("doc_search") is spec

    def test_get_missing_key_raises_key_error(self):
        with pytest.raises(KeyError):
            get_tool("does-not-exist")

    def test_get_missing_key_error_lists_available(self):
        register_tool(_make_spec(key="python_analysis"))
        with pytest.raises(KeyError, match="python_analysis"):
            get_tool("does-not-exist")


@pytest.mark.unit
class TestListTools:

    def test_list_empty_registry(self):
        assert list_tools() == []

    def test_list_returns_all_without_filter(self):
        register_tool(_make_spec(key="sql_query", category="tool"))
        register_tool(_make_spec(key="Finance Expert", category="persona"))
        assert len(list_tools()) == 2

    def test_list_filters_by_category(self):
        register_tool(_make_spec(key="sql_query", category="tool"))
        register_tool(_make_spec(key="Finance Expert", category="persona"))
        register_tool(_make_spec(key="amazon", category="connector"))

        personas = list_tools(category="persona")
        assert len(personas) == 1
        assert personas[0].key == "Finance Expert"

    def test_list_filter_with_no_matches_returns_empty(self):
        register_tool(_make_spec(key="sql_query", category="tool"))
        assert list_tools(category="connector") == []
