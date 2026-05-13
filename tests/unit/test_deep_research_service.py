"""Unit tests for api/deep_research_service.py.

All tests are CI-safe: no network calls, no API keys, no external services.
Coverage is intentionally narrow — the module is now a thin wrapper around
``deep_retrieve`` plus the ``_parse_json_list`` helper.
"""

import pytest

from api.deep_research_service import (
    MIN_CHUNKS_FOR_COVERAGE,
    _parse_json_list,
)


# ---------------------------------------------------------------------------
# _parse_json_list
# ---------------------------------------------------------------------------


class TestParseJsonList:
    def test_valid_json_array(self):
        raw = '["What is the salary?", "What is the start date?"]'
        result = _parse_json_list(raw, fallback=["original"])
        assert result == ["What is the salary?", "What is the start date?"]

    def test_markdown_fenced_json(self):
        raw = '```json\n["sub-q 1", "sub-q 2"]\n```'
        result = _parse_json_list(raw, fallback=["original"])
        assert result == ["sub-q 1", "sub-q 2"]

    def test_markdown_fenced_no_lang(self):
        raw = '```\n["a", "b", "c"]\n```'
        result = _parse_json_list(raw, fallback=["original"])
        assert result == ["a", "b", "c"]

    def test_malformed_json_falls_back(self):
        raw = "this is not json at all"
        result = _parse_json_list(raw, fallback=["original"])
        assert result == ["original"]

    def test_empty_string_uses_fallback(self):
        result = _parse_json_list("", fallback=["original"])
        assert result == ["original"]

    def test_empty_array_uses_fallback(self):
        result = _parse_json_list("[]", fallback=["original"])
        assert result == ["original"]

    def test_non_list_uses_fallback(self):
        result = _parse_json_list('{"key": "value"}', fallback=["original"])
        assert result == ["original"]

    def test_filters_non_string_items(self):
        raw = '["valid", 123, null, "also valid"]'
        result = _parse_json_list(raw, fallback=["original"])
        assert result == ["valid", "also valid"]

    def test_never_raises(self):
        """_parse_json_list must not raise under any string input."""
        for bad_input in ["", "[", "}{", "\x00\xff", "null", "123"]:
            try:
                result = _parse_json_list(bad_input, fallback=["fallback"])
                assert isinstance(result, list)
            except Exception as exc:
                pytest.fail(f"_parse_json_list raised on {bad_input!r}: {exc}")


def test_min_chunks_constant_exposed():
    assert MIN_CHUNKS_FOR_COVERAGE >= 1
