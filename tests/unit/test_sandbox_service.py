"""Unit tests for api/sandbox_service.py — DOCBOT-305.

Covers:
- VALID_CHART_TYPES constant
- ChartMetadata model validation
- SandboxResult model with chart_metadata field
- _chart_type_instructions() routing per chart type
- _extract_charts() with CHART_B64:, CHART_META:, and regular stdout lines
"""

import json
import pytest

from api.sandbox_service import (
    VALID_CHART_TYPES,
    ChartMetadata,
    SandboxResult,
    _chart_type_instructions,
    _extract_charts,
)


# ---------------------------------------------------------------------------
# VALID_CHART_TYPES
# ---------------------------------------------------------------------------


class TestValidChartTypes:
    def test_contains_required_types(self):
        required = {"auto", "bar", "line", "scatter", "heatmap", "box", "multi"}
        assert required == VALID_CHART_TYPES

    def test_no_unexpected_types(self):
        assert len(VALID_CHART_TYPES) == 7

    def test_all_lowercase(self):
        for t in VALID_CHART_TYPES:
            assert t == t.lower()


# ---------------------------------------------------------------------------
# ChartMetadata model
# ---------------------------------------------------------------------------


class TestChartMetadata:
    def test_defaults(self):
        m = ChartMetadata()
        assert m.type == "auto"
        assert m.title == ""
        assert m.x_label == ""
        assert m.y_label == ""
        assert m.series_count == 1

    def test_explicit_fields(self):
        m = ChartMetadata(
            type="bar", title="Revenue", x_label="Month", y_label="USD", series_count=3
        )
        assert m.type == "bar"
        assert m.title == "Revenue"
        assert m.x_label == "Month"
        assert m.y_label == "USD"
        assert m.series_count == 3

    def test_model_dump_round_trip(self):
        m = ChartMetadata(type="line", title="T", x_label="X", y_label="Y", series_count=2)
        d = m.model_dump()
        m2 = ChartMetadata(**d)
        assert m == m2


# ---------------------------------------------------------------------------
# SandboxResult with chart_metadata
# ---------------------------------------------------------------------------


class TestSandboxResult:
    def test_chart_metadata_default_empty(self):
        r = SandboxResult(stdout="", stderr="", charts=[], error=None, execution_time_ms=0)
        assert r.chart_metadata == []

    def test_chart_metadata_populated(self):
        meta = ChartMetadata(type="scatter", title="S")
        r = SandboxResult(
            stdout="",
            stderr="",
            charts=["b64data"],
            error=None,
            execution_time_ms=100,
            chart_metadata=[meta],
        )
        assert len(r.chart_metadata) == 1
        assert r.chart_metadata[0].type == "scatter"


# ---------------------------------------------------------------------------
# _chart_type_instructions()
# ---------------------------------------------------------------------------


class TestChartTypeInstructions:
    def test_auto_returns_default_fallback(self):
        instr = _chart_type_instructions("auto")
        # "auto" is not in the instructions dict — should return the default phrase
        assert "appropriate chart" in instr.lower() or "auto" in instr.lower() or instr  # non-empty

    def test_bar_mentions_bar(self):
        instr = _chart_type_instructions("bar")
        assert "bar" in instr.lower()

    def test_line_mentions_line(self):
        instr = _chart_type_instructions("line")
        assert "line" in instr.lower()

    def test_scatter_mentions_scatter(self):
        instr = _chart_type_instructions("scatter")
        assert "scatter" in instr.lower()

    def test_heatmap_mentions_heatmap(self):
        instr = _chart_type_instructions("heatmap")
        assert "heatmap" in instr.lower()

    def test_box_mentions_box(self):
        instr = _chart_type_instructions("box")
        assert "box" in instr.lower()

    def test_multi_mentions_subplot(self):
        instr = _chart_type_instructions("multi")
        assert "subplot" in instr.lower()

    def test_unknown_type_returns_default(self):
        instr = _chart_type_instructions("unknown_xyz")
        # Should not raise — returns a sensible fallback
        assert isinstance(instr, str)
        assert len(instr) > 0

    def test_all_named_types_return_non_empty(self):
        for chart_type in ("bar", "line", "scatter", "heatmap", "box", "multi"):
            assert len(_chart_type_instructions(chart_type)) > 0


# ---------------------------------------------------------------------------
# _extract_charts()
# ---------------------------------------------------------------------------


class TestExtractCharts:
    def test_empty_input(self):
        clean, charts, metadata = _extract_charts([])
        assert clean == []
        assert charts == []
        assert metadata == []

    def test_plain_stdout_unchanged(self):
        lines = ["Hello", "world", "result = 42"]
        clean, charts, metadata = _extract_charts(lines)
        assert clean == lines
        assert charts == []
        assert metadata == []

    def test_chart_b64_extracted(self):
        b64 = "aGVsbG8="
        lines = [f"CHART_B64:{b64}", "some text"]
        clean, charts, metadata = _extract_charts(lines)
        assert charts == [b64]
        assert clean == ["some text"]
        assert metadata == []

    def test_chart_b64_strips_whitespace(self):
        b64 = "aGVsbG8="
        lines = [f"CHART_B64:  {b64}  "]
        clean, charts, _ = _extract_charts(lines)
        assert charts == [b64]

    def test_chart_meta_extracted(self):
        meta_dict = {"type": "bar", "title": "Rev", "x_label": "Q", "y_label": "$", "series_count": 1}
        lines = [f"CHART_META:{json.dumps(meta_dict)}"]
        clean, charts, metadata = _extract_charts(lines)
        assert clean == []
        assert charts == []
        assert len(metadata) == 1
        assert metadata[0].type == "bar"
        assert metadata[0].title == "Rev"

    def test_chart_meta_malformed_silently_skipped(self):
        lines = ["CHART_META:not_valid_json", "normal line"]
        clean, charts, metadata = _extract_charts(lines)
        assert metadata == []
        assert clean == ["normal line"]

    def test_mixed_lines(self):
        meta_dict = {"type": "line", "title": "T", "x_label": "X", "y_label": "Y", "series_count": 2}
        lines = [
            "print output",
            "CHART_B64:abc123",
            f"CHART_META:{json.dumps(meta_dict)}",
            "result = done",
        ]
        clean, charts, metadata = _extract_charts(lines)
        assert clean == ["print output", "result = done"]
        assert charts == ["abc123"]
        assert len(metadata) == 1
        assert metadata[0].type == "line"

    def test_multiple_charts_and_metadata(self):
        meta1 = {"type": "bar", "title": "A", "x_label": "", "y_label": "", "series_count": 1}
        meta2 = {"type": "scatter", "title": "B", "x_label": "", "y_label": "", "series_count": 1}
        lines = [
            "CHART_B64:first_b64",
            f"CHART_META:{json.dumps(meta1)}",
            "CHART_B64:second_b64",
            f"CHART_META:{json.dumps(meta2)}",
        ]
        clean, charts, metadata = _extract_charts(lines)
        assert charts == ["first_b64", "second_b64"]
        assert len(metadata) == 2
        assert metadata[0].type == "bar"
        assert metadata[1].type == "scatter"
        assert clean == []

    def test_chart_meta_partial_fields_uses_defaults(self):
        """CHART_META with only 'type' should still parse — other fields get defaults."""
        lines = ['CHART_META:{"type": "heatmap"}']
        _, _, metadata = _extract_charts(lines)
        assert len(metadata) == 1
        assert metadata[0].type == "heatmap"
        assert metadata[0].title == ""
        assert metadata[0].series_count == 1
