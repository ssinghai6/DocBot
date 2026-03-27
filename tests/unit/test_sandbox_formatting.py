"""Tests for sandbox_service CSV formatting helpers and prompt changes."""

import textwrap

import pytest

from api.sandbox_service import _format_stdout_as_markdown


class TestFormatStdoutAsMarkdown:
    """Tests for _format_stdout_as_markdown helper."""

    def test_empty_input_returns_empty(self):
        assert _format_stdout_as_markdown("") == ""

    def test_none_like_empty(self):
        """Empty string is falsy — function should return it unchanged."""
        result = _format_stdout_as_markdown("")
        assert result == ""

    def test_preserves_markdown_tables(self):
        """Lines with pipes (markdown table format) should pass through unchanged."""
        table = textwrap.dedent("""\
            | col_a | col_b |
            |-------|-------|
            | 1     | 2     |
            | 3     | 4     |""")
        assert _format_stdout_as_markdown(table) == table

    def test_converts_section_headers_to_bold(self):
        """'--- Something ---' patterns should become **Something**."""
        text = "--- Summary Statistics ---\nsome data here"
        result = _format_stdout_as_markdown(text)
        assert "**Summary Statistics**" in result
        assert "---" not in result.replace("**", "").strip().split("\n")[0]

    def test_plain_text_unchanged(self):
        """Regular text without pipes or section markers passes through."""
        text = "Shape: 100 rows x 5 columns\nAll good"
        assert _format_stdout_as_markdown(text) == text

    def test_mixed_content(self):
        """Mix of section headers, pipes, and plain text."""
        text = (
            "--- Overview ---\n"
            "Shape: 50 rows\n"
            "| a | b |\n"
            "|---|---|\n"
            "| 1 | 2 |"
        )
        result = _format_stdout_as_markdown(text)
        assert "**Overview**" in result
        assert "Shape: 50 rows" in result
        assert "| a | b |" in result

    def test_dashes_only_line_not_converted(self):
        """A line of just dashes (no text between) should not produce an empty bold."""
        text = "------"
        result = _format_stdout_as_markdown(text)
        # Empty header stripped — line stays as-is
        assert result == "------"


class TestCsvPromptChanges:
    """Verify the CSV code-gen prompt no longer has the overly broad skip-chart instruction."""

    def test_no_broad_skip_chart_instruction(self):
        """The old broad 'skip chart' text should be replaced with a narrow version."""
        import inspect
        from api.sandbox_service import generate_csv_analysis_code

        source = inspect.getsource(generate_csv_analysis_code)
        # Old broad instruction should be gone
        assert "If question is about schema/columns/structure: print df.dtypes and df.shape, skip chart" not in source
        # New narrow instruction should be present
        assert "ONLY skip charting if the user explicitly asks about column names" in source

    def test_prompt_requires_narrative_summary(self):
        """Prompt should instruct LLM to print narrative summary, not raw DataFrame."""
        import inspect
        from api.sandbox_service import generate_csv_analysis_code

        source = inspect.getsource(generate_csv_analysis_code)
        assert "human-readable narrative summary" in source
        assert "Do NOT print raw DataFrame output" in source
        assert "to_markdown()" in source


class TestFallbackCodeImprovements:
    """Verify fallback code includes chart generation and markdown formatting."""

    def test_fallback_includes_chart(self):
        """Fallback code should include plt.show() for histogram generation."""
        import inspect
        from api.sandbox_service import run_csv_query_on_e2b

        source = inspect.getsource(run_csv_query_on_e2b)
        assert "plt.show()" in source
        assert "histogram" in source.lower()

    def test_fallback_uses_to_markdown(self):
        """Fallback code should use to_markdown() instead of to_string()."""
        import inspect
        from api.sandbox_service import run_csv_query_on_e2b

        source = inspect.getsource(run_csv_query_on_e2b)
        assert "to_markdown()" in source
