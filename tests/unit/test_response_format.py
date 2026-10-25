"""Unit tests for api/utils/response_format.py — DOCBOT-2001."""

from __future__ import annotations

import inspect
import random

import pytest

from api.personas import EXPERT_PERSONAS
from api.utils.response_format import (
    CITATION_FORMATS,
    FORMAT_CONTRACT_V1,
    KNOWN_SOURCES,
    SOURCE_CSV,
    SOURCE_DB,
    SOURCE_DOC,
    SOURCE_EDGAR,
    build_format_block,
    doc_sources_from_metadata,
    normalize_markdown,
    unclosed_fence_suffix,
)

pytestmark = pytest.mark.unit


# ── build_format_block ──────────────────────────────────────────────────────


class TestFormatContract:
    def test_contract_core_rules(self):
        c = FORMAT_CONTRACT_V1
        assert "GitHub-flavored Markdown" in c
        assert "raw HTML" in c
        assert "images" in c
        assert "6 columns" in c
        assert "$4.2M" in c and "$1.3B" in c
        assert "formula once" in c
        assert "LaTeX" in c and "$$...$$" in c

    def test_empty_sources_has_no_citation_rules(self):
        block = build_format_block(set())
        assert block == FORMAT_CONTRACT_V1
        for fmt in CITATION_FORMATS.values():
            assert fmt not in block

    def test_none_sources(self):
        assert build_format_block(None) == FORMAT_CONTRACT_V1

    @pytest.mark.parametrize("source", KNOWN_SOURCES)
    def test_single_source_only_its_citation(self, source):
        block = build_format_block({source})
        assert block.startswith(FORMAT_CONTRACT_V1)
        assert CITATION_FORMATS[source] in block
        for other, fmt in CITATION_FORMATS.items():
            if other != source:
                assert fmt not in block

    def test_citation_syntax(self):
        block = build_format_block(set(KNOWN_SOURCES))
        assert "[Source: <filename> p.<page>]" in block
        assert "[DB: <table>]" in block
        assert "[CSV: <file>]" in block
        assert "[EDGAR: <form>/<period>]" in block

    def test_order_is_deterministic(self):
        a = build_format_block([SOURCE_EDGAR, SOURCE_DOC, SOURCE_DB])
        b = build_format_block({SOURCE_DB, SOURCE_DOC, SOURCE_EDGAR})
        assert a == b
        assert a.index("[Source:") < a.index("[DB:") < a.index("[EDGAR:")

    def test_unknown_sources_ignored_and_case_insensitive(self):
        assert build_format_block({"bogus"}) == FORMAT_CONTRACT_V1
        assert CITATION_FORMATS[SOURCE_CSV] in build_format_block({"CSV"})

    def test_doc_sources_from_metadata(self):
        assert doc_sources_from_metadata([]) == {SOURCE_DOC}
        assert doc_sources_from_metadata([{"source": "a.pdf"}]) == {SOURCE_DOC}
        assert doc_sources_from_metadata([{"filing_type": "10-K"}]) == {SOURCE_EDGAR}
        assert doc_sources_from_metadata(
            [{"source": "a.pdf"}, {"filing_type": "10-Q"}, None]
        ) == {SOURCE_DOC, SOURCE_EDGAR}


# ── normalize_markdown ──────────────────────────────────────────────────────


class TestNormalizeMarkdown:
    def test_empty(self):
        assert normalize_markdown("") == ""

    def test_clean_text_unchanged(self):
        text = "## Summary\n\nRevenue grew **12%** to $4.2M.\n\n- one\n- two"
        assert normalize_markdown(text) == text

    def test_crlf_normalized(self):
        assert normalize_markdown("a\r\nb") == "a\nb"

    # HTML
    def test_strips_inline_html(self):
        assert normalize_markdown("Net <b>income</b> was <span style=\"x\">$3M</span>.") == (
            "Net income was $3M."
        )

    def test_block_html_becomes_line_breaks(self):
        out = normalize_markdown("<p>First</p><p>Second</p>")
        assert "<" not in out
        assert out.split("\n\n") == ["First", "Second"]

    def test_br_inside_table_row_becomes_space(self):
        out = normalize_markdown("| a | b |\n| --- | --- |\n| x<br>y | z |")
        assert "| x y | z |" in out

    def test_strips_comments_script_and_images(self):
        out = normalize_markdown(
            "a <!-- hidden --> b <script>alert(1)</script>c <img src=\"x.png\" alt=\"y\">"
        )
        assert out == "a  b c"

    def test_preserves_non_html_angle_brackets(self):
        text = "Cite as [DB: <table names>]; note a < b > c and x<5."
        assert normalize_markdown(text) == text

    def test_preserves_html_in_inline_code_and_fences(self):
        text = "Use `<div>` here.\n\n```html\n<div><b>x</b></div>\n```"
        assert normalize_markdown(text) == text

    # Fences
    def test_closes_unclosed_fence(self):
        out = normalize_markdown("Code:\n\n```python\nprint(1)\n")
        assert out == "Code:\n\n```python\nprint(1)\n```"

    def test_closes_unclosed_tilde_fence_with_same_marker(self):
        out = normalize_markdown("~~~~\nx")
        assert out.endswith("\n~~~~")

    def test_fence_contents_untouched(self):
        text = "```\n# not a heading\n| a |\n\n\n\n- x\n```"
        assert normalize_markdown(text) == text

    def test_unclosed_fence_suffix(self):
        assert unclosed_fence_suffix("a\n```py\nx") == "\n```"
        assert unclosed_fence_suffix("a\n```py\nx\n") == "```"
        assert unclosed_fence_suffix("```\nx\n```") == ""
        assert unclosed_fence_suffix("inline ```code``` only") == ""
        assert unclosed_fence_suffix("") == ""
        fixed = "a\n```py\nx" + unclosed_fence_suffix("a\n```py\nx")
        assert unclosed_fence_suffix(fixed) == ""

    # Block spacing
    def test_blank_line_before_heading_list_table(self):
        out = normalize_markdown(
            "Intro\n## Head\nText\n- a\n- b\nMore\n| h1 | h2 |\n| --- | --- |\n| 1 | 2 |\nAfter"
        )
        assert out == (
            "Intro\n\n## Head\n\nText\n\n- a\n- b\nMore\n\n"
            "| h1 | h2 |\n| --- | --- |\n| 1 | 2 |\n\nAfter"
        )

    def test_no_blank_between_list_items(self):
        text = "1. one\n2. two\n   detail\n3. three"
        assert normalize_markdown(text) == text

    def test_collapses_blank_runs(self):
        assert normalize_markdown("a\n\n\n\n\nb\n\n\nc") == "a\n\nb\n\nc"

    def test_math_block_untouched(self):
        text = "Formula:\n\n$$\n- x + y\n$$"
        assert normalize_markdown(text) == text

    # Tables
    def test_inserts_missing_separator(self):
        out = normalize_markdown("| Metric | Value |\n| Revenue | $4.2M |")
        assert out == "| Metric | Value |\n| --- | --- |\n| Revenue | $4.2M |"

    def test_repairs_short_separator(self):
        out = normalize_markdown("| a | b | c |\n|---|---\n| 1 | 2 | 3 |")
        assert out.split("\n")[1] == "| --- | --- | --- |"

    def test_repairs_bad_separator_chars_and_keeps_alignment(self):
        out = normalize_markdown("| a | b |\n|:==| —:|\n| 1 | 2 |")
        assert out.split("\n")[1] == "| :--- | ---: |"

    def test_single_pipe_line_left_alone(self):
        assert normalize_markdown("| just a line |") == "| just a line |"

    # Idempotence
    @pytest.mark.parametrize(
        "text",
        [
            "Here:<br>\n## H\ntext\n- a\n| x | y |\n|--|\n| 1 | 2 |\nend\n\n\n\n<div>z</div>",
            "```py\nunclosed",
            "<<b>div>nested</div>",
            "a\n```\ncode\n```\n# h\n| t |\n| u |",
            "- a\nlazy\n- b\n\n\n1. x",
        ],
    )
    def test_idempotent_examples(self, text):
        once = normalize_markdown(text)
        assert normalize_markdown(once) == once

    def test_idempotent_fuzz(self):
        atoms = [
            "# H", "## Sub", "text line", "- item", "1. item", "  indented", "", "",
            "| a | b |", "|---|---|", "| 1 | 2 |", "|:--|--:", "```", "~~~", "```py",
            "$$", "x = 1", "<br>", "<p>para</p>", "a<b>bold</b>", "`<i>`",
            "<!-- c -->", "| x <br> y |", "> quote", "---", "***",
        ]
        rng = random.Random(2001)
        for _ in range(3000):
            doc = "\n".join(rng.choice(atoms) for _ in range(rng.randint(1, 14)))
            once = normalize_markdown(doc)
            assert normalize_markdown(once) == once, repr(doc)


# ── Dedupe guard: old copy-pasted rules must not reappear ────────────────────

_LEGACY_RULE_MARKERS = (
    "Formatting: Answer directly and naturally",
    "Format large numbers concisely",
    "FORMAT RULES:",
    "Write formulas as LaTeX math",
    "Page X]",
    "STRUCTURE YOUR RESPONSE:",
    "Use markdown headers only when the response genuinely covers",
)


class TestNoDuplicatedRules:
    @pytest.mark.parametrize("name", sorted(EXPERT_PERSONAS))
    def test_persona_prompts_have_no_legacy_format_rules(self, name):
        persona_def = EXPERT_PERSONAS[name]["persona_def"]
        for marker in _LEGACY_RULE_MARKERS:
            assert marker not in persona_def, f"{name} still contains {marker!r}"
        assert "OUTPUT FORMAT RULES" not in persona_def

    def test_service_modules_have_no_legacy_format_rules(self):
        from api import autopilot_service, db_service, hybrid_service

        for module in (autopilot_service, hybrid_service, db_service):
            src = inspect.getsource(module)
            for marker in _LEGACY_RULE_MARKERS:
                assert marker not in src, f"{module.__name__} still contains {marker!r}"
            assert "build_format_block" in src

    def test_required_section_contract_still_injected(self):
        assert "## Executive Summary" in EXPERT_PERSONAS["Finance Expert"]["persona_def"]
        assert "OUTPUT FORMAT CONTRACT" in EXPERT_PERSONAS["Lawyer"]["persona_def"]
