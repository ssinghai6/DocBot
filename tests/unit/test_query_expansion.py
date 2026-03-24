"""Unit tests for api/utils/query_expansion.py.

These tests make no network calls, require no API keys, and have no external
dependencies — they must always pass in CI.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from api.utils.query_expansion import expand_query, deduplicate_docs


# ---------------------------------------------------------------------------
# expand_query
# ---------------------------------------------------------------------------

class TestExpandQuery:
    def test_original_query_is_first(self):
        q = "His position or title?"
        result = expand_query(q)
        assert result[0] == q

    def test_position_title_triggers_job_title_expansions(self):
        result = expand_query("His position or title?")
        # Must include at least one job-title synonym
        lowered = [r.lower() for r in result]
        assert any("job title" in s for s in lowered), f"Expected 'job title' in {result}"

    def test_soc_synonym_triggered(self):
        result = expand_query("What is the SOC code?")
        lowered = [r.lower() for r in result]
        assert any("soc" in s for s in lowered)

    def test_salary_synonym_triggered(self):
        result = expand_query("What is his wage?")
        lowered = [r.lower() for r in result]
        assert any("wage" in s or "salary" in s or "prevailing" in s for s in lowered)

    def test_no_duplicates_in_output(self):
        result = expand_query("What is the job title position?")
        lowered = [r.lower().strip() for r in result]
        assert len(lowered) == len(set(lowered)), "expand_query returned duplicate entries"

    def test_unrelated_query_returns_only_original(self):
        # A query about something not in our synonym map should return just itself
        result = expand_query("Summarise this document.")
        assert result == ["Summarise this document."]

    def test_result_is_list_of_strings(self):
        result = expand_query("His position or title?")
        assert isinstance(result, list)
        assert all(isinstance(r, str) for r in result)

    def test_empty_string_returns_list_with_empty(self):
        result = expand_query("")
        assert result == [""]

    def test_long_query_does_not_crash(self):
        long_q = "What is the job title and role of the employee named John in Section B?" * 5
        result = expand_query(long_q)
        assert result[0] == long_q
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# deduplicate_docs
# ---------------------------------------------------------------------------

def _make_doc(source: str, page: int, content: str):
    """Create a minimal mock LangChain Document."""
    doc = MagicMock()
    doc.metadata = {"source": source, "page": page}
    doc.page_content = content
    return doc


class TestDeduplicateDocs:
    def test_empty_input_returns_empty(self):
        assert deduplicate_docs([]) == []

    def test_single_list_preserved(self):
        docs = [_make_doc("file.pdf", 1, "hello world")]
        result = deduplicate_docs([docs])
        assert len(result) == 1

    def test_exact_duplicate_removed(self):
        d1 = _make_doc("file.pdf", 1, "Job Title: Data Science Engineer")
        d2 = _make_doc("file.pdf", 1, "Job Title: Data Science Engineer")
        result = deduplicate_docs([[d1], [d2]])
        assert len(result) == 1

    def test_different_pages_kept(self):
        d1 = _make_doc("file.pdf", 1, "Section A")
        d2 = _make_doc("file.pdf", 2, "Section B")
        result = deduplicate_docs([[d1], [d2]])
        assert len(result) == 2

    def test_different_sources_kept(self):
        d1 = _make_doc("a.pdf", 1, "same content here")
        d2 = _make_doc("b.pdf", 1, "same content here")
        result = deduplicate_docs([[d1], [d2]])
        assert len(result) == 2

    def test_first_list_has_priority(self):
        d1 = _make_doc("file.pdf", 1, "first occurrence")
        d2 = _make_doc("file.pdf", 1, "first occurrence")
        result = deduplicate_docs([[d1], [d2]])
        assert result[0] is d1

    def test_order_preserved_across_lists(self):
        d1 = _make_doc("file.pdf", 1, "page one")
        d2 = _make_doc("file.pdf", 2, "page two")
        d3 = _make_doc("file.pdf", 3, "page three")
        result = deduplicate_docs([[d1, d3], [d2]])
        pages = [d.metadata["page"] for d in result]
        assert pages == [1, 3, 2]

    def test_content_prefix_used_for_dedup(self):
        # Two docs with same source/page but IDENTICAL first 100 chars are treated as the
        # same chunk (dedup collapses them to one), which protects against near-duplicate
        # chunks produced by overlapping splits.
        base = "A" * 90
        d1 = _make_doc("f.pdf", 1, base + "DIFFERENT_SUFFIX_1" + "X" * 200)
        d2 = _make_doc("f.pdf", 1, base + "DIFFERENT_SUFFIX_2" + "Y" * 200)
        result = deduplicate_docs([[d1], [d2]])
        # Same (source, page, first-100-chars) → deduplicated to 1
        assert len(result) == 1

    def test_clearly_different_content_kept(self):
        # Two docs with the same source/page but clearly different content are both kept
        d1 = _make_doc("f.pdf", 1, "Section A: Job Title: Data Science Engineer")
        d2 = _make_doc("f.pdf", 1, "Section B: SOC Code: 15-2051.00 Data Scientists")
        result = deduplicate_docs([[d1], [d2]])
        assert len(result) == 2
