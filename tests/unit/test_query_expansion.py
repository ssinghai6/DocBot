"""Unit tests for api/utils/query_expansion.py — DOCBOT-1301.

These tests make no network calls, require no API keys, and have no external
dependencies — they must always pass in CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from api.utils.query_expansion import (
    deduplicate_docs,
    expand_query,
    expand_query_llm,
)

# Terms from the old H-1B/LCA immigration synonym map that must never appear
# in finance expansions again (DOCBOT-1301 regression guard).
_IMMIGRATION_TERMS = (
    "visa",
    "h-1b",
    "h1b",
    "lca",
    "soc code",
    "nonimmigrant",
    "labor condition application",
    "prevailing wage",
    "worksite",
)


def _assert_no_immigration_leak(results: list[str]) -> None:
    lowered = " ".join(r.lower() for r in results)
    for term in _IMMIGRATION_TERMS:
        assert term not in lowered, f"Immigration term '{term}' leaked into expansion: {results}"


# ---------------------------------------------------------------------------
# expand_query — finance vocab
# ---------------------------------------------------------------------------


class TestExpandQueryFinance:
    def test_original_query_is_first(self):
        q = "What was the operating margin trend?"
        result = expand_query(q)
        assert result[0] == q

    def test_margin_triggers_finance_expansions(self):
        result = expand_query("What was the operating margin trend?")
        lowered = [r.lower() for r in result]
        assert any("margin" in s for s in lowered)
        _assert_no_immigration_leak(result)

    def test_revenue_synonym_triggered(self):
        result = expand_query("How's topline growth this quarter?")
        lowered = [r.lower() for r in result]
        assert any("revenue" in s or "sales" in s for s in lowered)
        _assert_no_immigration_leak(result)

    def test_net_income_triggers_finance_not_wage(self):
        # "income" used to trigger the HR/immigration wage cluster
        # (prevailing wage, offered wage). It must now trigger net income.
        result = expand_query("What is the net income and margin?")
        lowered = [r.lower() for r in result]
        assert any("net income" in s or "earnings" in s for s in lowered)
        _assert_no_immigration_leak(result)

    def test_ebitda_synonym_triggered(self):
        result = expand_query("What was adjusted EBITDA?")
        lowered = [r.lower() for r in result]
        assert any("ebitda" in s for s in lowered)

    def test_segment_synonym_triggered(self):
        result = expand_query("Break down revenue by segment")
        lowered = [r.lower() for r in result]
        assert any("segment" in s for s in lowered)

    def test_guidance_synonym_triggered(self):
        result = expand_query("What is the FY2026 guidance?")
        lowered = [r.lower() for r in result]
        assert any("guidance" in s or "outlook" in s or "forecast" in s for s in lowered)

    def test_balance_sheet_synonym_triggered(self):
        result = expand_query("What are total assets and cash position?")
        lowered = [r.lower() for r in result]
        assert any("assets" in s or "cash" in s or "balance sheet" in s for s in lowered)

    def test_acquisition_synonym_triggered(self):
        result = expand_query("What acquisitions were completed?")
        lowered = [r.lower() for r in result]
        assert any("acquisition" in s or "merger" in s for s in lowered)

    def test_no_duplicates_in_output(self):
        result = expand_query("What is the revenue and net income margin?")
        lowered = [r.lower().strip() for r in result]
        assert len(lowered) == len(set(lowered)), "expand_query returned duplicate entries"

    def test_unrelated_query_returns_only_original(self):
        result = expand_query("Summarise this document.")
        assert result == ["Summarise this document."]

    def test_result_is_list_of_strings(self):
        result = expand_query("What was the operating margin trend?")
        assert isinstance(result, list)
        assert all(isinstance(r, str) for r in result)

    def test_empty_string_returns_list_with_empty(self):
        result = expand_query("")
        assert result == [""]

    def test_long_query_does_not_crash(self):
        long_q = "What was total revenue and net income and margin this quarter? " * 5
        result = expand_query(long_q)
        assert result[0] == long_q
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# expand_query_llm — LLM rewrite path with deterministic fallback
# ---------------------------------------------------------------------------


class TestExpandQueryLLM:
    def test_llm_success_merges_llm_and_deterministic_expansions(self):
        mock_response = "Total revenue\nNet sales figure"
        with patch(
            "api.utils.llm_provider.chat_completion", return_value=mock_response
        ) as mock_call:
            result = expand_query_llm("What was the revenue this quarter?")

        mock_call.assert_called_once()
        assert result[0] == "What was the revenue this quarter?"
        lowered = [r.lower() for r in result]
        assert any("total revenue" in s for s in lowered)
        assert any("net sales" in s for s in lowered)
        _assert_no_immigration_leak(result)

    def test_llm_failure_falls_back_to_deterministic_no_network(self):
        with patch(
            "api.utils.llm_provider.chat_completion",
            side_effect=Exception("groq rate limit"),
        ):
            result = expand_query_llm("What was the operating margin?")

        # Falls back to expand_query() deterministic output — no exception raised.
        assert result[0] == "What was the operating margin?"
        assert result == expand_query("What was the operating margin?")

    def test_llm_missing_key_falls_back_cleanly(self):
        with patch(
            "api.utils.llm_provider.chat_completion",
            side_effect=ValueError("groq_api_key not set"),
        ):
            result = expand_query_llm("What is EBITDA?")
        assert result == expand_query("What is EBITDA?")


# ---------------------------------------------------------------------------
# deduplicate_docs (unchanged behavior — signature must stay stable)
# ---------------------------------------------------------------------------


def _make_doc(source: str, page: int, content: str):
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
        d1 = _make_doc("file.pdf", 1, "Total Revenue: $42.1M")
        d2 = _make_doc("file.pdf", 1, "Total Revenue: $42.1M")
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
        base = "A" * 90
        d1 = _make_doc("f.pdf", 1, base + "DIFFERENT_SUFFIX_1" + "X" * 200)
        d2 = _make_doc("f.pdf", 1, base + "DIFFERENT_SUFFIX_2" + "Y" * 200)
        result = deduplicate_docs([[d1], [d2]])
        assert len(result) == 1

    def test_clearly_different_content_kept(self):
        d1 = _make_doc("f.pdf", 1, "Section A: Total Revenue: $42.1M")
        d2 = _make_doc("f.pdf", 1, "Section B: Net Income: $8.3M")
        result = deduplicate_docs([[d1], [d2]])
        assert len(result) == 2
