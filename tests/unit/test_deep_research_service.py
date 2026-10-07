"""Unit tests for api/deep_research_service.py.

All tests are CI-safe: no network calls, no API keys, no external services.
Coverage is intentionally narrow — the module is now a thin wrapper around
``deep_retrieve`` plus the ``_parse_json_list`` helper.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from api.deep_research_service import (
    MIN_CHUNKS_FOR_COVERAGE,
    _parse_json_list,
    deep_retrieve,
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


# ---------------------------------------------------------------------------
# DOCBOT-1508: per-session soft LLM cost ceiling gating
# ---------------------------------------------------------------------------


def _mock_vector_store() -> MagicMock:
    """A vector store whose retriever returns no documents for any query —
    keeps the gap-fill loop fast and deterministic in tests."""
    store = MagicMock()
    retriever = MagicMock()
    retriever.invoke.return_value = []
    store.as_retriever.return_value = retriever
    return store


class TestDeepRetrieveBudgetGate:
    def test_skips_planner_call_when_budget_exceeded(self, monkeypatch):
        """With the session already over budget, deep_retrieve must not call
        get_llm() (the sub-question decomposition call) and must fall back to
        the original question — a graceful degradation, not an error."""
        monkeypatch.setenv("groq_api_key", "test-key")

        with patch(
            "api.utils.llm_provider.is_session_budget_exceeded", return_value=True
        ), patch("api.utils.llm_provider.get_llm") as mock_get_llm:
            docs, sub_questions = asyncio.run(
                deep_retrieve(
                    "What was total revenue?",
                    _mock_vector_store(),
                    run_id="over-budget-run",
                )
            )

        mock_get_llm.assert_not_called()
        assert sub_questions == ["What was total revenue?"]
        assert docs == []

    def test_calls_planner_when_budget_not_exceeded(self, monkeypatch):
        """Sanity check: with the ceiling check mocked to False, the planner
        LLM call still happens as before (no change to the happy path)."""
        monkeypatch.setenv("groq_api_key", "test-key")

        mock_llm = MagicMock()
        with patch(
            "api.utils.llm_provider.is_session_budget_exceeded", return_value=False
        ), patch("api.utils.llm_provider.get_llm", return_value=mock_llm) as mock_get_llm, patch(
            "api.deep_research_service.ChatPromptTemplate"
        ) as mock_prompt_cls:
            # Build a fake chain whose ainvoke resolves to a JSON sub-question list.
            mock_chain = MagicMock()

            async def _ainvoke(_input):
                return '["What was total revenue?"]'

            mock_chain.ainvoke = _ainvoke
            mock_prompt_template = MagicMock()
            mock_prompt_template.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt_cls.from_messages.return_value = mock_prompt_template

            docs, sub_questions = asyncio.run(
                deep_retrieve(
                    "What was total revenue?",
                    _mock_vector_store(),
                    run_id="under-budget-run",
                )
            )

        mock_get_llm.assert_called_once()
        assert sub_questions == ["What was total revenue?"]
        assert docs == []

    def test_no_groq_key_skips_planner_regardless_of_budget(self, monkeypatch):
        """No groq_api_key at all → falls back to original question, same as
        before this ticket; the budget check is only ever reached when a key
        is configured."""
        monkeypatch.delenv("groq_api_key", raising=False)

        with patch("api.utils.llm_provider.get_llm") as mock_get_llm:
            docs, sub_questions = asyncio.run(
                deep_retrieve(
                    "What was total revenue?",
                    _mock_vector_store(),
                    run_id="no-key-run",
                )
            )

        mock_get_llm.assert_not_called()
        assert sub_questions == ["What was total revenue?"]
