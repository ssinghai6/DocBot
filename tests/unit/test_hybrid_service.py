"""Unit tests for api/hybrid_service.py — DOCBOT-401.

All tests are fully synchronous from the test runner's perspective; async
functions are driven by pytest-asyncio.  No real network calls are made: the
Groq client is replaced by an AsyncMock throughout.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.hybrid_service import (
    IntentClassification,
    classify_intent,
    classify_intent_safe,
    _hash_question,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_groq_response(content: str) -> MagicMock:
    """Build a minimal mock that looks like a groq ChatCompletion response."""
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


def _make_groq_client(content: str = "hybrid") -> AsyncMock:
    """Return an AsyncMock groq client whose completions.create returns *content*."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_make_groq_response(content)
    )
    return client


# A no-op async session factory used wherever DB logging is not under test.
_noop_session_factory = MagicMock()


# ---------------------------------------------------------------------------
# classify_intent — fallback rules
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestClassifyIntentFallbacks:

    async def test_no_db_returns_doc(self):
        """When has_db=False the classifier must return 'doc' without calling the LLM."""
        client = _make_groq_client()
        result = await classify_intent(
            question="What does the contract say about termination?",
            has_db=False,
            has_docs=True,
            session_id="sess-1",
            groq_client=client,
            async_session_factory=_noop_session_factory,
        )
        assert result == "doc"
        client.chat.completions.create.assert_not_called()

    async def test_no_docs_returns_sql(self):
        """When has_docs=False the classifier must return 'sql' without calling the LLM."""
        client = _make_groq_client()
        result = await classify_intent(
            question="How many orders were placed last month?",
            has_db=True,
            has_docs=False,
            session_id="sess-2",
            groq_client=client,
            async_session_factory=_noop_session_factory,
        )
        assert result == "sql"
        client.chat.completions.create.assert_not_called()

    async def test_both_false_returns_doc(self):
        """When neither source is available, has_db=False fires first → 'doc'."""
        client = _make_groq_client()
        result = await classify_intent(
            question="Anything at all",
            has_db=False,
            has_docs=False,
            session_id="sess-3",
            groq_client=client,
            async_session_factory=_noop_session_factory,
        )
        assert result == "doc"
        client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# classify_intent — LLM path
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestClassifyIntentLLM:

    async def test_llm_returns_sql(self):
        """LLM response 'sql' must be passed through as-is."""
        client = _make_groq_client("sql")
        result = await classify_intent(
            question="How many customers signed up this week?",
            has_db=True,
            has_docs=True,
            session_id="sess-4",
            groq_client=client,
            async_session_factory=_noop_session_factory,
        )
        assert result == "sql"
        client.chat.completions.create.assert_awaited_once()

    async def test_llm_returns_hybrid(self):
        """LLM response 'hybrid' must be passed through as-is."""
        client = _make_groq_client("hybrid")
        result = await classify_intent(
            question="Compare the policy document to the latest sales figures.",
            has_db=True,
            has_docs=True,
            session_id="sess-5",
            groq_client=client,
            async_session_factory=_noop_session_factory,
        )
        assert result == "hybrid"

    async def test_llm_returns_doc(self):
        """LLM response 'doc' must be passed through as-is."""
        client = _make_groq_client("doc")
        result = await classify_intent(
            question="Summarise the introduction chapter.",
            has_db=True,
            has_docs=True,
            session_id="sess-6",
            groq_client=client,
            async_session_factory=_noop_session_factory,
        )
        assert result == "doc"

    async def test_llm_unexpected_value_defaults_to_hybrid(self):
        """Any value other than sql/doc/hybrid must default to 'hybrid'."""
        for unexpected in ("maybe", "both", "", "  ", "none"):
            client = _make_groq_client(unexpected)
            result = await classify_intent(
                question="Some ambiguous question",
                has_db=True,
                has_docs=True,
                session_id="sess-7",
                groq_client=client,
                async_session_factory=_noop_session_factory,
            )
            assert result == "hybrid", f"Expected 'hybrid' for LLM output {unexpected!r}"

    async def test_llm_response_whitespace_stripped(self):
        """Leading/trailing whitespace in LLM output must be handled gracefully."""
        client = _make_groq_client("  sql  ")
        result = await classify_intent(
            question="Show me revenue by region",
            has_db=True,
            has_docs=True,
            session_id="sess-8",
            groq_client=client,
            async_session_factory=_noop_session_factory,
        )
        assert result == "sql"


# ---------------------------------------------------------------------------
# classify_intent_safe
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestClassifyIntentSafe:

    async def test_returns_intent_classification_model(self):
        """classify_intent_safe must return an IntentClassification instance."""
        client = _make_groq_client("sql")
        result = await classify_intent_safe(
            question="How many rows in orders?",
            has_db=True,
            has_docs=False,
            session_id="sess-9",
            groq_client=client,
            async_session_factory=_noop_session_factory,
        )
        assert isinstance(result, IntentClassification)

    async def test_fallback_applied_true_when_one_source_missing(self):
        """fallback_applied must be True when only one source is available."""
        client = _make_groq_client()
        result = await classify_intent_safe(
            question="Some question",
            has_db=False,
            has_docs=True,
            session_id="sess-10",
            groq_client=client,
            async_session_factory=_noop_session_factory,
        )
        assert result.fallback_applied is True
        assert result.intent == "doc"

    async def test_fallback_applied_false_when_both_sources_present(self):
        """fallback_applied must be False when both sources are available."""
        client = _make_groq_client("sql")
        result = await classify_intent_safe(
            question="Revenue this quarter?",
            has_db=True,
            has_docs=True,
            session_id="sess-11",
            groq_client=client,
            async_session_factory=_noop_session_factory,
        )
        assert result.fallback_applied is False
        assert result.intent == "sql"

    async def test_llm_exception_returns_hybrid_does_not_raise(self):
        """When the LLM call raises, classify_intent_safe must return 'hybrid' silently."""
        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("Groq service unavailable")
        )

        result = await classify_intent_safe(
            question="What is the churn rate?",
            has_db=True,
            has_docs=True,
            session_id="sess-12",
            groq_client=client,
            async_session_factory=_noop_session_factory,
        )
        assert result.intent == "hybrid"
        assert isinstance(result, IntentClassification)

    async def test_question_hash_is_16_hex_chars(self):
        """question_hash must be a 16-character hex string."""
        client = _make_groq_client("doc")
        result = await classify_intent_safe(
            question="Is this clause enforceable?",
            has_db=True,
            has_docs=True,
            session_id="sess-13",
            groq_client=client,
            async_session_factory=_noop_session_factory,
        )
        assert len(result.question_hash) == 16
        assert all(c in "0123456789abcdef" for c in result.question_hash)


# ---------------------------------------------------------------------------
# _hash_question
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHashQuestion:

    def test_deterministic(self):
        assert _hash_question("hello") == _hash_question("hello")

    def test_different_inputs_different_hashes(self):
        assert _hash_question("question A") != _hash_question("question B")

    def test_length_is_16(self):
        assert len(_hash_question("any question")) == 16
