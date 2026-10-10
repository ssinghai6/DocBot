"""Unit tests for api/hybrid_service.py — DOCBOT-401.

All tests are fully synchronous from the test runner's perspective; async
functions are driven by pytest-asyncio.  No real network calls are made: the
Groq client is replaced by an AsyncMock throughout.
"""

from __future__ import annotations

import asyncio
import json

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.hybrid_service import (
    IntentClassification,
    classify_intent,
    classify_intent_safe,
    hybrid_chat,
    _answer_addresses_discrepancies,
    _hash_question,
)
from api.utils.discrepancy_detector import DiscrepancyItem, DiscrepancyReport


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


# ---------------------------------------------------------------------------
# _answer_addresses_discrepancies — DOCBOT-1518 cheap substring gate check
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnswerAddressesDiscrepancies:
    def test_discrepancy_marker_is_detected(self):
        assert _answer_addresses_discrepancies(
            "Revenue was $330M. [DISCREPANCY] DB shows $325M."
        )

    def test_word_discrepancy_is_detected_case_insensitive(self):
        assert _answer_addresses_discrepancies("There is a Discrepancy between sources.")

    def test_plain_answer_is_not_addressed(self):
        assert not _answer_addresses_discrepancies("Revenue grew to $330M in Q4.")

    def test_empty_answer_is_not_addressed(self):
        assert not _answer_addresses_discrepancies("")


# ---------------------------------------------------------------------------
# hybrid_chat() — DOCBOT-1518 active discrepancy gate
#
# Drives the full generator with classify_intent_safe / rag_retrieve /
# _collect_sql_result / detect_discrepancies / the streaming + non-streaming
# LLM helpers all mocked, so only the gate logic itself (added by this
# ticket) is under test.
# ---------------------------------------------------------------------------


def _make_discrepancy_report(has_discrepancy: bool) -> DiscrepancyReport:
    if not has_discrepancy:
        return DiscrepancyReport()
    item = DiscrepancyItem(
        label="revenue",
        doc_value=330.0,
        db_value=325.0,
        delta=-5.0,
        pct=-1.5,
        doc_snippet="Revenue: $330M",
        db_snippet="revenue=325",
    )
    return DiscrepancyReport(discrepancies=[item], checked_pairs=1)


async def _drain_hybrid_chat(**overrides) -> list[dict]:
    """Run hybrid_chat() to completion and return the parsed SSE events."""
    kwargs = dict(
        question="What is Q4 revenue, and does it match the filing?",
        session_id="sess-disc",
        connection_id="conn-1",
        persona="Generalist",
        has_docs=True,
        messages_table=MagicMock(),
        sessions_table=MagicMock(),
        db_connections_table=MagicMock(),
        schema_cache_table=MagicMock(),
        query_history_table=MagicMock(),
        query_embeddings_table=MagicMock(),
        async_session_factory=MagicMock(),
        expert_personas={"Generalist": {"persona_def": "You are a helpful data analyst."}},
        vector_stores={},
    )
    kwargs.update(overrides)

    events: list[dict] = []
    async for chunk in hybrid_chat(**kwargs):
        for line in chunk.strip().splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    return events


_SQL_METADATA = {
    "sql_query": "SELECT revenue FROM financials WHERE quarter = 'Q4'",
    "result_preview": [{"revenue": 325}],
    "row_count": 1,
    "sources": ["financials"],
    "chart_events": [],
    "analysis_code_event": None,
}


@pytest.mark.unit
class TestHybridChatActiveDiscrepancyGate:
    def _run(self, *, has_discrepancy: bool, stream_tokens: list[str]):
        classification = IntentClassification(
            intent="hybrid", fallback_applied=False, question_hash="x" * 16
        )
        with patch(
            "api.hybrid_service.classify_intent_safe",
            new=AsyncMock(return_value=classification),
        ), patch(
            "api.hybrid_service.rag_retrieve",
            new=AsyncMock(return_value=("Revenue: $330M per the 10-K.", [])),
        ), patch(
            "api.hybrid_service._collect_sql_result",
            new=AsyncMock(return_value=dict(_SQL_METADATA)),
        ), patch(
            "api.utils.discrepancy_detector.detect_discrepancies",
            new=MagicMock(return_value=_make_discrepancy_report(has_discrepancy)),
        ), patch(
            "api.utils.llm_provider.chat_completion_stream",
            new=MagicMock(return_value=iter(stream_tokens)),
        ) as mock_stream, patch(
            "api.utils.llm_provider.chat_completion",
            new=MagicMock(return_value="Revenue was $330M per the filing. [DISCREPANCY] DB shows $325M."),
        ) as mock_correction, patch(
            "api.trace_service.log_trace",
            new=AsyncMock(return_value="trace-id-1"),
        ), patch(
            "api.lineage_service.emit_lineage",
            new=MagicMock(return_value=None),
        ):
            events = asyncio.run(_drain_hybrid_chat())
        return events, mock_stream, mock_correction

    def test_discrepancy_not_addressed_triggers_one_resynthesis(self):
        """detect_discrepancies finds a real delta, the streamed answer never
        mentions it -> exactly one corrective re-synthesis call is made and
        its output is appended as an extra token event."""
        events, mock_stream, mock_correction = self._run(
            has_discrepancy=True,
            stream_tokens=["Revenue grew to $330M in Q4."],
        )

        mock_correction.assert_called_once()
        token_events = [e for e in events if e.get("type") == "token"]
        combined = "".join(e["content"] for e in token_events)
        assert "[DISCREPANCY]" in combined

    def test_discrepancy_already_addressed_is_unchanged(self):
        """The streamed answer already surfaces the discrepancy -> no
        corrective call is made, and the tokens are exactly what streamed."""
        events, mock_stream, mock_correction = self._run(
            has_discrepancy=True,
            stream_tokens=[
                "Revenue was $330M per the filing. [DISCREPANCY] DB shows $325M."
            ],
        )

        mock_correction.assert_not_called()
        token_events = [e for e in events if e.get("type") == "token"]
        combined = "".join(e["content"] for e in token_events)
        assert combined == "Revenue was $330M per the filing. [DISCREPANCY] DB shows $325M."

    def test_no_discrepancy_detected_is_unchanged(self):
        """detect_discrepancies finds nothing -> gate never fires, no
        corrective call is made regardless of answer content."""
        events, mock_stream, mock_correction = self._run(
            has_discrepancy=False,
            stream_tokens=["Revenue grew to $330M in Q4."],
        )

        mock_correction.assert_not_called()
        token_events = [e for e in events if e.get("type") == "token"]
        combined = "".join(e["content"] for e in token_events)
        assert combined == "Revenue grew to $330M in Q4."
