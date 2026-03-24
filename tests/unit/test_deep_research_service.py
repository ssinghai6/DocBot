"""Unit tests for api/deep_research_service.py.

All tests are CI-safe: no network calls, no API keys, no external services.
"""

import pytest

from api.deep_research_service import (
    MIN_CHUNKS_FOR_COVERAGE,
    DeepResearchState,
    _parse_json_list,
    gap_router,
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
        # falls back to line-splitting or fallback
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_empty_string_uses_fallback(self):
        result = _parse_json_list("", fallback=["original"])
        assert result == ["original"]

    def test_numbered_list_parsed(self):
        raw = "1. What is the job title?\n2. What is the wage level?\n3. When does it start?"
        result = _parse_json_list(raw, fallback=["original"])
        assert len(result) == 3
        assert "What is the job title?" in result

    def test_bullet_list_parsed(self):
        raw = "- What is the job title?\n- What is the wage level?"
        result = _parse_json_list(raw, fallback=["original"])
        assert len(result) == 2

    def test_never_raises(self):
        """_parse_json_list must not raise under any input."""
        for bad_input in [None, 123, [], {}, "\x00\xff"]:
            try:
                result = _parse_json_list(str(bad_input), fallback=["fallback"])
                assert isinstance(result, list)
            except Exception as exc:
                pytest.fail(f"_parse_json_list raised on {bad_input!r}: {exc}")


# ---------------------------------------------------------------------------
# gap_router
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> DeepResearchState:
    base: DeepResearchState = {
        "question": "test",
        "session_id": "s1",
        "persona_def": "You are helpful.",
        "vector_store": None,
        "sub_questions": ["q1", "q2", "q3"],
        "retrieved_chunks": {},
        "gaps": [],
        "iterations": 0,
        "final_answer": "",
        "citations": [],
    }
    base.update(overrides)
    return base


class TestGapRouter:
    def test_loops_when_gaps_and_under_limit(self):
        state = _make_state(gaps=["q3"], iterations=0)
        assert gap_router(state) == "parallel_retriever"

    def test_proceeds_when_no_gaps(self):
        state = _make_state(gaps=[], iterations=0)
        assert gap_router(state) == "synthesizer"

    def test_proceeds_when_iterations_at_max(self):
        state = _make_state(gaps=["q3"], iterations=2)
        assert gap_router(state) == "synthesizer"

    def test_proceeds_when_iterations_over_max(self):
        state = _make_state(gaps=["q1", "q2"], iterations=5)
        assert gap_router(state) == "synthesizer"

    def test_loops_when_gaps_and_iterations_one(self):
        state = _make_state(gaps=["q1"], iterations=1)
        assert gap_router(state) == "parallel_retriever"
