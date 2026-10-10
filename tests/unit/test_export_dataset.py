"""Unit tests for finetune/export_dataset.py — DOCBOT-1523.

Pure-logic tests against ``build_examples_for_trace`` (no DB, no network) —
mirrors this repo's convention that tests exercising only deterministic
functions live in tests/unit/, even though the module under test lives
outside api/ (finetune/ is standalone offline ML tooling, not FastAPI
runtime code — see CLAUDE.md).
"""

import json

from finetune.export_dataset import (
    _extract_corrected_tool,
    _infer_source_flags,
    _parse_tools_used,
    build_examples_for_trace,
)


def _trace(**overrides):
    base = {
        "id": "trace-1",
        "question": "Forecast next quarter's revenue based on trend.",
        "tool_chosen": "sql_query",
    }
    base.update(overrides)
    return base


def _judgment(**overrides):
    base = {
        "trace_id": "trace-1",
        "tool_selection_verdict": "correct",
        "tool_selection_reason": "sql_query was a reasonable choice.",
        "judged_at": "2026-01-01T00:00:00Z",
    }
    base.update(overrides)
    return base


class TestCorrectVerdict:
    def test_correct_verdict_produces_positive_example(self):
        trace = _trace(tool_chosen="sql_query")
        judgment = _judgment(tool_selection_verdict="correct")

        examples, reasons = build_examples_for_trace(trace, judgment)

        assert reasons == []
        assert len(examples) == 1
        messages = examples[0]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Forecast next quarter's revenue" in messages[1]["content"]
        assert messages[2]["role"] == "assistant"
        assert json.loads(messages[2]["content"]) == {"tool": "sql_query"}

    def test_correct_verdict_multi_tool_emits_one_example_per_tool(self):
        trace = _trace(tool_chosen="sql_query, python_analysis")
        judgment = _judgment(tool_selection_verdict="correct")

        examples, reasons = build_examples_for_trace(trace, judgment)

        assert reasons == []
        assert len(examples) == 2
        labels = {
            json.loads(ex["messages"][2]["content"])["tool"] for ex in examples
        }
        assert labels == {"sql_query", "python_analysis"}

    def test_correct_verdict_with_unparseable_tool_chosen_is_skipped(self):
        trace = _trace(tool_chosen=None)
        judgment = _judgment(tool_selection_verdict="correct")

        examples, reasons = build_examples_for_trace(trace, judgment)

        assert examples == []
        assert len(reasons) == 1
        assert "no parseable tool_chosen" in reasons[0]


class TestIncorrectVerdict:
    def test_incorrect_verdict_with_derivable_correction_is_relabeled(self):
        trace = _trace(tool_chosen="sql_query")
        judgment = _judgment(
            tool_selection_verdict="incorrect",
            tool_selection_reason=(
                "The question is a forecast, not a lookup — python_analysis "
                "would have been the better choice here."
            ),
        )

        examples, reasons = build_examples_for_trace(trace, judgment)

        assert reasons == []
        assert len(examples) == 1
        assert json.loads(examples[0]["messages"][2]["content"]) == {
            "tool": "python_analysis"
        }
        assert "hindsight-corrected" in examples[0]["_meta"]["note"]

    def test_incorrect_verdict_with_no_confident_correction_is_skipped_and_logged(self):
        trace = _trace(tool_chosen="sql_query")
        judgment = _judgment(
            tool_selection_verdict="incorrect",
            tool_selection_reason="The question is about financial analysis, not a SQL query.",
        )

        examples, reasons = build_examples_for_trace(trace, judgment)

        assert examples == []
        assert len(reasons) == 1
        assert "no confident correction derivable" in reasons[0]

    def test_incorrect_verdict_with_ambiguous_multiple_mentions_is_skipped(self):
        trace = _trace(tool_chosen="sql_query")
        judgment = _judgment(
            tool_selection_verdict="incorrect",
            tool_selection_reason=(
                "Could have been doc_search or python_analysis instead."
            ),
        )

        examples, reasons = build_examples_for_trace(trace, judgment)

        assert examples == []
        assert len(reasons) == 1
        assert "no confident correction derivable" in reasons[0]


class TestUncertainOrMissingVerdict:
    def test_uncertain_verdict_is_skipped(self):
        trace = _trace()
        judgment = _judgment(tool_selection_verdict="uncertain")

        examples, reasons = build_examples_for_trace(trace, judgment)

        assert examples == []
        assert len(reasons) == 1
        assert "uncertain" in reasons[0]

    def test_missing_judgment_is_skipped(self):
        trace = _trace()

        examples, reasons = build_examples_for_trace(trace, None)

        assert examples == []
        assert len(reasons) == 1
        assert "ungraded" in reasons[0]

    def test_empty_question_is_skipped_even_with_correct_verdict(self):
        trace = _trace(question="")
        judgment = _judgment(tool_selection_verdict="correct")

        examples, reasons = build_examples_for_trace(trace, judgment)

        assert examples == []
        assert "empty question" in reasons[0]


class TestHelpers:
    def test_parse_tools_used_dedupes_and_drops_unknown_tokens(self):
        assert _parse_tools_used("sql_query, sql_query, bogus_tool, doc_search") == [
            "sql_query",
            "doc_search",
        ]

    def test_parse_tools_used_handles_none(self):
        assert _parse_tools_used(None) == []

    def test_infer_source_flags_csv_only_when_sole_tool_is_python_analysis(self):
        has_db, has_docs, has_csv = _infer_source_flags(["python_analysis"])
        assert (has_db, has_docs, has_csv) == (False, False, True)

    def test_infer_source_flags_sql_present_means_has_db(self):
        has_db, has_docs, has_csv = _infer_source_flags(["sql_query", "python_analysis"])
        assert has_db is True
        assert has_csv is False

    def test_extract_corrected_tool_returns_none_for_zero_or_multiple_mentions(self):
        assert _extract_corrected_tool("no tool names here", "sql_query") is None
        assert (
            _extract_corrected_tool("could be doc_search or python_analysis", "sql_query")
            is None
        )

    def test_extract_corrected_tool_returns_single_mention(self):
        assert (
            _extract_corrected_tool("doc_search would fit better here", "sql_query")
            == "doc_search"
        )
