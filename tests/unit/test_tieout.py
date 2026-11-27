"""Unit tests for api/utils/tieout.py — DOCBOT-2002.

Covers the marker → card end-to-end path: detector DiscrepancyItem → inline
marker token → lineage TieOutCard payload, plus the LLM-output rewriters
(``replace_discrepancy_tags``, ``missing_marker_tail``) and the autopilot
step-based detection entrypoint. Pure logic — no network, no API keys.
"""

from __future__ import annotations

import json

from api.utils.discrepancy_detector import DiscrepancyItem, detect_discrepancies
from api.utils.lineage import LineageCollector, LineageDiscrepancy
from api.utils.tieout import (
    MARKER_RE,
    NEUTRAL_LABEL,
    detect_from_autopilot_steps,
    match_item,
    missing_marker_tail,
    referenced_ids,
    replace_discrepancy_tags,
)

DOC_CONTEXT = "Source: 10-K.pdf, Page 42\nQ4 2025 Net Income: $330M"
SQL_META = {
    "result_preview": [{"net_income": 325}],
    "sources": ["financials"],
    "sql_query": "SELECT net_income FROM financials",
}


def _item(**overrides) -> DiscrepancyItem:
    base = dict(
        label="net income",
        doc_value=330.0,
        db_value=325.0,
        delta=-5.0,
        pct=-1.5,
        doc_snippet="Net Income: $330M",
        db_snippet="net_income=325",
        unit="currency",
    )
    base.update(overrides)
    return DiscrepancyItem(**base)


# ---------------------------------------------------------------------------
# Marker → card end-to-end
# ---------------------------------------------------------------------------


class TestMarkerToCard:

    def test_detector_item_carries_marker_and_card_fields(self):
        report = detect_discrepancies(DOC_CONTEXT, SQL_META)
        assert report.has_discrepancies
        item = report.discrepancies[0]
        assert MARKER_RE.fullmatch(item.marker)
        assert item.display_label == "Net Income (Q4 2025)"
        assert item.severity == "high"
        assert item.unit == "currency"
        assert item.doc_file == "10-K.pdf" and item.doc_page == 42
        assert item.db_table == "financials"
        assert item.db_query == "SELECT net_income FROM financials"

    def test_marker_into_collector_yields_tie_out_card(self):
        item = _item(
            period="Q4 2025",
            doc_file="10-K.pdf",
            doc_page=42,
            db_table="financials",
            db_query="SELECT net_income FROM financials",
        )
        c = LineageCollector("r1", "hybrid")
        c.add_discrepancy(item)
        lineage = c.build()

        assert len(lineage.discrepancies) == 1
        card = lineage.discrepancies[0]
        assert card.id == item.id
        assert card.metric == "net income"
        assert card.period == "Q4 2025"
        assert card.report_value == 330.0
        assert card.books_value == 325.0
        assert card.delta == -5.0
        assert card.delta_pct == -1.5
        assert card.severity == "high"
        assert card.unit == "currency"
        assert card.doc_citation.file == "10-K.pdf" and card.doc_citation.page == 42
        assert card.db_citation.table == "financials"
        assert card.db_citation.query == "SELECT net_income FROM financials"

    def test_card_is_json_serializable_for_sse(self):
        c = LineageCollector("r1", "hybrid")
        c.add_discrepancy(_item(period="Q4 2025", doc_file="10-K.pdf", doc_page=42))
        payload = c.build().model_dump(mode="json")
        json.dumps(payload)  # must not raise
        card = payload["discrepancies"][0]
        assert card["id"].startswith("d-")
        assert card["report_value"] == 330.0
        assert card["doc_citation"]["page"] == 42


# ---------------------------------------------------------------------------
# match_item
# ---------------------------------------------------------------------------


class TestMatchItem:

    def test_single_item_fallback(self):
        items = [_item()]
        assert match_item("some unlabelled line [DISCREPANCY]", items) is items[0]

    def test_marker_id_wins(self):
        a, b = _item(), _item(label="total revenue", doc_value=1_000.0, db_value=900.0, delta=-100.0, pct=-10.0)
        assert match_item(f"see {a.marker}", [a, b]) is a

    def test_token_and_value_scoring(self):
        rev = _item(label="total revenue", doc_value=1_000_000.0, db_value=1_200_000.0, delta=200_000.0, pct=20.0)
        ni = _item()
        ctx = "Total revenue was 1,000,000 vs 1,200,000 in the books [DISCREPANCY]"
        assert match_item(ctx, [ni, rev]) is rev

    def test_no_items_returns_none(self):
        assert match_item("[DISCREPANCY]", []) is None


# ---------------------------------------------------------------------------
# replace_discrepancy_tags / referenced_ids / missing_marker_tail
# ---------------------------------------------------------------------------


class TestReplaceTags:

    def test_legacy_open_tag_rewritten_to_marker(self):
        items = [_item(label="net income")]
        out = replace_discrepancy_tags(
            "Net Income came in lower. [DISCREPANCY] DB shows 325.", items
        )
        assert "[DISCREPANCY]" not in out
        assert items[0].marker in out

    def test_legacy_block_with_close_tag_dropped(self):
        items = [_item(label="net income")]
        out = replace_discrepancy_tags(
            "[DISCREPANCY] Net income lower than the filing [/DISCREPANCY]", items
        )
        assert "DISCREPANCY" not in out.replace("Discrepancy", "")
        assert items[0].marker in out

    def test_echoed_prompt_block_collapsed_to_marker(self):
        item = _item(label="net income", period="Q4 2025")
        text = (
            "[DISCREPANCY] Net Income: Report says 330 [Source: 10-K]. "
            "Books show 325 [DB: financials]. Delta: -5 (-1.5%)"
        )
        out = replace_discrepancy_tags(text, [item])
        assert "[DISCREPANCY]" not in out
        assert f"**{item.display_label}** {item.marker}" in out

    def test_unmatched_tag_becomes_neutral_label(self):
        out = replace_discrepancy_tags("[DISCREPANCY] something odd", [])
        assert out == f"{NEUTRAL_LABEL} something odd"

    def test_tag_dropped_when_marker_already_on_line(self):
        item = _item()
        line = f"Net income lower. {item.marker} [DISCREPANCY]"
        out = replace_discrepancy_tags(line, [item])
        assert "[DISCREPANCY]" not in out
        assert out.count(item.marker) == 1

    def test_case_insensitive_and_multiline(self):
        items = [_item(label="net income"), _item(label="total revenue", doc_value=1_000.0, db_value=1_100.0, delta=100.0, pct=10.0)]
        text = (
            "First finding:\n[Discrepancy] net income below plan\n\n"
            "Second finding:\n[DISCREPANCY] total revenue differs"
        )
        out = replace_discrepancy_tags(text, items)
        assert "[Discrepancy]" not in out and "[DISCREPANCY]" not in out
        for item in items:
            assert item.marker in out

    def test_fast_path_without_keyword(self):
        assert replace_discrepancy_tags("plain answer", []) == "plain answer"


class TestTail:

    def test_all_referenced_returns_empty(self):
        item = _item()
        text = f"Lower. {item.marker}"
        assert missing_marker_tail(text, [item]) == ""

    def test_missing_items_appended(self):
        a, b = _item(label="net income", period="Q4 2025"), _item(
            label="total revenue", doc_value=1_000.0, db_value=1_100.0, delta=100.0, pct=10.0
        )
        tail = missing_marker_tail("answer text", [a, b])
        assert tail.startswith("\n\n**Tie-out:**")
        for item in (a, b):
            assert item.display_label in tail and item.marker in tail

    def test_referenced_ids(self):
        item = _item()
        assert referenced_ids(f"x {item.marker} y") == {item.id}
        assert referenced_ids("no markers") == set()


# ---------------------------------------------------------------------------
# detect_from_autopilot_steps
# ---------------------------------------------------------------------------


class TestDetectFromAutopilotSteps:

    def test_doc_only_returns_none(self):
        steps = [{"tieout_doc_context": DOC_CONTEXT, "error": None}]
        assert detect_from_autopilot_steps(steps) is None

    def test_sql_only_returns_none(self):
        steps = [{"tieout_sql": SQL_META, "error": None}]
        assert detect_from_autopilot_steps(steps) is None

    def test_both_sides_detected(self):
        steps = [
            {"tieout_doc_context": DOC_CONTEXT, "error": None, "tool": "doc_search"},
            {"tieout_sql": SQL_META, "error": None, "tool": "sql_query"},
        ]
        report = detect_from_autopilot_steps(steps)
        assert report is not None and report.has_discrepancies
        item = report.discrepancies[0]
        assert item.doc_file == "10-K.pdf"
        assert item.db_table == "financials"

    def test_error_steps_skipped(self):
        steps = [
            {"tieout_doc_context": DOC_CONTEXT, "error": "boom"},
            {"tieout_sql": SQL_META, "error": None},
        ]
        assert detect_from_autopilot_steps(steps) is None

    def test_non_dict_sql_ignored(self):
        steps = [
            {"tieout_doc_context": DOC_CONTEXT, "error": None},
            {"tieout_sql": "garbage", "error": None},
        ]
        assert detect_from_autopilot_steps(steps) is None


# ---------------------------------------------------------------------------
# Legacy lineage payloads
# ---------------------------------------------------------------------------


class TestLegacyUpgrade:

    def test_pre_2002_row_validates_to_card(self):
        card = LineageDiscrepancy.model_validate(
            {"label": "net income", "doc_value": 330.0, "db_value": 325.0, "delta": -5.0, "pct": -1.5}
        )
        assert card.id.startswith("d-")
        assert MARKER_RE.fullmatch(f"[[discrepancy:{card.id}]]")
        assert card.report_value == 330.0 and card.books_value == 325.0


def test_4021_autopilot_detection_passes_use_full_rows():
    """Both autopilot detection passes read every books row, not the 10-row sample."""
    from api.autopilot_service import _executed_rows_preview

    rows = [{"segment": f"S{i}", "revenue": i} for i in range(25)]
    state = {"data_results": {"0": {"tool": "sql_query", "sample_rows": rows[:10], "tieout_rows": rows}}}
    assert _executed_rows_preview(state) == rows
    legacy = {"data_results": {"0": {"tool": "sql_query", "sample_rows": rows[:10]}}}
    assert _executed_rows_preview(legacy) == rows[:10]
