"""Unit tests for api/utils/discrepancy_detector.py — DOCBOT-403."""

import pytest
from api.utils.discrepancy_detector import (
    DiscrepancyReport,
    detect_discrepancies,
    _extract_from_text,
    _extract_from_rows,
    _label_similarity,
)


# ---------------------------------------------------------------------------
# Label similarity
# ---------------------------------------------------------------------------


class TestLabelSimilarity:

    def test_identical_labels(self):
        assert _label_similarity("revenue", "revenue") == 1.0

    def test_synonym_normalization(self):
        # "sales" → "revenue" and "revenues" → "revenue" via synonyms
        sim = _label_similarity("total sales", "total revenue")
        assert sim >= 0.5

    def test_unrelated_labels(self):
        sim = _label_similarity("headcount", "revenue")
        assert sim < 0.3

    def test_partial_overlap(self):
        sim = _label_similarity("net income", "net profit")
        # "net" overlaps; "income"/"profit" are synonyms after normalization
        assert sim > 0.4

    def test_empty_labels(self):
        assert _label_similarity("", "revenue") == 0.0


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


class TestExtractFromText:

    def test_colon_separated(self):
        text = "Revenue: $1,200,000"
        values = _extract_from_text(text, source="doc")
        assert any(abs(v.value - 1_200_000) < 1 for v in values)

    def test_million_suffix(self):
        text = "Net income: 4.5M"
        values = _extract_from_text(text, source="doc")
        assert any(abs(v.value - 4_500_000) < 1 for v in values)

    def test_billion_suffix(self):
        text = "Total assets: $2.1B"
        values = _extract_from_text(text, source="doc")
        assert any(abs(v.value - 2_100_000_000) < 1 for v in values)

    def test_k_suffix(self):
        text = "Marketing spend: 500K"
        values = _extract_from_text(text, source="doc")
        assert any(abs(v.value - 500_000) < 1 for v in values)

    def test_inline_pattern(self):
        text = "Total revenue was $3,500,000 last quarter."
        values = _extract_from_text(text, source="doc")
        assert any(abs(v.value - 3_500_000) < 1 for v in values)

    def test_negative_value(self):
        text = "Net loss: -$200,000"
        values = _extract_from_text(text, source="doc")
        assert any(abs(v.value - (-200_000)) < 1 for v in values)

    def test_no_false_positives_on_dates(self):
        text = "As of 2024-01-15, the company reported."
        values = _extract_from_text(text, source="doc")
        # Should not extract standalone year numbers as metrics
        assert not any(abs(v.value - 2024) < 1 and v.label.strip() in ("", "as") for v in values)

    def test_deduplicates_same_value(self):
        text = "Revenue: $1,000,000\nRevenue: $1,000,000"
        values = _extract_from_text(text, source="doc")
        revenue_matches = [v for v in values if "revenue" in v.label]
        assert len(revenue_matches) == 1


# ---------------------------------------------------------------------------
# Row extraction
# ---------------------------------------------------------------------------


class TestExtractFromRows:

    def test_integer_column(self):
        rows = [{"total_revenue": 5_000_000, "customer": "ACME"}]
        values = _extract_from_rows(rows)
        assert any(abs(v.value - 5_000_000) < 1 for v in values)

    def test_float_column(self):
        rows = [{"net_income": 1_234_567.89}]
        values = _extract_from_rows(rows)
        assert any(abs(v.value - 1_234_567.89) < 0.01 for v in values)

    def test_string_numeric(self):
        rows = [{"amount": "2,500.00"}]
        values = _extract_from_rows(rows)
        assert any(abs(v.value - 2500.0) < 0.01 for v in values)

    def test_skips_non_numeric(self):
        rows = [{"name": "Alice", "status": "active", "count": 3}]
        values = _extract_from_rows(rows)
        assert len(values) == 1
        assert abs(values[0].value - 3) < 0.01

    def test_empty_rows(self):
        assert _extract_from_rows([]) == []


# ---------------------------------------------------------------------------
# detect_discrepancies — full pipeline
# ---------------------------------------------------------------------------


class TestDetectDiscrepancies:

    def _make_sql_meta(self, rows=None, csv_answer=None):
        meta = {}
        if rows is not None:
            meta["result_preview"] = rows
        if csv_answer is not None:
            meta["csv_answer"] = csv_answer
        return meta

    def test_real_discrepancy_detected(self):
        # A realistic reporting discrepancy (8%) — within the plausible band.
        # Deltas >25% are treated as unit/granularity mismatches and suppressed.
        doc = "Revenue: $1,000,000 for Q1 2024."
        sql = self._make_sql_meta(rows=[{"revenue": 1_080_000}])
        report = detect_discrepancies(doc, sql)
        assert report.has_discrepancies
        item = report.discrepancies[0]
        assert abs(item.doc_value - 1_000_000) < 1
        assert abs(item.db_value - 1_080_000) < 1
        assert abs(item.delta - 80_000) < 1
        assert item.pct is not None
        assert abs(item.pct - 8.0) < 0.1

    def test_large_delta_suppressed_as_unit_mismatch(self):
        """A 100%+ gap on the same label is a unit mismatch, not a discrepancy."""
        doc = "Total revenue: $5,200,000,000"
        sql = self._make_sql_meta(rows=[{"total_revenue": 1180}])  # DB in millions
        report = detect_discrepancies(doc, sql)
        assert not report.has_discrepancies

    def test_no_discrepancy_when_values_match(self):
        doc = "Total revenue: $2,000,000"
        sql = self._make_sql_meta(rows=[{"total_revenue": 2_000_000}])
        report = detect_discrepancies(doc, sql)
        assert not report.has_discrepancies

    def test_small_rounding_not_flagged(self):
        """0.5% difference should not be flagged (below 1% default threshold)."""
        doc = "Net income: $1,000,000"
        sql = self._make_sql_meta(rows=[{"net_income": 1_005_000}])
        report = detect_discrepancies(doc, sql, delta_threshold=0.01)
        assert not report.has_discrepancies

    def test_csv_answer_source(self):
        doc = "Revenue: $500,000"
        sql = self._make_sql_meta(csv_answer="revenue: 540,000")
        report = detect_discrepancies(doc, sql)
        assert report.has_discrepancies
        assert abs(report.discrepancies[0].delta - 40_000) < 1

    def test_no_match_when_unrelated_labels(self):
        doc = "Employee count: 150"
        sql = self._make_sql_meta(rows=[{"revenue": 5_000_000}])
        report = detect_discrepancies(doc, sql)
        assert not report.has_discrepancies

    def test_prompt_block_format(self):
        # DOCBOT-2002: each discrepancy carries an inline marker token instead
        # of a literal [DISCREPANCY] tag.
        doc = "Revenue: $1,000,000"
        sql = self._make_sql_meta(rows=[{"revenue": 1_200_000}])
        report = detect_discrepancies(doc, sql)
        block = report.to_prompt_block()
        assert "[[discrepancy:d-" in block
        assert "1,000,000" in block
        assert "1,200,000" in block
        assert "200,000" in block
        assert "20.0%" in block
        # The literal tag only appears in the "never write this" instruction,
        # never as a per-item prefix.
        item_lines = [l for l in block.splitlines() if l.strip().startswith("[[discrepancy:")]
        assert item_lines and all("[DISCREPANCY]" not in l for l in item_lines)

    def test_empty_doc_context(self):
        report = detect_discrepancies("", {"result_preview": [{"revenue": 1_000}]})
        assert not report.has_discrepancies

    def test_empty_sql_metadata(self):
        report = detect_discrepancies("Revenue: $1,000,000", {})
        assert not report.has_discrepancies

    def test_checked_pairs_counted(self):
        doc = "Revenue: $1,000,000\nNet income: $200,000"
        sql = self._make_sql_meta(rows=[{"revenue": 999_000, "net_income": 200_000}])
        report = detect_discrepancies(doc, sql)
        assert report.checked_pairs >= 1

    def test_negative_delta_direction(self):
        """DB value lower than doc value — delta should be negative."""
        doc = "Total expenses: $800,000"
        sql = self._make_sql_meta(rows=[{"total_expenses": 600_000}])
        report = detect_discrepancies(doc, sql)
        assert report.has_discrepancies
        assert report.discrepancies[0].delta < 0

    def test_report_no_discrepancies_returns_empty_block(self):
        doc = "Revenue: $1,000,000"
        sql = self._make_sql_meta(rows=[{"revenue": 1_000_000}])
        report = detect_discrepancies(doc, sql)
        assert report.to_prompt_block() == ""


# ---------------------------------------------------------------------------
# DOCBOT-2002 — periods, citations, severity, tie-out semantics
# ---------------------------------------------------------------------------


class TestPeriodAwareness:

    def test_quarter_mismatch_not_compared(self):
        """A Q1 report figure never ties out against a Q4 books row."""
        doc = "Q1 2025 revenue: $1,000,000"
        sql = {"result_preview": [{"quarter": "Q4 2025", "revenue": 1_200_000}]}
        report = detect_discrepancies(doc, sql)
        assert not report.has_discrepancies

    def test_quarter_match_compared(self):
        doc = "Q4 2025 revenue: $1,000,000"
        sql = {"result_preview": [{"quarter": "Q4 2025", "revenue": 1_100_000}]}
        report = detect_discrepancies(doc, sql)
        assert report.has_discrepancies
        assert report.discrepancies[0].period == "Q4 2025"

    def test_period_extracted_from_line_prefix(self):
        doc = "Q4 2025: Net Income $330M, Revenue $1,440M"
        values = _extract_from_text(doc, source="doc")
        by_label = {v.label: v for v in values}
        assert any(v.period == "Q4 2025" for v in by_label.values())

    def test_fy_mention_after_figure_not_misread(self):
        """'(up 18% from $4.4B in FY2024)' must not set FY2024 on the figure."""
        doc = "Revenue: $1,440M (up 18% from $4.4 billion in FY2024)"
        values = _extract_from_text(doc, source="doc")
        revenue = [v for v in values if "revenue" in v.label and abs(v.value - 1_440_000_000) < 1]
        assert revenue and revenue[0].period is None


class TestCitationsAndSeverity:

    def test_doc_citation_from_source_header(self):
        doc = "Source: 10-K.pdf, Page 42\nNet income: $330M"
        sql = {"result_preview": [{"net_income": 325}]}
        report = detect_discrepancies(doc, sql)
        assert report.has_discrepancies
        item = report.discrepancies[0]
        assert item.doc_file == "10-K.pdf"
        assert item.doc_page == 42

    def test_db_citation_from_sources_and_query(self):
        doc = "Net income: $330M"
        sql = {
            "result_preview": [{"net_income": 325}],
            "sources": ["financials", "quarters"],
            "sql_query": "SELECT net_income FROM financials",
        }
        report = detect_discrepancies(doc, sql)
        item = report.discrepancies[0]
        assert item.db_table == "financials, quarters"
        assert item.db_query == "SELECT net_income FROM financials"

    def test_severity_bands(self):
        from api.utils.discrepancy_detector import severity_for

        assert severity_for(None) == "high"
        assert severity_for(0.2) == "low"
        assert severity_for(-0.7) == "medium"
        assert severity_for(1.5) == "high"
        assert severity_for(-3.0) == "high"

    def test_item_ids_stable_and_marker_shaped(self):
        from api.utils.discrepancy_detector import DiscrepancyItem, discrepancy_id

        a = DiscrepancyItem(label="revenue", doc_value=100.0, db_value=105.0,
                            delta=5.0, pct=5.0, doc_snippet="d", db_snippet="b")
        b = DiscrepancyItem(label="revenue", doc_value=100.0, db_value=105.0,
                            delta=5.0, pct=5.0, doc_snippet="d", db_snippet="b")
        assert a.id == b.id == discrepancy_id("revenue", None, 100.0, 105.0)
        import re as _re

        assert _re.fullmatch(r"d-[0-9a-f]{8}", a.id)
        assert a.marker == f"[[discrepancy:{a.id}]]"

    def test_currency_never_ties_out_against_percent(self):
        doc = "Revenue: $1,000,000"
        sql = {"result_preview": [{"growth_rate": 20}]}
        report = detect_discrepancies(doc, sql)
        assert not report.has_discrepancies

    def test_plausible_value_ties_out(self):
        """If ANY books row agrees with the report figure, no card is raised."""
        doc = "Revenue: $1,000,000"
        sql = {"result_preview": [{"revenue": 1_000_000}, {"revenue": 1_800_000}]}
        report = detect_discrepancies(doc, sql)
        assert not report.has_discrepancies

    def test_dimension_qualified_label(self):
        """SQL rows with a segment dimension match that segment's doc figure."""
        doc = "Restaurant advertising revenue: $890M"
        sql = {"result_preview": [
            {"segment": "Delivery", "advertising_revenue": 700_000_000},
            {"segment": "Restaurant", "advertising_revenue": 860_000_000},
        ]}
        report = detect_discrepancies(doc, sql)
        assert report.has_discrepancies
        item = report.discrepancies[0]
        assert abs(item.db_value - 860_000_000) < 1
        assert "restaurant" in item.label.lower()
