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
        doc = "Revenue: $1,000,000 for Q1 2024."
        sql = self._make_sql_meta(rows=[{"revenue": 1_500_000}])
        report = detect_discrepancies(doc, sql)
        assert report.has_discrepancies
        item = report.discrepancies[0]
        assert abs(item.doc_value - 1_000_000) < 1
        assert abs(item.db_value - 1_500_000) < 1
        assert abs(item.delta - 500_000) < 1
        assert item.pct is not None
        assert abs(item.pct - 50.0) < 0.1

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
        sql = self._make_sql_meta(csv_answer="revenue: 750,000")
        report = detect_discrepancies(doc, sql)
        assert report.has_discrepancies
        assert abs(report.discrepancies[0].delta - 250_000) < 1

    def test_no_match_when_unrelated_labels(self):
        doc = "Employee count: 150"
        sql = self._make_sql_meta(rows=[{"revenue": 5_000_000}])
        report = detect_discrepancies(doc, sql)
        assert not report.has_discrepancies

    def test_prompt_block_format(self):
        doc = "Revenue: $1,000,000"
        sql = self._make_sql_meta(rows=[{"revenue": 1_200_000}])
        report = detect_discrepancies(doc, sql)
        block = report.to_prompt_block()
        assert "[DISCREPANCY]" in block
        assert "1,000,000" in block
        assert "1,200,000" in block
        assert "200,000" in block
        assert "20.0%" in block

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
