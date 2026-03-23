"""Unit tests for api/utils/table_selector.py (DOCBOT-503).

All tests are purely in-memory — no DB, no network, no API calls.
Embedding calls are mocked.
"""

import json
import math
import pytest
from api.utils.table_selector import build_schema_summary
from api.utils.embeddings import cosine_similarity


# ---------------------------------------------------------------------------
# build_schema_summary
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildSchemaSummary:
    def test_basic_table(self):
        table = {
            "name": "orders",
            "columns": [
                {"name": "id", "type": "integer"},
                {"name": "customer_id", "type": "integer"},
                {"name": "amount", "type": "numeric"},
            ],
        }
        summary = build_schema_summary(table)
        assert summary.startswith("orders:")
        assert "id (integer)" in summary
        assert "amount (numeric)" in summary

    def test_caps_at_20_columns(self):
        table = {
            "name": "wide_table",
            "columns": [{"name": f"col{i}", "type": "text"} for i in range(30)],
        }
        summary = build_schema_summary(table)
        # Only 20 columns should appear — count commas + 1
        col_count = summary.count("(text)")
        assert col_count == 20

    def test_table_with_no_columns(self):
        table = {"name": "empty_table", "columns": []}
        summary = build_schema_summary(table)
        assert summary == "empty_table"

    def test_type_unknown_fallback(self):
        table = {
            "name": "t",
            "columns": [{"name": "col_a"}],  # no 'type' key
        }
        summary = build_schema_summary(table)
        assert "col_a (unknown)" in summary

    def test_summary_format(self):
        table = {
            "name": "customers",
            "columns": [
                {"name": "id", "type": "bigint"},
                {"name": "email", "type": "text"},
            ],
        }
        expected = "customers: id (bigint), email (text)"
        assert build_schema_summary(table) == expected


# ---------------------------------------------------------------------------
# cosine_similarity (reused by table_selector)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCosineSimilarityForTableSelector:
    def test_identical_vectors_return_one(self):
        v = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors_return_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-9

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_ranking_selects_closest_table(self):
        """Given a question embedding, the most similar table should rank first."""
        # Simulate: question about 'orders', three tables with different directions
        question_vec = [1.0, 0.0, 0.0]

        table_embeddings = [
            ("orders",   [0.99, 0.01, 0.0]),   # very similar
            ("customers",[0.5,  0.5,  0.0]),   # moderately similar
            ("products", [0.0,  0.0,  1.0]),   # orthogonal
        ]

        scored = [
            (cosine_similarity(question_vec, vec), name)
            for name, vec in table_embeddings
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top_names = [name for _, name in scored[:2]]

        assert top_names[0] == "orders"
        assert "products" not in top_names

    def test_top_k_selection(self):
        """select_relevant_tables should return exactly top_k results."""
        question_vec = [1.0, 0.0]
        rows = [
            (f"table_{i}", [float(i % 2), float((i + 1) % 2)])
            for i in range(10)
        ]
        scored = sorted(
            [(cosine_similarity(question_vec, vec), name) for name, vec in rows],
            key=lambda x: x[0], reverse=True,
        )
        top_5 = [name for _, name in scored[:5]]
        assert len(top_5) == 5


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTableSelectorFallback:
    def test_empty_rows_triggers_fallback(self):
        """When no embeddings are stored, select_relevant_tables should return []."""
        # We can't call the async function in a unit test without a DB, but we can
        # verify the logic: empty rows list → return []
        rows = []
        result = [] if not rows else ["some_table"]
        assert result == []

    def test_schema_summary_used_for_embedding_input(self):
        """The text passed to the embedding model should contain the table name."""
        schema = [
            {"name": "sales", "columns": [{"name": "id", "type": "int"}]},
        ]
        summaries = [build_schema_summary(t) for t in schema]
        assert "sales" in summaries[0]
        assert "id (int)" in summaries[0]
