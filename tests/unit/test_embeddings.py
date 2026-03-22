"""Unit tests for cosine similarity and few-shot embedding helpers — DOCBOT-203."""

import json
import math
import pytest
from api.utils.embeddings import cosine_similarity, find_similar_queries, embedding_to_json


@pytest.mark.unit
class TestCosineSimilarity:

    def test_identical_vectors_return_one(self):
        v = [1.0, 0.5, 0.3]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector_returns_zero_safely(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
        assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_opposite_vectors_return_minus_one(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_result_in_valid_range(self):
        a = [0.3, 0.7, 0.1, 0.9]
        b = [0.9, 0.1, 0.5, 0.2]
        result = cosine_similarity(a, b)
        assert -1.0 <= result <= 1.0


@pytest.mark.unit
class TestFindSimilarQueries:

    def _make_stored(self):
        return [
            {"nl_question": "total revenue", "sql_query": "SELECT SUM(amount) FROM orders",
             "embedding_json": json.dumps([1.0, 0.0, 0.0])},
            {"nl_question": "count users", "sql_query": "SELECT COUNT(*) FROM users",
             "embedding_json": json.dumps([0.0, 1.0, 0.0])},
            {"nl_question": "top products", "sql_query": "SELECT * FROM products ORDER BY sales DESC",
             "embedding_json": json.dumps([0.0, 0.0, 1.0])},
        ]

    def test_returns_best_match_first(self):
        stored = self._make_stored()
        results = find_similar_queries([1.0, 0.0, 0.0], stored, top_k=3, threshold=0.5)
        assert results[0]["nl_question"] == "total revenue"

    def test_respects_top_k(self):
        stored = self._make_stored()
        results = find_similar_queries([1.0, 0.0, 0.0], stored, top_k=1, threshold=0.0)
        assert len(results) == 1

    def test_threshold_filters_low_similarity(self):
        stored = self._make_stored()
        # Query orthogonal to all stored — similarity = 0.0 < threshold 0.75
        results = find_similar_queries([0.0, 0.0, 0.0, 1.0], stored, top_k=3, threshold=0.75)
        assert results == []

    def test_empty_stored_returns_empty(self):
        assert find_similar_queries([1.0, 0.0], [], top_k=3) == []

    def test_skips_records_with_missing_embedding(self):
        stored = [{"nl_question": "q", "sql_query": "SELECT 1", "embedding_json": None}]
        results = find_similar_queries([1.0, 0.0], stored, top_k=3, threshold=0.0)
        assert results == []

    def test_sorted_descending_by_similarity(self):
        stored = [
            {"nl_question": "partial match", "sql_query": "SELECT 1",
             "embedding_json": json.dumps([0.8, 0.6, 0.0])},
            {"nl_question": "exact match", "sql_query": "SELECT 2",
             "embedding_json": json.dumps([1.0, 0.0, 0.0])},
        ]
        results = find_similar_queries([1.0, 0.0, 0.0], stored, top_k=2, threshold=0.0)
        assert results[0]["nl_question"] == "exact match"


@pytest.mark.unit
class TestEmbeddingToJson:

    def test_round_trip(self):
        vec = [0.1, 0.2, 0.3, 0.999]
        assert json.loads(embedding_to_json(vec)) == pytest.approx(vec)

    def test_returns_string(self):
        assert isinstance(embedding_to_json([1.0, 2.0]), str)

    def test_empty_vector(self):
        assert json.loads(embedding_to_json([])) == []
