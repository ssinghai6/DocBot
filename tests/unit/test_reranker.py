"""Unit tests for api/utils/reranker.py — DOCBOT-1002.

All tests are CI-safe: InferenceClient is mocked, no network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from api.utils.reranker import rerank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_docs(contents: list[str]) -> list:
    """Return minimal Document-like objects with .page_content."""
    docs = []
    for text in contents:
        doc = MagicMock()
        doc.page_content = text
        docs.append(doc)
    return docs


def _patch_client(scores: list[float]):
    """Patch InferenceClient to return given scores from sentence_similarity."""
    mock_client = MagicMock()
    mock_client.sentence_similarity.return_value = scores
    return patch(
        "api.utils.reranker.InferenceClient",
        return_value=mock_client,
    )


# ---------------------------------------------------------------------------
# Happy-path: sorted by score descending, capped at top_k
# ---------------------------------------------------------------------------


class TestRerankSorting:
    def test_returns_top_k_sorted_by_score(self):
        docs = _make_docs(["doc A", "doc B", "doc C", "doc D", "doc E"])
        scores = [0.1, 0.9, 0.4, 0.7, 0.3]

        with _patch_client(scores):
            result = rerank("query", docs, hf_api_key="hf_test", top_k=3)

        assert len(result) == 3
        assert result[0].page_content == "doc B"
        assert result[1].page_content == "doc D"
        assert result[2].page_content == "doc C"

    def test_top_k_capped_at_doc_count(self):
        docs = _make_docs(["a", "b"])
        scores = [0.5, 0.8]

        with _patch_client(scores):
            result = rerank("q", docs, hf_api_key="key", top_k=10)

        assert len(result) == 2

    def test_client_called_with_correct_args(self):
        docs = _make_docs(["passage one", "passage two"])
        scores = [0.3, 0.7]

        with _patch_client(scores) as mock_cls:
            rerank("my question", docs, hf_api_key="hf_abc", top_k=5)

        mock_cls.assert_called_once()
        mock_instance = mock_cls.return_value
        call_args = mock_instance.sentence_similarity.call_args
        assert call_args[0][0] == "my question"
        assert call_args[1]["other_sentences"] == ["passage one", "passage two"]


# ---------------------------------------------------------------------------
# Fallback: empty key → no API call, original order returned
# ---------------------------------------------------------------------------


class TestRerankEmptyKey:
    def test_empty_key_skips_api(self):
        docs = _make_docs(["x", "y", "z"])

        with _patch_client([]) as mock_cls:
            result = rerank("q", docs, hf_api_key="", top_k=5)
            mock_cls.assert_not_called()

        assert [d.page_content for d in result] == ["x", "y", "z"]

    def test_empty_key_honours_top_k(self):
        docs = _make_docs(["a", "b", "c", "d"])
        result = rerank("q", docs, hf_api_key="", top_k=2)
        assert len(result) == 2
        assert result[0].page_content == "a"


# ---------------------------------------------------------------------------
# Fallback: client raises an exception
# ---------------------------------------------------------------------------


class TestRerankFallbackOnException:
    def test_fallback_on_client_error(self):
        docs = _make_docs(["p", "q", "r"])

        mock_client = MagicMock()
        mock_client.sentence_similarity.side_effect = Exception("timeout")
        with patch("api.utils.reranker.InferenceClient", return_value=mock_client):
            result = rerank("query", docs, hf_api_key="hf_key", top_k=5)

        assert [d.page_content for d in result] == ["p", "q", "r"]

    def test_fallback_on_auth_error(self):
        docs = _make_docs(["alpha", "beta"])

        mock_client = MagicMock()
        mock_client.sentence_similarity.side_effect = Exception("403 Forbidden")
        with patch("api.utils.reranker.InferenceClient", return_value=mock_client):
            result = rerank("q", docs, hf_api_key="bad_key", top_k=5)

        assert [d.page_content for d in result] == ["alpha", "beta"]

    def test_fallback_on_unexpected_response_shape(self):
        """If HF returns wrong number of scores, fall back gracefully."""
        docs = _make_docs(["one", "two", "three"])
        # Only 2 scores for 3 docs → shape mismatch
        with _patch_client([0.5, 0.9]):
            result = rerank("q", docs, hf_api_key="hf_key", top_k=5)

        assert [d.page_content for d in result] == ["one", "two", "three"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestRerankEdgeCases:
    def test_empty_docs_list(self):
        result = rerank("q", [], hf_api_key="hf_key", top_k=5)
        assert result == []

    def test_single_doc(self):
        docs = _make_docs(["only doc"])
        scores = [0.75]

        with _patch_client(scores):
            result = rerank("q", docs, hf_api_key="hf_key", top_k=5)

        assert len(result) == 1
        assert result[0].page_content == "only doc"
