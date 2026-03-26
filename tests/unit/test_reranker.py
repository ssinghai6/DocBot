"""Unit tests for api/utils/reranker.py — DOCBOT-1002.

All tests are CI-safe: httpx.post is mocked, no network calls.
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


def _mock_hf_response(scores: list[float]) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = scores
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# Happy-path: sorted by score descending, capped at top_k
# ---------------------------------------------------------------------------


class TestRerankSorting:
    def test_returns_top_k_sorted_by_score(self):
        docs = _make_docs(["doc A", "doc B", "doc C", "doc D", "doc E"])
        scores = [0.1, 0.9, 0.4, 0.7, 0.3]

        with patch("api.utils.reranker.httpx.post") as mock_post:
            mock_post.return_value = _mock_hf_response(scores)
            result = rerank("query", docs, hf_api_key="hf_test", top_k=3)

        assert len(result) == 3
        # Highest score (0.9) = doc B should be first
        assert result[0].page_content == "doc B"
        # Second-highest (0.7) = doc D
        assert result[1].page_content == "doc D"
        # Third-highest (0.4) = doc C
        assert result[2].page_content == "doc C"

    def test_top_k_capped_at_doc_count(self):
        docs = _make_docs(["a", "b"])
        scores = [0.5, 0.8]

        with patch("api.utils.reranker.httpx.post") as mock_post:
            mock_post.return_value = _mock_hf_response(scores)
            result = rerank("q", docs, hf_api_key="key", top_k=10)

        assert len(result) == 2

    def test_httpx_called_with_correct_payload(self):
        docs = _make_docs(["passage one", "passage two"])
        scores = [0.3, 0.7]

        with patch("api.utils.reranker.httpx.post") as mock_post:
            mock_post.return_value = _mock_hf_response(scores)
            rerank("my question", docs, hf_api_key="hf_abc", top_k=5)

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs["json"] if call_kwargs.kwargs else call_kwargs[1]["json"]
        assert payload["inputs"]["query"] == "my question"
        assert payload["inputs"]["passages"] == ["passage one", "passage two"]


# ---------------------------------------------------------------------------
# Fallback: empty key → no API call, original order returned
# ---------------------------------------------------------------------------


class TestRerankEmptyKey:
    def test_empty_key_skips_api(self):
        docs = _make_docs(["x", "y", "z"])

        with patch("api.utils.reranker.httpx.post") as mock_post:
            result = rerank("q", docs, hf_api_key="", top_k=5)
            mock_post.assert_not_called()

        # Original order preserved
        assert [d.page_content for d in result] == ["x", "y", "z"]

    def test_empty_key_honours_top_k(self):
        docs = _make_docs(["a", "b", "c", "d"])
        result = rerank("q", docs, hf_api_key="", top_k=2)
        assert len(result) == 2
        assert result[0].page_content == "a"


# ---------------------------------------------------------------------------
# Fallback: httpx raises an exception
# ---------------------------------------------------------------------------


class TestRerankFallbackOnException:
    def test_fallback_on_httpx_error(self):
        docs = _make_docs(["p", "q", "r"])

        with patch("api.utils.reranker.httpx.post", side_effect=Exception("timeout")):
            result = rerank("query", docs, hf_api_key="hf_key", top_k=5)

        # Original order returned unchanged
        assert [d.page_content for d in result] == ["p", "q", "r"]

    def test_fallback_on_raise_for_status(self):
        docs = _make_docs(["alpha", "beta"])
        resp = MagicMock()
        resp.raise_for_status.side_effect = Exception("403 Forbidden")

        with patch("api.utils.reranker.httpx.post", return_value=resp):
            result = rerank("q", docs, hf_api_key="bad_key", top_k=5)

        assert [d.page_content for d in result] == ["alpha", "beta"]

    def test_fallback_on_unexpected_response_shape(self):
        """If HF returns wrong number of scores, fall back gracefully."""
        docs = _make_docs(["one", "two", "three"])
        # HF returns only 2 scores for 3 docs → shape mismatch
        with patch("api.utils.reranker.httpx.post") as mock_post:
            mock_post.return_value = _mock_hf_response([0.5, 0.9])
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

        with patch("api.utils.reranker.httpx.post") as mock_post:
            mock_post.return_value = _mock_hf_response(scores)
            result = rerank("q", docs, hf_api_key="hf_key", top_k=5)

        assert len(result) == 1
        assert result[0].page_content == "only doc"
