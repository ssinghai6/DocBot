"""Unit tests for api/utils/reranker.py — DOCBOT-1002 / DOCBOT-1405.

All tests are CI-safe: httpx.post is mocked, no network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from api.utils import reranker
from api.utils.reranker import rerank, rerank_scored


@pytest.fixture(autouse=True)
def _hf_provider(monkeypatch):
    """Existing tests exercise the HF endpoint path; local provider is tested below."""
    monkeypatch.setenv("RERANKER_PROVIDER", "hf")


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


def _hf_response(scores: list[float]) -> MagicMock:
    """Build a mock httpx.Response matching the real HF cross-encoder shape:
    a single-element outer list wrapping one score dict per input pair,
    in input order.
    """
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = [
        [{"label": "LABEL_0", "score": s} for s in scores]
    ]
    return resp


def _patch_post(scores: list[float]):
    return patch("api.utils.reranker.httpx.post", return_value=_hf_response(scores))


# ---------------------------------------------------------------------------
# Happy-path: sorted by score descending, capped at top_k
# ---------------------------------------------------------------------------


class TestRerankSorting:
    def test_returns_top_k_sorted_by_score(self):
        docs = _make_docs(["doc A", "doc B", "doc C", "doc D", "doc E"])
        scores = [0.1, 0.9, 0.4, 0.7, 0.3]

        with _patch_post(scores):
            result = rerank("query", docs, hf_api_key="hf_test", top_k=3)

        assert len(result) == 3
        assert result[0].page_content == "doc B"
        assert result[1].page_content == "doc D"
        assert result[2].page_content == "doc C"

    def test_top_k_capped_at_doc_count(self):
        docs = _make_docs(["a", "b"])
        scores = [0.5, 0.8]

        with _patch_post(scores):
            result = rerank("q", docs, hf_api_key="key", top_k=10)

        assert len(result) == 2

    def test_client_called_with_correct_pair_payload(self):
        docs = _make_docs(["passage one", "passage two"])
        scores = [0.3, 0.7]

        with _patch_post(scores) as mock_post:
            rerank("my question", docs, hf_api_key="hf_abc", top_k=5)

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["headers"] == {"Authorization": "Bearer hf_abc"}
        assert kwargs["json"] == {
            "inputs": [
                {"text": "my question", "text_pair": "passage one"},
                {"text": "my question", "text_pair": "passage two"},
            ]
        }


# ---------------------------------------------------------------------------
# Fallback: empty key → no API call, original order returned
# ---------------------------------------------------------------------------


class TestRerankEmptyKey:
    def test_empty_key_skips_api(self):
        docs = _make_docs(["x", "y", "z"])

        with _patch_post([]) as mock_post:
            result = rerank("q", docs, hf_api_key="", top_k=5)
            mock_post.assert_not_called()

        assert [d.page_content for d in result] == ["x", "y", "z"]

    def test_empty_key_honours_top_k(self):
        docs = _make_docs(["a", "b", "c", "d"])
        result = rerank("q", docs, hf_api_key="", top_k=2)
        assert len(result) == 2
        assert result[0].page_content == "a"


# ---------------------------------------------------------------------------
# Fallback: request raises / returns an error status
# ---------------------------------------------------------------------------


class TestRerankFallbackOnException:
    def test_fallback_on_request_error(self):
        docs = _make_docs(["p", "q", "r"])

        with patch(
            "api.utils.reranker.httpx.post",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            result = rerank("query", docs, hf_api_key="hf_key", top_k=5)

        assert [d.page_content for d in result] == ["p", "q", "r"]

    def test_fallback_on_http_status_error(self):
        docs = _make_docs(["alpha", "beta"])

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "403 Forbidden", request=MagicMock(), response=MagicMock()
        )
        with patch("api.utils.reranker.httpx.post", return_value=mock_response):
            result = rerank("q", docs, hf_api_key="bad_key", top_k=5)

        assert [d.page_content for d in result] == ["alpha", "beta"]

    def test_fallback_on_unexpected_response_shape(self):
        """If HF returns wrong number of scores, fall back gracefully."""
        docs = _make_docs(["one", "two", "three"])
        # Only 2 scores for 3 docs → shape mismatch
        with _patch_post([0.5, 0.9]):
            result = rerank("q", docs, hf_api_key="hf_key", top_k=5)

        assert [d.page_content for d in result] == ["one", "two", "three"]

    def test_fallback_on_missing_score_key(self):
        docs = _make_docs(["one", "two"])
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [[{"label": "LABEL_0"}, {"label": "LABEL_0"}]]
        with patch("api.utils.reranker.httpx.post", return_value=mock_response):
            result = rerank("q", docs, hf_api_key="hf_key", top_k=5)

        assert [d.page_content for d in result] == ["one", "two"]


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

        with _patch_post(scores):
            result = rerank("q", docs, hf_api_key="hf_key", top_k=5)

        assert len(result) == 1
        assert result[0].page_content == "only doc"


# ---------------------------------------------------------------------------
# DOCBOT-1511: local (free) provider
# ---------------------------------------------------------------------------


class TestLocalProvider:
    @pytest.fixture(autouse=True)
    def _local(self, monkeypatch):
        monkeypatch.setenv("RERANKER_PROVIDER", "local")

    def test_local_scores_rank_docs_and_need_no_key(self):
        docs = _make_docs(["cat", "net income was 325", "revenue"])
        with patch("api.utils.reranker._local_scores", return_value=[0.01, 0.99, 0.2]) as local, \
             patch("api.utils.reranker.httpx.post") as post:
            out = rerank_scored("q", docs, "", top_k=2)
        assert [d.page_content for d, _ in out] == ["net income was 325", "revenue"]
        assert out[0][1] == pytest.approx(0.99)
        post.assert_not_called()
        local.assert_called_once()

    def test_local_failure_keeps_retrieval_order_with_none_scores(self):
        docs = _make_docs(["a", "b", "c"])
        with patch("api.utils.reranker._local_scores", return_value=None):
            out = rerank_scored("q", docs, "", top_k=2)
        assert [d.page_content for d, _ in out] == ["a", "b"]
        assert all(score is None for _, score in out)

    def test_length_mismatch_is_treated_as_failure(self):
        docs = _make_docs(["a", "b"])
        with patch("api.utils.reranker._local_scores", return_value=[0.5]):
            out = rerank_scored("q", docs, "", top_k=2)
        assert [d.page_content for d, _ in out] == ["a", "b"]

    def test_off_provider_skips_everything(self, monkeypatch):
        monkeypatch.setenv("RERANKER_PROVIDER", "off")
        docs = _make_docs(["a", "b"])
        with patch("api.utils.reranker._local_scores") as local:
            out = rerank_scored("q", docs, "key", top_k=1)
        local.assert_not_called()
        assert [d.page_content for d, _ in out] == ["a"]

    def test_local_scores_applies_sigmoid_and_survives_missing_fastembed(self, monkeypatch):
        monkeypatch.setattr(reranker, "_local_encoder", MagicMock(rerank=lambda q, t: [8.0, -8.0]))
        monkeypatch.setattr(reranker, "_local_unavailable", False)
        scores = reranker._local_scores("q", ["a", "b"])
        assert scores is not None and scores[0] > 0.99 and scores[1] < 0.01

    def test_inference_error_returns_none(self, monkeypatch):
        boom = MagicMock()
        boom.rerank.side_effect = RuntimeError("onnx exploded")
        monkeypatch.setattr(reranker, "_local_encoder", boom)
        monkeypatch.setattr(reranker, "_local_unavailable", False)
        assert reranker._local_scores("q", ["a"]) is None
