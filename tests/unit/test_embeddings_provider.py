"""DOCBOT-1511: embeddings provider (local ONNX default, HF opt-in with fallback).

CI-safe: the ONNX model and the HF endpoint are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from api.utils import embeddings_provider as ep


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    ep.reset_embeddings_cache()
    monkeypatch.delenv("EMBEDDINGS_PROVIDER", raising=False)
    monkeypatch.delenv("huggingface_api_key", raising=False)
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    yield
    ep.reset_embeddings_cache()


def _fake_fn(batch):
    return [[float(len(t))] + [0.0] * (ep.EMBEDDING_DIM - 1) for t in batch]


class TestLocalOnnxEmbeddings:
    def test_embed_documents_returns_float_lists(self):
        emb = ep.LocalOnnxEmbeddings()
        emb._fn = _fake_fn
        out = emb.embed_documents(["ab", "abcd"])
        assert len(out) == 2 and out[1][0] == 4.0
        assert all(isinstance(x, float) for x in out[0])

    def test_empty_input_returns_empty_without_loading_model(self):
        emb = ep.LocalOnnxEmbeddings()
        assert emb.embed_documents([]) == []
        assert emb._fn is None

    def test_blank_text_is_replaced_so_the_model_never_gets_empty_string(self):
        seen = []
        emb = ep.LocalOnnxEmbeddings()
        emb._fn = lambda batch: (seen.extend(batch), _fake_fn(batch))[1]
        emb.embed_documents([""])
        assert seen == [" "]

    def test_batches_large_inputs(self):
        calls = []
        emb = ep.LocalOnnxEmbeddings()
        emb._fn = lambda batch: (calls.append(len(batch)), _fake_fn(batch))[1]
        out = emb.embed_documents(["x"] * (ep._BATCH_SIZE * 2 + 5))
        assert calls == [ep._BATCH_SIZE, ep._BATCH_SIZE, 5]
        assert len(out) == ep._BATCH_SIZE * 2 + 5

    def test_embed_query_returns_single_vector(self):
        emb = ep.LocalOnnxEmbeddings()
        emb._fn = _fake_fn
        assert len(emb.embed_query("hello")) == ep.EMBEDDING_DIM


class TestProviderSelection:
    def test_default_is_local_and_cached(self):
        a = ep.get_embeddings()
        assert isinstance(a, ep.LocalOnnxEmbeddings)
        assert ep.get_embeddings() is a

    def test_hf_without_token_falls_back_to_local(self, monkeypatch):
        monkeypatch.setenv("EMBEDDINGS_PROVIDER", "hf")
        assert isinstance(ep.get_embeddings(), ep.LocalOnnxEmbeddings)

    def test_hf_key_alone_does_not_switch_provider(self, monkeypatch):
        """Having a key set must not silently route to the paid API."""
        monkeypatch.setenv("huggingface_api_key", "hf_x")
        assert isinstance(ep.get_embeddings(), ep.LocalOnnxEmbeddings)

    def test_hf_provider_falls_back_to_local_on_error(self, monkeypatch):
        monkeypatch.setenv("EMBEDDINGS_PROVIDER", "hf")
        monkeypatch.setenv("huggingface_api_key", "hf_x")
        failing = MagicMock()
        failing.embed_documents.side_effect = RuntimeError("402 Payment Required")
        failing.embed_query.side_effect = RuntimeError("402 Payment Required")
        with patch("langchain_huggingface.HuggingFaceEndpointEmbeddings", return_value=failing):
            emb = ep.get_embeddings()
            assert isinstance(emb, ep.HfWithLocalFallbackEmbeddings)
            emb._local._fn = _fake_fn
            assert emb.embed_documents(["abc"])[0][0] == 3.0
            assert emb.embed_query("abcd")[0] == 4.0

    def test_hf_provider_uses_hf_when_healthy(self, monkeypatch):
        monkeypatch.setenv("EMBEDDINGS_PROVIDER", "hf")
        monkeypatch.setenv("huggingface_api_key", "hf_x")
        ok = MagicMock()
        ok.embed_documents.return_value = [[9.0]]
        with patch("langchain_huggingface.HuggingFaceEndpointEmbeddings", return_value=ok):
            emb = ep.get_embeddings()
            assert emb.embed_documents(["a"]) == [[9.0]]
