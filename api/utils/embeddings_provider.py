"""DOCBOT-1511: free, local embeddings (no API key, no credits).

Default provider runs ``all-MiniLM-L6-v2`` through ONNX Runtime using the
copy that ships with ``chromadb`` (already a dependency), so no new package
is needed. It is the same model the HuggingFace endpoint served, so vectors
already stored in Chroma / Postgres stay usable.

Env ``EMBEDDINGS_PROVIDER``:
  * ``local`` (default): ONNX on this machine. No network after first model download.
  * ``hf``: HuggingFace Inference API, falling back to local on any error.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384
_HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_BATCH_SIZE = 64


class LocalOnnxEmbeddings(Embeddings):
    """all-MiniLM-L6-v2 on ONNX Runtime. Thread-safe, lazily loaded."""

    def __init__(self) -> None:
        self._fn = None
        self._lock = threading.Lock()

    def _get_fn(self):
        if self._fn is None:
            with self._lock:
                if self._fn is None:
                    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

                    self._fn = ONNXMiniLM_L6_V2()
        return self._fn

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        fn = self._get_fn()
        out: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = [t if t else " " for t in texts[start : start + _BATCH_SIZE]]
            out.extend([[float(x) for x in vec] for vec in fn(batch)])
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class HfWithLocalFallbackEmbeddings(Embeddings):
    """HF Inference API first; local ONNX when it errors (402, 5xx, timeout)."""

    def __init__(self, hf_token: str, local: LocalOnnxEmbeddings) -> None:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        self._hf = HuggingFaceEndpointEmbeddings(
            model=_HF_MODEL,
            task="feature-extraction",
            huggingfacehub_api_token=hf_token,
        )
        self._local = local

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._hf.embed_documents(texts)
        except Exception as exc:  # HF client raises varied HTTP/network errors
            logger.warning("embeddings: HF failed (%s); using local ONNX", type(exc).__name__)
            return self._local.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._hf.embed_query(text)
        except Exception as exc:
            logger.warning("embeddings: HF failed (%s); using local ONNX", type(exc).__name__)
            return self._local.embed_query(text)


_cache: Optional[Embeddings] = None
_cache_lock = threading.Lock()


def get_embeddings() -> Embeddings:
    """Return the process-wide embeddings model for the configured provider."""
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                local = LocalOnnxEmbeddings()
                provider = os.getenv("EMBEDDINGS_PROVIDER", "local").strip().lower()
                token = os.getenv("huggingface_api_key") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
                if provider == "hf" and token:
                    _cache = HfWithLocalFallbackEmbeddings(token, local)
                else:
                    if provider == "hf":
                        logger.warning("embeddings: EMBEDDINGS_PROVIDER=hf but no HF token; using local")
                    _cache = local
    return _cache


def reset_embeddings_cache() -> None:
    """Drop the cached model (used by tests)."""
    global _cache
    with _cache_lock:
        _cache = None
