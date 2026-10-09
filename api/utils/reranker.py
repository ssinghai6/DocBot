"""Cross-encoder reranker for post-retrieval document re-scoring.

Uses BAAI/bge-reranker-base via the HuggingFace Inference API (no local
model download -- keeps the Railway container lean, per CLAUDE.md).
Falls back to original order if the API is unavailable.

DOCBOT-1405: the original implementation called
``InferenceClient(...).sentence_similarity()`` against
``sentence-transformers/all-MiniLM-L6-v2`` -- the *same* bi-encoder model
already used for retrieval embeddings (see api/utils/chunker.py /
api/utils/embeddings.py). A bi-encoder embeds the query and each passage
independently and compares vectors; it cannot perform the joint
query+passage attention a real cross-encoder does, so reranking with it
added a ~5s network round trip for almost no new ranking signal beyond
what retrieval already provided.

``cross-encoder/ms-marco-MiniLM-L-6-v2`` -- the natural first choice for a
lightweight cross-encoder -- has no ``inferenceProviderMapping`` on the Hub
at all (confirmed via ``HfApi().model_info(..., expand=["inferenceProviderMapping"])``
returning ``[]``), so it cannot be reached through *any* HF Inference
Provider, old or new client API. ``BAAI/bge-reranker-base`` is a real
cross-encoder that IS live on the ``hf-inference`` provider (task
``text-classification``), verified with a real API call: it scored a
clearly relevant passage at 0.997 and two irrelevant passages at ~0.00004,
a far sharper signal than the old bi-encoder's 0.01-0.84 spread on the same
inputs.

``huggingface_hub`` 0.36's ``InferenceClient.text_classification()`` only
accepts a single ``text: str`` (no ``text_pair``), so it cannot express a
cross-encoder query/passage pair. We instead POST directly to the
``hf-inference`` router with the pair-batch payload shape the underlying
HF text-classification pipeline expects for cross-encoders:
``{"inputs": [{"text": query, "text_pair": passage}, ...]}``, scoring all
passages in a single request. This was verified against the real HF
Inference API (not just mocked) before landing.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_MODEL = "BAAI/bge-reranker-base"
_INFERENCE_URL = f"https://router.huggingface.co/hf-inference/models/{_MODEL}"
_TIMEOUT_SECONDS = 10

# DOCBOT-1511: free local reranker (ONNX, ~90MB). Env ``RERANKER_PROVIDER``:
#   local (default) | hf (HuggingFace Inference API, needs a key) | off
_LOCAL_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
_local_encoder = None
_local_lock = threading.Lock()
_local_unavailable = False


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


def _local_scores(query: str, texts: list[str]) -> Optional[list[float]]:
    """Score ``texts`` against ``query`` with the local cross-encoder (0-1).

    Returns None when fastembed is missing or inference fails, so the caller
    falls back to retrieval order.
    """
    global _local_encoder, _local_unavailable
    if _local_unavailable:
        return None
    try:
        if _local_encoder is None:
            with _local_lock:
                if _local_encoder is None:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    _local_encoder = TextCrossEncoder(_LOCAL_MODEL)
        return [_sigmoid(float(x)) for x in _local_encoder.rerank(query, texts)]
    except ImportError:
        _local_unavailable = True
        logger.warning("rerank: fastembed not installed; keeping retrieval order")
        return None
    except Exception as exc:  # onnx/model-download errors vary
        logger.warning("rerank: local cross-encoder failed (%s); keeping retrieval order", exc)
        return None


def rerank_scored(
    query: str,
    docs: list,
    hf_api_key: str = "",
    top_k: int = 5,
) -> list[tuple]:
    """Like :func:`rerank` but returns ``(doc, score)`` pairs.

    DOCBOT-1510: the Inspector lineage shows each chunk's cross-encoder
    score. ``score`` is ``None`` whenever no cross-encoder ran (empty key,
    API failure, unexpected response) so callers can tell "not reranked"
    apart from "scored low".
    """
    if not docs:
        return []

    provider = os.getenv("RERANKER_PROVIDER", "local").strip().lower()
    if provider == "off":
        return [(d, None) for d in docs[:top_k]]
    if provider != "hf":
        scores = _local_scores(query, [d.page_content for d in docs])
        if scores is None or len(scores) != len(docs):
            return [(d, None) for d in docs[:top_k]]
        ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
        return [(doc, float(score)) for doc, score in ranked[:top_k]]

    if not hf_api_key:
        logger.debug("rerank: hf_api_key is empty — skipping cross-encoder")
        return [(d, None) for d in docs[:top_k]]

    payload = {
        "inputs": [
            {"text": query, "text_pair": doc.page_content} for doc in docs
        ]
    }

    try:
        response = httpx.post(
            _INFERENCE_URL,
            headers={"Authorization": f"Bearer {hf_api_key}"},
            json=payload,
            timeout=_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        result = response.json()

        if (
            not isinstance(result, list)
            or len(result) != 1
            or not isinstance(result[0], list)
            or len(result[0]) != len(docs)
        ):
            logger.warning(
                "rerank: unexpected response shape from HF API — "
                "falling back to original order. response=%r",
                result,
            )
            return [(d, None) for d in docs[:top_k]]

        scores = [item["score"] for item in result[0]]

        ranked = sorted(
            zip(docs, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [(doc, float(score)) for doc, score in ranked[:top_k]]

    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        logger.warning(
            "rerank: cross-encoder call failed (%s) — "
            "falling back to original retrieval order",
            exc,
        )
        return [(d, None) for d in docs[:top_k]]


def rerank(
    query: str,
    docs: list,
    hf_api_key: str = "",
    top_k: int = 5,
) -> list:
    """Re-score retrieved documents with a cross-encoder and return top_k.

    Parameters
    ----------
    query:
        The natural-language question used for retrieval.
    docs:
        List of LangChain Document objects (must have ``.page_content``).
    hf_api_key:
        HuggingFace Inference API key.  When empty the function returns
        ``docs[:top_k]`` without making any network call.
    top_k:
        Maximum number of documents to return after re-ranking.

    Returns
    -------
    list
        Up to ``top_k`` Document objects, sorted by cross-encoder score
        descending.  On any failure the original order is preserved.
    """
    return [doc for doc, _ in rerank_scored(query, docs, hf_api_key, top_k)]
