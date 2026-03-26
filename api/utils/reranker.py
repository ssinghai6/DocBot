"""Cross-encoder reranker for post-retrieval document re-scoring.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 via HuggingFace Inference API
(no local model download — keeps Railway container lean).
Falls back to original order if API unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_HF_API_URL = (
    "https://api-inference.huggingface.co/models/"
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
_TIMEOUT_SECONDS = 5.0


def rerank(
    query: str,
    docs: list,
    hf_api_key: str,
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
    if not hf_api_key:
        logger.debug("rerank: hf_api_key is empty — skipping cross-encoder")
        return docs[:top_k]

    if not docs:
        return docs

    passages: list[str] = [doc.page_content for doc in docs]

    try:
        response = httpx.post(
            _HF_API_URL,
            headers={"Authorization": f"Bearer {hf_api_key}"},
            json={"inputs": {"query": query, "passages": passages}},
            timeout=_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        scores: Any = response.json()

        # HF returns a list of floats, one per passage.
        if not isinstance(scores, list) or len(scores) != len(docs):
            logger.warning(
                "rerank: unexpected response shape from HF API — "
                "falling back to original order. response=%r",
                scores,
            )
            return docs[:top_k]

        ranked = sorted(
            zip(scores, docs),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [doc for _, doc in ranked[:top_k]]

    except Exception as exc:
        logger.warning(
            "rerank: cross-encoder call failed (%s) — "
            "falling back to original retrieval order",
            exc,
        )
        return docs[:top_k]
