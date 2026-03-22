"""Cosine similarity helpers for few-shot query retrieval."""

import json
import math
from typing import List, Optional


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Return cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def find_similar_queries(
    query_embedding: List[float],
    stored: List[dict],
    top_k: int = 3,
    threshold: float = 0.75,
) -> List[dict]:
    """
    Rank *stored* query records by cosine similarity to *query_embedding*.

    Each record in *stored* must have an ``embedding_json`` key (JSON-encoded list).

    Returns up to *top_k* records with similarity >= *threshold*, sorted descending.
    """
    scored = []
    for record in stored:
        raw = record.get("embedding_json")
        if not raw:
            continue
        try:
            vec = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            continue
        score = cosine_similarity(query_embedding, vec)
        if score >= threshold:
            scored.append((score, record))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:top_k]]


def embedding_to_json(embedding: List[float]) -> str:
    """Serialize an embedding vector to a compact JSON string for storage."""
    return json.dumps(embedding)
