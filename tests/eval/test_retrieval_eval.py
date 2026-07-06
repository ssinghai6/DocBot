"""Retrieval quality evaluation — Recall@k over the TechCorp demo 10-K.

Builds a real vector store from the demo document chunks, runs a gold set of
questions, and measures how often the correct source page is retrieved in the
top-k. Requires HuggingFace embeddings (huggingface_api_key), so it is marked
external and skipped in CI.

Run:  pytest tests/eval/eval_retrieval.py -s -m external
  or: python -m tests.eval.eval_retrieval
"""

from __future__ import annotations

import os

import pytest

# (question, set of acceptable source pages) — pages mirror api/demo_service.py.
GOLD_QA: list[tuple[str, set[int]]] = [
    ("What was TechCorp's total revenue in FY2024?", {3}),
    ("What is TechCorp's net income and net margin?", {3}),
    ("How did each quarter perform in 2024?", {7}),
    ("What was Q4 2024 net income?", {7}),
    ("Break down revenue by business segment", {5}),
    ("How much revenue came from Professional Services?", {5}),
    ("What are TechCorp's total assets and cash position?", {9}),
    ("What acquisitions did TechCorp complete?", {12}),
    ("What is the FY2025 financial guidance?", {14}),
    ("What is driving TechCorp's revenue growth?", {3}),
]


def _build_store():
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
    from api.demo_service import DEMO_DOCUMENT_CHUNKS
    from api.utils.vector_store import create_store

    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("huggingface_api_key")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )
    return create_store("eval_retrieval", DEMO_DOCUMENT_CHUNKS, embeddings)


def evaluate(k_values=(1, 3, 5)) -> dict[int, float]:
    """Return {k: recall@k} over the gold set."""
    store = _build_store()
    max_k = max(k_values)
    retriever = store.as_retriever(search_kwargs={"k": max_k})

    hits = {k: 0 for k in k_values}
    detail = []
    for question, gold_pages in GOLD_QA:
        docs = retriever.invoke(question)
        ranked_pages = [d.metadata.get("page") for d in docs]
        for k in k_values:
            if gold_pages & set(ranked_pages[:k]):
                hits[k] += 1
        detail.append((question, gold_pages, ranked_pages[:max_k]))

    n = len(GOLD_QA)
    recall = {k: hits[k] / n for k in k_values}

    print("\n=== Retrieval Evaluation (TechCorp demo 10-K) ===")
    print(f"questions={n}")
    for k in k_values:
        print(f"  Recall@{k}: {recall[k]:.2f}")
    print()
    for q, gold, ranked in detail:
        ok = "OK " if gold & set(ranked[:max(k_values)]) else "MISS"
        print(f"  {ok} gold={sorted(gold)} ranked_pages={ranked}  | {q}")
    return recall


@pytest.mark.external
def test_retrieval_recall():
    if not (os.getenv("huggingface_api_key") or os.getenv("HUGGINGFACEHUB_API_TOKEN")):
        pytest.skip("huggingface_api_key not set")
    recall = evaluate()
    # A well-tuned retriever should surface the right page in the top-5 for the
    # large majority of demo questions.
    assert recall[5] >= 0.7, f"Recall@5 regressed to {recall[5]:.2f}"


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    evaluate()
