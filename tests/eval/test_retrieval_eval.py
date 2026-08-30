"""Retrieval quality evaluation — Recall@k over the QuickBite demo 10-K.

Builds a real vector store from the demo document chunks, runs a gold set of
questions, and measures how often the correct source page is retrieved in the
top-k. Requires HuggingFace embeddings (huggingface_api_key), so it is marked
external and skipped in CI.

DOCBOT-1301 baseline: `evaluate_with_expansion()` additionally runs each gold
question through `api.utils.query_expansion.expand_query()` (multi-query
retrieval + dedup), mirroring how `hybrid_service.py` and `index.py` actually
retrieve. This captures whether query expansion helps or hurts finance
retrieval — the pre-rewrite baseline the finance query-rewrite change must
beat before merge.

Run:  pytest tests/eval/eval_retrieval.py -s -m external
  or: python -m tests.eval.eval_retrieval
"""

from __future__ import annotations

import os

import pytest

# (question, set of acceptable source pages) — pages mirror api/demo_service.py.
# Includes overview/cross-section questions (not just single-fact lookups) so
# the baseline isn't trivially easy — a rewrite that only helps clean lookups
# would show a false win otherwise.
GOLD_QA: list[tuple[str, set[int]]] = [
    ("What was QuickBite's total revenue in FY2025?", {3}),
    ("What is QuickBite's net income and net margin?", {3}),
    ("How did each quarter perform in 2025?", {7}),
    ("What was Q4 2025 net income?", {7}),
    ("Break down revenue by business segment", {5}),
    ("How much revenue came from Restaurant Advertising?", {5}),
    ("What are QuickBite's total assets and cash position?", {9}),
    ("What acquisitions did QuickBite complete?", {9}),
    ("What is the FY2026 financial guidance?", {14}),
    ("What is driving QuickBite's revenue growth?", {3}),
    # Overview/cross-section — plausible answer spans multiple pages.
    ("Give an overview of QuickBite's FY2025 financial performance", {3, 7}),
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
    """Return {k: recall@k} over the gold set using a single raw query."""
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

    print("\n=== Retrieval Evaluation (QuickBite demo 10-K, raw query) ===")
    print(f"questions={n}")
    for k in k_values:
        print(f"  Recall@{k}: {recall[k]:.2f}")
    print()
    for q, gold, ranked in detail:
        ok = "OK " if gold & set(ranked[:max(k_values)]) else "MISS"
        print(f"  {ok} gold={sorted(gold)} ranked_pages={ranked}  | {q}")
    return recall


def evaluate_with_expansion(k_values=(1, 3, 5)) -> dict[int, float]:
    """Return {k: recall@k} using expand_query() + dedup, mirroring the real
    hybrid_service.py / index.py retrieval path (multi-query fan-out, merged).
    """
    from api.utils.query_expansion import expand_query, deduplicate_docs

    store = _build_store()
    max_k = max(k_values)
    retriever = store.as_retriever(search_kwargs={"k": max_k})

    hits = {k: 0 for k in k_values}
    detail = []
    for question, gold_pages in GOLD_QA:
        expanded = expand_query(question)
        doc_lists = [retriever.invoke(q) for q in expanded]
        merged = deduplicate_docs(doc_lists)
        ranked_pages = [d.metadata.get("page") for d in merged]
        for k in k_values:
            if gold_pages & set(ranked_pages[:k]):
                hits[k] += 1
        detail.append((question, gold_pages, ranked_pages[:max_k], len(expanded)))

    n = len(GOLD_QA)
    recall = {k: hits[k] / n for k in k_values}

    print("\n=== Retrieval Evaluation (QuickBite demo 10-K, expand_query) ===")
    print(f"questions={n}")
    for k in k_values:
        print(f"  Recall@{k}: {recall[k]:.2f}")
    print()
    for q, gold, ranked, n_expansions in detail:
        ok = "OK " if gold & set(ranked[:max(k_values)]) else "MISS"
        print(f"  {ok} gold={sorted(gold)} ranked_pages={ranked} n_expansions={n_expansions}  | {q}")
    return recall


@pytest.mark.external
def test_retrieval_recall():
    if not (os.getenv("huggingface_api_key") or os.getenv("HUGGINGFACEHUB_API_TOKEN")):
        pytest.skip("huggingface_api_key not set")
    recall = evaluate()
    # A well-tuned retriever should surface the right page in the top-5 for the
    # large majority of demo questions.
    assert recall[5] >= 0.7, f"Recall@5 regressed to {recall[5]:.2f}"


@pytest.mark.external
def test_retrieval_recall_expansion_does_not_regress():
    """DOCBOT-1301 baseline gate: query expansion must not hurt recall vs raw.

    Pre-rewrite, expand_query() injects immigration/HR vocabulary (e.g. "income"
    triggers prevailing-wage phrasing) into finance queries — this is expected
    to be neutral-to-harmful. Once the finance rewrite lands, this test's
    recall_with_expansion should meet or beat raw recall.
    """
    if not (os.getenv("huggingface_api_key") or os.getenv("HUGGINGFACEHUB_API_TOKEN")):
        pytest.skip("huggingface_api_key not set")
    raw = evaluate()
    expanded = evaluate_with_expansion()
    print("\n=== Baseline comparison (raw vs expand_query) ===")
    for k in (1, 3, 5):
        print(f"  Recall@{k}: raw={raw[k]:.2f}  expanded={expanded[k]:.2f}  delta={expanded[k]-raw[k]:+.2f}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    evaluate()
    evaluate_with_expansion()
