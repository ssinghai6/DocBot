"""
Query expansion utilities for RAG retrieval quality — finance vertical.

Addresses the short-query / semantic-mismatch problem where a natural language
question like "How's the top line doing?" fails to match structured filing
text like "Total Revenue: $42.1M" because the embeddings are too dissimilar in
the all-MiniLM-L6-v2 vector space.

Strategy: lightweight multi-query expansion — no LLM call, no extra deps.
DOCBOT-1301: replaced the original H-1B/LCA immigration-vocabulary rules
(dead weight on the finance vertical DocBot is positioned for) with financial
statement / 10-K terminology. An optional LLM rewrite path (`expand_query_llm`)
is available for callers that want it, with deterministic fallback so no
network call is required in tests or when the LLM is unavailable.

The caller retrieves top-k documents for each query, then deduplicates by
document ID so the final candidate set is the union of all result sets.  This
improves recall without hurting precision because the LLM still reads all
returned chunks and must ground its answer in them.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synonym map: maps a pattern (lowercased words) to expansion phrases.
# Each entry is: (set_of_trigger_words, list_of_expansion_templates)
# ---------------------------------------------------------------------------

_SYNONYM_RULES: list[tuple[set[str], list[str]]] = [
    # Revenue / sales
    (
        {"revenue", "sales", "topline", "top-line", "turnover"},
        [
            "total revenue",
            "net sales",
            "total sales",
            "revenue by segment",
        ],
    ),
    # Margin / profitability
    (
        {"margin", "profitability", "profitable"},
        [
            "operating margin",
            "net margin",
            "gross margin",
            "profit margin",
        ],
    ),
    # Net income / earnings
    (
        {"income", "earnings", "profit", "bottomline", "bottom-line"},
        [
            "net income",
            "net earnings",
            "bottom line profit",
        ],
    ),
    # EBITDA
    (
        {"ebitda"},
        [
            "EBITDA",
            "adjusted EBITDA",
            "earnings before interest taxes depreciation and amortization",
        ],
    ),
    # Segment / business unit
    (
        {"segment", "division", "unit", "vertical"},
        [
            "business segment",
            "reportable segment",
            "segment revenue",
        ],
    ),
    # Growth / trend / YoY
    (
        {"growth", "yoy", "trend", "increase", "decline"},
        [
            "year-over-year growth",
            "YoY growth",
            "revenue growth trend",
        ],
    ),
    # Guidance / outlook / forecast
    (
        {"guidance", "outlook", "forecast", "projection", "projected"},
        [
            "financial guidance",
            "forward guidance",
            "outlook forecast",
        ],
    ),
    # Opex / capex / expenses
    (
        {"opex", "capex", "expense", "expenditure", "spending", "cost"},
        [
            "operating expenses",
            "capital expenditures",
            "operating costs",
        ],
    ),
    # Balance sheet / assets / cash
    (
        {"assets", "liabilities", "balance", "cash", "equity"},
        [
            "total assets",
            "cash and cash equivalents",
            "balance sheet",
            "stockholders equity",
        ],
    ),
    # Acquisitions / M&A
    (
        {"acquisition", "acquisitions", "merger", "acquired", "m&a"},
        [
            "acquisitions completed",
            "mergers and acquisitions",
            "business combinations",
        ],
    ),
    # Cash flow
    (
        {"cashflow", "fcf"},
        [
            "cash flow from operations",
            "free cash flow",
            "operating cash flow",
        ],
    ),
]


def expand_query(question: str) -> list[str]:
    """Return a deduplicated list of query strings for the given question.

    The original question is always the first element.  Additional expansions
    are appended based on synonym rules that match tokens in the question.
    Deterministic and network-free.

    Parameters
    ----------
    question:
        The user's natural-language question, e.g. "How's the top line doing?"

    Returns
    -------
    list[str]
        [original_question, expansion_1, expansion_2, ...]
        Guaranteed to have at least one element (the original).
    """
    queries: list[str] = [question]
    seen: set[str] = {question.lower().strip()}

    lower_q = question.lower()
    tokens = set(re.findall(r"\b[\w&-]+\b", lower_q))

    for trigger_words, expansions in _SYNONYM_RULES:
        if tokens & trigger_words:  # any overlap
            for exp in expansions:
                norm = exp.lower().strip()
                if norm not in seen:
                    seen.add(norm)
                    queries.append(exp)

    return queries


def expand_query_llm(question: str, *, hf_api_key: str | None = None) -> list[str]:
    """Like `expand_query`, but tries an LLM rewrite pass first for finance
    terminology the static synonym map doesn't cover (e.g. non-GAAP framing,
    "run rate", specific line-item names).

    Falls back to the deterministic `expand_query` on any LLM failure (no
    key, rate limit, timeout) so callers always get a usable result and
    callers/tests never require network access to exercise the fallback path.
    """
    try:
        from api.utils.llm_provider import chat_completion

        response = chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user's finance question as 2-3 alternate "
                        "phrasings that would match how the answer is written in "
                        "a 10-K or financial statement (e.g. line-item names, "
                        "GAAP terminology). One phrasing per line, no numbering, "
                        "no explanation."
                    ),
                },
                {"role": "user", "content": question},
            ],
            max_tokens=150,
        )
        llm_expansions = [line.strip() for line in response.splitlines() if line.strip()]
    except Exception as exc:
        logger.warning("expand_query_llm: LLM rewrite failed (%s), using deterministic fallback", exc)
        return expand_query(question)

    queries: list[str] = [question]
    seen: set[str] = {question.lower().strip()}
    for exp in llm_expansions + expand_query(question)[1:]:
        norm = exp.lower().strip()
        if norm not in seen:
            seen.add(norm)
            queries.append(exp)
    return queries


def deduplicate_docs(doc_lists: list[list]) -> list:
    """Merge multiple lists of LangChain Document objects, deduplicating by
    (source, page, first-100-chars).

    Earlier lists have priority — their documents appear first in the output.
    """
    seen: set[str] = set()
    merged: list = []
    for docs in doc_lists:
        for doc in docs:
            key = (
                doc.metadata.get("source", ""),
                doc.metadata.get("page", 0),
                doc.page_content[:100],
            )
            if key not in seen:
                seen.add(key)
                merged.append(doc)
    return merged
