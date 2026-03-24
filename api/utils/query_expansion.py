"""
Query expansion utilities for RAG retrieval quality.

Addresses the short-query / semantic-mismatch problem where a 4-word natural
language question like "His position or title?" fails to match structured form
text like "Job Title: Data Science Engineer" because the embeddings are too
dissimilar in the all-MiniLM-L6-v2 vector space.

Strategy: lightweight multi-query expansion — no LLM call, no extra deps.

For each incoming question we generate 2-4 additional phrasings that cover:
  - Synonym substitutions for common HR/legal/finance field names
  - Declarative restatements that match how form data is written (not how
    questions are asked)
  - Specific term variants (e.g. "role" → "job title", "designation",
    "position", "occupation")

The caller retrieves top-k documents for each query, then deduplicates by
document ID so the final candidate set is the union of all result sets.  This
improves recall without hurting precision because the LLM still reads all
returned chunks and must ground its answer in them.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Synonym map: maps a pattern (lowercased words) to expansion phrases.
# Each entry is: (set_of_trigger_words, list_of_expansion_templates)
#
# Templates may contain "{subject}" which is replaced with any detected
# subject noun phrase (e.g. "his", "her", "their", the person's name).
# ---------------------------------------------------------------------------

_SYNONYM_RULES: list[tuple[set[str], list[str]]] = [
    # Job title / role / position
    (
        {"position", "title", "role", "job", "designation", "occupation", "work"},
        [
            "job title",
            "position title",
            "job title occupation title",
            "SOC occupation title",
            "role designation",
            "current job title",
        ],
    ),
    # Name / identity
    (
        {"name", "full name", "who"},
        [
            "full legal name",
            "applicant name",
            "employee name",
            "worker name",
        ],
    ),
    # Salary / wages / compensation
    (
        {"salary", "wage", "compensation", "pay", "income", "earning"},
        [
            "wage rate",
            "prevailing wage",
            "offered wage",
            "annual salary",
            "hourly wage",
            "compensation amount",
        ],
    ),
    # Location / address / workplace
    (
        {"location", "address", "workplace", "office", "site", "place", "where"},
        [
            "work location",
            "place of employment",
            "worksite address",
            "employer address",
        ],
    ),
    # Employer / company
    (
        {"employer", "company", "organization", "organisation", "firm"},
        [
            "employer name",
            "company name",
            "legal business name",
        ],
    ),
    # Dates / period / duration
    (
        {"date", "when", "period", "duration", "start", "end", "begin", "expire"},
        [
            "employment period",
            "validity period",
            "begin date end date",
            "start date",
        ],
    ),
    # Visa / status / classification
    (
        {"visa", "status", "classification", "h1b", "h-1b", "lca", "immigration"},
        [
            "visa classification",
            "nonimmigrant classification",
            "H-1B classification",
            "labor condition application",
        ],
    ),
    # SOC code
    (
        {"soc", "code", "occupation code"},
        [
            "SOC code",
            "SOC occupation code",
            "occupational classification code",
        ],
    ),
    # Hours / schedule
    (
        {"hours", "schedule", "weekly", "part", "full", "time"},
        [
            "hours per week",
            "work schedule",
            "full-time part-time",
        ],
    ),
]


def _extract_subject(question: str) -> str:
    """Return a best-guess subject noun phrase from the question."""
    lower = question.lower()
    for pronoun in ("his", "her", "their", "its", "my", "your"):
        if pronoun in lower:
            return pronoun
    # If a proper name is present (capitalised word not at sentence start), use it
    words = question.split()
    for i, w in enumerate(words):
        cleaned = w.strip("?,.")
        if i > 0 and cleaned and cleaned[0].isupper():
            return cleaned
    return ""


def expand_query(question: str) -> list[str]:
    """Return a deduplicated list of query strings for the given question.

    The original question is always the first element.  Additional expansions
    are appended based on synonym rules that match tokens in the question.

    Parameters
    ----------
    question:
        The user's natural-language question, e.g. "His position or title?"

    Returns
    -------
    list[str]
        [original_question, expansion_1, expansion_2, ...]
        Guaranteed to have at least one element (the original).
    """
    queries: list[str] = [question]
    seen: set[str] = {question.lower().strip()}

    lower_q = question.lower()
    tokens = set(re.findall(r"\b\w+\b", lower_q))

    for trigger_words, expansions in _SYNONYM_RULES:
        if tokens & trigger_words:  # any overlap
            for exp in expansions:
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
