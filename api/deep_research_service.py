"""Deep retrieval helpers used by Autopilot's doc_search tool.

This module previously hosted a full LangGraph-driven Deep Research route
(query_planner → parallel_retriever → evidence_evaluator → synthesizer).
That standalone route was retired when Autopilot adopted the same retrieval
primitive: only ``deep_retrieve()`` is kept here, callable from
``api/autopilot_service.py``.

LLM call budget for ``deep_retrieve``: 1 (sub-question planner). The gap-fill
loop is pure vector search.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from api.utils.query_expansion import deduplicate_docs, expand_query

logger = logging.getLogger(__name__)

MIN_CHUNKS_FOR_COVERAGE = 2
MAX_ITERATIONS = 2


def _get_llm(groq_api_key: str, streaming: bool = False):
    """Return a ChatGroq LLM bound to the documented coordination model."""
    from langchain_groq import ChatGroq

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=groq_api_key,
        streaming=streaming,
    )


def _parse_json_list(raw: str, fallback: list[str]) -> list[str]:
    """Parse a JSON array of strings out of an LLM response, with fallback.

    Strips markdown fences and trailing prose. If parsing fails or the result
    is not a non-empty list of strings, ``fallback`` is returned.
    """
    if not raw:
        return fallback

    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return fallback

    if not isinstance(parsed, list):
        return fallback

    items = [item for item in parsed if isinstance(item, str) and item.strip()]
    return items if items else fallback


async def deep_retrieve(
    question: str,
    vector_store: Any,
    max_iterations: int = 2,
) -> tuple[list[Document], list[str]]:
    """Run the deep retrieval pipeline without LangGraph overhead.

    Performs sub-question decomposition, parallel retrieval with query
    expansion, coverage evaluation, and gap-fill re-retrieval.

    Parameters
    ----------
    question:
        The natural-language question to retrieve documents for.
    vector_store:
        A LangChain-compatible vector store (must support ``.as_retriever()``).
    max_iterations:
        Maximum number of retrieval passes (initial + gap-fill loops).

    Returns
    -------
    tuple[list[Document], list[str]]
        (deduplicated_documents, sub_questions_used)
    """
    groq_api_key = os.getenv("groq_api_key", "")

    # ── Step 1: Decompose question into sub-questions ─────────────────────
    sub_questions: list[str] = [question]
    if groq_api_key:
        try:
            llm = _get_llm(groq_api_key)
            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are a research planning assistant. Given a user question, decompose it "
                    "into 3-5 focused sub-questions that together fully cover the original question. "
                    "Each sub-question must be answerable independently from a document. "
                    "Return ONLY a valid JSON array of strings — no commentary, no markdown fences.\n"
                    'Example: ["What are the eligibility requirements?", "What is the deadline?"]'
                )),
                ("human", "{question}"),
            ])
            chain = prompt | llm | StrOutputParser()
            raw = await asyncio.wait_for(
                chain.ainvoke({"question": question}),
                timeout=15.0,
            )
            sub_questions = _parse_json_list(raw, fallback=[question])[:5]
        except Exception as exc:
            logger.warning("deep_retrieve planner failed, using original question: %s", exc)
            sub_questions = [question]

    logger.info("deep_retrieve sub-questions: %s", sub_questions)

    # ── Step 2: Parallel retrieval with query expansion ───────────────────
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )

    async def _fetch(sub_q: str) -> tuple[str, list[Document]]:
        expanded = expand_query(sub_q)
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=min(len(expanded), 6)) as pool:
            result_lists = await asyncio.gather(
                *[
                    asyncio.wait_for(
                        loop.run_in_executor(pool, retriever.invoke, q),
                        timeout=5.0,
                    )
                    for q in expanded
                ],
                return_exceptions=True,
            )
        valid = [r for r in result_lists if isinstance(r, list)]
        return sub_q, deduplicate_docs(valid)

    retrieved_chunks: dict[str, list[Document]] = {}
    iterations = 0

    while iterations < max_iterations:
        iterations += 1

        # First pass: search all sub-questions. Subsequent passes: only gaps.
        search_targets = (
            [sq for sq in retrieved_chunks if len(retrieved_chunks[sq]) < MIN_CHUNKS_FOR_COVERAGE]
            if iterations > 1
            else sub_questions
        )

        if not search_targets:
            break

        results = await asyncio.gather(*[_fetch(sq) for sq in search_targets])

        for sub_q, docs in results:
            prior = retrieved_chunks.get(sub_q, [])
            retrieved_chunks[sub_q] = deduplicate_docs([prior, docs])

        # ── Step 3: Evaluate coverage — check for gaps ────────────────────
        gaps = [
            sq for sq, docs in retrieved_chunks.items()
            if len(docs) < MIN_CHUNKS_FOR_COVERAGE
        ]
        logger.info(
            "deep_retrieve iteration %d: %d sub-questions, %d gaps",
            iterations, len(retrieved_chunks), len(gaps),
        )
        if not gaps:
            break

    # ── Step 4: Flatten and deduplicate all documents ─────────────────────
    all_docs: list[Document] = []
    for docs in retrieved_chunks.values():
        all_docs.extend(docs)

    all_docs = deduplicate_docs([all_docs])

    return all_docs, sub_questions
