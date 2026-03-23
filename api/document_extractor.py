"""
DOCBOT-406: LangExtract — Structured Financial Document Extraction

Detects financial documents at upload time and extracts typed numeric values
with span verification. Each extracted value is anchored to its verbatim source
span so the hybrid synthesis prompt can perform precise arithmetic comparisons
instead of vague text-level reasoning.

Public API:
    is_financial_document(text) -> bool
    extract_financial_fields(text, session_id, groq_api_key) -> list[ExtractedField]
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Financial document heuristics
# ---------------------------------------------------------------------------

_FINANCIAL_KEYWORDS = {
    "revenue", "arr", "mrr", "ebitda", "margin", "forecast", "budget",
    "projection", "target", "actuals", "gross", "net", "profit", "loss",
    "p&l", "income", "expense", "capex", "opex", "cogs", "arpu", "ltv",
    "cac", "runway", "burn", "headcount", "yoy", "qoq", "cagr", "fy",
    "quarter", "q1", "q2", "q3", "q4", "fiscal", "annual", "monthly",
}

# At least this fraction of whitespace-split tokens must contain a digit
_NUMERIC_DENSITY_THRESHOLD = 0.015  # 1.5%
_MIN_KEYWORD_HITS = 3


def is_financial_document(text: str) -> bool:
    """Return True if heuristics suggest this is a financial document.

    Checks keyword presence (≥3 financial terms) and numeric density
    (≥1.5% of tokens contain a digit). Both conditions must hold.
    """
    if not text:
        return False

    lower = text.lower()
    tokens = lower.split()
    if len(tokens) < 50:
        return False

    keyword_hits = sum(1 for kw in _FINANCIAL_KEYWORDS if kw in lower)
    if keyword_hits < _MIN_KEYWORD_HITS:
        return False

    numeric_tokens = sum(1 for t in tokens if re.search(r"\d", t))
    density = numeric_tokens / len(tokens)
    return density >= _NUMERIC_DENSITY_THRESHOLD


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------


class ExtractedField(BaseModel):
    """A single typed financial value extracted from a document."""

    name: str                  # e.g. "ARR target", "Gross Margin"
    value: float               # numeric value (e.g. 2400000.0)
    unit: str                  # "USD", "%", "units", "headcount", etc.
    span_text: str             # verbatim substring from the source document
    page: int                  # 0 if unknown
    verified: bool             # True if span_text was found in source text


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = (
    "You are a financial data extractor. Extract key financial metrics from the text.\n"
    "Return a JSON array where each element has EXACTLY these fields:\n"
    '  {"name": "metric name", "value": 1234567.0, "unit": "USD|%|units", '
    '"span_text": "exact verbatim quote from text containing the number", "page": 0}\n\n'
    "Rules:\n"
    "- Extract only numeric financial values: revenues, targets, margins, headcount, etc.\n"
    "- value must be a float (convert $2.4M → 2400000.0, 15% → 15.0, $1B → 1000000000.0)\n"
    "- span_text must be an exact substring that appears verbatim in the input text\n"
    "- page is 0 if not identifiable from the text\n"
    "- Return [] if no clear financial values are found\n"
    "- Output ONLY the JSON array — no markdown fences, no explanation"
)


async def extract_financial_fields(
    text: str,
    session_id: str,
    groq_api_key: str,
) -> list[ExtractedField]:
    """Extract typed financial values from document text.

    Uses Groq LLM to identify financial metrics, then span-verifies each
    extracted value against the original text. Only verified fields
    (span_text found verbatim) are returned.

    Returns an empty list on any error — never raises.
    """
    if not text or not groq_api_key:
        return []

    # Sample first 8 000 chars — enough for most board decks / P&L pages
    sample = text[:8_000]

    try:
        import groq as groq_module

        client = groq_module.Groq(api_key=groq_api_key)
        loop = asyncio.get_running_loop()

        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Extract financial metrics from this text:\n\n{sample}",
                    },
                ],
                max_tokens=1_000,
                temperature=0,
            ),
        )

        raw = response.choices[0].message.content.strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:]).rstrip("` \n")

        fields_raw = json.loads(raw)
        if not isinstance(fields_raw, list):
            return []

        results: list[ExtractedField] = []
        for item in fields_raw:
            if not isinstance(item, dict):
                continue
            try:
                span = item.get("span_text", "")
                verified = bool(span) and span in text
                field = ExtractedField(
                    name=str(item.get("name", "")).strip(),
                    value=float(item.get("value", 0)),
                    unit=str(item.get("unit", "USD")).strip(),
                    span_text=span,
                    page=int(item.get("page", 0)),
                    verified=verified,
                )
                results.append(field)
            except (ValueError, TypeError):
                continue

        verified_count = sum(1 for f in results if f.verified)
        logger.info(
            "extract_financial_fields: session=%s extracted=%d verified=%d",
            session_id,
            len(results),
            verified_count,
        )
        return results

    except Exception as exc:
        logger.warning("extract_financial_fields failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Formatting helper for synthesis prompts
# ---------------------------------------------------------------------------


def format_extracted_fields_for_prompt(fields: list[ExtractedField]) -> str:
    """Render verified ExtractedFields as a compact block for LLM prompts.

    Only includes verified fields to avoid hallucinated values polluting the
    discrepancy detection logic.
    """
    verified = [f for f in fields if f.verified]
    if not verified:
        return ""

    lines = ["Typed financial values extracted from this document (span-verified):"]
    for f in verified:
        lines.append(
            f'  • {f.name}: {f.value:,.2f} {f.unit} '
            f'(page {f.page}, span: "{f.span_text[:80]}")'
        )
    return "\n".join(lines)
