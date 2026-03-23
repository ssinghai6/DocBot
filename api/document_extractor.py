"""
DOCBOT-406: LangExtract — Structured Financial Document Extraction

Detects financial documents at upload time and extracts typed numeric values
using Google LangExtract (Gemini-backed). Each extracted value is anchored to
its exact char_interval in the source text, providing precise span verification
for the hybrid synthesis discrepancy detection logic.

Public API:
    is_financial_document(text) -> bool
    extract_financial_fields(text, session_id, gemini_api_key) -> list[ExtractedField]
    format_extracted_fields_for_prompt(fields) -> str
"""

from __future__ import annotations

import logging
import re
import textwrap
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

_NUMERIC_DENSITY_THRESHOLD = 0.015  # 1.5%
_MIN_KEYWORD_HITS = 3


def is_financial_document(text: str) -> bool:
    """Return True if heuristics suggest this is a financial document.

    Checks keyword presence (>=3 financial terms) and numeric density
    (>=1.5% of tokens contain a digit). Both conditions must hold.
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

    name: str        # e.g. "ARR target", "Gross Margin"
    value: float     # numeric value (e.g. 2400000.0)
    unit: str        # "USD", "%", "units", "headcount", etc.
    span_text: str   # verbatim substring from the source document
    page: int        # 0 if unknown
    verified: bool   # True if char_interval was resolved by LangExtract


# ---------------------------------------------------------------------------
# LangExtract prompt + examples
# ---------------------------------------------------------------------------

_PROMPT = textwrap.dedent("""
    Extract all numeric financial metrics from the text.
    For each metric provide:
      - The exact metric name (e.g. "ARR", "Gross Margin", "Headcount")
      - The numeric value as a float (convert $2.4M to 2400000.0, 15% to 15.0, $1B to 1000000000.0)
      - The unit (USD, %, units, headcount, etc.)
    Use the exact verbatim text containing the number as the extraction_text.
    Do not paraphrase or overlap entities.
""").strip()

_EXAMPLES_RAW = [
    {
        "text": "Annual Recurring Revenue (ARR) reached $2.4M, up 35% YoY.",
        "extractions": [
            {"class": "financial_metric", "text": "$2.4M", "name": "ARR", "value": 2400000.0, "unit": "USD"},
            {"class": "financial_metric", "text": "35%", "name": "ARR YoY Growth", "value": 35.0, "unit": "%"},
        ],
    },
    {
        "text": "Gross margin improved to 68% in Q3 FY2024 from 61% in Q3 FY2023.",
        "extractions": [
            {"class": "financial_metric", "text": "68%", "name": "Gross Margin Q3 FY2024", "value": 68.0, "unit": "%"},
            {"class": "financial_metric", "text": "61%", "name": "Gross Margin Q3 FY2023", "value": 61.0, "unit": "%"},
        ],
    },
]


def _build_examples():
    """Build lx.data.ExampleData list for lx.extract()."""
    import langextract as lx

    examples = []
    for ex in _EXAMPLES_RAW:
        extractions = [
            lx.data.Extraction(
                extraction_class=e["class"],
                extraction_text=e["text"],
                attributes={"name": e["name"], "value": str(e["value"]), "unit": e["unit"]},
            )
            for e in ex["extractions"]
        ]
        examples.append(lx.data.ExampleData(text=ex["text"], extractions=extractions))
    return examples


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


async def extract_financial_fields(
    text: str,
    session_id: str,
    gemini_api_key: str,
) -> list[ExtractedField]:
    """Extract typed financial values from document text using LangExtract.

    Uses Gemini via LangExtract for full-document chunked extraction with
    precise char_interval source grounding. Only fields with a resolved
    char_interval (verified=True) are returned.

    Returns an empty list on any error — never raises.
    """
    if not text or not gemini_api_key:
        return []

    try:
        import asyncio
        import langextract as lx

        examples = _build_examples()
        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(
            None,
            lambda: lx.extract(
                text_or_documents=text,
                prompt_description=_PROMPT,
                examples=examples,
                model_id="gemini-2.5-flash",
                api_key=gemini_api_key,
                show_progress=False,
                max_workers=2,  # free tier: 10 RPM cap — stay well within limit
            ),
        )

        # lx.extract() returns AnnotatedDocument for single text input
        raw_extractions = result.extractions if result.extractions else []

        fields: list[ExtractedField] = []
        for extraction in raw_extractions:
            if extraction.char_interval is None:
                continue  # not grounded in source — skip

            attrs = extraction.attributes or {}
            name = attrs.get("name", extraction.extraction_class or "unknown").strip()
            unit = attrs.get("unit", "USD").strip()

            raw_value = attrs.get("value", "")
            try:
                value = float(raw_value)
            except (ValueError, TypeError):
                value = _parse_numeric(extraction.extraction_text)
                if value is None:
                    continue

            span_text = extraction.extraction_text or ""
            verified = bool(span_text) and span_text in text

            fields.append(ExtractedField(
                name=name,
                value=value,
                unit=unit,
                span_text=span_text,
                page=0,
                verified=verified,
            ))

        verified_count = sum(1 for f in fields if f.verified)
        logger.info(
            "extract_financial_fields: session=%s extracted=%d verified=%d",
            session_id,
            len(fields),
            verified_count,
        )
        return fields

    except Exception as exc:
        logger.warning("extract_financial_fields failed: %s", exc)
        return []


def _parse_numeric(text: str) -> Optional[float]:
    """Best-effort numeric parse from a raw span string like '$2.4M' or '15%'."""
    if not text:
        return None
    cleaned = re.sub(r"[$,%]", "", text.replace(",", "")).strip()
    multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    lower = cleaned.lower()
    for suffix, mult in multipliers.items():
        if lower.endswith(suffix):
            try:
                return float(lower[:-1]) * mult
            except ValueError:
                return None
    try:
        return float(cleaned)
    except ValueError:
        return None


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
            f'  - {f.name}: {f.value:,.2f} {f.unit} '
            f'(page {f.page}, span: "{f.span_text[:80]}")'
        )
    return "\n".join(lines)
