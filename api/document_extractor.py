"""
SCRUM-391: Broadened LangExtract — Universal Structured Extraction

Detects document type at upload time and extracts typed, span-verified values
using Google LangExtract (Gemini 2.5 Flash). Works across five document types:

  financial  — revenue, margins, ARR, forecasts, budgets
  legal      — parties, dates, obligations, penalties, durations
  medical    — diagnoses, medications, lab values, dosages, durations
  research   — findings, sample sizes, p-values, effect sizes, confidence intervals
  general    — any key numeric/categorical fact the LLM can ground in the text

Public API:
    detect_document_type(text) -> str
    is_extractable_document(text) -> bool
    extract_document_fields(text, session_id, gemini_api_key) -> list[ExtractedField]
    format_extracted_fields_for_prompt(fields) -> str

    # backward-compat aliases
    is_financial_document(text) -> bool
    extract_financial_fields(text, session_id, gemini_api_key) -> list[ExtractedField]
"""

from __future__ import annotations

import logging
import re
import textwrap
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Document type detection
# ---------------------------------------------------------------------------

_KEYWORDS: dict[str, set[str]] = {
    "financial": {
        "revenue", "arr", "mrr", "ebitda", "margin", "forecast", "budget",
        "projection", "target", "actuals", "gross", "net", "profit", "loss",
        "p&l", "income", "expense", "capex", "opex", "cogs", "arpu", "ltv",
        "cac", "runway", "burn", "headcount", "yoy", "qoq", "cagr", "fy",
        "quarter", "q1", "q2", "q3", "q4", "fiscal", "annual", "monthly",
    },
    "legal": {
        "agreement", "contract", "party", "parties", "clause", "liability",
        "indemnify", "indemnification", "warrant", "warranty", "breach",
        "termination", "jurisdiction", "governing law", "obligation",
        "penalty", "damages", "arbitration", "confidential", "whereas",
        "hereinafter", "executed", "effective date", "term", "covenant",
    },
    "medical": {
        "patient", "diagnosis", "treatment", "medication", "dosage", "mg",
        "prescription", "symptom", "clinical", "laboratory", "lab",
        "blood pressure", "glucose", "cholesterol", "hemoglobin", "bmi",
        "icd", "cpt", "procedure", "surgery", "prognosis", "chronic",
        "acute", "physician", "pharmacist", "dose", "mmhg", "bpm",
    },
    "research": {
        "abstract", "hypothesis", "methodology", "sample size", "p-value",
        "confidence interval", "statistical significance", "regression",
        "correlation", "cohort", "randomized", "controlled trial",
        "literature review", "findings", "conclusion", "journal",
        "participants", "mean", "median", "standard deviation", "effect size",
        "peer-reviewed", "citation", "bibliography",
    },
}

_NUMERIC_DENSITY_THRESHOLD = 0.015  # 1.5% of tokens contain a digit
_MIN_KEYWORD_HITS = 3


def detect_document_type(text: str) -> str:
    """Classify document into one of: financial, legal, medical, research, general.

    Scores each category by keyword hits. Returns the highest-scoring category
    if it meets the minimum threshold, otherwise returns 'general'.
    """
    if not text:
        return "general"

    lower = text.lower()
    tokens = lower.split()
    if len(tokens) < 50:
        return "general"

    scores = {
        dtype: sum(1 for kw in kws if kw in lower)
        for dtype, kws in _KEYWORDS.items()
    }

    best_type = max(scores, key=lambda k: scores[k])
    if scores[best_type] >= _MIN_KEYWORD_HITS:
        return best_type
    return "general"


def is_extractable_document(text: str) -> bool:
    """Return True if the document has enough content for structured extraction.

    Requires >= 50 tokens and numeric density >= 1.5% OR a detected non-general
    document type (legal/medical/research documents may have low numeric density
    but still contain extractable structured facts).
    """
    if not text:
        return False
    tokens = text.lower().split()
    if len(tokens) < 50:
        return False

    doc_type = detect_document_type(text)
    if doc_type != "general":
        return True

    numeric_tokens = sum(1 for t in tokens if re.search(r"\d", t))
    return (numeric_tokens / len(tokens)) >= _NUMERIC_DENSITY_THRESHOLD


# backward-compat alias
def is_financial_document(text: str) -> bool:
    return detect_document_type(text) == "financial"


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------


class ExtractedField(BaseModel):
    """A single typed value extracted and span-verified from a document."""

    name: str        # e.g. "ARR target", "Penalty Clause", "Systolic BP"
    value: float     # numeric value; 0.0 for non-numeric facts
    unit: str        # "USD", "%", "mg", "days", "mmHg", "n/a", etc.
    span_text: str   # verbatim substring from source document
    page: int        # 0 if unknown
    verified: bool   # True if char_interval was resolved by LangExtract


# ---------------------------------------------------------------------------
# Per-type prompts and examples
# ---------------------------------------------------------------------------

_PROMPTS: dict[str, str] = {
    "financial": textwrap.dedent("""
        Extract all numeric financial metrics from the text.
        For each metric provide:
          - The exact metric name (e.g. "ARR", "Gross Margin", "Headcount")
          - The numeric value as a float (convert $2.4M → 2400000.0, 15% → 15.0, $1B → 1000000000.0)
          - The unit (USD, %, units, headcount, etc.)
        Use the exact verbatim text containing the number as the extraction_text.
        Do not paraphrase or overlap entities.
    """).strip(),

    "legal": textwrap.dedent("""
        Extract all key structured facts from the legal document.
        For each fact provide:
          - The exact field name (e.g. "Effective Date", "Termination Notice Period",
            "Penalty Amount", "Contract Duration", "Governing Law Jurisdiction")
          - The numeric value as a float where applicable (e.g. days, USD amounts);
            use 0.0 for non-numeric facts like jurisdiction names
          - The unit (days, USD, years, n/a)
        Use the exact verbatim text as the extraction_text.
        Do not paraphrase or overlap entities.
    """).strip(),

    "medical": textwrap.dedent("""
        Extract all clinical measurements, medications, and diagnoses from the text.
        For each item provide:
          - The exact field name (e.g. "Systolic BP", "Fasting Glucose", "Metformin Dose",
            "HbA1c", "BMI", "Heart Rate")
          - The numeric value as a float where applicable; use 0.0 for non-numeric entries
          - The unit (mmHg, mg/dL, mg, bpm, kg/m², %, n/a)
        Use the exact verbatim text as the extraction_text.
        Do not paraphrase or overlap entities.
    """).strip(),

    "research": textwrap.dedent("""
        Extract all quantitative research findings and statistical values from the text.
        For each item provide:
          - The exact field name (e.g. "Sample Size", "P-value", "Effect Size",
            "Confidence Interval Lower", "Mean Score", "Standard Deviation")
          - The numeric value as a float
          - The unit (n, p, %, SD, CI, n/a)
        Use the exact verbatim text as the extraction_text.
        Do not paraphrase or overlap entities.
    """).strip(),

    "general": textwrap.dedent("""
        Extract all key structured facts, numbers, dates, and named entities from the text.
        For each item provide:
          - A descriptive field name
          - The numeric value as a float where applicable; use 0.0 for non-numeric facts
          - The unit (USD, %, days, years, n/a, or whatever is most appropriate)
        Use the exact verbatim text as the extraction_text.
        Do not paraphrase or overlap entities.
    """).strip(),
}

_EXAMPLES_RAW: dict[str, list[dict]] = {
    "financial": [
        {
            "text": "Annual Recurring Revenue (ARR) reached $2.4M, up 35% YoY.",
            "extractions": [
                {"class": "metric", "text": "$2.4M", "name": "ARR", "value": 2400000.0, "unit": "USD"},
                {"class": "metric", "text": "35%", "name": "ARR YoY Growth", "value": 35.0, "unit": "%"},
            ],
        },
        {
            "text": "Gross margin improved to 68% in Q3 FY2024 from 61% in Q3 FY2023.",
            "extractions": [
                {"class": "metric", "text": "68%", "name": "Gross Margin Q3 FY2024", "value": 68.0, "unit": "%"},
                {"class": "metric", "text": "61%", "name": "Gross Margin Q3 FY2023", "value": 61.0, "unit": "%"},
            ],
        },
    ],
    "legal": [
        {
            "text": "The agreement shall be effective as of January 1, 2024 and continue for a term of 24 months.",
            "extractions": [
                {"class": "metric", "text": "January 1, 2024", "name": "Effective Date", "value": 0.0, "unit": "n/a"},
                {"class": "metric", "text": "24 months", "name": "Contract Duration", "value": 24.0, "unit": "months"},
            ],
        },
        {
            "text": "In the event of breach, the defaulting party shall pay a penalty of $50,000.",
            "extractions": [
                {"class": "metric", "text": "$50,000", "name": "Breach Penalty", "value": 50000.0, "unit": "USD"},
            ],
        },
    ],
    "medical": [
        {
            "text": "Patient presents with blood pressure of 145/92 mmHg and fasting glucose of 126 mg/dL.",
            "extractions": [
                {"class": "metric", "text": "145/92 mmHg", "name": "Blood Pressure", "value": 145.0, "unit": "mmHg"},
                {"class": "metric", "text": "126 mg/dL", "name": "Fasting Glucose", "value": 126.0, "unit": "mg/dL"},
            ],
        },
        {
            "text": "Prescribed Metformin 500mg twice daily for type 2 diabetes management.",
            "extractions": [
                {"class": "metric", "text": "500mg", "name": "Metformin Dose", "value": 500.0, "unit": "mg"},
            ],
        },
    ],
    "research": [
        {
            "text": "The study included 342 participants (n=342) and found a statistically significant effect (p=0.003).",
            "extractions": [
                {"class": "metric", "text": "342", "name": "Sample Size", "value": 342.0, "unit": "n"},
                {"class": "metric", "text": "p=0.003", "name": "P-value", "value": 0.003, "unit": "p"},
            ],
        },
    ],
    "general": [
        {
            "text": "The project was completed in 18 months at a total cost of $1.2M with a team of 12 engineers.",
            "extractions": [
                {"class": "metric", "text": "18 months", "name": "Project Duration", "value": 18.0, "unit": "months"},
                {"class": "metric", "text": "$1.2M", "name": "Total Cost", "value": 1200000.0, "unit": "USD"},
                {"class": "metric", "text": "12 engineers", "name": "Team Size", "value": 12.0, "unit": "headcount"},
            ],
        },
    ],
}


def _build_examples(doc_type: str):
    """Build lx.data.ExampleData list for the given doc_type."""
    import langextract as lx

    raw = _EXAMPLES_RAW.get(doc_type, _EXAMPLES_RAW["general"])
    examples = []
    for ex in raw:
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


async def extract_document_fields(
    text: str,
    session_id: str,
    gemini_api_key: str,
) -> list[ExtractedField]:
    """Extract typed structured values from any document type using LangExtract.

    Detects document type first, then uses the matching prompt and examples.
    Uses Gemini 2.5 Flash via LangExtract for full-document chunked extraction
    with precise char_interval source grounding.

    Returns an empty list on any error — never raises.
    """
    if not text or not gemini_api_key:
        return []

    doc_type = detect_document_type(text)

    try:
        import asyncio
        import langextract as lx

        examples = _build_examples(doc_type)
        prompt = _PROMPTS[doc_type]
        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(
            None,
            lambda: lx.extract(
                text_or_documents=text,
                prompt_description=prompt,
                examples=examples,
                model_id="gemini-2.5-flash",
                api_key=gemini_api_key,
                show_progress=False,
                max_workers=2,  # free tier: 10 RPM cap
            ),
        )

        raw_extractions = result.extractions if result.extractions else []

        fields: list[ExtractedField] = []
        for extraction in raw_extractions:
            if extraction.char_interval is None:
                continue

            attrs = extraction.attributes or {}
            name = attrs.get("name", extraction.extraction_class or "unknown").strip()
            unit = attrs.get("unit", "n/a").strip()

            raw_value = attrs.get("value", "")
            try:
                value = float(raw_value)
            except (ValueError, TypeError):
                value = _parse_numeric(extraction.extraction_text)
                if value is None:
                    value = 0.0

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
            "extract_document_fields: session=%s type=%s extracted=%d verified=%d",
            session_id,
            doc_type,
            len(fields),
            verified_count,
        )
        return fields

    except Exception as exc:
        logger.warning("extract_document_fields failed (type=%s): %s", doc_type, exc)
        return []


# backward-compat alias
async def extract_financial_fields(
    text: str,
    session_id: str,
    gemini_api_key: str,
) -> list[ExtractedField]:
    return await extract_document_fields(text, session_id, gemini_api_key)


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

    Only includes verified fields to avoid hallucinated values polluting
    discrepancy detection logic.
    """
    verified = [f for f in fields if f.verified]
    if not verified:
        return ""

    lines = ["Structured values extracted from this document (span-verified):"]
    for f in verified:
        value_str = f"{f.value:,.2f}" if f.unit != "n/a" else f.span_text[:80]
        lines.append(
            f'  - {f.name}: {value_str} {f.unit} '
            f'(span: "{f.span_text[:80]}")'
        )
    return "\n".join(lines)
