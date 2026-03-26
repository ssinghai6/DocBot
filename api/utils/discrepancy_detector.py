"""
DocBot Discrepancy Detector — DOCBOT-403 (real implementation)

Extracts numeric values from document context and SQL result rows, matches them
by label similarity, and computes concrete deltas BEFORE the synthesis prompt is
built.  The LLM receives pre-computed facts, not instructions to compute them.

This replaces the stub prompt-only approach that allowed the LLM to hallucinate
discrepancies or miss real ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class NumericValue:
    """A numeric value extracted from text, with its surrounding label."""
    label: str          # e.g. "revenue", "total sales", "net income"
    value: float
    raw_text: str       # original snippet for citation
    source: str         # "doc" | "db"


@dataclass
class DiscrepancyItem:
    """A confirmed discrepancy between doc and DB for the same metric."""
    label: str
    doc_value: float
    db_value: float
    delta: float        # db_value - doc_value
    pct: float | None   # percentage delta relative to doc_value; None if doc_value == 0
    doc_snippet: str
    db_snippet: str


@dataclass
class DiscrepancyReport:
    """Full result returned to the synthesis layer."""
    discrepancies: list[DiscrepancyItem] = field(default_factory=list)
    checked_pairs: int = 0      # how many (doc_label, db_label) pairs were evaluated

    @property
    def has_discrepancies(self) -> bool:
        return bool(self.discrepancies)

    def to_prompt_block(self) -> str:
        """
        Return a ready-to-inject prompt block with pre-computed deltas.
        Empty string if nothing to report.
        """
        if not self.discrepancies:
            return ""

        lines = ["\n\nPRE-COMPUTED DISCREPANCY ANALYSIS (use these exact numbers):"]
        for d in self.discrepancies:
            sign = "+" if d.delta >= 0 else ""
            pct_str = f" ({sign}{d.pct:.1f}%)" if d.pct is not None else ""
            lines.append(
                f"  [DISCREPANCY] {d.label}: "
                f"Doc says {_fmt(d.doc_value)} [Source: {d.doc_snippet}]. "
                f"DB shows {_fmt(d.db_value)} [DB: {d.db_snippet}]. "
                f"Delta: {sign}{_fmt(d.delta)}{pct_str}"
            )
        lines.append(
            "Flag each of the above with [DISCREPANCY] in your answer. "
            "Do NOT invent additional discrepancies beyond those listed."
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_discrepancies(
    doc_context: str,
    sql_metadata: dict[str, Any],
    *,
    match_threshold: float = 0.55,
    delta_threshold: float = 0.01,
) -> DiscrepancyReport:
    """
    Extract numeric values from both sources, match by label similarity, and
    return a DiscrepancyReport with pre-computed deltas.

    Args:
        doc_context:      Raw document context string passed to the synthesizer.
        sql_metadata:     The dict returned by _collect_sql_result() — contains
                          either result_preview rows or csv_answer text.
        match_threshold:  Jaccard token similarity needed to consider two labels
                          as referring to the same metric (0–1).
        delta_threshold:  Minimum relative difference to flag as a discrepancy.
                          0.01 = 1 % difference.
    """
    doc_values = _extract_from_text(doc_context, source="doc")

    db_values: list[NumericValue] = []
    csv_answer = sql_metadata.get("csv_answer")
    if csv_answer:
        db_values = _extract_from_text(csv_answer, source="db")
    else:
        rows = sql_metadata.get("result_preview", [])
        db_values = _extract_from_rows(rows)

    report = DiscrepancyReport()
    pairs_checked = 0

    for dv in doc_values:
        for sv in db_values:
            sim = _label_similarity(dv.label, sv.label)
            if sim < match_threshold:
                continue
            pairs_checked += 1
            delta = sv.value - dv.value
            pct = (delta / dv.value * 100) if dv.value != 0 else None
            rel_diff = abs(delta / dv.value) if dv.value != 0 else (abs(delta) if delta != 0 else 0)
            if rel_diff >= delta_threshold:
                report.discrepancies.append(
                    DiscrepancyItem(
                        label=_canonical_label(dv.label, sv.label),
                        doc_value=dv.value,
                        db_value=sv.value,
                        delta=delta,
                        pct=pct,
                        doc_snippet=dv.raw_text[:80],
                        db_snippet=sv.raw_text[:80],
                    )
                )

    report.checked_pairs = pairs_checked
    return report


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

# Matches patterns like:
#   Revenue: $1,234,567  |  net income of 4.5M  |  total: 12,300  |  -$500K
_LABEL_VALUE_RE = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z0-9 _/\-]{1,60}?)"          # label (2-60 chars)
    r"\s*[:\-=]\s*"                                           # separator
    r"(?P<sign>[-–]?)\s*"
    r"[\$£€¥]?\s*"
    r"(?P<number>[0-9][0-9,]*(?:\.[0-9]+)?)"                  # number
    r"\s*(?P<suffix>[KkMmBbTt](?:illion|n|)?)?",             # optional K/M/B suffix
    re.IGNORECASE,
)

# Also match inline: "revenue of $1.2M" or "total sales were 45,000"
_INLINE_VALUE_RE = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z0-9 _/\-]{1,60}?)\s+"
    r"(?:was|were|is|are|of|totaling?|equaling?|reached?|hit|at)\s+"
    r"(?P<sign>[-–]?)\s*"
    r"[\$£€¥]?\s*"
    r"(?P<number>[0-9][0-9,]*(?:\.[0-9]+)?)"
    r"\s*(?P<suffix>[KkMmBbTt](?:illion|n|)?)?",
    re.IGNORECASE,
)

_SUFFIX_MULTIPLIERS = {
    "k": 1_000, "K": 1_000,
    "m": 1_000_000, "M": 1_000_000,
    "b": 1_000_000_000, "B": 1_000_000_000,
    "t": 1_000_000_000_000, "T": 1_000_000_000_000,
}


def _parse_number(number_str: str, sign: str, suffix: str | None) -> float:
    value = float(number_str.replace(",", ""))
    if suffix:
        multiplier = _SUFFIX_MULTIPLIERS.get(suffix[0].upper(), 1)
        value *= multiplier
    if sign in ("-", "–"):
        value = -value
    return value


def _extract_from_text(text: str, source: str) -> list[NumericValue]:
    """Extract (label, numeric_value) pairs from free text."""
    results: list[NumericValue] = []
    seen: set[tuple[str, float]] = set()

    for pattern in (_LABEL_VALUE_RE, _INLINE_VALUE_RE):
        for m in pattern.finditer(text):
            try:
                value = _parse_number(m.group("number"), m.group("sign"), m.group("suffix"))
            except (ValueError, KeyError):
                continue

            label = _normalize_label(m.group("label"))
            if not label or len(label) < 3:
                continue

            key = (label, round(value, 2))
            if key in seen:
                continue
            seen.add(key)

            results.append(NumericValue(
                label=label,
                value=value,
                raw_text=m.group(0).strip(),
                source=source,
            ))

    return results


def _extract_from_rows(rows: list[dict]) -> list[NumericValue]:
    """
    Extract numeric values from SQL result_preview rows.
    Each key becomes the label; each numeric value is extracted.
    """
    results: list[NumericValue] = []
    seen: set[tuple[str, float]] = set()

    for row in rows:
        if not isinstance(row, dict):
            continue
        for col, val in row.items():
            if not _is_numeric_like(val):
                continue
            try:
                num = float(str(val).replace(",", "").replace("$", ""))
            except (ValueError, TypeError):
                continue

            label = _normalize_label(str(col))
            if not label or len(label) < 2:
                continue

            key = (label, round(num, 2))
            if key in seen:
                continue
            seen.add(key)

            results.append(NumericValue(
                label=label,
                value=num,
                raw_text=f"{col}: {val}",
                source="db",
            ))

    return results


def _is_numeric_like(val: Any) -> bool:
    if isinstance(val, (int, float)):
        return True
    if isinstance(val, str):
        cleaned = val.replace(",", "").replace("$", "").strip()
        try:
            float(cleaned)
            return True
        except ValueError:
            return False
    return False


# ---------------------------------------------------------------------------
# Label normalization + similarity
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "the", "a", "an", "of", "in", "for", "and", "or", "to", "by",
    "with", "at", "from", "per", "as", "is", "are", "was", "were",
    "total", "net", "gross",  # kept but de-weighted by being stopwords
})

# Synonyms to normalize before comparison
_LABEL_SYNONYMS: dict[str, str] = {
    "revenues": "revenue",
    "sales": "revenue",
    "turnover": "revenue",
    "earnings": "income",
    "profit": "income",
    "profits": "income",
    "losses": "loss",
    "expenditure": "expense",
    "expenditures": "expenses",
    "costs": "cost",
    "headcount": "employees",
    "workforce": "employees",
    "staff": "employees",
}


def _normalize_label(label: str) -> str:
    label = label.lower().strip()
    label = re.sub(r"[^a-z0-9 ]", " ", label)
    tokens = label.split()
    tokens = [_LABEL_SYNONYMS.get(t, t) for t in tokens]
    return " ".join(t for t in tokens if t)


def _label_tokens(label: str) -> set[str]:
    return set(_normalize_label(label).split()) - _STOP_WORDS


def _label_similarity(a: str, b: str) -> float:
    """Jaccard similarity between label token sets."""
    ta, tb = _label_tokens(a), _label_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _canonical_label(doc_label: str, db_label: str) -> str:
    """Pick the more descriptive of the two labels."""
    return doc_label if len(doc_label) >= len(db_label) else db_label


# ---------------------------------------------------------------------------
# Formatting helper
# ---------------------------------------------------------------------------


def _fmt(value: float) -> str:
    """Format a number with commas; show as integer if no fractional part."""
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:,.2f}"
