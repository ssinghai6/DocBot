"""
DocBot Discrepancy Detector — DOCBOT-403 (real implementation)

Extracts numeric values from document context and SQL result rows, matches them
by label similarity, and computes concrete deltas BEFORE the synthesis prompt is
built.  The LLM receives pre-computed facts, not instructions to compute them.

This replaces the stub prompt-only approach that allowed the LLM to hallucinate
discrepancies or miss real ones.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Severity = Literal["low", "medium", "high"]
Unit = Literal["currency", "percent", "number"]

# DOCBOT-2002: inline marker the synthesizer places in the answer text; the UI
# swaps it for a badge linking to the structured TieOutCard.
MARKER_TEMPLATE = "[[discrepancy:{id}]]"

# Severity bands on |delta %| (relative to the reported figure).
SEVERITY_HIGH_PCT = 1.0
SEVERITY_MEDIUM_PCT = 0.5

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
    period: Optional[str] = None      # canonical "Q4 2025" / "FY2025"
    kind: str = "plain"               # "currency" | "percent" | "plain"
    doc_file: Optional[str] = None    # doc side: "Source: X, Page N" header
    doc_page: Optional[int] = None
    dimension: str = ""               # db side: row entity, e.g. "lab services"


@dataclass
class DiscrepancyItem:
    """A confirmed discrepancy between doc and DB for the same metric.

    ``doc_*`` = the reported figure (filing / report), ``db_*`` = the books.
    ``id`` and ``severity`` are derived when not supplied.
    """
    label: str
    doc_value: float
    db_value: float
    delta: float        # db_value - doc_value
    pct: float | None   # percentage delta relative to doc_value; None if doc_value == 0
    doc_snippet: str
    db_snippet: str
    period: Optional[str] = None
    unit: Unit = "number"
    doc_file: Optional[str] = None
    doc_page: Optional[int] = None
    db_table: Optional[str] = None
    db_query: Optional[str] = None
    id: str = ""
    severity: Severity = "medium"

    def __post_init__(self) -> None:
        if not self.id:
            self.id = discrepancy_id(self.label, self.period, self.doc_value, self.db_value)
        self.severity = severity_for(self.pct)

    @property
    def marker(self) -> str:
        return MARKER_TEMPLATE.format(id=self.id)

    @property
    def display_label(self) -> str:
        label = self.label.strip().title() if self.label.islower() else self.label.strip()
        return f"{label} ({self.period})" if self.period else label


def discrepancy_id(label: str, period: Optional[str], doc_value: float, db_value: float) -> str:
    """Stable, URL/markdown-safe id: ``d-`` + 8 hex chars."""
    raw = f"{_normalize_label(label)}|{period or ''}|{doc_value:.4f}|{db_value:.4f}"
    return "d-" + hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]


def severity_for(pct: Optional[float]) -> Severity:
    """Materiality band on |delta %|. Unknown % (doc value 0) is high."""
    if pct is None:
        return "high"
    mag = abs(pct)
    if mag >= SEVERITY_HIGH_PCT:
        return "high"
    if mag >= SEVERITY_MEDIUM_PCT:
        return "medium"
    return "low"


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

        DOCBOT-2002: each discrepancy carries an inline marker token
        (``[[discrepancy:<id>]]``). The model mentions it in one sentence and
        places the marker; the UI renders the full tie-out card from the
        structured lineage payload, so the breakdown is never repeated as a
        literal block in the answer text.
        """
        if not self.discrepancies:
            return ""

        lines = ["\n\nPRE-COMPUTED TIE-OUT DISCREPANCIES (use these exact numbers):"]
        for d in self.discrepancies:
            sign = "+" if d.delta >= 0 else ""
            pct_str = f" ({sign}{d.pct:.1f}%)" if d.pct is not None else ""
            doc_ref = d.doc_file + (f" p{d.doc_page}" if d.doc_page is not None else "") if d.doc_file else d.doc_snippet
            db_ref = d.db_table or d.db_snippet
            lines.append(
                f"  {d.marker} {d.display_label}: "
                f"Report says {_fmt(d.doc_value)} [Source: {doc_ref}]. "
                f"Books show {_fmt(d.db_value)} [DB: {db_ref}]. "
                f"Delta: {sign}{_fmt(d.delta)}{pct_str}"
            )
        example = self.discrepancies[0].marker
        lines.append(
            "For EACH discrepancy above, write ONE short sentence naming the metric and "
            f"end it with its marker token copied exactly (e.g. {example}). The interface "
            "renders a full tie-out card from the marker, so do not repeat the breakdown "
            "and NEVER write the literal text [DISCREPANCY]. "
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
    delta_threshold: float = 0.005,
    max_delta_threshold: float = 0.25,
) -> DiscrepancyReport:
    """
    Extract numeric values from both sources, match by label similarity, and
    return a DiscrepancyReport with pre-computed deltas.

    Args:
        doc_context:      Raw document context string passed to the synthesizer.
                          ``Source: <file>, Page <n>`` chunk headers (as built by
                          rag_retrieve / autopilot doc_search) are parsed into
                          per-value doc citations.
        sql_metadata:     The dict returned by _collect_sql_result() — contains
                          either result_preview rows or csv_answer text, plus
                          ``sources`` / ``sql_query`` used for the db citation.
        match_threshold:  Jaccard token similarity needed to consider two labels
                          as referring to the same metric (0–1).
        delta_threshold:  Minimum relative difference to flag as a discrepancy.
                          0.005 = 0.5 % (DOCBOT-2002: tie-out sensitivity — a
                          $5M gap on an $890M segment line is a real break).
        max_delta_threshold: Upper bound on relative difference. Deltas larger
                          than this are almost never true reporting
                          discrepancies — they signal a unit mismatch (e.g. the
                          DB stores revenue in millions while the doc quotes it
                          in absolute dollars), a granularity mismatch (annual
                          doc figure vs quarterly DB rows), or a parse error.
                          Such pairs are skipped to avoid nonsensical output
                          like "Delta: -100%" or "+19566%". 0.25 = 25 %.

    Matching (DOCBOT-2002):
      * Period-aware: when both sides carry a period (``Q4 2025`` / ``FY2025``)
        they must agree, so Q1–Q3 report figures are never compared with a Q4
        row.
      * Entity-aware: SQL rows with a text dimension (e.g. a segment name) are
        labelled ``<dimension> <column>`` so "Restaurant Advertising: $890M"
        matches that segment's revenue row, not every revenue row.
      * Tie-out semantics: if ANY plausible books value agrees with the report
        figure, it ties out; otherwise the best-matching plausible value is
        reported as the discrepancy.
    """
    doc_values = _extract_from_text(doc_context, source="doc")

    db_values: list[NumericValue] = []
    csv_answer = sql_metadata.get("csv_answer")
    if csv_answer:
        db_values = _extract_from_text(csv_answer, source="db")
    else:
        rows = sql_metadata.get("result_preview", [])
        sql = sql_metadata.get("sql_query")
        db_values = _extract_from_rows(
            rows, fallback_dimension=_where_dimension(sql), tag_columns=_literal_columns(sql)
        )

    sources = [str(x) for x in (sql_metadata.get("sources") or []) if x]
    db_table = ", ".join(sources[:3]) if sources else None
    raw_query = sql_metadata.get("sql_query")
    db_query = str(raw_query)[:2000] if raw_query else None

    report = DiscrepancyReport()
    pairs_checked = 0
    seen_ids: set[str] = set()

    for dv in doc_values:
        candidates: list[tuple[bool, float, float, int, NumericValue]] = []
        for idx, sv in enumerate(db_values):
            if not _kinds_compatible(dv, sv):
                continue
            if dv.period and sv.period and dv.period != sv.period:
                continue
            label_sim = _label_similarity(dv.label, sv.label)
            # DOCBOT-2542: a row's entity alone names the figure ("Lab Services:
            # $104.9M" vs {segment: "Lab Services", segment_revenue: ...}); the
            # column name only adds tokens that dilute Jaccard below threshold.
            # DOCBOT-2585: "|" separates WHERE literals ("Sales and marketing|opex");
            # each may name the figure on its own, or all of them together.
            sim = max(
                [label_sim] + [_label_similarity(dv.label, d) for d in (sv.dimension, *sv.dimension.split("|"))]
            ) if sv.dimension else label_sim
            if sim < match_threshold:
                continue
            period_match = bool(dv.period and sv.period == dv.period)
            candidates.append((period_match, sim, label_sim, -idx, sv))
        if not candidates:
            continue

        pairs_checked += 1
        candidates.sort(key=lambda c: c[:4], reverse=True)

        # Align unit scale before comparing. Docs quote figures with explicit
        # units ("$330M", "$5.2 billion") that parse to absolute dollars, while
        # DB columns often store the same metric in millions (e.g. 325). Snap
        # the smaller value up by the nearest power of 1000 so genuine
        # reporting discrepancies (330M vs 325M → 1.5%) surface while true
        # mismatches (annual vs quarterly) stay large and get filtered.
        evaluated: list[tuple[NumericValue, float, float, float, float]] = []
        for *_rank, sv in candidates:
            doc_val, db_val = _align_scale(dv.value, sv.value)
            delta = db_val - doc_val
            rel_diff = abs(delta / doc_val) if doc_val != 0 else (abs(delta) if delta != 0 else 0)
            evaluated.append((sv, doc_val, db_val, delta, rel_diff))

        if any(e[4] < delta_threshold for e in evaluated):
            continue  # a books value agrees with the report figure — ties out

        chosen = next((e for e in evaluated if e[4] <= max_delta_threshold), None)
        if chosen is None:
            continue
        sv, doc_val, db_val, delta, _rel = chosen
        pct = (delta / doc_val * 100) if doc_val != 0 else None

        item = DiscrepancyItem(
            label=_canonical_label(dv.label, sv.label, dv.raw_text),
            doc_value=doc_val,
            db_value=db_val,
            delta=delta,
            pct=pct,
            doc_snippet=dv.raw_text[:80],
            db_snippet=sv.raw_text[:80],
            period=dv.period or sv.period,
            unit=_unit_for(dv, sv),
            doc_file=dv.doc_file,
            doc_page=dv.doc_page,
            db_table=db_table,
            db_query=db_query,
        )
        if item.id in seen_ids:
            continue
        seen_ids.add(item.id)
        report.discrepancies.append(item)

    report.checked_pairs = pairs_checked
    return report


def _kinds_compatible(dv: NumericValue, sv: NumericValue) -> bool:
    """A currency figure never ties out against a percentage column (and v.v.)."""
    if dv.kind == "percent" and sv.kind != "percent":
        return False
    if dv.kind == "currency" and sv.kind == "percent":
        return False
    return True


def _unit_for(dv: NumericValue, sv: NumericValue) -> Unit:
    if dv.kind == "currency":
        return "currency"
    if dv.kind == "percent" or sv.kind == "percent":
        return "percent"
    return "number"


# ---------------------------------------------------------------------------
# Period helpers
# ---------------------------------------------------------------------------

_PERIOD_Q_RE = re.compile(r"\bQ([1-4])\s*(?:FY\s*)?'?(\d{4}|\d{2})\b", re.IGNORECASE)
_PERIOD_Q_REV_RE = re.compile(r"\b((?:19|20)\d{2})\s*[-/ ]?\s*Q([1-4])\b", re.IGNORECASE)
_PERIOD_FY_RE = re.compile(r"\b(?:FY|fiscal\s+year)\s*'?(\d{4}|\d{2})\b", re.IGNORECASE)
_BARE_QUARTER_RE = re.compile(r"^\s*Q([1-4])\s*$", re.IGNORECASE)
# DOCBOT-2637: filing chunks without a printed page use ", Section S" (doc_refs.chunk_header)
_SOURCE_HEADER_RE = re.compile(
    r"^Source:\s*(?P<file>.+?)(?:,\s*Page\s+(?P<page>\d+)|,\s*Section\s+.*)?\s*$", re.MULTILINE
)

_YEAR_COLS = frozenset({"year", "fiscal_year", "fy", "yr", "calendar_year"})
_PERIOD_COLS = _YEAR_COLS | frozenset({
    "quarter", "fiscal_quarter", "qtr", "period", "fiscal_period", "month", "date",
})
_PERCENT_TOKENS = frozenset({"pct", "percent", "percentage", "ratio", "margin", "rate", "growth"})


def _norm_year(y: str) -> int:
    n = int(y)
    return n + 2000 if n < 100 else n


def _find_periods(text: str) -> list[tuple[int, int, str]]:
    """All (start, end, canonical) period mentions in ``text``, in order."""
    found: list[tuple[int, int, str]] = []
    for m in _PERIOD_Q_RE.finditer(text):
        found.append((m.start(), m.end(), f"Q{m.group(1)} {_norm_year(m.group(2))}"))
    for m in _PERIOD_Q_REV_RE.finditer(text):
        found.append((m.start(), m.end(), f"Q{m.group(2)} {_norm_year(m.group(1))}"))
    for m in _PERIOD_FY_RE.finditer(text):
        found.append((m.start(), m.end(), f"FY{_norm_year(m.group(1))}"))
    found.sort()
    return found


def extract_period(text: str) -> Optional[str]:
    """First canonical period mentioned in ``text`` (``Q4 2025`` / ``FY2025``)."""
    periods = _find_periods(text or "")
    return periods[0][2] if periods else None


def _strip_periods(label: str) -> str:
    out = label
    for start, end, _canon in reversed(_find_periods(label)):
        out = out[:start] + " " + out[end:]
    return out


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
    r"\s*(?P<suffix>[KkMmBbTt](?:illion|n|)?\b)?",             # optional K/M/B suffix
    re.IGNORECASE,
)

# Also match inline: "revenue of $1.2M" or "total sales were 45,000"
_INLINE_VALUE_RE = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z0-9 _/\-]{1,60}?)\s+"
    r"(?:was|were|is|are|of|totaling?|equaling?|reached?|hit|at)\s+"
    r"(?P<sign>[-–]?)\s*"
    r"[\$£€¥]?\s*"
    r"(?P<number>[0-9][0-9,]*(?:\.[0-9]+)?)"
    r"\s*(?P<suffix>[KkMmBbTt](?:illion|n|)?\b)?",
    re.IGNORECASE,
)

# Match "Label $value" with no connector, e.g. "Net Income $330M",
# "Professional Services: $890M" (comma/list-separated financial lines). Requires
# a currency symbol so bare numbers ("Operating Margin 27") don't false-match.
_LABEL_CURRENCY_RE = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z0-9 _/\-]{1,40}?)\s*[:\-]?\s+"
    r"(?P<sign>[-–]?)\s*"
    r"[\$£€¥]\s*"                                             # currency REQUIRED
    r"(?P<number>[0-9][0-9,]*(?:\.[0-9]+)?)"
    r"\s*(?P<suffix>[KkMmBbTt](?:illion|n|)?\b)?",
    re.IGNORECASE,
)

# Table rows from filings rendered as "Label | 914,830 | 856,565" (DOCBOT-2634).
# The first numeric cell is the row's value: statements list the latest period first.
_TABLE_ROW_RE = re.compile(
    r"^(?P<label>[A-Za-z][^|\n]{1,80}?)\s*\|\s*"
    r"(?P<sign>[-–(]?)\s*"
    r"[\$£€¥]?\s*"
    r"(?P<number>[0-9][0-9,]*(?:\.[0-9]+)?)\)?"
    r"(?P<suffix>)\s*(?=\||$)",
    re.MULTILINE,
)

_SUFFIX_MULTIPLIERS = {
    "k": 1_000, "K": 1_000,
    "m": 1_000_000, "M": 1_000_000,
    "b": 1_000_000_000, "B": 1_000_000_000,
    "t": 1_000_000_000_000, "T": 1_000_000_000_000,
}


def _align_scale(a: float, b: float) -> tuple[float, float]:
    """Snap two values to the same order of magnitude when they differ by a
    near-power-of-1000 factor (a unit mismatch, e.g. absolute dollars vs
    millions). Returns the values with the smaller one scaled up.

    Only aligns when the ratio is >~31x (rounds to at least 1000^1); genuine
    reporting discrepancies (a few %) are left untouched.
    """
    if a <= 0 or b <= 0:
        return a, b
    hi, lo = (a, b) if a >= b else (b, a)
    k = round(math.log(hi / lo) / math.log(1000))
    if k < 1:
        return a, b
    lo_scaled = lo * (1000 ** k)
    return (a, lo_scaled) if a >= b else (lo_scaled, b)


def _parse_number(number_str: str, sign: str, suffix: str | None) -> float:
    value = float(number_str.replace(",", ""))
    if suffix:
        multiplier = _SUFFIX_MULTIPLIERS.get(suffix[0].upper(), 1)
        value *= multiplier
    if sign in ("-", "–", "("):
        value = -value
    return value


def _doc_context_for(text: str, start: int, number_start: int) -> tuple[Optional[str], Optional[str], Optional[int]]:
    """(period, file, page) for a value at ``start`` in a doc context string.

    Period: the same line up to the number ("Q4 2025: Revenue $1,440M, Net
    Income $330M"), else the nearest earlier mention inside the same chunk
    (a section header like "REVENUE BY SEGMENT — FY2025"). Text *after* the
    number is ignored so "(up 18% from $4.4B in FY2024)" is not misread as
    the period of the figure it follows.
    """
    header: Optional[re.Match[str]] = None
    for h in _SOURCE_HEADER_RE.finditer(text, 0, start):
        header = h
    chunk_start = header.end() if header else 0
    file = header.group("file").strip()[:200] if header else None
    page = int(header.group("page")) if header and header.group("page") else None

    line_start = text.rfind("\n", 0, start) + 1
    period = extract_period(text[max(line_start, chunk_start):number_start])
    if period is None and line_start > chunk_start:
        # Only heading-like lines ("REVENUE BY SEGMENT — FY2025": no digits
        # left once the period is removed) set a section period; data lines
        # such as "... from $4.4 billion in FY2024)" must not leak downward.
        for line in reversed(text[chunk_start:line_start].splitlines()):
            periods = _find_periods(line)
            if periods and not re.search(r"\d", _strip_periods(line)):
                period = periods[-1][2]
                break
    return period, file, page


def _extract_from_text(text: str, source: str) -> list[NumericValue]:
    """Extract (label, numeric_value) pairs from free text."""
    results: list[NumericValue] = []
    seen: set[tuple[str, float, Optional[str]]] = set()

    for pattern in (_TABLE_ROW_RE, _LABEL_VALUE_RE, _INLINE_VALUE_RE, _LABEL_CURRENCY_RE):
        for m in pattern.finditer(text):
            try:
                value = _parse_number(m.group("number"), m.group("sign"), m.group("suffix"))
            except (ValueError, KeyError):
                continue

            label = _normalize_label(_strip_periods(m.group("label")))
            if not label or len(label) < 3:
                continue

            period, doc_file, doc_page = _doc_context_for(text, m.start(), m.start("number"))

            key = (label, round(value, 2), period)
            if key in seen:
                continue
            seen.add(key)

            matched = m.group(0)
            tail = text[m.end():m.end() + 2].lstrip()
            if tail.startswith("%"):
                kind = "percent"
            elif any(sym in matched for sym in "$£€¥"):
                kind = "currency"
            else:
                kind = "plain"

            results.append(NumericValue(
                label=label,
                value=value,
                raw_text=matched.strip(),
                source=source,
                period=period,
                kind=kind,
                doc_file=doc_file if source == "doc" else None,
                doc_page=doc_page if source == "doc" else None,
            ))

    return results


def _row_period_and_dimension(row: dict) -> tuple[Optional[str], str]:
    """Derive (period, entity dimension) from a SQL row's non-metric columns."""
    period: Optional[str] = None
    bare_quarter: Optional[str] = None
    year: Optional[int] = None
    dims: list[str] = []
    for col, val in row.items():
        col_l = str(col).lower()
        if isinstance(val, str) and not _is_numeric_like(val):
            p = extract_period(val)
            if p:
                period = period or p
                continue
            bq = _BARE_QUARTER_RE.match(val)
            if bq:
                bare_quarter = bq.group(1)
                continue
            if col_l in _PERIOD_COLS:
                continue
            if re.search(r"[A-Za-z]", val) and len(val) <= 60:
                dims.append(val)
        elif col_l in _YEAR_COLS and _is_numeric_like(val):
            try:
                n = int(float(str(val)))
            except (TypeError, ValueError):
                continue
            if 1900 <= n <= 2100:
                year = n
    if period is None and bare_quarter and year:
        period = f"Q{bare_quarter} {year}"
    elif period is None and year:
        period = f"FY{year}"
    return period, " ".join(dims[:2])


def _where_dimension(sql: Any) -> str:
    """Text the SQL filtered on, e.g. ``WHERE category = 'Sales and marketing'``.

    DOCBOT-2322: when a query selects only ``period, amount`` the metric name
    lives solely in the WHERE clause, so the row alone labels the value
    "amount" and nothing ties out. Read equality / LIKE string literals on
    non-period columns from the sqlglot AST (never regex) and use them as the
    row dimension, up to two joined by "|". Returns "" when nothing applies or
    the SQL does not parse.
    """
    if not sql:
        return ""
    import sqlglot
    import sqlglot.expressions as exp

    try:
        tree = sqlglot.parse_one(str(sql))
    except sqlglot.errors.SqlglotError:
        return ""
    if tree is None:
        return ""
    values: list[str] = []
    for node in tree.find_all(exp.EQ, exp.Like):
        col = node.find(exp.Column)
        lit = node.expression if isinstance(node.expression, exp.Literal) else node.this
        if col is None or not isinstance(lit, exp.Literal) or not lit.is_string:
            continue
        if col.name.lower() in _PERIOD_COLS:
            continue
        text = lit.this.strip("%").strip()
        if not text or extract_period(text) or _BARE_QUARTER_RE.match(text):
            continue
        if re.search(r"[A-Za-z]", text) and len(text) <= 60 and text not in values:
            values.append(text)
    return "|".join(values[:2])


def _literal_columns(sql: Any) -> frozenset[str]:
    """Lower-cased output columns the SQL fills with a string constant.

    DOCBOT-2585: the generator sometimes tags rows itself, e.g.
    ``SELECT period, 'MD&A' AS source, amount ... WHERE category = 'Sales and
    marketing' UNION ALL ...``. Such a tag is not a data entity, so it must not
    hide the WHERE-clause metric name from the row label.
    """
    if not sql:
        return frozenset()
    import sqlglot
    import sqlglot.expressions as exp

    try:
        tree = sqlglot.parse_one(str(sql))
    except sqlglot.errors.SqlglotError:
        return frozenset()
    if tree is None:
        return frozenset()
    literal: set[str] = set()
    data: set[str] = set()
    for select in tree.find_all(exp.Select):
        for proj in select.expressions:
            is_tag = isinstance(proj, exp.Alias) and isinstance(proj.this, exp.Literal) and proj.this.is_string
            (literal if is_tag else data).add(proj.alias_or_name.lower())
    # "SELECT segment ... UNION ALL SELECT 'Total' AS segment": real data in one branch, not a tag.
    return frozenset(literal - data)


def _extract_from_rows(
    rows: list[dict], fallback_dimension: str = "", tag_columns: frozenset[str] = frozenset()
) -> list[NumericValue]:
    """
    Extract numeric values from SQL result_preview rows.
    Each key becomes the label; each numeric value is extracted. Text
    dimension columns (segment / product names) are prefixed to the label and
    period columns (quarter / year) become the value's period.
    ``tag_columns`` (constant-literal columns, see ``_literal_columns``) only
    name the row when neither data columns nor the WHERE clause do.
    """
    results: list[NumericValue] = []
    seen: set[tuple[str, float, Optional[str]]] = set()

    for row in rows:
        if not isinstance(row, dict):
            continue
        period, dimension = _row_period_and_dimension(row)
        if tag_columns:
            data_dimension = _row_period_and_dimension(
                {k: v for k, v in row.items() if str(k).lower() not in tag_columns}
            )[1]
            dimension = data_dimension or fallback_dimension or dimension
        dimension = dimension or fallback_dimension
        dim_label = _normalize_label(dimension) if dimension else ""
        # Ledger account codes ("4100 Revenue - Lab Services") never appear in prose.
        entity = "|".join(
            " ".join(t for t in _normalize_label(part).split() if not t.isdigit())
            for part in dimension.split("|")
        ) if dimension else ""
        for col, val in row.items():
            if str(col).lower() in _PERIOD_COLS:
                continue
            if not _is_numeric_like(val):
                continue
            try:
                num = float(str(val).replace(",", "").replace("$", ""))
            except (ValueError, TypeError):
                continue

            col_label = _normalize_label(str(col))
            if not col_label or len(col_label) < 2:
                continue
            label = f"{dim_label} {col_label}".strip() if dim_label else col_label

            key = (label, round(num, 2), period)
            if key in seen:
                continue
            seen.add(key)

            col_tokens = set(col_label.split())
            prefix = f"{dimension.replace('|', ', ')} · " if dimension else ""
            results.append(NumericValue(
                label=label,
                value=num,
                raw_text=f"{prefix}{col}: {val}",
                source="db",
                period=period,
                kind="percent" if col_tokens & _PERCENT_TOKENS else "plain",
                dimension=entity,
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
    # Unit-suffix tokens that SQL column names carry (e.g. "revenue_m",
    # "net_income_pct") but doc prose never uses as content words. Left in,
    # these dilute Jaccard similarity enough to push a genuine match below
    # match_threshold — e.g. "net_income_m" vs doc "Net Income" scored 0.5
    # (just under the 0.55 default) until "m" was stripped here.
    "m", "k", "b", "t", "pct", "percent", "amt", "amount", "usd", "value",
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


_SYNONYMS_OF: dict[str, list[str]] = {}
for _src, _dst in _LABEL_SYNONYMS.items():
    _SYNONYMS_OF.setdefault(_dst, []).append(_src)


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


def _canonical_label(doc_label: str, db_label: str, doc_raw: str = "") -> str:
    """Pick the more descriptive of the two labels for display.

    Prefer the doc label — it's author-written prose ("Net Income"), while
    the db label is usually derived from a SQL column name and carries
    unit-suffix noise ("net income m") even after that noise is stripped for
    Jaccard matching. Fall back to db_label only if the doc label is empty.

    DOCBOT-2002: when the raw doc snippet still contains the label in its
    original casing ("Restaurant Advertising"), return that casing so the
    tie-out card reads like the filing.
    """
    if not doc_label.strip():
        return db_label
    if doc_raw:
        words = doc_label.split()
        # Match synonym-normalized words back to the original wording
        # ("revenue and marketing" -> "Sales and marketing").
        pattern = r"\W+".join(
            "(?:" + "|".join(re.escape(x) for x in [w, *_SYNONYMS_OF.get(w, ())]) + ")"
            for w in words
        )
        m = re.search(pattern, doc_raw, re.IGNORECASE)
        if m:
            return m.group(0)
    return doc_label


# ---------------------------------------------------------------------------
# Formatting helper
# ---------------------------------------------------------------------------


def _fmt(value: float) -> str:
    """Format a number with commas; show as integer if no fractional part."""
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:,.2f}"
