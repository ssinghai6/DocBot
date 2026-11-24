"""DOCBOT-2637: year-over-year changes computed in code from filing tables.

Synthesis used to work out trends itself from raw table text and got them
wrong (Etsy: impairment "steady at $101.7M" when FY2024 was "—", tax "rose"
from $107.5M to $83.7M, opex "+5%" when it was +8.8%). As with discrepancies
(DOCBOT-403), the model now only reports changes computed here.

Reads the ``a | b | c`` tables that edgar_connector.html_to_text renders and
chunk_filing keeps whole: a header row of years (``2025 | 2024 | 2023``), then
label rows. A cell that is only a dash means nil (zero), as in SEC filings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from api.utils.discrepancy_detector import _SOURCE_HEADER_RE

MAX_FACTS = 60
_YEAR_RE = re.compile(r"(?:19|20)\d\d")
_NIL = {"—", "–", "-", "−"}
_NUMBER_RE = re.compile(r"^(\()?\$?\s*(-|−)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(%)?\s*(\))?$")
_UNITS_RE = re.compile(r"\(\s*in (thousands|millions|billions)", re.IGNORECASE)


@dataclass(frozen=True)
class YoYFact:
    label: str
    year: str
    prior_year: str
    value: float
    prior: float
    percent_kind: bool
    source: str
    units: str

    def line(self) -> str:
        where = ", ".join(p for p in (self.source, f"in {self.units}" if self.units else "") if p)
        cur, pri = _fmt(self.value, self.percent_kind), _fmt(self.prior, self.percent_kind)
        head = f"- {self.label} ({where}): FY{self.year} {cur} vs FY{self.prior_year} {pri}"
        diff = self.value - self.prior
        if self.percent_kind:
            return f"{head} -> {_direction(diff)} {abs(diff):,.1f} percentage points" if diff else f"{head} -> unchanged"
        if diff == 0:
            return f"{head} -> unchanged"
        if self.prior == 0:
            return f"{head} -> nil in FY{self.prior_year}, so no % change ({_direction(diff)} {_fmt(abs(diff))})"
        if self.value == 0:
            return f"{head} -> nil in FY{self.year} ({_direction(diff)} {_fmt(abs(diff))})"
        if (self.value > 0) != (self.prior > 0):
            return f"{head} -> changed sign ({_direction(diff)} {_fmt(abs(diff))}); no % change"
        if self.value < 0:
            # both in parentheses: "lower" would read as "the expense fell"
            grew = abs(self.value) - abs(self.prior)
            pct = grew / abs(self.prior) * 100
            word = "larger" if grew > 0 else "smaller"
            return f"{head} -> both negative; the amount in parentheses is {word} by {_fmt(abs(grew))} ({pct:+.1f}%)"
        pct = diff / abs(self.prior) * 100
        return f"{head} -> {_direction(diff)} {_fmt(abs(diff))} ({pct:+.1f}%)"


def _direction(diff: float) -> str:
    return "higher by" if diff > 0 else "lower by"


def _fmt(value: float, percent: bool = False) -> str:
    if percent:
        return f"{value:,.1f}%"
    if value == 0:
        return "nil"
    text = f"{abs(value):,.0f}" if value == int(value) else f"{abs(value):,.2f}"
    return f"({text})" if value < 0 else text


def _cell_value(cell: str) -> Optional[tuple[float, bool]]:
    """(value, is_percent) of a numeric cell; dashes are nil; None if not numeric."""
    cell = cell.strip()
    if cell in _NIL:
        return 0.0, False
    m = _NUMBER_RE.match(cell)
    if not m:
        return None
    value = float(m.group(3).replace(",", ""))
    if m.group(1) and m.group(5) or m.group(2):
        value = -value
    return value, bool(m.group(4))


def _header_years(cells: list[str]) -> Optional[list[Optional[str]]]:
    """Year per column for a header row (None for non-year cells), else None."""
    years = [c if _YEAR_RE.fullmatch(c) else None for c in cells]
    if sum(y is not None for y in years) < 2:
        return None
    if any(y is None and re.search(r"\d", c) for y, c in zip(years, cells)):
        return None  # a data row, not a header
    return years


def _chunks(doc_context: str) -> list[tuple[str, str]]:
    """(source label, chunk text) per ``Source: ...`` header."""
    heads = list(_SOURCE_HEADER_RE.finditer(doc_context))
    if not heads:
        return [("", doc_context)]
    out = []
    for h, nxt in zip(heads, heads[1:] + [None]):
        page = h.group("page")
        label = h.group("file").strip() + (f" p{page}" if page else "")
        out.append((label, doc_context[h.end():nxt.start() if nxt else len(doc_context)]))
    return out


def yoy_facts(doc_context: str) -> list[YoYFact]:
    """Latest-year vs prior-year change for every filing table row in the context."""
    facts: list[YoYFact] = []
    seen: set[tuple[str, float, float]] = set()
    for source, chunk in _chunks(doc_context):
        units_match = _UNITS_RE.search(chunk)
        units = units_match.group(1).lower() if units_match else ""
        years: Optional[list[Optional[str]]] = None
        for line in chunk.splitlines():
            if not line.strip():
                years = None  # a blank line ends the table
                continue
            if " | " not in line:
                continue  # sub-headings inside a table ("Operating expenses:")
            cells = [c.strip() for c in line.split(" | ")]
            header = _header_years(cells)
            if header is not None:
                years = header
                continue
            if years is None or not re.match(r"[A-Za-z]", cells[0]):
                continue
            if len(cells) - 1 == len(years):  # header has no label column
                cols = years
            elif len(cells) == len(years) and years[0] is None:
                cols = years[1:]
            else:
                continue
            values = cells[1:]
            dated = [(y, _cell_value(v)) for y, v in zip(cols, values) if y is not None]
            dated = [(y, v) for y, v in dated if v is not None]
            if len(dated) < 2:
                continue
            dated.sort(key=lambda yv: yv[0], reverse=True)
            (year, (value, pct)), (prior_year, (prior, prior_pct)) = dated[0], dated[1]
            key = (cells[0].lower(), value, prior)
            if key in seen:
                continue
            seen.add(key)
            facts.append(YoYFact(cells[0], year, prior_year, value, prior, pct or prior_pct, source, units))
            if len(facts) >= MAX_FACTS:
                return facts
    return facts


def yoy_prompt_block(doc_context: str) -> str:
    """Prompt section listing the computed changes, or '' when there are none."""
    facts = yoy_facts(doc_context)
    if not facts:
        return ""
    lines = "\n".join(f.line() for f in facts)
    return (
        "\n\nPRE-COMPUTED YEAR-OVER-YEAR CHANGES (computed in code from the filing tables "
        "above; \"nil\" is a dash in the filing, meaning zero):\n" + lines + "\n"
    )


YOY_RULE = (
    "- Any statement that a figure rose, fell, stayed steady, or changed by an amount or "
    "percentage between years must match the PRE-COMPUTED YEAR-OVER-YEAR CHANGES. Never "
    "compute your own growth rates or year-over-year differences, and do not describe the "
    "trend of a line that is not listed there. In filing tables a dash (—) means nil (zero).\n"
)
