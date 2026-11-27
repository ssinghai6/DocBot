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
# "(18,509)", "$(18,509)" (DOCBOT-4021: edgar _row_cells output) and "-18,509" are negative
_NUMBER_RE = re.compile(r"^(\()?\$?\s*(\()?\s*(-|−)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(%)?\s*(\))?$")
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
    table: str = ""

    def line(self) -> str:
        where = ", ".join(p for p in (self.source, f"in {self.units}" if self.units else "") if p)
        cur, pri = _fmt(self.value, self.percent_kind), _fmt(self.prior, self.percent_kind)
        # the table matters: "Marketing" is also a row of the stock-comp table
        name = f"{self.label} [{self.table}]" if self.table else self.label
        head = f"- {name} ({where}): FY{self.year} {cur} vs FY{self.prior_year} {pri}"
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
    value = float(m.group(4).replace(",", ""))
    if (m.group(1) or m.group(2)) and m.group(6) or m.group(3):
        value = -value
    return value, bool(m.group(5))


def _header_years(cells: list[str]) -> Optional[list[Optional[str]]]:
    """Year per column for a header row (None for non-year cells), else None."""
    years = [c if _YEAR_RE.fullmatch(c) else None for c in cells]
    if sum(y is not None for y in years) < 2:
        return None
    if any(y is None and re.search(r"\d", c) for y, c in zip(years, cells)):
        return None  # a data row, not a header
    return years


_NOT_CAPTION = re.compile(r"^(\(|(year|years|fiscal year|three months|six months|nine months)s? ended|as of|december|table of contents|\d)", re.I)


def _is_caption(line: str) -> bool:
    """A table title line ("Consolidated Statements of Operations"), not a
    units / period / footer line."""
    text = line.strip()
    return len(text) >= 10 and not _NOT_CAPTION.match(text) and re.search(r"[A-Za-z]{4}", text) is not None


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
        caption = table = ""
        for line in chunk.splitlines():
            if not line.strip():
                years = None  # a blank line ends the table
                continue
            if " | " not in line:
                if years is None and _is_caption(line):
                    caption = re.split(r"\s+(?:is|are) as follows|\s*\(|:", line.strip())[0][:90]
                continue  # sub-headings inside a table ("Operating expenses:")
            cells = [c.strip() for c in line.split(" | ")]
            header = _header_years(cells)
            if header is not None:
                years, table = header, caption
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
            facts.append(YoYFact(cells[0], year, prior_year, value, prior, pct or prior_pct, source, units, table))
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


# ── answer check (DOCBOT-2637) ────────────────────────────────────────────────
# Line names as answers write them, beyond the filing caption itself.
_ALIASES = {
    "general and administrative": ("g&a", "general & administrative"),
    "income from operations": ("operating income",),
    "provision for income taxes": ("tax provision", "income tax expense"),
}
# no bare "up"/"down": "identical down to the dollar", "drill-down"
_UP_WORDS = r"(?:rose|rises?|risen|increased|increases|grew|grows|higher|jumped|climbed)"
_DOWN_WORDS = r"(?:fell|falls?|decreased|decreases|declined|declines|decline|dropped|drops?|lower|shrank)"
_STEADY_WORDS = r"(?:steady|unchanged|flat|no change)"
# allocation tables ("Stock-based compensation expense included in ...") reuse
# line names for a slice of each line, so their rows are not the line itself
_ALLOCATION_TABLE = re.compile(r"\bincluded in\b|\ballocated\b|\bby (function|line item)\b", re.I)
# ratios, shares and hypotheticals are not statements about the line itself
# ratios, shares, hypotheticals and filing-vs-books comparisons ("$5M lower in
# the filing than in the books") are not year-over-year statements
_SKIP_CLAUSE = re.compile(
    r"margin|ratio|÷|share of|% of|\bper\b|\bif\b|\bcould\b|\bmay\b|\bmight\b|\bwould\b|\bshould\b"
    r"|\bbooks?\b|\bledger\b|\bdatabase\b|\broll.?up\b|\bthan\b"
)
_CLAUSE_RE = re.compile(r"[.;:,\n|(]|\bwhile\b|\bbut\b|\byet\b|\bwhereas\b|\bdespite\b|\bthough\b|\band\b")


def _names(label: str) -> tuple[str, ...]:
    key = re.sub(r"\s*\(\d+\)$", "", label.lower()).strip()
    return (key, *_ALIASES.get(key, ()))


def yoy_contradictions(answer: str, facts: list[YoYFact]) -> list[str]:
    """Statements like "Marketing fell" / "higher tax provision" that contradict
    a computed change.

    Conservative on purpose, since a hit triggers a rewrite: the direction word
    must sit right next to the line name ("<line> rose", "a decline in <line>",
    "higher <line>"), the clause must name only that line, and a line name
    that appears in two tables with different directions is never checked.
    """
    all_facts = facts
    # "Marketing" in a stock-comp allocation table is not marketing expense
    facts = [f for f in facts if not _ALLOCATION_TABLE.search(f.table)]
    direction: dict[str, set[int]] = {}
    by_name: dict[str, YoYFact] = {}
    for fact in facts:
        if fact.percent_kind:
            continue
        diff = fact.value - fact.prior
        if fact.value < 0 and fact.prior < 0:
            diff = abs(fact.value) - abs(fact.prior)  # parenthesised lines: size
        sign = (diff > 0) - (diff < 0)
        for name in _names(fact.label):
            direction.setdefault(name, set()).add(sign)
            by_name.setdefault(name, fact)
    names = [n for n, signs in direction.items() if len(signs) == 1]
    # every line name any table uses, longest first: "cost of revenue" is masked
    # as one token before "revenue" is looked for
    every = sorted({n for f in all_facts for n in _names(f.label)}, key=len, reverse=True)
    out: list[str] = []
    for clause in _CLAUSE_RE.split(answer.lower()):
        if _SKIP_CLAUSE.search(clause):
            continue
        tokens: dict[str, str] = {}
        for i, n in enumerate(every):
            token = f"line{i}x"
            masked = re.sub(rf"(?<![a-z]){re.escape(n)}(?![a-z])", token, clause)
            if masked != clause:
                clause, tokens[n] = masked, token
        present = [n for n in names if n in tokens]
        if len(present) != 1 or len(tokens) != 1:
            continue
        name = tokens[present[0]]
        said = None
        if re.search(rf"{name}\W+(?:\w+\W+){{0,4}}?{_UP_WORDS}\b|\b(?:higher|increase in|rise in|growth in)\s+(?:\w+\s+)?{name}", clause):
            said = 1
        if re.search(rf"{name}\W+(?:\w+\W+){{0,4}}?{_DOWN_WORDS}\b|\b(?:lower|decline in|decrease in|drop in|reduction in)\s+(?:\w+\s+)?{name}", clause):
            said = None if said == 1 else -1
        if said is None and re.search(rf"{name}\W+(?:\w+\W+){{0,4}}?{_STEADY_WORDS}\b", clause):
            said = 0
        if said is None:
            continue
        actual = next(iter(direction[present[0]]))
        if said != actual:
            text = clause.replace(name, present[0]).strip()
            out.append(f'"{text[:160]}" contradicts: {by_name[present[0]].line()[2:]}')
    return out
