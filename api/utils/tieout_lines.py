"""DOCBOT-2637: put back filing lines the books query skipped.

The SQL step filters an account column with ``IN (...)`` and sometimes leaves
out a line the filing shows in the same table (Etsy: "Asset impairment
charges" between G&A and "Total operating expenses"). Synthesis then tells the
user that line is not in the books. This module finds such skipped lines
deterministically and adds them to the query's ``IN`` list:

* only columns whose complete value list is known (schema cache, DOCBOT-2633),
  so a value is added only if the books really hold it;
* only filing table rows in a short gap (at most MAX_GAP rows) between two rows
  the query already selected, in the same chunk, so a skipped line inside a
  block comes back but an opex question never pulls in revenue or net income.

Pure functions, no I/O: hybrid_service re-executes the rewritten query.
"""
from __future__ import annotations

import re
from typing import Any, Optional

import sqlglot
from sqlglot import exp

from api.utils.discrepancy_detector import _SOURCE_HEADER_RE, _TABLE_ROW_RE
from api.utils.sql_validator import _DIALECT_ALIASES

MAX_GAP = 2
MAX_ADDED = 20


def _complete_values(schema_subset: list[dict[str, Any]]) -> dict[str, list[str]]:
    """lower(column name) -> every value it holds, for columns with a full list."""
    out: dict[str, list[str]] = {}
    for table in schema_subset:
        for col in table.get("columns", []):
            if col.get("values_complete") and col.get("sample_values"):
                out[col["name"].lower()] = [str(v) for v in col["sample_values"]]
    return out


def _row_sequences(doc_context: str) -> list[list[str]]:
    """Table-row labels in order, one list per retrieved chunk."""
    starts = [0] + [m.start() for m in _SOURCE_HEADER_RE.finditer(doc_context)]
    seqs = []
    for start, end in zip(starts, starts[1:] + [len(doc_context)]):
        labels = [m.group("label").strip() for m in _TABLE_ROW_RE.finditer(doc_context, start, end)]
        if labels:
            seqs.append(labels)
    return seqs


def _norm(label: str) -> str:
    """Case, punctuation and plural-insensitive line name."""
    return " ".join(t.rstrip("s") for t in re.sub(r"[^a-z0-9]+", " ", label.lower()).split())


def _match(label: str, values: list[str]) -> Optional[str]:
    """The value naming the same line. Exact on purpose: the detector's
    synonym matching treats "Gross profit" as "Net income"."""
    key = _norm(label)
    return next((v for v in values if _norm(v) == key), None)


def add_skipped_lines(
    sql: str, schema_subset: list[dict[str, Any]], doc_context: str, dialect: str,
) -> tuple[str, list[str]]:
    """Return (sql, added values). ``sql`` is unchanged when nothing was skipped."""
    complete = _complete_values(schema_subset)
    seqs = _row_sequences(doc_context) if complete else []
    if not seqs:
        return sql, []
    read = _DIALECT_ALIASES.get(dialect, dialect)
    try:
        tree = sqlglot.parse_one(sql, read=read)
    except sqlglot.errors.ParseError:
        return sql, []

    added: list[str] = []
    for node in tree.find_all(exp.In):
        col = node.this
        values = complete.get(col.name.lower()) if isinstance(col, exp.Column) else None
        if not values or node.args.get("query") or isinstance(node.parent, exp.Not):
            continue
        # only row filters: an IN inside a computed column (CASE ... IN) changes what it computes
        if node.find_ancestor(exp.Where) is None or node.find_ancestor(exp.Case) is not None:
            continue
        present = [e.this for e in node.expressions if isinstance(e, exp.Literal) and e.is_string]
        missing = [v for v in values if v not in present]
        new: list[str] = []
        for labels in seqs:
            hits = [i for i, lab in enumerate(labels) if _match(lab, present)]
            for a, b in zip(hits, hits[1:]):
                if b - a - 1 > MAX_GAP:
                    continue  # far-apart picks (revenue ... net income) are a choice, not a skip
                for lab in labels[a + 1:b]:
                    value = _match(lab, [v for v in missing if v not in new])
                    if value is not None:
                        new.append(value)
        new = new[:MAX_ADDED - len(added)]
        if new:
            node.set("expressions", node.expressions + [exp.Literal.string(v) for v in new])
            added.extend(new)
    if not added:
        return sql, []
    return tree.sql(dialect=read), added
