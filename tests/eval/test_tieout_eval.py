"""DOCBOT-2541: tie-out accuracy eval -- scorer + gold-set consistency (CI),
live N-run pass rate (external).

CI:    pytest tests/eval/test_tieout_eval.py
Live:  pytest tests/eval/test_tieout_eval.py -m external -s   (needs groq_api_key)
"""

from __future__ import annotations

import asyncio
import os
import re
import sqlite3

import pytest

from tests.eval.gold_tieout_set import GOLD_TIEOUT, M
from tests.eval.tieout_eval import (
    RunRecord,
    _chunks_for,
    _sqlite_for,
    close,
    report,
    run_eval,
    score_run,
)

SEG = next(q for q in GOLD_TIEOUT if q.id == "segment_note12")
LAB_SERVICES = {
    "metric": "Lab Services",
    "period": "FY2025",
    "report_value": 104.9 * M,
    "books_value": 102.7 * M,
    "doc_citation": {"file": "Tidewell-Draft-10K-FY2025.pdf", "page": 9},
    "db_citation": {"table": "revenue_by_segment", "query": "SELECT ..."},
}
SEG_ROWS = [
    {"segment": "Commercial Instruments", "amount": 248_400_000},
    {"segment": "Lab Services", "amount": 102_700_000},
    {"segment": "Software and Analytics", "amount": 61_500_000},
]


def _good() -> RunRecord:
    return RunRecord(intent="hybrid", discrepancies=[dict(LAB_SERVICES)], sql_rows=SEG_ROWS, doc_pages={9})


# --- scorer ----------------------------------------------------------------


def test_close_tolerance_and_unit_alignment():
    assert close(325, 325 * M)            # books in $M vs report in $
    assert close(104.9 * M, 104.95 * M)   # 0.05% -> ties
    assert not close(54.9 * M, 55.2 * M)  # 0.55% planted gap is never a tie
    assert not close(None, 1.0)


def test_correct_run_passes():
    assert score_run(SEG, _good()) == (True, [])


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda r: setattr(r, "discrepancies", []), "Lab Services: expected 1 discrepancy, got 0"),
        (lambda r: r.discrepancies[0].update(books_value=104.9 * M), "Lab Services: values"),
        (lambda r: r.discrepancies[0].update(doc_citation={"file": "x.pdf", "page": 4}), "doc citation"),
        (lambda r: r.discrepancies[0].update(db_citation={}), "no db citation"),
        (lambda r: setattr(r, "doc_pages", set()), "doc page 9 not cited"),
        (lambda r: setattr(r, "sql_rows", SEG_ROWS[1:]), "Commercial Instruments: books value"),
        (lambda r: r.discrepancies.append({**LAB_SERVICES, "metric": "Commercial Instruments"}),
         "Commercial Instruments: flagged but ties out"),
        (lambda r: r.discrepancies.append({**LAB_SERVICES, "metric": "Gross profit"}), "false positive"),
        (lambda r: setattr(r, "error", "TimeoutError"), "error: TimeoutError"),
    ],
)
def test_each_check_can_fail_a_run(mutate, reason):
    run = _good()
    mutate(run)
    passed, fails = score_run(SEG, run)
    assert not passed
    assert any(reason in f for f in fails), fails


def test_report_pass_rate_per_question():
    text = report({"segment_note12": [(True, []), (False, ["x"]), (True, [])]}, "offline")
    assert "2/3" in text and "first failure: x" in text and "overall 2/3" in text


# --- gold set is consistent with the demo data ------------------------------


def test_gold_set_size_and_ids():
    assert len(GOLD_TIEOUT) >= 10
    assert len({q.id for q in GOLD_TIEOUT}) == len(GOLD_TIEOUT)
    assert sum(f.flag for q in GOLD_TIEOUT for f in q.figures) >= 5
    assert sum(not f.flag for q in GOLD_TIEOUT for f in q.figures) >= 5


@pytest.mark.parametrize("q", GOLD_TIEOUT, ids=lambda q: q.id)
def test_gold_values_match_demo_data(q):
    pages = {c.metadata["page"]: c.page_content for c in _chunks_for(q.dataset)}
    path = _sqlite_for(q.dataset)
    try:
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(q.books_sql)]
        conn.close()
    finally:
        path.unlink(missing_ok=True)
    for fig in q.figures:
        assert fig.label.lower() in pages[fig.page].lower(), (fig.label, fig.page)
        stated = [float(n.replace(",", "")) for n in re.findall(r"\$([0-9][0-9,]*(?:\.[0-9]+)?)", pages[fig.page])]
        assert any(close(s, fig.report) for s in stated), (fig.label, stated)
        assert any(close(v, fig.books) for r in rows for v in r.values() if isinstance(v, (int, float))), fig.label
        assert fig.flag == (not close(fig.report, fig.books))


# --- live ------------------------------------------------------------------


@pytest.mark.external
def test_live_tieout_pass_rate(capsys):
    """N live runs per question (EVAL_TIEOUT_RUNS, default 10); >= 90% each."""
    if not os.getenv("groq_api_key"):
        pytest.skip("groq_api_key not set")
    runs = int(os.getenv("EVAL_TIEOUT_RUNS", "10"))
    results = asyncio.run(run_eval("live", runs))
    with capsys.disabled():
        print("\n" + report(results, "live"))
    low = {qid: sum(p for p, _ in o) for qid, o in results.items() if sum(p for p, _ in o) < 0.9 * runs}
    assert not low, f"below 90% pass rate: {low}"
