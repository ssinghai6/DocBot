"""DOCBOT-2604: pure helpers of scripts/golden_path.py (no network)."""

import importlib.util
import io
from pathlib import Path

import openpyxl

spec = importlib.util.spec_from_file_location("golden_path", Path(__file__).parents[2] / "scripts" / "golden_path.py")
gp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gp)

CITED = {"metric": "Net income", "report_value": 330, "books_value": 325,
         "doc_citation": {"source": "10K.pdf"}, "db_citation": {"table": "quarterly"}}


def test_parse_sse_skips_noise():
    ev = gp.parse_sse(['data: {"type":"token"}', ": ping", "data: nope", 'data: {"type":"done","run_id":"r1"}'])
    assert [e["type"] for e in ev] == ["token", "done"]
    assert gp.run_id_of(ev) == "r1"


def test_planted_discrepancy_ok_and_raw_dollars():
    assert gp.planted_discrepancy([{"type": "lineage", "discrepancies": [CITED]}])[0]
    raw = {**CITED, "report_value": 330e6, "books_value": 325e6}
    assert gp.planted_discrepancy([{"type": "lineage", "discrepancies": [raw]}])[0]


def test_planted_discrepancy_needs_both_citations_and_values():
    no_db = {**CITED, "db_citation": {}}
    assert not gp.planted_discrepancy([{"type": "lineage", "discrepancies": [no_db]}])[0]
    other = {**CITED, "report_value": 1, "books_value": 2}
    assert not gp.planted_discrepancy([{"type": "lineage", "discrepancies": [other]}])[0]


def _xlsx(rows):
    wb = openpyxl.Workbook()
    for r in rows:
        wb.active.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_check_xlsx():
    assert gp.check_xlsx(_xlsx([["a", "b"], [1, 2]]))[0]
    assert not gp.check_xlsx(b"garbage")[0]


def test_check_autopilot():
    ok = [{"type": "step"}, {"type": "answer", "content": "x"}, {"type": "done"}]
    assert gp.check_autopilot(ok)[0]
    assert not gp.check_autopilot(ok + [{"type": "error", "content": "boom"}])[0]
    assert not gp.check_autopilot([{"type": "done"}])[0]
