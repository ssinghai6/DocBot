"""DOCBOT-2732: tied doc/books lines survive detector -> lineage -> workpaper export. No network."""

from __future__ import annotations

import io
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openpyxl import load_workbook

from api import demo_service as d
from api import tieout_export
from api.routes import tieout_routes
from api.utils.discrepancy_detector import detect_discrepancies
from api.utils.lineage import Lineage, LineageCollector

DOC_CONTEXT = "\n\n".join(
    f"Source: {c.metadata['source']}, Page {c.metadata['page']}\n{c.page_content}"
    for c in d.TIEOUT_DOCUMENT_CHUNKS
)
SQL = "SELECT segment, amount AS segment_revenue FROM revenue_by_segment WHERE period = 'FY2025'"


@pytest.fixture(scope="module")
def report():
    path = d._create_tieout_sqlite()
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(SQL)]
    conn.close()
    path.unlink(missing_ok=True)
    return detect_discrepancies(
        DOC_CONTEXT, {"result_preview": rows, "sources": ["revenue_by_segment"], "sql_query": SQL}
    )


def test_detector_keeps_ties_and_still_flags_planted_break(report):
    assert [i.label for i in report.discrepancies] == ["Lab Services"]
    assert all(i.status == "exception" for i in report.discrepancies)
    assert report.ties, "agreeing segment lines must be kept as ties"
    tie = report.ties[0]
    assert tie.status == "ties" and tie.doc_file and tie.db_query == SQL
    assert abs(tie.delta) < abs(report.discrepancies[0].delta)
    assert "Lab Services" not in {t.label for t in report.ties}
    assert len({t.id for t in report.ties}) == len(report.ties)


def test_lineage_round_trip_with_and_without_ties(report):
    col = LineageCollector("run-1", "hybrid")
    for item in [*report.discrepancies, *report.ties]:
        col.add_discrepancy(item)
    built = col.build()
    assert len(built.discrepancies) == 1 and len(built.ties) == len(report.ties)
    assert built.ties[0].db_citation.row
    assert Lineage.model_validate_json(built.model_dump_json()) == built
    old = Lineage.model_validate_json('{"run_id": "r", "mode": "hybrid", "discrepancies": []}')
    assert old.ties == []


def _lineage(ties: int = 1, exceptions: int = 1) -> Lineage:
    line = {"metric": "Revenue", "report_value": 100.0, "books_value": 100.0, "delta": 0.0, "delta_pct": 0.0}
    return Lineage.model_validate({
        "run_id": "run-t", "mode": "hybrid",
        "ties": [{**line, "id": f"d-t{i}", "metric": "-Ties"} for i in range(ties)],
        "discrepancies": [
            {**line, "id": f"d-e{i}", "books_value": 110.0, "delta": 10.0} for i in range(exceptions)
        ],
    })


def test_export_lists_exceptions_first_then_ties():
    rows = tieout_export.tieout_rows(_lineage())
    assert [r[-1] for r in rows] == ["Exception", "Ties out"]
    ws = load_workbook(io.BytesIO(tieout_export.build_xlsx(_lineage()))).active
    assert ws["J5"].value == "Exception" and ws["J6"].value == "Ties out"
    assert ws["A6"].value == "-Ties" and ws["A6"].data_type == "s"
    assert tieout_export.build_pdf(_lineage()).startswith(b"%PDF")


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    monkeypatch.setattr(tieout_routes, "_state", {"lineage": None, "sessions": None, "connections": None, "factory": None})
    app = FastAPI()
    app.include_router(tieout_routes.router)
    c = TestClient(app)
    c.cookies.set("docbot_session", "tok")
    return c


@pytest.mark.parametrize("ties,exceptions,status", [(1, 0, 200), (0, 1, 200), (0, 0, 404)])
def test_route_404_only_when_no_lines(client, ties, exceptions, status):
    with (
        patch.object(tieout_routes, "run_row_accessible", new_callable=AsyncMock, return_value=True),
        patch.object(tieout_routes, "get_lineage", new_callable=AsyncMock, return_value=_lineage(ties, exceptions)),
        patch("api.auth_service.get_user_from_session", new_callable=AsyncMock,
              return_value=SimpleNamespace(id="u-1", org_id="org-1", role="viewer")),
    ):
        assert client.get("/api/lineage/run-t/tieout.xlsx").status_code == status
