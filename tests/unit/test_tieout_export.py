"""DOCBOT-2014 — tie-out workpaper export (XLSX/PDF) + download route guards. No network, no DB."""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pymupdf
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openpyxl import load_workbook

from api import tieout_export
from api.routes import tieout_routes
from api.utils.lineage import Lineage

LINEAGE = Lineage.model_validate({
    "run_id": "run-abc",
    "mode": "hybrid",
    "question": "Does Q4 net income tie out?",
    "discrepancies": [
        {
            "id": "d-00000001", "metric": "Q4 net income", "period": "FY2024",
            "report_value": 325_000_000.0, "books_value": 330_000_000.0,
            "delta": 5_000_000.0, "delta_pct": 1.54, "severity": "low", "unit": "currency",
            "doc_citation": {"file": "techcorp-10k.pdf", "page": 42},
            "db_citation": {"table": "income_statement", "query": "SELECT net_income FROM income_statement"},
        },
        {
            "id": "d-00000002", "metric": "=HYPERLINK(\"http://evil\")", "report_value": 1.0,
            "books_value": None, "delta": None,
        },
    ],
})


def test_rows_carry_both_citations_and_status():
    rows = tieout_export.tieout_rows(LINEAGE)
    assert rows[0] == (
        "Q4 net income", "FY2024", 325_000_000.0, "techcorp-10k.pdf, p. 42",
        330_000_000.0, "income_statement | SELECT net_income FROM income_statement",
        5_000_000.0, 1.54, "low", "Exception",
    )
    assert rows[1][-1] == "Needs review"


def test_status_ties_out_within_rounding():
    lin = Lineage.model_validate({"run_id": "r", "mode": "hybrid", "discrepancies": [
        {"id": "d-1", "metric": "Revenue", "report_value": 1.0, "books_value": 1.001, "delta": 0.001},
    ]})
    assert tieout_export.tieout_rows(lin)[0][-1] == "Ties out"


def test_xlsx_round_trip_and_no_formula_injection():
    ws = load_workbook(io.BytesIO(tieout_export.build_xlsx(LINEAGE))).active
    assert [c.value for c in ws[4]] == list(tieout_export.HEADERS)
    assert ws["A5"].value == "Q4 net income"
    assert ws["C5"].value == 325_000_000 and ws["G5"].value == 5_000_000
    assert ws["D5"].value == "techcorp-10k.pdf, p. 42"
    assert ws["A6"].data_type == "s"  # text from the document never becomes a formula


def test_pdf_contains_rows():
    data = tieout_export.build_pdf(LINEAGE)
    assert data.startswith(b"%PDF")
    text = pymupdf.open(stream=data, filetype="pdf")[0].get_text()
    assert "Q4 net income" in text and "techcorp-10k.pdf" in text


# ── route ────────────────────────────────────────────────────────────────────


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    monkeypatch.setattr(tieout_routes, "_state", {"lineage": None, "sessions": None, "connections": None, "factory": None})
    app = FastAPI()
    app.include_router(tieout_routes.router)
    return TestClient(app)


def _patches(accessible: bool, lineage=LINEAGE):
    return (
        patch.object(tieout_routes, "run_row_accessible", new_callable=AsyncMock, return_value=accessible),
        patch.object(tieout_routes, "get_lineage", new_callable=AsyncMock, return_value=lineage),
        patch("api.auth_service.get_user_from_session", new_callable=AsyncMock,
              return_value=SimpleNamespace(id="u-1", org_id="org-1", role="viewer")),
    )


@pytest.mark.parametrize("fmt,ctype", [
    ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ("pdf", "application/pdf"),
])
def test_download_ok(client, fmt, ctype):
    a, b, c = _patches(True)
    client.cookies.set("docbot_session", "tok")
    with a, b, c:
        resp = client.get(f"/api/lineage/run-abc/tieout.{fmt}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == ctype
    assert f'filename="tieout-run-abc.{fmt}"' in resp.headers["content-disposition"]


def test_foreign_run_is_404_and_never_loaded(client):
    a, b, c = _patches(False)
    client.cookies.set("docbot_session", "tok")
    with a, b as loader, c:
        resp = client.get("/api/lineage/run-abc/tieout.xlsx")
    assert resp.status_code == 404
    loader.assert_not_called()


def test_no_discrepancies_is_404(client):
    a, b, c = _patches(True, Lineage(run_id="r", mode="hybrid"))
    client.cookies.set("docbot_session", "tok")
    with a, b, c:
        assert client.get("/api/lineage/r/tieout.xlsx").status_code == 404


def test_unknown_format_rejected(client):
    a, b, c = _patches(True)
    client.cookies.set("docbot_session", "tok")
    with a, b, c:
        assert client.get("/api/lineage/run-abc/tieout.csv").status_code == 422


def test_anonymous_rejected_when_auth_required(client, monkeypatch):
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    assert client.get("/api/lineage/run-abc/tieout.xlsx").status_code == 401


@pytest.mark.asyncio
async def test_wait_pending_writes_lets_inflight_lineage_land():
    # DOCBOT-2604: export clicked before the background insert committed 404'd on prod
    import asyncio

    from api import lineage_service

    landed = []

    async def slow_write():
        await asyncio.sleep(0.05)
        landed.append(True)

    task = asyncio.get_running_loop().create_task(slow_write())
    lineage_service._pending_writes.add(task)
    task.add_done_callback(lineage_service._pending_writes.discard)
    await lineage_service.wait_pending_writes()
    assert landed == [True]
    await lineage_service.wait_pending_writes()  # nothing pending is a no-op
