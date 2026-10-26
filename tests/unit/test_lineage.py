"""Unit tests for DOCBOT-1510 lineage collector, persistence and reranker scores."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api import lineage_service
from api.utils.lineage import SNIPPET_MAX_CHARS, LineageCollector, safe_snippet
from api.utils.reranker import rerank, rerank_scored


class _Doc:
    def __init__(self, text: str) -> None:
        self.page_content = text
        self.metadata = {"source": "a.pdf", "page": 1}


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class TestSafeSnippet:
    def test_masks_pii(self):
        out = safe_snippet("Contact jane.doe@example.com for details")
        assert out is not None and "jane.doe@example.com" not in out

    def test_truncates(self):
        out = safe_snippet("word " * 500)
        assert out is not None and len(out) <= SNIPPET_MAX_CHARS + 1
        assert out.endswith("…")

    def test_empty_returns_none(self):
        assert safe_snippet("") is None
        assert safe_snippet(None) is None


class TestCollector:
    def test_sources_deduped_by_kind_label_page(self):
        c = LineageCollector("r1", "docs", "q")
        c.add_source(kind="pdf", label="a.pdf", page=1, snippet="x")
        c.add_source(kind="pdf", label="a.pdf", page=1, snippet="y")
        c.add_source(kind="pdf", label="a.pdf", page=2, snippet="z")
        assert len(c.build().sources) == 2

    def test_same_source_in_different_autopilot_steps_kept(self):
        c = LineageCollector("r1", "autopilot")
        c.add_source(kind="pdf", label="a.pdf", page=1, step_num=1)
        c.add_source(kind="pdf", label="a.pdf", page=1, step_num=2)
        assert len(c.build().sources) == 2

    def test_standalone_query_ignored_when_unchanged(self):
        c = LineageCollector("r1", "db", "same")
        c.set_standalone_query("same")
        assert c.build().standalone_query is None
        c.set_standalone_query("different")
        assert c.build().standalone_query == "different"

    def test_step_context_manager_records_ok_and_error(self):
        c = LineageCollector("r1", "db")
        with c.step("fine", tool="t"):
            pass
        with pytest.raises(RuntimeError):
            with c.step("boom"):
                raise RuntimeError("x")
        steps = c.build().steps
        assert [s.status for s in steps] == ["ok", "error"]
        assert steps[0].latency_ms is not None

    def test_set_sql_adds_table_sources_and_caps_preview(self):
        c = LineageCollector("r1", "db")
        c.set_sql(
            sql="SELECT 1",
            tables_selected=["orders"],
            tables_considered=["orders", "products"],
            result_preview=[{"i": i} for i in range(50)],
        )
        built = c.build()
        assert built.sql is not None and len(built.sql.result_preview) == 10
        assert any(s.kind == "sql_table" and s.label == "orders" for s in built.sources)

    def test_discrepancy_recorded(self):
        # DOCBOT-2002: add_discrepancy takes a detector DiscrepancyItem and
        # stores the structured tie-out (id/metric/report/books/citations).
        from api.utils.discrepancy_detector import DiscrepancyItem

        c = LineageCollector("r1", "hybrid")
        item = DiscrepancyItem(
            label="Net income",
            doc_value=330.0,
            db_value=325.0,
            delta=-5.0,
            pct=-1.5,
            doc_snippet="Net income: $330M",
            db_snippet="net_income=325",
            period="Q4 2025",
            unit="currency",
            doc_file="10-K.pdf",
            doc_page=42,
            db_table="financials",
            db_query="SELECT net_income FROM financials",
        )
        c.add_discrepancy(item)
        d = c.build().discrepancies[0]
        assert (d.report_value, d.books_value, d.delta) == (330.0, 325.0, -5.0)
        assert d.id == item.id
        assert d.metric == "Net income"
        assert d.period == "Q4 2025"
        assert d.severity == "high"  # |pct| = 1.5 >= 1.0 band
        assert d.unit == "currency"
        assert d.doc_citation.file == "10-K.pdf" and d.doc_citation.page == 42
        assert d.db_citation.table == "financials"
        assert d.db_citation.query == "SELECT net_income FROM financials"

    def test_discrepancy_deduped_by_id(self):
        from api.utils.discrepancy_detector import DiscrepancyItem, discrepancy_id

        c = LineageCollector("r1", "hybrid")
        item = DiscrepancyItem(
            label="Net income", doc_value=330.0, db_value=325.0,
            delta=-5.0, pct=-1.5, doc_snippet="d", db_snippet="b",
        )
        c.add_discrepancy(item)
        c.add_discrepancy(item)
        built = c.build()
        assert len(built.discrepancies) == 1
        assert built.discrepancies[0].id == discrepancy_id("net income", None, 330.0, 325.0)

    def test_discrepancy_legacy_row_upgrade(self):
        """Pre-2002 persisted rows ({label, doc_value, ...}) validate + get an id."""
        from api.utils.lineage import LineageDiscrepancy

        d = LineageDiscrepancy.model_validate(
            {"label": "net income", "doc_value": 330.0, "db_value": 325.0, "pct": -1.5, "delta": -5.0}
        )
        assert d.metric == "net income"
        assert d.report_value == 330.0 and d.books_value == 325.0
        assert d.delta_pct == -1.5
        assert d.id.startswith("d-")
        # Round-trips through the SSE JSON payload
        import json as _json

        assert _json.loads(d.model_dump_json())["id"] == d.id

    def test_build_attaches_model_calls_and_skips_malformed(self):
        c = LineageCollector("r1", "hybrid")
        built = c.build(
            [
                {"provider": "groq", "model": "llama", "input_tokens": 10, "success": True},
                {"provider": None, "model": None, "latency_ms": "not-a-number"},
            ]
        )
        assert len(built.model_calls) == 1
        assert built.total_latency_ms is not None

    def test_mark_step_updates_latest_matching_step(self):
        c = LineageCollector("r1", "hybrid")
        c.add_step("run_sql_pipeline")
        c.add_step("synthesize")
        c.mark_step("run_sql_pipeline", "error", "boom")
        steps = c.build().steps
        assert steps[0].status == "error" and steps[0].detail == "boom"
        assert steps[1].status == "ok"
        c.mark_step("missing", "error")  # no-op, must not raise

    def test_mode_can_change(self):
        c = LineageCollector("r1", "db")
        c.set_mode("csv")
        assert c.build().mode == "csv"


# ---------------------------------------------------------------------------
# Reranker scores
# ---------------------------------------------------------------------------


class TestRerankScored:
    @pytest.fixture(autouse=True)
    def _hf_provider(self, monkeypatch):
        monkeypatch.setenv("RERANKER_PROVIDER", "hf")

    def test_no_key_returns_none_scores(self):
        docs = [_Doc("a"), _Doc("b")]
        out = rerank_scored("q", docs, "", top_k=1)
        assert len(out) == 1 and out[0][1] is None

    def test_scores_sorted_descending(self):
        docs = [_Doc("low"), _Doc("high")]
        resp = MagicMock()
        resp.json.return_value = [[{"score": 0.1}, {"score": 0.9}]]
        with patch("api.utils.reranker.httpx.post", return_value=resp):
            out = rerank_scored("q", docs, "key", top_k=2)
        assert [d.page_content for d, _ in out] == ["high", "low"]
        assert out[0][1] == pytest.approx(0.9)

    def test_rerank_contract_unchanged(self):
        docs = [_Doc("low"), _Doc("high")]
        resp = MagicMock()
        resp.json.return_value = [[{"score": 0.1}, {"score": 0.9}]]
        with patch("api.utils.reranker.httpx.post", return_value=resp):
            out = rerank("q", docs, "key", top_k=1)
        assert [d.page_content for d in out] == ["high"]


# ---------------------------------------------------------------------------
# Persistence + SSE
# ---------------------------------------------------------------------------


@pytest.fixture
async def wired_lineage():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    metadata = MetaData()
    table = lineage_service.register_lineage_table(metadata)
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    lineage_service.wire_lineage_store(table, factory)
    yield table
    lineage_service._lineage_table = None
    lineage_service._async_session_factory = None
    await engine.dispose()


class TestPersistence:
    async def test_emit_returns_sse_and_round_trips(self, wired_lineage):
        c = LineageCollector("run-1", "docs", "what is revenue?")
        c.add_source(kind="pdf", label="10k.pdf", page=3, snippet="Revenue was $1B")
        line = lineage_service.emit_lineage(c, "sess-1")
        assert line.startswith("data: ") and line.endswith("\n\n")
        payload = json.loads(line[6:])
        assert payload["type"] == "lineage" and payload["run_id"] == "run-1"

        # wait for the fire-and-forget write
        import asyncio
        await asyncio.gather(*list(lineage_service._pending_writes))

        loaded = await lineage_service.get_lineage("run-1")
        assert loaded is not None
        assert loaded.sources[0].label == "10k.pdf"

    async def test_upsert_overwrites_same_run(self, wired_lineage):
        import asyncio
        for q in ("first", "second"):
            c = LineageCollector("run-2", "docs", q)
            lineage_service.emit_lineage(c, "s")
            await asyncio.gather(*list(lineage_service._pending_writes))
        loaded = await lineage_service.get_lineage("run-2")
        assert loaded is not None and loaded.question == "second"

    async def test_missing_run_returns_none(self, wired_lineage):
        assert await lineage_service.get_lineage("nope") is None

    async def test_emit_without_store_still_returns_event(self):
        lineage_service._lineage_table = None
        c = LineageCollector("run-3", "db")
        assert lineage_service.emit_lineage(c).startswith("data: ")

    def test_emit_outside_event_loop_does_not_raise(self):
        c = LineageCollector("run-4", "db")
        assert lineage_service.emit_lineage(c)
