"""DOCBOT-2722: batch EDGAR ingest returns ONE session holding every filing."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from api import edgar_service
from api.utils import vector_store as vs


class _FakeEmb(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(t) % 7), 1.0, 0.5] for t in texts]

    def embed_query(self, text):
        return [1.0, 1.0, 0.5]


@pytest.fixture()
def chroma_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(vs, "_PERSIST_DIR", tmp_path)
    return tmp_path


def _make_filing(sid: str, date: str, n: int):
    docs = [
        Document(page_content=f"{date} chunk {i}", metadata={"source": f"ETSY-10-K-{date}", "filing_date": date})
        for i in range(n)
    ]
    return vs.create_store(sid, docs, _FakeEmb())


def _factory():
    session = AsyncMock()
    session.begin = MagicMock()
    session.begin.return_value.__aenter__ = AsyncMock()
    session.begin.return_value.__aexit__ = AsyncMock(return_value=False)
    f = MagicMock()
    f.return_value.__aenter__ = AsyncMock(return_value=session)
    f.return_value.__aexit__ = AsyncMock(return_value=False)
    return f


def _combine(results, owner, stores, fields, inserted):
    def _fake_insert(_t):
        stmt = MagicMock()
        stmt.values = MagicMock(side_effect=lambda **kw: inserted.append(kw) or MagicMock())
        return stmt

    with patch("api.edgar_service.insert", side_effect=_fake_insert):
        return asyncio.run(edgar_service.combine_filing_sessions(
            results, ticker="ETSY", filing_type="10-K", owner=owner,
            sessions_table=MagicMock(), async_session_factory=_factory(),
            vector_stores=stores, get_embeddings_fn=_FakeEmb, extracted_fields_store=fields,
        ))


def test_three_filings_one_session_with_per_filing_metadata(chroma_dir):
    _make_filing("a1", "2025-02-20", 3)
    _make_filing("b2", "2024-02-22", 2)
    _make_filing("c3", "2023-02-23", 4)
    results = [{"session_id": s} for s in ("a1", "b2", "c3")]
    stores: dict = {}
    fields = {"a1": [{"f": 1}], "c3": [{"f": 3}]}
    inserted: list[dict] = []
    owner = {"user_id": "u1", "org_id": "o1", "anon_owner": None}

    sid = _combine(results, owner, stores, fields, inserted)

    assert sid and sid not in ("a1", "b2", "c3")
    data = stores[sid]._collection.get(include=["metadatas"])
    assert len(data["ids"]) == 9 and len(set(data["ids"])) == 9
    dates = sorted(m["filing_date"] for m in data["metadatas"])
    assert dates == ["2023-02-23"] * 4 + ["2024-02-22"] * 2 + ["2025-02-20"] * 3
    assert fields[sid] == [{"f": 1}, {"f": 3}]
    # session row is stamped with the caller's owner columns
    assert inserted[0]["session_id"] == sid
    assert inserted[0]["org_id"] == "o1" and inserted[0]["user_id"] == "u1"


def test_partial_failure_combines_only_successes(chroma_dir):
    _make_filing("a1", "2025-02-20", 3)
    _make_filing("c3", "2023-02-23", 2)
    results = [{"session_id": "a1"}, {"accession_number": "x", "error": "boom"}, {"session_id": "c3"}]
    stores: dict = {}
    sid = _combine(results, {"user_id": "u"}, stores, {}, [])
    metas = stores[sid]._collection.get(include=["metadatas"])["metadatas"]
    assert len(metas) == 5
    assert {m["filing_date"] for m in metas} == {"2025-02-20", "2023-02-23"}


def test_single_success_reused_and_none_when_all_fail(chroma_dir):
    inserted: list[dict] = []
    assert _combine([{"session_id": "only"}, {"error": "x"}], {"user_id": "u"}, {}, {}, inserted) == "only"
    assert _combine([{"error": "x"}], {"user_id": "u"}, {}, {}, inserted) is None
    assert inserted == []


def test_combined_never_reads_sessions_outside_results(chroma_dir):
    """Another caller's session (not in this batch's results) is never read."""
    _make_filing("mine1", "2025-02-20", 1)
    _make_filing("mine2", "2024-02-22", 1)
    _make_filing("theirs", "1999-01-01", 5)
    stores: dict = {}
    sid = _combine([{"session_id": "mine1"}, {"session_id": "mine2"}], {"user_id": "u"}, stores, {}, [])
    metas = stores[sid]._collection.get(include=["metadatas"])["metadatas"]
    assert len(metas) == 2 and all(m["filing_date"] != "1999-01-01" for m in metas)
