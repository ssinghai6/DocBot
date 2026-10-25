"""DOCBOT-1611: clone_store copies a persisted Chroma collection (real Chroma, temp dir).

Used by the EDGAR cache-hit path so each caller gets their own collection.
No network: embeddings are a deterministic fake, stored vectors are copied.
"""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_chroma")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402

from api.utils import vector_store  # noqa: E402

SRC = "11111111-1111-1111-1111-111111111111"
DST_A = "22222222-2222-2222-2222-222222222222"
DST_B = "33333333-3333-3333-3333-333333333333"


class _FakeEmbeddings(Embeddings):
    def _vec(self, text: str) -> list[float]:
        return [float(len(text) % 7), float(text.count("revenue")), 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture
def persist_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "_PERSIST_DIR", tmp_path / "chroma")
    return tmp_path


def test_clone_copies_documents_into_independent_collections(persist_dir):
    emb = _FakeEmbeddings()
    docs = [
        Document(page_content="revenue grew 10%", metadata={"source": "AAPL-10-K", "page": 1}),
        Document(page_content="risk factors", metadata={"source": "AAPL-10-K", "page": 2}),
    ]
    vector_store.create_store(SRC, docs, emb)

    cloned_a = vector_store.clone_store(SRC, DST_A, emb)
    cloned_b = vector_store.clone_store(SRC, DST_B, emb)
    assert cloned_a is not None and cloned_b is not None
    store_a, count_a = cloned_a
    assert count_a == 2

    hits = store_a.similarity_search("revenue", k=2)
    assert {h.page_content for h in hits} == {"revenue grew 10%", "risk factors"}
    assert all(h.metadata["source"] == "AAPL-10-K" for h in hits)

    # Copies are independent: deleting the source or a sibling leaves A intact.
    vector_store.delete_store(SRC)
    vector_store.delete_store(DST_B)
    reloaded = vector_store.load_store(DST_A, emb)
    assert reloaded is not None
    assert len(reloaded.similarity_search("risk", k=5)) == 2


def test_clone_missing_source_returns_none(persist_dir):
    assert vector_store.clone_store(SRC, DST_A, _FakeEmbeddings()) is None
