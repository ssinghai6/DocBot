"""DOCBOT-2911 part B: cross-source matching. No network: embeddings and
list_org_ontology are mocked."""

from __future__ import annotations

import re

import numpy as np
import pytest

from api import cross_source_service as xs

_VOCAB: dict[str, int] = {}


def _fake_embed(texts):
    """Bag-of-words vectors: identical token sets -> cosine 1.0."""
    m = np.zeros((len(texts), 64))
    for r, t in enumerate(texts):
        for tok in re.findall(r"[a-z0-9]+", t.lower()):
            m[r, _VOCAB.setdefault(tok, len(_VOCAB) % 64)] += 1.0
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def _col(name, type_="text", tag="other", hashes=None):
    return {"name": name, "type": type_, "semantic_tag": tag, "null_pct": 0.0,
            "value_hashes": hashes or []}


def _table(conn, table, cols, dialect="csv"):
    return {"connection_id": conn, "dialect": dialect, "connection_name": f"name-{conn}",
            "table_name": table, "columns": cols, "embedding": []}


@pytest.fixture
def ontology(monkeypatch):
    """Install a fake list_org_ontology; returns (tables list, calls list)."""
    tables: list[dict] = []
    calls: list[str] = []

    async def fake(org_id):
        calls.append(org_id)
        return tables

    from api import ontology_service
    monkeypatch.setattr(ontology_service, "list_org_ontology", fake, raising=False)
    monkeypatch.setattr(xs, "_embed", _fake_embed)
    monkeypatch.setattr(xs, "_term_vecs_cache", None)
    return tables, calls


async def test_never_pairs_within_same_connection(ontology):
    tables, _ = ontology
    tables += [
        _table("c1", "orders", [_col("customer_id", hashes=["h1", "h2"])]),
        _table("c1", "customers", [_col("customer_id", hashes=["h1", "h2"])]),
    ]
    assert await xs.match_across_sources("org-1") == []


async def test_cross_connection_value_match(ontology):
    tables, _ = ontology
    tables += [
        _table("c1", "orders", [_col("customer_id", hashes=["h1", "h2", "h3"])]),
        _table("c2", "crm", [_col("customer_id", hashes=["h1", "h2", "h9"])], dialect="postgresql"),
    ]
    [m] = await xs.match_across_sources("org-1")
    assert {m["left"]["connection_id"], m["right"]["connection_id"]} == {"c1", "c2"}
    assert m["evidence"] == "values"
    assert m["containment"] == pytest.approx(0.667, abs=1e-3)


async def test_value_confirmed_ranks_above_name_only(ontology):
    tables, _ = ontology
    tables += [
        _table("c1", "a", [_col("region"), _col("cust no", hashes=["x", "y"])]),
        # Both pairs score 1.0 on name; only "cust no" is value-confirmed.
        _table("c2", "b", [_col("region"), _col("customer number", hashes=["x", "y", "z"])]),
    ]
    matches = await xs.match_across_sources("org-1")
    assert [m["evidence"] for m in matches] == ["values", "name+type"]
    assert matches[0]["left"]["column"] == "cust no"


async def test_disjoint_ids_are_rejected(ontology):
    tables, _ = ontology
    tables += [
        _table("c1", "a", [_col("order_id", hashes=["a1", "a2"])]),
        _table("c2", "b", [_col("order_id", hashes=["b1", "b2"])]),
    ]
    assert await xs.match_across_sources("org-1") == []


async def test_incompatible_types_not_paired(ontology):
    tables, _ = ontology
    tables += [
        _table("c1", "a", [_col("amount", type_="float64")]),
        _table("c2", "b", [_col("amount", type_="varchar")]),
    ]
    assert await xs.match_across_sources("org-1") == []


async def test_org_id_passthrough_and_none_short_circuits(ontology):
    _, calls = ontology
    await xs.match_across_sources("org-42")
    assert calls == ["org-42"]
    assert await xs.match_across_sources(None) == []
    assert await xs.cross_source_context(None, "q") == ("", {})
    assert calls == ["org-42"]  # None never reaches part A


async def test_finance_terms_map_across_sources(ontology):
    tables, _ = ontology
    tables += [
        _table("c1", "a", [_col("net income", type_="float64")]),
        _table("c2", "b", [_col("net_income", type_="numeric")]),
    ]
    terms = await xs.map_finance_terms("org-1")
    assert {r["connection_id"] for r in terms["net_income"]} == {"c1", "c2"}


async def test_context_block_and_measure_terms(ontology):
    tables, _ = ontology
    tables += [
        _table("c1", "pl", [_col("net income", type_="float64"),
                            _col("customer id", hashes=["h1"])]),
        _table("c2", "crm", [_col("customer id", hashes=["h1"])]),
    ]
    block, terms = await xs.cross_source_context("org-1", "customer id net income", "c1")
    assert "Cross-source column mappings" in block
    assert "name-c1.pl.customer id = name-c2.crm.customer id" in block
    assert len(block.strip().splitlines()) <= xs.BLOCK_MAX_LINES
    assert terms == {"net income": "net_income"}


async def test_single_connection_gives_no_block(ontology):
    tables, _ = ontology
    tables += [_table("c1", "pl", [_col("net income", type_="float64")])]
    block, terms = await xs.cross_source_context("org-1", "net income", "c1")
    assert block == ""
    assert terms == {"net income": "net_income"}


async def test_context_skips_silently_when_part_a_raises(monkeypatch):
    async def boom(org_id):
        raise RuntimeError("db down")

    from api import ontology_service
    monkeypatch.setattr(ontology_service, "list_org_ontology", boom, raising=False)
    assert await xs.cross_source_context("org-1", "q", "c1") == ("", {})


async def test_context_skips_when_part_a_function_missing(monkeypatch):
    from api import ontology_service
    monkeypatch.delattr(ontology_service, "list_org_ontology", raising=False)
    assert await xs.cross_source_context("org-1", "q", "c1") == ("", {})


def test_canonicalize_renames_measures_without_clobbering():
    meta = {"result_preview": [{"net_sales": 10, "revenue": 5, "quarter": "Q1"}], "sources": ["t"]}
    out = xs.canonicalize_sql_metadata(meta, {"net_sales": "revenue", "cost": "cogs"})
    assert out["result_preview"] == [{"net_sales": 10, "revenue": 5, "quarter": "Q1"}]
    out = xs.canonicalize_sql_metadata({"result_preview": [{"net_sales": 10}]}, {"net_sales": "revenue"})
    assert out["result_preview"] == [{"revenue": 10}]
    assert xs.canonicalize_sql_metadata(meta, {}) is meta


def test_humanize_expands_abbreviations():
    assert xs._humanize("custNo") == "customer number"
    assert xs._humanize("txn_amt") == "transaction amount"


def test_canonicalize_renames_tieout_rows_too():
    meta = {"result_preview": [{"net_sales": 10}], "tieout_rows": [{"net_sales": 10}, {"net_sales": 20}]}
    out = xs.canonicalize_sql_metadata(meta, {"net_sales": "revenue"})
    assert out["tieout_rows"] == [{"revenue": 10}, {"revenue": 20}]
    assert meta["tieout_rows"] == [{"net_sales": 10}, {"net_sales": 20}]
