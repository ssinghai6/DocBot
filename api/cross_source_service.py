"""Cross-source ontology matching — DOCBOT-2911 part B.

Finds columns in DIFFERENT connections of one org that likely mean the same
business thing, and maps columns onto a small fixed finance vocabulary.

Bounded and deterministic: zero LLM calls. One batched local-ONNX embedding
pass over column names, a capped candidate list, then value-containment
confirmation (idea from NVIDIA/auto-ontology, see api/utils/value_containment).

Input is ``api.ontology_service.list_org_ontology(org_id)`` (part A), which is
already org-scoped; this module only ever reads that one org's entries and
never sees connection strings or raw values (only value hashes).

shortcut: no per-org result cache — recomputed per call (one DB read + one
embedding batch). Add a cache keyed on org_id, invalidated on ontology
rebuild, if this shows up in latency traces.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Optional

import numpy as np

from api.utils.value_containment import containment

logger = logging.getLogger(__name__)

MAX_COLUMNS = 400          # hard cap on columns considered per org
MAX_CANDIDATES = 50        # top name-similar pairs that get value checks
CANDIDATE_SIM = 0.45       # min name similarity to be a candidate at all
NAME_ONLY_SIM = 0.75       # min name similarity to accept without value proof
VALUE_CONFIRM = 0.5        # containment needed for evidence="values"
VALUE_CONTRADICT = 0.1     # non-numeric cols with this little overlap are not the same field
TERM_SIM = 0.5             # min similarity to a finance term description
TERM_MARGIN = 0.05         # best term must beat the runner-up by this much
RELEVANCE_SIM = 0.3        # min question similarity for a line to enter the prompt
BLOCK_MAX_LINES = 15

FINANCE_TERMS: dict[str, str] = {
    "revenue": "revenue: total sales or income from customers",
    "cogs": "cost of goods sold: direct cost of products sold",
    "gross_profit": "gross profit: revenue minus cost of goods sold",
    "opex": "operating expenses: selling general and administrative costs",
    "net_income": "net income: profit after all expenses",
    "date": "date or period: when the transaction happened",
    "customer_id": "customer id: identifier of a customer",
    "order_id": "order id: identifier of an order",
    "amount": "amount: monetary value of a transaction",
}
# Terms that are numeric measures: only these are used to relabel SQL rows for
# discrepancy detection (ids/dates must keep their names — the detector reads
# period/dimension columns by name).
MEASURE_TERMS = frozenset({"revenue", "cogs", "gross_profit", "opex", "net_income"})
_NUMERIC_TERMS = MEASURE_TERMS | {"amount"}

# MiniLM embeds abbreviations poorly ("cust no" vs "customer id" ~0.1).
_ABBREV = {
    "cust": "customer", "amt": "amount", "txn": "transaction", "qty": "quantity",
    "rev": "revenue", "opex": "operating expenses", "cogs": "cost of goods sold",
    "no": "number", "num": "number", "nbr": "number", "dt": "date",
    "acct": "account", "inv": "invoice", "ord": "order", "prod": "product",
}


def _humanize(name: str) -> str:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
    tokens = re.split(r"[^A-Za-z0-9]+", s.lower())
    return " ".join(_ABBREV.get(t, t) for t in tokens if t)


def _family(col_type: str) -> str:
    t = (col_type or "").lower()
    if not t:
        return ""
    if "date" in t or "time" in t:
        return "date"
    if any(h in t for h in ("int", "float", "double", "decimal", "numeric", "real", "money")):
        return "numeric"
    return "text"


def _clean(s: Any) -> str:
    """Column/table names are user-controlled (CSV headers): keep them one-line and short."""
    return re.sub(r"[\r\n`]+", " ", str(s or ""))[:64]


async def _load(org_id: Optional[str]) -> list[dict]:
    if not org_id:
        return []
    from api import ontology_service

    tables = await ontology_service.list_org_ontology(org_id)
    cols: list[dict] = []
    for t in tables or []:
        for c in t.get("columns") or []:
            if not c.get("name"):
                continue
            cols.append({
                "connection_id": t.get("connection_id"),
                "connection_name": t.get("connection_name") or t.get("dialect") or "source",
                "table": t.get("table_name"),
                "column": c["name"],
                "family": _family(c.get("type", "")),
                "tag": c.get("semantic_tag") or "",
                "hashes": c.get("value_hashes") or [],
                "text": _humanize(c["name"]),
            })
            if len(cols) >= MAX_COLUMNS:  # shortcut: truncates huge orgs, raise cap if needed
                return cols
    return cols


def _embed(texts: list[str]) -> np.ndarray:
    from api.utils.embeddings_provider import get_embeddings

    m = np.asarray(get_embeddings().embed_documents(texts), dtype=float)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def _ref(c: dict) -> dict:
    return {"connection_id": c["connection_id"], "table": c["table"], "column": c["column"]}


def _match(cols: list[dict], vecs: np.ndarray) -> list[tuple[dict, int, int]]:
    sims = vecs @ vecs.T
    cands: list[tuple[float, int, int]] = []
    for i in range(len(cols)):
        a = cols[i]
        for j in range(i + 1, len(cols)):
            b = cols[j]
            if a["connection_id"] == b["connection_id"]:
                continue
            if a["family"] and b["family"] and a["family"] != b["family"]:
                continue
            if a["tag"] and b["tag"] and a["tag"] != b["tag"] and "other" not in (a["tag"], b["tag"]):
                continue
            s = 1.0 if a["text"] == b["text"] else float(sims[i, j])
            if s >= CANDIDATE_SIM:
                cands.append((s, i, j))
    cands.sort(key=lambda x: x[0], reverse=True)

    out: list[tuple[dict, int, int]] = []
    for s, i, j in cands[:MAX_CANDIDATES]:
        a, b = cols[i], cols[j]
        c = containment(a["hashes"], b["hashes"])
        if c is not None and c >= VALUE_CONFIRM:
            evidence, score = "values", 0.6 * c + 0.4 * s
        elif c is not None and c < VALUE_CONTRADICT and "numeric" not in (a["family"], b["family"]):
            continue  # ids/categories with disjoint values are different fields
        elif s >= NAME_ONLY_SIM:
            evidence, score = "name+type", s
        else:
            continue
        out.append(({
            "left": _ref(a), "right": _ref(b), "score": round(score, 3),
            "evidence": evidence, "containment": None if c is None else round(c, 3),
        }, i, j))
    out.sort(key=lambda m: (m[0]["evidence"] == "values", m[0]["score"]), reverse=True)
    return out


_term_vecs_cache: Optional[np.ndarray] = None


def _term_vecs() -> np.ndarray:
    global _term_vecs_cache
    if _term_vecs_cache is None:
        _term_vecs_cache = _embed(list(FINANCE_TERMS.values()))
    return _term_vecs_cache


def _terms(cols: list[dict], vecs: np.ndarray) -> dict[str, list[int]]:
    names = list(FINANCE_TERMS)
    sims = vecs @ _term_vecs().T
    out: dict[str, list[int]] = {}
    for i, row in enumerate(sims):
        order = np.argsort(row)[::-1]
        best = int(order[0])
        runner_up = float(row[order[1]]) if len(order) > 1 else 0.0
        if row[best] < TERM_SIM or row[best] - runner_up < TERM_MARGIN:
            continue
        term, fam = names[best], cols[i]["family"]
        if term in _NUMERIC_TERMS and fam not in ("", "numeric"):
            continue
        if term == "date" and fam == "numeric":
            continue
        out.setdefault(term, []).append(i)
    return out


def _compute(cols: list[dict]) -> tuple[np.ndarray, list[tuple[dict, int, int]], dict[str, list[int]]]:
    vecs = _embed([c["text"] for c in cols])
    return vecs, _match(cols, vecs), _terms(cols, vecs)


async def match_across_sources(org_id: Optional[str]) -> list[dict]:
    """Likely-same column pairs across different connections of one org.

    Value-confirmed matches (evidence="values") rank above name-only ones.
    """
    cols = await _load(org_id)
    if len(cols) < 2:
        return []
    vecs = await asyncio.to_thread(_embed, [c["text"] for c in cols])
    return [m for m, _, _ in _match(cols, vecs)]


async def map_finance_terms(org_id: Optional[str]) -> dict[str, list[dict]]:
    """{canonical finance term: [column refs across sources]}; deterministic, no LLM."""
    cols = await _load(org_id)
    if not cols:
        return {}
    vecs = await asyncio.to_thread(_embed, [c["text"] for c in cols])
    return {t: [_ref(cols[i]) for i in idx] for t, idx in _terms(cols, vecs).items()}


def _label(c: dict) -> str:
    return f"{_clean(c['connection_name'])}.{_clean(c['table'])}.{_clean(c['column'])}"


async def cross_source_context(
    org_id: Optional[str], question: str, connection_id: Optional[str] = None
) -> tuple[str, dict[str, str]]:
    """Prompt block of question-relevant cross-source mappings + measure-term map.

    Returns ``(block, {column_name: measure_term})``; the map covers only
    ``connection_id``'s columns. The block is "" unless the org has >= 2
    connections. Never raises: any failure (part A missing, DB down, model
    load) logs and returns ``("", {})``.
    """
    try:
        cols = await _load(org_id)
        if not cols:
            return "", {}
        vecs, matches, terms = await asyncio.to_thread(_compute, cols)

        col_terms: dict[str, str] = {}
        if connection_id:
            for term, idx in terms.items():
                if term in MEASURE_TERMS:
                    for i in idx:
                        if cols[i]["connection_id"] == connection_id:
                            col_terms.setdefault(cols[i]["column"], term)

        if len({c["connection_id"] for c in cols}) < 2:
            return "", col_terms

        q = (await asyncio.to_thread(_embed, [question]))[0]
        qs = vecs @ q
        lines: list[tuple[float, str]] = []
        for m, i, j in matches:
            rel = float(max(qs[i], qs[j]))
            if rel >= RELEVANCE_SIM:
                how = (f"values overlap {m['containment']:.0%}" if m["evidence"] == "values"
                       else "name/type match")
                lines.append((rel, f"- {_label(cols[i])} = {_label(cols[j])} ({how})"))
        term_q = _term_vecs() @ q
        names = list(FINANCE_TERMS)
        for t, idx in terms.items():
            srcs = {cols[i]["connection_id"] for i in idx}
            if len(srcs) >= 2 and term_q[names.index(t)] >= RELEVANCE_SIM:
                lines.append((float(term_q[names.index(t)]),
                              f"- {t}: " + ", ".join(_label(cols[i]) for i in idx[:4])))
        if not lines:
            return "", col_terms
        lines.sort(key=lambda x: x[0], reverse=True)
        header = ("\n\nCross-source column mappings (same business field across the "
                  "org's connected sources; use to compare like-for-like):")
        body = "\n".join(text for _, text in lines[: BLOCK_MAX_LINES - 1])
        return f"{header}\n{body}\n", col_terms
    except Exception as exc:  # non-fatal by contract: never break a chat turn
        logger.warning("cross_source_context skipped (%s: %s)", type(exc).__name__, exc)
        return "", {}


def canonicalize_sql_metadata(sql_metadata: dict, col_terms: dict[str, str]) -> dict:
    """Copy of sql_metadata whose result rows use canonical measure names.

    Lets the discrepancy detector match a doc's "Revenue" to a DB column named
    e.g. ``net_sales``. Never renames onto an existing key.
    """
    rows = sql_metadata.get("result_preview")
    if not col_terms or not rows:
        return sql_metadata
    renamed = []
    for row in rows:
        if not isinstance(row, dict):
            renamed.append(row)
            continue
        new: dict = {}
        for k, v in row.items():
            t = col_terms.get(k)
            new[t if t and t not in row and t not in new else k] = v
        renamed.append(new)
    return {**sql_metadata, "result_preview": renamed}
