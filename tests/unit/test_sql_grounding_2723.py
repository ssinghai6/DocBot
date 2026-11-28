"""DOCBOT-2723: on the real San Diego ledger the SQL filtered `account LIKE
'%Energy & Utilities%'` (NULL) instead of account_class = 'Energy and
Utilities', the NULL went out as "not in the books", and wrong queries were
stored as few-shot examples for later questions."""

import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api import db_service
from tests.unit.test_db_service_helpers import (
    _FakeConnRow, _fake_db_connections_table, _make_conn_lookup_session_factory,
)

# (account_type, account_class, account) — rows sorted so the late classes
# sit past a small scan, like the real file (sorted by department).
LINES = [
    ("Personnel", "Personnel Cost", "Salaried Wages"),
    ("Personnel", "Fringe Benefits", "Retirement ADC"),
    ("Non-Personnel", "Supplies", "Office Supplies"),
    ("Non-Personnel", "Energy and Utilities", "Electric Services"),
    ("Non-Personnel", "Energy and Utilities", "Water Services"),
    ("Non-Personnel", "Transfers Out", "Transfers to Other Funds"),
    ("Transient Occupancy Taxes", "Transient Occupancy Taxes", "Transient Occupancy Tax"),
    ("Property Tax Revenue", "Property Taxes", "Property Tax - Secured"),
]


def _gl_db(tmp_path):
    path = tmp_path / "sd.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE gl_actuals (fiscal_year INTEGER, dept_name TEXT, account TEXT, "
        "account_type TEXT, account_class TEXT, amount REAL)"
    )
    rows = [
        (2023, f"Dept {d}", f"{acct} {d}", t, c, 100.0 + d)
        for t, c, acct in LINES for d in range(60)
    ]
    conn.executemany("INSERT INTO gl_actuals VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return path


async def _schema(tmp_path):
    return await db_service._introspect_schema_from_url(f"sqlite:///{_gl_db(tmp_path)}", "sqlite")


async def _prompt(question, schema, few_shot=()):
    with patch("api.utils.llm_provider.achat_completion", new=AsyncMock(return_value="SELECT 1")) as llm, \
            patch("api.utils.llm_cache.get_cached_response", new=AsyncMock(return_value=None)), \
            patch("api.utils.llm_cache.set_cached_response", new=AsyncMock()):
        await db_service._generate_sql(question, schema, list(few_shot))
    return llm.await_args.args[0][0]["content"]


# ── 1. category grounding ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hint_names_the_category_column_not_the_account_name(tmp_path):
    schema = await _schema(tmp_path)
    cols = {c["name"]: c for c in schema[0]["columns"]}
    assert "sample_values" not in cols["account"]  # 480 distinct: no list
    q = "Does Energy & Utilities expenditure for FY2023 in the report agree with the books?"
    assert db_service._category_hints(q, schema) == [("gl_actuals", "account_class", "Energy and Utilities")]
    prompt = await _prompt(q, schema)
    assert "- gl_actuals.account_class = 'Energy and Utilities'" in prompt


@pytest.mark.asyncio
async def test_hint_keeps_longest_value_and_both_columns_holding_it(tmp_path):
    schema = await _schema(tmp_path)
    hints = db_service._category_hints("transient occupancy taxes vs the report", schema)
    assert sorted(c for _, c, _ in hints) == ["account_class", "account_type"]
    assert {v for _, _, v in hints} == {"Transient Occupancy Taxes"}
    assert db_service._category_hints("total general fund revenue", schema) == []


@pytest.mark.asyncio
async def test_scan_reaches_categories_past_5000_rows(tmp_path):
    """Old 5000-row scan saw only the first classes and showed 5 of them."""
    path = tmp_path / "big.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE gl (account_class TEXT, amount REAL)")
    conn.executemany("INSERT INTO gl VALUES (?, 1)", [("Personnel Cost",)] * 6000 + [("Energy and Utilities",)] * 10)
    conn.commit()
    conn.close()
    schema = await db_service._introspect_schema_from_url(f"sqlite:///{path}", "sqlite")
    col = schema[0]["columns"][0]
    assert col["sample_values"] == ["Energy and Utilities", "Personnel Cost"]
    assert col["values_complete"] is True


@pytest.mark.asyncio
async def test_prompt_has_grounding_and_no_unrequested_filter_rules():
    prompt = await _prompt("q", [], [{"nl_question": "old", "sql_query": "SELECT 1 WHERE amount > 0"}])
    assert "look for it first among the listed column values" in prompt
    assert "Use LIKE on a column without listed values" in prompt
    assert "no sign filters (amount > 0)" in prompt
    assert "never sum revenue and expense rows together" in prompt
    assert "do not copy their filters" in prompt
    assert db_service.PROMPT_VERSION_SQL_GEN == "v9"


# ── 2. empty / NULL result ───────────────────────────────────────────────────

@pytest.mark.parametrize("sql,rows,empty", [
    ("SELECT SUM(amount) FROM t", [], True),
    ("SELECT SUM(amount) AS total FROM t WHERE account LIKE '%x%'", [(None,)], True),
    ("SELECT 'E&U' AS line, SUM(amount) FROM t", [("E&U", None)], True),
    ("SELECT COALESCE(SUM(amount), 0) FROM t", [(0,)], True),
    ("SELECT 'a', SUM(x) FROM t UNION ALL SELECT 'b', SUM(y) FROM t", [("a", None), ("b", None)], True),
    ("SELECT 'a', SUM(x) FROM t UNION ALL SELECT 'b', SUM(y) FROM t", [("a", 5), ("b", None)], False),
    ("SELECT SUM(amount) FROM t", [(0,)], False),  # a real zero total
    ("SELECT name, notes FROM t", [("x", None)], False),  # no aggregate
])
def test_is_empty_result(sql, rows, empty):
    assert db_service._is_empty_result(sql, "sqlite", rows) is empty


async def _run_pipeline(gen_replies, exec_results):
    schema = [{"name": "gl_actuals", "columns": [
        {"name": "account", "type": "TEXT"},
        {"name": "account_class", "type": "TEXT", "sample_values": ["Energy and Utilities", "Supplies"],
         "values_complete": True},
        {"name": "amount", "type": "REAL"},
    ]}]
    gen = AsyncMock(side_effect=gen_replies)
    store = AsyncMock()
    answer_kwargs = {}

    async def _answer(question, sql, result_dicts, persona_def, **kw):
        answer_kwargs.update(kw, sql=sql)
        yield "ok"

    with (
        patch.object(db_service, "get_schema", AsyncMock(return_value=schema)),
        patch.object(db_service, "_select_relevant_tables", AsyncMock(return_value=["gl_actuals"])),
        patch.object(db_service, "_retrieve_few_shot", AsyncMock(return_value=[])),
        patch.object(db_service, "_generate_sql", gen),
        patch.object(db_service, "_execute_query", AsyncMock(side_effect=exec_results)),
        patch.object(db_service, "_store_query_history", store),
        patch.object(db_service, "_stream_answer", side_effect=_answer),
        patch.object(db_service, "_get_embeddings_model", return_value=MagicMock()),
        patch.object(db_service, "_get_embedding", AsyncMock(return_value=[0.1])),
        patch.object(db_service, "decrypt_credentials", return_value={}),
        patch.object(db_service, "_resolve_connection", return_value=("sqlite:///:memory:", None)),
    ):
        events = [c async for c in db_service.run_sql_pipeline(
            connection_id="conn-1",
            question="Does Energy & Utilities expenditure agree with the books?",
            persona="Finance Expert", db_connections_table=_fake_db_connections_table(),
            schema_cache_table=MagicMock(), query_history_table=MagicMock(),
            query_embeddings_table=MagicMock(), async_session_factory=_make_conn_lookup_session_factory(_FakeConnRow()),
            expert_personas={}, table_embeddings_table=None,
        )]
    meta = next(json.loads(e[6:]) for e in events if '"metadata"' in e)
    return gen, store, answer_kwargs, meta


BAD = "SELECT SUM(amount) FROM gl_actuals WHERE account LIKE '%Energy & Utilities%'"
GOOD = "SELECT SUM(amount) FROM gl_actuals WHERE account_class = 'Energy and Utilities'"


@pytest.mark.asyncio
async def test_null_total_retries_once_with_feedback_and_uses_fixed_sql():
    gen, store, answer, meta = await _run_pipeline(
        [BAD, GOOD], [([(None,)], ["total"]), ([(64016064.0,)], ["total"])]
    )
    assert gen.await_count == 2
    feedback = gen.await_args_list[1].kwargs["feedback"]
    assert "LIKE '%Energy & Utilities%'" in feedback and "found nothing" in feedback
    assert "gl_actuals.account_class = 'Energy and Utilities'" in feedback
    assert "account_class = 'Energy and Utilities'" in meta["sql_query"]
    assert meta["empty_result"] is False and answer["empty"] is False
    # retried run never becomes a few-shot example
    assert not store.await_args.args[4].endswith(db_service._VERIFIED_SUFFIX)


@pytest.mark.asyncio
async def test_still_empty_after_retry_answers_could_not_determine():
    gen, store, answer, meta = await _run_pipeline([BAD, BAD], [([(None,)], ["t"]), ([(None,)], ["t"])])
    assert gen.await_count == 2  # exactly one retry
    assert meta["empty_result"] is True and answer["empty"] is True
    assert not store.await_args.args[4].endswith(db_service._VERIFIED_SUFFIX)


@pytest.mark.asyncio
async def test_first_try_non_empty_is_stored_as_verified():
    gen, store, answer, meta = await _run_pipeline([GOOD], [([(64016064.0,)], ["t"])])
    assert gen.await_count == 1
    assert store.await_args.args[4] == "1 rows" + db_service._VERIFIED_SUFFIX
    assert db_service._parse_row_count(store.await_args.args[4]) == 1


@pytest.mark.asyncio
async def test_empty_answer_prompt_says_could_not_be_determined():
    seen = {}

    def _stream(messages, **kw):
        seen["prompt"] = messages[0]["content"]
        yield "x"

    with patch("api.utils.llm_provider.chat_completion_stream", side_effect=_stream):
        _ = [t async for t in db_service._stream_answer("q", "SELECT 1", [{"t": None}], "p", empty=True)]
    assert "could not be determined" in seen["prompt"]
    assert "do NOT say the item is absent from the books" in seen["prompt"]
    with patch("api.utils.llm_provider.chat_completion_stream", side_effect=_stream):
        _ = [t async for t in db_service._stream_answer("q", "SELECT 1", [{"t": 1}], "p")]
    assert "could not be determined" not in seen["prompt"]


# ── 3. few-shot hygiene ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_few_shot_retrieval_only_reads_verified_rows():
    from sqlalchemy import Column, MetaData, String, Table, Text, insert
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from api.utils.embeddings import embedding_to_json

    md = MetaData()
    qh = Table("query_history", md, Column("id", String, primary_key=True), Column("connection_id", String),
               Column("nl_question", Text), Column("sql_query", Text), Column("result_summary", Text))
    qe = Table("query_embeddings", md, Column("query_id", String, primary_key=True), Column("embedding_json", Text))
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(md.create_all)
        for qid, summary in (("bad", "1 rows"), ("good", "1 rows" + db_service._VERIFIED_SUFFIX)):
            await conn.execute(insert(qh).values(id=qid, connection_id="c", nl_question=qid,
                                                 sql_query=f"SELECT '{qid}'", result_summary=summary))
            await conn.execute(insert(qe).values(query_id=qid, embedding_json=embedding_to_json([1.0, 0.0])))
    out = await db_service._retrieve_few_shot("c", [1.0, 0.0], qh, qe, async_sessionmaker(engine))
    await engine.dispose()
    assert [ex["nl_question"] for ex in out] == ["good"]


def test_hybrid_synthesis_gets_empty_result_rule():
    import inspect
    from api import hybrid_service

    src = inspect.getsource(hybrid_service)
    assert 'sql_metadata.get("empty_result")' in src and "EMPTY_RESULT_RULE" in src
    assert hybrid_service.PROMPT_VERSION_HYBRID_SYNTHESIS == "v3"
