"""Unit tests for api.ontology_service — persistent data ontology (DOCBOT-1517).

Uses an in-memory SQLite async engine so no live PostgreSQL is required
(mirrors tests/unit/test_trace_service.py's approach for the sibling
agent_traces table). All DB and embeddings calls are mocked/faked — no
network, no real Groq/HF calls.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, Text, insert, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api import ontology_service
from api.utils.value_fingerprint import fingerprint_values


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def wired_store():
    """Create an in-memory SQLite-backed data_ontology table and wire the module."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    metadata = MetaData()
    table = ontology_service.register_ontology_table(metadata)

    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    ontology_service.wire_ontology_store(
        table, MagicMock(), MagicMock(), session_factory
    )

    yield table, session_factory

    # Reset module globals so tests don't leak state into each other.
    ontology_service._ontology_table = None
    ontology_service._db_connections_table = None
    ontology_service._schema_cache_table = None
    ontology_service._async_session_factory = None
    await engine.dispose()


_SCHEMA = [
    {
        "name": "orders",
        "columns": [
            {"name": "id", "type": "INTEGER"},
            {"name": "customer_id", "type": "INTEGER"},
            {"name": "amount", "type": "NUMERIC"},
            {"name": "created_at", "type": "TIMESTAMP"},
            {"name": "status", "type": "VARCHAR"},
        ],
        "is_view": False,
    },
    {
        "name": "customers",
        "columns": [
            {"name": "id", "type": "INTEGER"},
            {"name": "name", "type": "VARCHAR"},
        ],
        "is_view": False,
    },
]


def _fake_embeddings_model(dim: int = 4):
    """Deterministic fake embeddings: each text's vector is a one-hot-ish
    vector seeded by its length, so cosine similarity ranking is stable and
    checkable without any real embedding model."""
    model = MagicMock()

    def _embed_documents(texts):
        return [[float((hash(t) % 97) + i) for i in range(dim)] for t in texts]

    def _embed_query(text):
        return [float((hash(text) % 97) + i) for i in range(dim)]

    model.embed_documents = MagicMock(side_effect=_embed_documents)
    model.embed_query = MagicMock(side_effect=_embed_query)
    return model


# ---------------------------------------------------------------------------
# Semantic tag / heuristic helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSemanticTagHeuristics:
    def test_id_column_tagged_entity(self):
        assert ontology_service._infer_semantic_tag("customer_id", "INTEGER") == "entity"
        assert ontology_service._infer_semantic_tag("id", "INTEGER") == "entity"

    def test_date_column_tagged_date(self):
        assert ontology_service._infer_semantic_tag("created_at", "TIMESTAMP") == "date"
        assert ontology_service._infer_semantic_tag("order_date", "DATE") == "date"

    def test_amount_column_tagged_amount(self):
        assert ontology_service._infer_semantic_tag("total_amount", "NUMERIC") == "amount"
        assert ontology_service._infer_semantic_tag("price", "FLOAT") == "amount"

    def test_category_column_tagged_category(self):
        assert ontology_service._infer_semantic_tag("status", "VARCHAR") == "category"

    def test_unrecognized_column_tagged_other(self):
        assert ontology_service._infer_semantic_tag("notes", "TEXT") == "other"


@pytest.mark.unit
class TestJoinHints:
    def test_matches_table_by_singular_and_plural_naming(self):
        column_metadata = [{"name": "customer_id", "type": "INTEGER", "semantic_tag": "entity"}]
        hints = ontology_service._infer_join_hints(
            "orders", column_metadata, ["orders", "customers"]
        )
        assert len(hints) == 1
        assert hints[0]["likely_references_table"] == "customers"
        assert hints[0]["basis"] == "naming_convention"

    def test_no_match_when_table_absent(self):
        column_metadata = [{"name": "widget_id", "type": "INTEGER", "semantic_tag": "entity"}]
        hints = ontology_service._infer_join_hints("orders", column_metadata, ["orders"])
        assert hints == []

    def test_plain_id_column_is_not_a_join_hint(self):
        column_metadata = [{"name": "id", "type": "INTEGER", "semantic_tag": "entity"}]
        hints = ontology_service._infer_join_hints("orders", column_metadata, ["orders", "id"])
        assert hints == []


# ---------------------------------------------------------------------------
# Table registration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegisterTable:
    def test_table_has_expected_columns(self):
        metadata = MetaData()
        table = ontology_service.register_ontology_table(metadata)
        col_names = {c.name for c in table.columns}
        assert col_names == {
            "id", "connection_id", "table_name", "column_metadata_json",
            "embedding", "data_card_json", "join_hints_json", "updated_at",
        }

    def test_table_name(self):
        metadata = MetaData()
        table = ontology_service.register_ontology_table(metadata)
        assert table.name == "data_ontology"

    def test_unique_constraint_on_connection_and_table(self):
        metadata = MetaData()
        table = ontology_service.register_ontology_table(metadata)
        constraint_cols = {
            tuple(c.name for c in uc.columns) for uc in table.constraints
            if hasattr(uc, "columns") and len(uc.columns) == 2
        }
        assert ("connection_id", "table_name") in constraint_cols


# ---------------------------------------------------------------------------
# build_ontology
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestBuildOntology:
    async def test_noop_when_unwired(self):
        """Never raises, never does anything when the module hasn't been wired."""
        ontology_service._ontology_table = None
        ontology_service._async_session_factory = None
        await ontology_service.build_ontology("conn-unwired")  # must not raise

    async def test_upserts_one_row_per_table(self, wired_store):
        table, session_factory = wired_store

        with (
            patch("api.db_service.get_schema", AsyncMock(return_value=_SCHEMA)),
            patch(
                "api.utils.embeddings_provider.get_embeddings",
                return_value=_fake_embeddings_model(),
            ),
        ):
            await ontology_service.build_ontology("conn-1")

        async with session_factory() as session:
            result = await session.execute(
                select(table).where(table.c.connection_id == "conn-1")
            )
            rows = result.fetchall()

        assert {r.table_name for r in rows} == {"orders", "customers"}
        for row in rows:
            assert json.loads(row.column_metadata_json)
            assert json.loads(row.embedding)
            assert row.data_card_json is not None
            assert row.join_hints_json is not None

    async def test_column_metadata_includes_semantic_tags(self, wired_store):
        table, session_factory = wired_store

        with (
            patch("api.db_service.get_schema", AsyncMock(return_value=_SCHEMA)),
            patch(
                "api.utils.embeddings_provider.get_embeddings",
                return_value=_fake_embeddings_model(),
            ),
        ):
            await ontology_service.build_ontology("conn-1")

        async with session_factory() as session:
            result = await session.execute(
                select(table).where(
                    table.c.connection_id == "conn-1", table.c.table_name == "orders"
                )
            )
            row = result.first()

        cols = {c["name"]: c["semantic_tag"] for c in json.loads(row.column_metadata_json)}
        assert cols["customer_id"] == "entity"
        assert cols["created_at"] == "date"
        assert cols["amount"] == "amount"

    async def test_join_hints_reference_customers_table(self, wired_store):
        table, session_factory = wired_store

        with (
            patch("api.db_service.get_schema", AsyncMock(return_value=_SCHEMA)),
            patch(
                "api.utils.embeddings_provider.get_embeddings",
                return_value=_fake_embeddings_model(),
            ),
        ):
            await ontology_service.build_ontology("conn-1")

        async with session_factory() as session:
            result = await session.execute(
                select(table).where(
                    table.c.connection_id == "conn-1", table.c.table_name == "orders"
                )
            )
            row = result.first()

        join_hints = json.loads(row.join_hints_json)
        assert any(h["likely_references_table"] == "customers" for h in join_hints)

    async def test_rerun_updates_existing_rows_instead_of_duplicating(self, wired_store):
        """Calling build_ontology twice for the same connection must upsert,
        not create duplicate rows."""
        table, session_factory = wired_store

        with (
            patch("api.db_service.get_schema", AsyncMock(return_value=_SCHEMA)),
            patch(
                "api.utils.embeddings_provider.get_embeddings",
                return_value=_fake_embeddings_model(),
            ),
        ):
            await ontology_service.build_ontology("conn-1")
            await ontology_service.build_ontology("conn-1")

        async with session_factory() as session:
            result = await session.execute(
                select(table).where(table.c.connection_id == "conn-1")
            )
            rows = result.fetchall()

        assert len(rows) == 2  # still exactly one row per table, not four

    async def test_empty_schema_is_a_noop(self, wired_store):
        table, session_factory = wired_store

        with patch("api.db_service.get_schema", AsyncMock(return_value=[])):
            await ontology_service.build_ontology("conn-empty")  # must not raise

        async with session_factory() as session:
            result = await session.execute(
                select(table).where(table.c.connection_id == "conn-empty")
            )
            assert result.fetchall() == []

    async def test_never_raises_on_embeddings_failure(self, wired_store):
        with (
            patch("api.db_service.get_schema", AsyncMock(return_value=_SCHEMA)),
            patch(
                "api.utils.embeddings_provider.get_embeddings",
                side_effect=RuntimeError("embeddings model unavailable"),
            ),
        ):
            await ontology_service.build_ontology("conn-boom")  # must not raise

    async def test_never_raises_on_get_schema_failure(self, wired_store):
        with patch(
            "api.db_service.get_schema", AsyncMock(side_effect=RuntimeError("DB down"))
        ):
            await ontology_service.build_ontology("conn-boom")  # must not raise


# ---------------------------------------------------------------------------
# ontology_lookup — the critical fallback-safety path
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestOntologyLookup:
    async def test_returns_none_when_unwired(self):
        """Critical fallback-safety test: when the module has never been
        wired (e.g. the DB migration/registration hasn't run, or startup
        wiring failed), ontology_lookup() must return None — the explicit
        'not built yet' signal callers use to fall back to their existing
        behavior unchanged."""
        ontology_service._ontology_table = None
        ontology_service._async_session_factory = None
        result = await ontology_service.ontology_lookup("conn-1", "How many orders?")
        assert result is None

    async def test_returns_none_when_wired_but_no_rows_for_connection(self, wired_store):
        """Critical fallback-safety test: the module is wired and the table
        exists, but this specific connection has no ontology rows yet (never
        built, or a different connection's rows exist) — still None."""
        table, session_factory = wired_store

        with (
            patch("api.db_service.get_schema", AsyncMock(return_value=_SCHEMA)),
            patch(
                "api.utils.embeddings_provider.get_embeddings",
                return_value=_fake_embeddings_model(),
            ),
        ):
            await ontology_service.build_ontology("conn-other")

        result = await ontology_service.ontology_lookup("conn-1", "How many orders?")
        assert result is None

    async def test_returns_top_k_result_when_ontology_exists(self, wired_store):
        table, session_factory = wired_store

        with (
            patch("api.db_service.get_schema", AsyncMock(return_value=_SCHEMA)),
            patch(
                "api.utils.embeddings_provider.get_embeddings",
                return_value=_fake_embeddings_model(),
            ),
        ):
            await ontology_service.build_ontology("conn-1")

            result = await ontology_service.ontology_lookup("conn-1", "How many orders?", top_k=1)

        assert result is not None
        assert len(result) == 1
        hit = result[0]
        assert hit["table_name"] in {"orders", "customers"}
        assert "column_metadata" in hit
        assert "data_card" in hit
        assert "join_hints" in hit
        assert "score" in hit

    async def test_top_k_caps_result_length(self, wired_store):
        table, session_factory = wired_store

        with (
            patch("api.db_service.get_schema", AsyncMock(return_value=_SCHEMA)),
            patch(
                "api.utils.embeddings_provider.get_embeddings",
                return_value=_fake_embeddings_model(),
            ),
        ):
            await ontology_service.build_ontology("conn-1")

            result = await ontology_service.ontology_lookup("conn-1", "How many orders?", top_k=1)
            result_all = await ontology_service.ontology_lookup("conn-1", "How many orders?", top_k=8)

        assert len(result) == 1
        assert len(result_all) == 2  # only 2 tables exist for conn-1

    async def test_never_raises_returns_none_on_db_error(self, wired_store):
        table, session_factory = wired_store

        async def _boom():
            raise RuntimeError("simulated DB failure")

        broken_factory = MagicMock(side_effect=RuntimeError("DB down"))
        ontology_service._async_session_factory = broken_factory

        result = await ontology_service.ontology_lookup("conn-1", "anything")
        assert result is None

    async def test_never_raises_returns_none_on_embeddings_error(self, wired_store):
        table, session_factory = wired_store

        with (
            patch("api.db_service.get_schema", AsyncMock(return_value=_SCHEMA)),
            patch(
                "api.utils.embeddings_provider.get_embeddings",
                return_value=_fake_embeddings_model(),
            ),
        ):
            await ontology_service.build_ontology("conn-1")

        with patch(
            "api.utils.embeddings_provider.get_embeddings",
            side_effect=RuntimeError("embeddings model unavailable"),
        ):
            result = await ontology_service.ontology_lookup("conn-1", "anything")

        assert result is None


# ---------------------------------------------------------------------------
# DOCBOT-2911: CSV/SQLite upload coverage, value fingerprints, org scope
# ---------------------------------------------------------------------------

@pytest.fixture
async def upload_store():
    """Ontology store plus minimal db_connections / schema_cache tables."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    metadata = MetaData()
    onto = ontology_service.register_ontology_table(metadata)
    conns = Table(
        "db_connections", metadata,
        Column("id", String, primary_key=True),
        Column("session_id", String, nullable=False),
        Column("dialect", String, nullable=False),
        Column("host", String, nullable=False),
        Column("port", Integer, nullable=False),
        Column("dbname", String, nullable=False),
        Column("credentials_blob", Text, nullable=False),
        Column("user_id", String),
        Column("org_id", String),
    )
    cache = Table(
        "schema_cache", metadata,
        Column("connection_id", String, primary_key=True),
        Column("schema_json", Text),
        Column("expires_at", String),
    )
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    ontology_service.wire_ontology_store(onto, conns, cache, factory)

    yield onto, conns, cache, factory

    ontology_service._ontology_table = None
    ontology_service._db_connections_table = None
    ontology_service._schema_cache_table = None
    ontology_service._async_session_factory = None
    await engine.dispose()


async def _drain_background_tasks() -> None:
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    await asyncio.gather(*pending, return_exceptions=True)


async def _all_rows(onto, factory):
    async with factory() as session:
        return (await session.execute(select(onto))).fetchall()


_CSV = (
    b"customer_id,email,status,amount\n"
    b"1,alice@example.com,paid,100.5\n"
    b"2.0,bob@example.com,pending,200\n"
    b"3,carol@example.com,paid,50\n"
)

_MULTI_CSV = (
    b"EXHIBIT 1 Revenue,,\n"
    b"region,revenue,units\n"
    b"East,100,1\n"
    b"West,200,2\n"
    b"EXHIBIT 2 Costs,,\n"
    b"region,cost,units\n"
    b"East,50,1\n"
)


@pytest.mark.unit
@pytest.mark.asyncio
class TestCsvUploadOntology:
    async def test_csv_upload_builds_ontology_with_hashes_and_no_raw_values(self, upload_store):
        from api.file_upload_service import upload_csv

        onto, conns, cache, factory = upload_store
        with patch(
            "api.utils.embeddings_provider.get_embeddings", return_value=_fake_embeddings_model()
        ):
            result = await upload_csv(_CSV, "clients.csv", "sess-1", conns, cache, factory)
            await _drain_background_tasks()

        rows = await _all_rows(onto, factory)
        assert [(r.connection_id, r.table_name) for r in rows] == [(result["connection_id"], "clients")]
        cols = {c["name"]: c for c in json.loads(rows[0].column_metadata_json)}
        assert cols["customer_id"]["semantic_tag"] == "entity"
        assert cols["customer_id"]["value_hashes"] == fingerprint_values(["1", "2", "3"])
        assert cols["status"]["value_hashes"] == fingerprint_values(["paid", "pending"])
        assert cols["email"]["null_pct"] == 0.0

        stored = " ".join(
            f"{r.column_metadata_json} {r.data_card_json} {r.join_hints_json}" for r in rows
        )
        for raw in ("alice@example.com", "bob", "pending", "100.5"):
            assert raw not in stored

    async def test_build_failure_does_not_fail_upload(self, upload_store):
        from api.file_upload_service import upload_csv

        onto, conns, cache, factory = upload_store
        with patch(
            "api.utils.embeddings_provider.get_embeddings",
            side_effect=RuntimeError("embeddings model unavailable"),
        ):
            result = await upload_csv(_CSV, "clients.csv", "sess-1", conns, cache, factory)
            await _drain_background_tasks()

        assert result["connection_id"]
        assert await _all_rows(onto, factory) == []

    async def test_multi_section_csv_gets_one_row_per_section(self, upload_store):
        from api.file_upload_service import upload_csv

        onto, conns, cache, factory = upload_store
        with patch(
            "api.utils.embeddings_provider.get_embeddings", return_value=_fake_embeddings_model()
        ):
            await upload_csv(_MULTI_CSV, "report.csv", "sess-1", conns, cache, factory)
            await _drain_background_tasks()

        rows = sorted(await _all_rows(onto, factory), key=lambda r: r.table_name)
        assert len(rows) == 2
        assert rows[0].table_name.startswith("report__0_")
        assert rows[1].table_name.startswith("report__1_")
        east_west = fingerprint_values(["east", "west"])
        cols0 = {c["name"]: c for c in json.loads(rows[0].column_metadata_json)}
        cols1 = {c["name"]: c for c in json.loads(rows[1].column_metadata_json)}
        assert cols0["region"]["value_hashes"] == east_west
        assert cols1["region"]["value_hashes"] == fingerprint_values(["east"])

    async def test_refresh_does_not_overwrite_csv_ontology(self, upload_store):
        from api.file_upload_service import upload_csv

        onto, conns, cache, factory = upload_store
        with patch(
            "api.utils.embeddings_provider.get_embeddings", return_value=_fake_embeddings_model()
        ):
            result = await upload_csv(_CSV, "clients.csv", "sess-1", conns, cache, factory)
            await _drain_background_tasks()
            before = (await _all_rows(onto, factory))[0].column_metadata_json
            with patch("api.db_service.get_schema", AsyncMock()) as get_schema:
                await ontology_service.build_ontology(result["connection_id"])
            get_schema.assert_not_called()

        assert (await _all_rows(onto, factory))[0].column_metadata_json == before


@pytest.mark.unit
@pytest.mark.asyncio
class TestSqlValueHashes:
    async def test_only_id_and_category_columns_get_hashes(self, upload_store, sqlite_url):
        _onto, conns, _cache, _factory = upload_store
        per_table = [{
            "table_name": "orders",
            "column_metadata": ontology_service._build_column_metadata({"columns": [
                {"name": "id", "type": "INTEGER"},
                {"name": "status", "type": "TEXT"},
                {"name": "amount", "type": "REAL"},
            ]}),
        }]
        conn_row = MagicMock(dialect="sqlite")
        with (
            patch("api.db_service._decrypt_conn_credentials_lazy", AsyncMock(return_value={})),
            patch("api.db_service._resolve_connection", return_value=(sqlite_url, None)),
        ):
            await ontology_service._attach_sql_value_hashes("conn-sql", conn_row, per_table)

        cols = {c["name"]: c for c in per_table[0]["column_metadata"]}
        assert cols["id"]["value_hashes"] == fingerprint_values([1, 2, 3, 4])
        assert sorted(cols["status"]["value_hashes"]) == sorted(fingerprint_values(["paid", "pending"]))
        assert cols["amount"]["value_hashes"] == []

    async def test_credential_failure_skips_silently(self, upload_store):
        per_table = [{
            "table_name": "orders",
            "column_metadata": ontology_service._build_column_metadata(
                {"columns": [{"name": "id", "type": "INTEGER"}]}
            ),
        }]
        with patch(
            "api.db_service._decrypt_conn_credentials_lazy", AsyncMock(side_effect=RuntimeError("bad key"))
        ):
            await ontology_service._attach_sql_value_hashes("c", MagicMock(dialect="sqlite"), per_table)
        assert per_table[0]["column_metadata"][0]["value_hashes"] == []


@pytest.mark.unit
@pytest.mark.asyncio
class TestListOrgOntology:
    async def _seed(self, conns, onto, factory):
        async with factory() as session:
            async with session.begin():
                for cid, user, org, dialect in (
                    ("c-a", "u1", "org-a", "postgresql"),
                    ("c-b", "u2", "org-b", "csv"),
                    ("c-anon", None, "org-a", "csv"),  # backfilled anonymous row
                ):
                    await session.execute(insert(conns).values(
                        id=cid, session_id="s", dialect=dialect, host="__local_file__" if dialect == "csv" else "db",
                        port=0, dbname=f"{cid}.csv" if dialect == "csv" else "analytics",
                        credentials_blob="x", user_id=user, org_id=org,
                    ))
                    await session.execute(insert(onto).values(
                        id=f"o-{cid}", connection_id=cid, table_name="t",
                        column_metadata_json=json.dumps([{
                            "name": "id", "type": "INTEGER", "semantic_tag": "entity",
                            "null_pct": None, "value_hashes": ["abcd"],
                        }]),
                        embedding=json.dumps([0.1, 0.2]),
                    ))

    async def test_returns_only_callers_org_user_owned_rows(self, upload_store):
        onto, conns, _cache, factory = upload_store
        await self._seed(conns, onto, factory)

        result = await ontology_service.list_org_ontology("org-a")

        assert result == [{
            "connection_id": "c-a",
            "dialect": "postgresql",
            "connection_name": "analytics",
            "table_name": "t",
            "columns": [{
                "name": "id", "type": "INTEGER", "semantic_tag": "entity",
                "null_pct": None, "value_hashes": ["abcd"],
            }],
            "embedding": [0.1, 0.2],
        }]
        assert [r["connection_id"] for r in await ontology_service.list_org_ontology("org-b")] == ["c-b"]

    async def test_none_or_unknown_org_returns_empty(self, upload_store):
        onto, conns, _cache, factory = upload_store
        await self._seed(conns, onto, factory)
        assert await ontology_service.list_org_ontology(None) == []
        assert await ontology_service.list_org_ontology("org-zzz") == []

    async def test_unwired_returns_empty(self):
        assert await ontology_service.list_org_ontology("org-a") == []
