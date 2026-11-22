"""Unit tests for api.utils.llm_cache — exact-match LLM response cache
(DOCBOT-1506).

Uses an in-memory SQLite async engine so no live PostgreSQL is required
(mirrors tests/unit/test_llm_trace_service.py's approach for the sibling
llm_calls table).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import MetaData, select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from api.utils import llm_cache
from api.utils.billing_context import Principal, bind_principal


@pytest.fixture
async def wired_cache():
    """Create an in-memory SQLite-backed llm_response_cache table and wire
    the module."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    metadata = MetaData()
    table = llm_cache.register_llm_cache_table(metadata)

    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    llm_cache.wire_llm_cache(table, session_factory)

    yield table, session_factory

    # Reset module globals so tests don't leak state into each other.
    llm_cache._llm_cache_table = None
    llm_cache._async_session_factory = None
    await engine.dispose()


@pytest.fixture(autouse=True)
def clean_metrics():
    llm_cache.reset_metrics_for_tests()
    yield
    llm_cache.reset_metrics_for_tests()


# ---------------------------------------------------------------------------
# Table registration
# ---------------------------------------------------------------------------


class TestRegisterTable:
    def test_table_has_expected_columns(self):
        metadata = MetaData()
        table = llm_cache.register_llm_cache_table(metadata)
        col_names = {c.name for c in table.columns}
        assert col_names == {
            "id", "prompt_hash", "model", "response", "created_at", "expires_at",
        }

    def test_table_name(self):
        metadata = MetaData()
        table = llm_cache.register_llm_cache_table(metadata)
        assert table.name == "llm_response_cache"

    def test_unique_constraint_on_prompt_hash_and_model(self):
        metadata = MetaData()
        table = llm_cache.register_llm_cache_table(metadata)
        constraint_cols = {
            tuple(c.name for c in uc.columns) for uc in table.constraints
            if hasattr(uc, "columns") and len(uc.columns) == 2
        }
        assert ("prompt_hash", "model") in constraint_cols


# ---------------------------------------------------------------------------
# hash_prompt — stable, deterministic
# ---------------------------------------------------------------------------


class TestHashPrompt:
    def test_same_text_same_hash(self):
        assert llm_cache.hash_prompt("hello world") == llm_cache.hash_prompt("hello world")

    def test_different_text_different_hash(self):
        assert llm_cache.hash_prompt("hello world") != llm_cache.hash_prompt("hello world!")

    def test_returns_hex_digest(self):
        digest = llm_cache.hash_prompt("anything")
        assert len(digest) == 64
        int(digest, 16)  # does not raise


# ---------------------------------------------------------------------------
# get_cached_response / set_cached_response — unwired behavior
# ---------------------------------------------------------------------------


class TestUnwired:
    @pytest.mark.asyncio
    async def test_get_returns_none_when_unwired(self):
        llm_cache._llm_cache_table = None
        llm_cache._async_session_factory = None
        result = await llm_cache.get_cached_response("abc", "model-x")
        assert result is None
        assert llm_cache.get_cache_metrics()["misses"] == 1

    @pytest.mark.asyncio
    async def test_set_is_a_noop_when_unwired(self):
        llm_cache._llm_cache_table = None
        llm_cache._async_session_factory = None
        # Must not raise.
        await llm_cache.set_cached_response("abc", "model-x", "some response")
        assert llm_cache.get_cache_metrics()["sets"] == 0


# ---------------------------------------------------------------------------
# Round trip: miss -> set -> hit
# ---------------------------------------------------------------------------


class TestRoundTrip:
    @pytest.mark.asyncio
    async def test_miss_then_hit(self, wired_cache):
        prompt_hash = llm_cache.hash_prompt("SELECT ...")
        model = "openai/gpt-oss-20b"

        miss = await llm_cache.get_cached_response(prompt_hash, model)
        assert miss is None

        await llm_cache.set_cached_response(prompt_hash, model, "SELECT * FROM orders LIMIT 10")

        hit = await llm_cache.get_cached_response(prompt_hash, model)
        assert hit == "SELECT * FROM orders LIMIT 10"

        metrics = llm_cache.get_cache_metrics()
        assert metrics["misses"] == 1
        assert metrics["hits"] == 1
        assert metrics["sets"] == 1
        assert metrics["hit_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_different_model_is_a_separate_cache_key(self, wired_cache):
        prompt_hash = llm_cache.hash_prompt("same prompt text")
        await llm_cache.set_cached_response(prompt_hash, "model-a", "response-a")

        result = await llm_cache.get_cached_response(prompt_hash, "model-b")
        assert result is None

    @pytest.mark.asyncio
    async def test_different_prompt_hash_is_a_separate_cache_key(self, wired_cache):
        await llm_cache.set_cached_response(llm_cache.hash_prompt("prompt one"), "model-a", "resp-1")
        result = await llm_cache.get_cached_response(llm_cache.hash_prompt("prompt two"), "model-a")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_overwrites_existing_entry_for_same_key(self, wired_cache):
        prompt_hash = llm_cache.hash_prompt("same prompt")
        model = "model-a"
        await llm_cache.set_cached_response(prompt_hash, model, "first response")
        await llm_cache.set_cached_response(prompt_hash, model, "second response")

        result = await llm_cache.get_cached_response(prompt_hash, model)
        assert result == "second response"

        # No duplicate rows left behind by the delete-then-insert upsert.
        table, session_factory = wired_cache
        async with session_factory() as session:
            rows = (await session.execute(select(table))).fetchall()
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_empty_response_is_never_cached(self, wired_cache):
        prompt_hash = llm_cache.hash_prompt("some prompt")
        await llm_cache.set_cached_response(prompt_hash, "model-a", "")

        table, session_factory = wired_cache
        async with session_factory() as session:
            rows = (await session.execute(select(table))).fetchall()
        assert rows == []
        assert llm_cache.get_cache_metrics()["sets"] == 0


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------


class TestTTLExpiry:
    @pytest.mark.asyncio
    async def test_expired_entry_is_not_returned(self, wired_cache):
        table, session_factory = wired_cache
        prompt_hash = llm_cache.hash_prompt("expiring prompt")
        model = "model-a"

        # Insert directly with an already-past expires_at — avoids depending
        # on a negative-ttl code path in set_cached_response.
        from sqlalchemy import insert as sa_insert
        import uuid

        past = datetime.now(timezone.utc) - timedelta(seconds=10)
        async with session_factory() as session:
            async with session.begin():
                await session.execute(
                    sa_insert(table).values(
                        id=str(uuid.uuid4()),
                        prompt_hash=llm_cache._scoped_key(prompt_hash),
                        model=model,
                        response="stale response",
                        expires_at=past,
                    )
                )

        result = await llm_cache.get_cached_response(prompt_hash, model)
        assert result is None
        assert llm_cache.get_cache_metrics()["misses"] == 1

    @pytest.mark.asyncio
    async def test_unexpired_entry_is_returned(self, wired_cache):
        prompt_hash = llm_cache.hash_prompt("fresh prompt")
        model = "model-a"
        await llm_cache.set_cached_response(prompt_hash, model, "fresh response", ttl_seconds=3600)

        result = await llm_cache.get_cached_response(prompt_hash, model)
        assert result == "fresh response"


# ---------------------------------------------------------------------------
# Failure isolation — a cache error must never raise
# ---------------------------------------------------------------------------


class TestFailureIsolation:
    @pytest.mark.asyncio
    async def test_get_degrades_to_miss_on_db_error(self, wired_cache, monkeypatch):
        table, session_factory = wired_cache

        def _broken_factory():
            raise RuntimeError("simulated DB outage")

        monkeypatch.setattr(llm_cache, "_async_session_factory", _broken_factory)

        result = await llm_cache.get_cached_response("abc", "model-x")
        assert result is None
        assert llm_cache.get_cache_metrics()["errors"] == 1

    @pytest.mark.asyncio
    async def test_set_does_not_raise_on_db_error(self, wired_cache, monkeypatch):
        def _broken_factory():
            raise RuntimeError("simulated DB outage")

        monkeypatch.setattr(llm_cache, "_async_session_factory", _broken_factory)

        # Must not raise.
        await llm_cache.set_cached_response("abc", "model-x", "some response")
        assert llm_cache.get_cache_metrics()["errors"] == 1


# ---------------------------------------------------------------------------
# get_cache_metrics
# ---------------------------------------------------------------------------


class TestGetCacheMetrics:
    def test_hit_rate_is_none_with_no_calls(self):
        metrics = llm_cache.get_cache_metrics()
        assert metrics == {"hits": 0, "misses": 0, "sets": 0, "errors": 0, "hit_rate": None}

    @pytest.mark.asyncio
    async def test_hit_rate_reflects_hits_and_misses(self, wired_cache):
        prompt_hash = llm_cache.hash_prompt("p")
        await llm_cache.get_cached_response(prompt_hash, "m")  # miss
        await llm_cache.set_cached_response(prompt_hash, "m", "r")
        await llm_cache.get_cached_response(prompt_hash, "m")  # hit
        await llm_cache.get_cached_response(prompt_hash, "m")  # hit

        metrics = llm_cache.get_cache_metrics()
        assert metrics["misses"] == 1
        assert metrics["hits"] == 2
        assert metrics["hit_rate"] == round(2 / 3, 4)


# ---------------------------------------------------------------------------
# Tenant scoping — DOCBOT-1604 (SEC-25)
# ---------------------------------------------------------------------------

_ORG_A = Principal(user_id="u-a", org_id="org-a", auth_method="session")
_ORG_B = Principal(user_id="u-b", org_id="org-b", auth_method="session")


class TestTenantScoping:
    @pytest.mark.asyncio
    async def test_other_org_misses(self, wired_cache):
        h = llm_cache.hash_prompt("SELECT secret")
        with bind_principal(_ORG_A):
            await llm_cache.set_cached_response(h, "m", "org-a sql")
        with bind_principal(_ORG_B):
            assert await llm_cache.get_cached_response(h, "m") is None

    @pytest.mark.asyncio
    async def test_same_org_hits_across_users(self, wired_cache):
        h = llm_cache.hash_prompt("SELECT shared")
        with bind_principal(_ORG_A):
            await llm_cache.set_cached_response(h, "m", "org-a sql")
        teammate = Principal(user_id="u-a2", org_id="org-a", auth_method="session")
        with bind_principal(teammate):
            assert await llm_cache.get_cached_response(h, "m") == "org-a sql"

    @pytest.mark.asyncio
    async def test_anonymous_separated_from_orgs(self, wired_cache):
        h = llm_cache.hash_prompt("intent q")
        await llm_cache.set_cached_response(h, "m", "anon resp")  # no principal bound
        with bind_principal(_ORG_A):
            assert await llm_cache.get_cached_response(h, "m") is None
            await llm_cache.set_cached_response(h, "m", "org resp")
        assert await llm_cache.get_cached_response(h, "m") == "anon resp"
        demo = Principal(user_id=None, org_id=None, auth_method="demo", is_demo=True)
        with bind_principal(demo):
            assert await llm_cache.get_cached_response(h, "m") == "anon resp"

    @pytest.mark.asyncio
    async def test_orgless_user_scoped_to_user(self, wired_cache):
        h = llm_cache.hash_prompt("q")
        with bind_principal(Principal(user_id="solo", org_id=None)):
            await llm_cache.set_cached_response(h, "m", "solo resp")
        with bind_principal(Principal(user_id="other", org_id=None)):
            assert await llm_cache.get_cached_response(h, "m") is None
        assert await llm_cache.get_cached_response(h, "m") is None  # not anon either

    @pytest.mark.asyncio
    async def test_legacy_unscoped_row_never_served(self, wired_cache):
        from sqlalchemy import insert as sa_insert
        import uuid

        table, session_factory = wired_cache
        h = llm_cache.hash_prompt("legacy prompt")
        async with session_factory() as session:
            async with session.begin():
                await session.execute(
                    sa_insert(table).values(
                        id=str(uuid.uuid4()),
                        prompt_hash=h,  # pre-1604 key: bare prompt hash
                        model="m",
                        response="leaked",
                        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                    )
                )
        assert await llm_cache.get_cached_response(h, "m") is None
        with bind_principal(_ORG_A):
            assert await llm_cache.get_cached_response(h, "m") is None


# ---------------------------------------------------------------------------
# DOCBOT-2585: keyed on the model that actually serves the slot
# ---------------------------------------------------------------------------


async def test_bedrock_model_swap_misses_old_rows(wired_cache, monkeypatch):
    monkeypatch.setenv("LLM_PRIMARY", "bedrock")
    monkeypatch.setenv("BEDROCK_MODEL", "openai.gpt-oss-20b-1:0")
    await llm_cache.set_cached_response("h", "openai/gpt-oss-20b", "SELECT 1")
    assert await llm_cache.get_cached_response("h", "openai/gpt-oss-20b") == "SELECT 1"

    monkeypatch.setenv("BEDROCK_MODEL", "openai.gpt-oss-120b-1:0")
    assert await llm_cache.get_cached_response("h", "openai/gpt-oss-20b") is None
