"""Integration tests for artifact_service (DOCBOT-501).

These tests hit a real SQLite in-memory database via SQLAlchemy async.
No network calls, no API keys required.
"""

import json
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import (
    MetaData, Table, Column, String, Text, Integer, DateTime, func,
)

from api.artifact_service import (
    save_artifact,
    list_artifacts,
    get_artifact,
    ArtifactSummary,
    ArtifactDetail,
    _MAX_ROWS,
)


# ---------------------------------------------------------------------------
# Fixtures — in-memory SQLite DB with session_artifacts table
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def artifact_db():
    """Create an in-memory SQLite engine with session_artifacts table."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    metadata = MetaData()

    session_artifacts_table = Table(
        "session_artifacts", metadata,
        Column("id", String, primary_key=True),
        Column("session_id", String, nullable=False),
        Column("turn_id", Integer, nullable=False),
        Column("artifact_type", String, nullable=False),
        Column("name", Text, nullable=False),
        Column("data_json", Text),
        Column("chart_b64", Text),
        Column("row_count", Integer),
        Column("columns", Text),
        Column("created_at", DateTime, server_default=func.now(), nullable=False),
    )

    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield session_artifacts_table, factory

    await engine.dispose()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_and_get_artifact_with_data(artifact_db):
    table, factory = artifact_db
    rows = [{"month": "Jan", "revenue": 100}, {"month": "Feb", "revenue": 200}]

    artifact_id = await save_artifact(
        session_id="sess-1",
        turn_id=1,
        artifact_type="sql_result",
        name="Revenue by Month",
        result_dicts=rows,
        chart_b64="base64data",
        session_artifacts_table=table,
        async_session_factory=factory,
    )

    assert artifact_id  # non-empty UUID
    detail = await get_artifact(artifact_id, table, factory)
    assert detail is not None
    assert detail.session_id == "sess-1"
    assert detail.turn_id == 1
    assert detail.artifact_type == "sql_result"
    assert detail.name == "Revenue by Month"
    assert detail.row_count == 2
    assert detail.columns == ["month", "revenue"]
    assert detail.has_chart is True
    assert detail.chart_b64 == "base64data"
    assert json.loads(detail.data_json) == rows


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_chart_only_artifact(artifact_db):
    table, factory = artifact_db

    artifact_id = await save_artifact(
        session_id="sess-2",
        turn_id=1,
        artifact_type="chart",
        name="Heatmap",
        result_dicts=None,
        chart_b64="png_base64",
        session_artifacts_table=table,
        async_session_factory=factory,
    )

    detail = await get_artifact(artifact_id, table, factory)
    assert detail.data_json is None
    assert detail.chart_b64 == "png_base64"
    assert detail.row_count is None
    assert detail.columns is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_artifacts_ordered_by_turn(artifact_db):
    table, factory = artifact_db

    await save_artifact(
        session_id="sess-3", turn_id=2, artifact_type="sql_result",
        name="Second query", result_dicts=[{"x": 1}],
        session_artifacts_table=table, async_session_factory=factory,
    )
    await save_artifact(
        session_id="sess-3", turn_id=1, artifact_type="sql_result",
        name="First query", result_dicts=[{"y": 2}],
        session_artifacts_table=table, async_session_factory=factory,
    )

    summaries = await list_artifacts("sess-3", table, factory)
    assert len(summaries) == 2
    assert summaries[0].turn_id == 1
    assert summaries[1].turn_id == 2
    assert summaries[0].name == "First query"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_artifacts_isolated_by_session(artifact_db):
    table, factory = artifact_db

    await save_artifact(
        session_id="sess-A", turn_id=1, artifact_type="dataframe",
        name="A data", result_dicts=[{"col": 1}],
        session_artifacts_table=table, async_session_factory=factory,
    )
    await save_artifact(
        session_id="sess-B", turn_id=1, artifact_type="dataframe",
        name="B data", result_dicts=[{"col": 2}],
        session_artifacts_table=table, async_session_factory=factory,
    )

    a_artifacts = await list_artifacts("sess-A", table, factory)
    b_artifacts = await list_artifacts("sess-B", table, factory)
    assert len(a_artifacts) == 1
    assert len(b_artifacts) == 1
    assert a_artifacts[0].name == "A data"
    assert b_artifacts[0].name == "B data"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_nonexistent_artifact_returns_none(artifact_db):
    table, factory = artifact_db
    result = await get_artifact("nonexistent-id", table, factory)
    assert result is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_truncation_at_max_rows(artifact_db):
    table, factory = artifact_db

    big_result = [{"id": i} for i in range(600)]
    artifact_id = await save_artifact(
        session_id="sess-4",
        turn_id=1,
        artifact_type="sql_result",
        name="Large result",
        result_dicts=big_result,
        session_artifacts_table=table,
        async_session_factory=factory,
    )

    detail = await get_artifact(artifact_id, table, factory)
    # row_count reflects original count
    assert detail.row_count == 600
    # data_json is truncated to _MAX_ROWS
    stored_rows = json.loads(detail.data_json)
    assert len(stored_rows) == _MAX_ROWS
    assert stored_rows[0]["id"] == 0
    assert stored_rows[-1]["id"] == _MAX_ROWS - 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_empty_session_returns_empty_list(artifact_db):
    table, factory = artifact_db
    result = await list_artifacts("nonexistent-session", table, factory)
    assert result == []
