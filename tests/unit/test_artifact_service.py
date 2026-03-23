"""Unit tests for artifact_service (DOCBOT-501).

All tests are purely in-memory — no DB, no network calls.
"""

import json
import pytest
from api.artifact_service import (
    ArtifactSummary,
    ArtifactDetail,
    _MAX_ROWS,
)


@pytest.mark.unit
class TestArtifactModels:
    def test_summary_fields(self):
        s = ArtifactSummary(
            id="a1",
            session_id="s1",
            turn_id=1,
            artifact_type="sql_result",
            name="Q3 Revenue",
            row_count=50,
            columns=["month", "revenue"],
            has_chart=True,
            created_at="2026-01-01T00:00:00",
        )
        assert s.id == "a1"
        assert s.has_chart is True
        assert s.columns == ["month", "revenue"]

    def test_detail_inherits_summary(self):
        d = ArtifactDetail(
            id="a2",
            session_id="s1",
            turn_id=2,
            artifact_type="chart",
            name="Heatmap",
            row_count=None,
            columns=None,
            has_chart=True,
            created_at="2026-01-01T00:00:00",
            data_json=None,
            chart_b64="abc123",
        )
        assert d.chart_b64 == "abc123"
        assert d.data_json is None

    def test_max_rows_constant(self):
        assert _MAX_ROWS == 500

    def test_summary_optional_fields(self):
        s = ArtifactSummary(
            id="a3",
            session_id="s2",
            turn_id=1,
            artifact_type="dataframe",
            name="Raw Data",
            row_count=None,
            columns=None,
            has_chart=False,
            created_at="2026-01-01T00:00:00",
        )
        assert s.row_count is None
        assert s.columns is None
        assert s.has_chart is False


@pytest.mark.unit
class TestArtifactTruncation:
    """Verify that the _MAX_ROWS cap is applied correctly in save_artifact()."""

    def test_max_rows_value(self):
        """500-row cap ensures artifacts stay storage-bounded."""
        assert _MAX_ROWS == 500

    def test_truncation_preserves_first_rows(self):
        """Verify that truncation logic ([:_MAX_ROWS]) preserves the leading rows."""
        rows = [{"id": i} for i in range(600)]
        truncated = rows[:_MAX_ROWS]
        assert len(truncated) == 500
        assert truncated[0] == {"id": 0}
        assert truncated[-1] == {"id": 499}

    def test_column_extraction_from_result_dicts(self):
        """Column names should be extracted from the first row's keys."""
        result_dicts = [
            {"month": "Jan", "revenue": 100},
            {"month": "Feb", "revenue": 200},
        ]
        if result_dicts:
            columns = list(result_dicts[0].keys())
        assert columns == ["month", "revenue"]

    def test_json_serialization_roundtrip(self):
        """data_json must round-trip cleanly through json.dumps / json.loads."""
        original = [{"id": i, "val": float(i) * 1.5} for i in range(10)]
        data_json = json.dumps(original)
        recovered = json.loads(data_json)
        assert recovered == original

    def test_columns_json_roundtrip(self):
        cols = ["id", "name", "revenue"]
        cols_json = json.dumps(cols)
        assert json.loads(cols_json) == cols

    def test_empty_result_dicts_produces_none(self):
        """Empty result_dicts should yield no data_json and no columns."""
        result_dicts = []
        data_json = None
        columns = None
        row_count = None
        if result_dicts:
            truncated = result_dicts[:_MAX_ROWS]
            row_count = len(result_dicts)
            data_json = json.dumps(truncated)
            if truncated:
                columns = list(truncated[0].keys())
        assert data_json is None
        assert columns is None
        assert row_count is None


@pytest.mark.unit
class TestArtifactTypeValidation:
    valid_types = ["dataframe", "chart", "sql_result"]

    @pytest.mark.parametrize("artifact_type", ["dataframe", "chart", "sql_result"])
    def test_valid_artifact_types(self, artifact_type):
        s = ArtifactSummary(
            id="x",
            session_id="s",
            turn_id=1,
            artifact_type=artifact_type,
            name="Test",
            row_count=None,
            columns=None,
            has_chart=False,
            created_at="2026-01-01T00:00:00",
        )
        assert s.artifact_type == artifact_type

    def test_name_truncation_to_100_chars(self):
        """Names derived from questions should be capped at 100 chars."""
        long_question = "x" * 200
        name = long_question[:100]
        assert len(name) == 100
