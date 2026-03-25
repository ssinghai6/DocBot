"""Integration tests for file upload service — DOCBOT-206, DOCBOT-207.

DOCBOT-207 now uses the CSV→E2B pandas pipeline (no SQLite intermediate).
Tests cover _parse_csv_metadata (schema detection) and _validate_sqlite_magic.
"""

import pytest

from api.file_upload_service import (
    _validate_sqlite_magic,
    _parse_csv_metadata,
    _SQLITE_MAX_BYTES,
    _CSV_MAX_BYTES,
)


@pytest.mark.integration
class TestSQLiteMagicValidation:

    def test_valid_sqlite_file_passes(self, sqlite_db_path):
        _validate_sqlite_magic(sqlite_db_path)  # must not raise

    def test_invalid_file_raises_value_error(self, tmp_path):
        bad = tmp_path / "fake.sqlite"
        bad.write_bytes(b"this is not a sqlite file at all")
        with pytest.raises(ValueError, match="not a valid SQLite"):
            _validate_sqlite_magic(bad)

    def test_invalid_file_is_deleted_after_rejection(self, tmp_path):
        bad = tmp_path / "bad.sqlite"
        bad.write_bytes(b"garbage bytes here 1234567890")
        with pytest.raises(ValueError):
            _validate_sqlite_magic(bad)
        assert not bad.exists()

    def test_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.sqlite"
        empty.write_bytes(b"")
        with pytest.raises(ValueError):
            _validate_sqlite_magic(empty)


@pytest.mark.integration
class TestCSVMetadataParsing:
    """Tests for _parse_csv_metadata — schema detection without SQLite write."""

    def test_basic_csv_parsed(self, simple_csv_bytes):
        table_name, row_count, columns, sections, manifest = _parse_csv_metadata(simple_csv_bytes, "sales.csv")
        assert table_name == "sales"
        assert row_count == 3
        assert "product" in columns
        assert "revenue" in columns
        assert len(sections) == 1  # single-table CSV

    def test_bom_encoding_handled(self, bom_csv_bytes):
        _, _, columns, _, _ = _parse_csv_metadata(bom_csv_bytes, "bom.csv")
        assert "name" in columns   # BOM stripped from first column name
        assert "value" in columns

    def test_messy_csv_column_names_sanitised(self, messy_csv_bytes):
        _, row_count, columns, _, _ = _parse_csv_metadata(messy_csv_bytes, "messy.csv")
        assert row_count == 2
        assert len(columns) == 3
        for col in columns:
            assert col == col.lower()
            assert " " not in col

    def test_column_names_sanitised(self):
        csv = b"First Name,Last Name,Total $\nAlice,Smith,100\n"
        _, _, columns, _, _ = _parse_csv_metadata(csv, "data.csv")
        for col in columns:
            assert col == col.lower()
            assert " " not in col

    def test_table_name_derived_from_filename(self, simple_csv_bytes):
        table_name, _, _, _, _ = _parse_csv_metadata(simple_csv_bytes, "monthly_report_2024.csv")
        assert table_name == "monthly_report_2024"

    def test_empty_csv_raises_value_error(self):
        with pytest.raises(ValueError, match="empty|parsed"):
            _parse_csv_metadata(b"col1,col2\n", "empty.csv")

    def test_returns_column_list(self, simple_csv_bytes):
        _, _, columns, _, _ = _parse_csv_metadata(simple_csv_bytes, "sales.csv")
        assert isinstance(columns, list)
        assert len(columns) > 0

    def test_returns_correct_row_count(self, simple_csv_bytes):
        _, row_count, _, _, _ = _parse_csv_metadata(simple_csv_bytes, "sales.csv")
        assert row_count == 3

    def test_size_limits_are_correct(self):
        assert _SQLITE_MAX_BYTES == 100 * 1024 * 1024
        assert _CSV_MAX_BYTES == 50 * 1024 * 1024
