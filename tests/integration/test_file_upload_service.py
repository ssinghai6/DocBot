"""Integration tests for file upload service — DOCBOT-206, DOCBOT-207."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from api.file_upload_service import (
    _validate_sqlite_magic,
    _csv_bytes_to_sqlite,
    _infer_and_cast_columns,
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
class TestCSVToSQLite:

    def test_basic_csv_parsed(self, simple_csv_bytes, tmp_path):
        out = tmp_path / "out.sqlite"
        table_name, row_count, columns = _csv_bytes_to_sqlite(simple_csv_bytes, "sales.csv", out)
        assert table_name == "sales"
        assert row_count == 3
        assert "product" in columns
        assert "revenue" in columns

    def test_sqlite_file_is_queryable(self, simple_csv_bytes, tmp_path):
        out = tmp_path / "out.sqlite"
        table_name, _, _ = _csv_bytes_to_sqlite(simple_csv_bytes, "sales.csv", out)
        conn = sqlite3.connect(str(out))
        rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()
        conn.close()
        assert len(rows) == 3

    def test_bom_encoding_handled(self, bom_csv_bytes, tmp_path):
        out = tmp_path / "bom.sqlite"
        _, _, columns = _csv_bytes_to_sqlite(bom_csv_bytes, "bom.csv", out)
        assert "name" in columns  # BOM stripped from first column name
        assert "value" in columns

    def test_messy_csv_with_quoted_commas(self, messy_csv_bytes, tmp_path):
        out = tmp_path / "messy.sqlite"
        table_name, row_count, columns = _csv_bytes_to_sqlite(messy_csv_bytes, "messy.csv", out)
        assert row_count == 2
        assert len(columns) == 3

    def test_column_names_sanitised(self, tmp_path):
        csv = b"First Name,Last Name,Total $\nAlice,Smith,100\n"
        out = tmp_path / "out.sqlite"
        _, _, columns = _csv_bytes_to_sqlite(csv, "data.csv", out)
        for col in columns:
            assert col == col.lower()
            assert " " not in col

    def test_table_name_derived_from_filename(self, simple_csv_bytes, tmp_path):
        out = tmp_path / "out.sqlite"
        table_name, _, _ = _csv_bytes_to_sqlite(simple_csv_bytes, "monthly_report_2024.csv", out)
        assert table_name == "monthly_report_2024"

    def test_empty_csv_raises_value_error(self, tmp_path):
        out = tmp_path / "out.sqlite"
        with pytest.raises(ValueError, match="empty"):
            _csv_bytes_to_sqlite(b"col1,col2\n", "empty.csv", out)

    def test_size_limits_are_correct(self):
        assert _SQLITE_MAX_BYTES == 100 * 1024 * 1024
        assert _CSV_MAX_BYTES == 50 * 1024 * 1024


@pytest.mark.integration
class TestTypeInference:

    def test_integer_columns_inferred(self):
        import pandas as pd
        df = pd.DataFrame({"qty": ["1", "2", "3"], "name": ["a", "b", "c"]})
        result = _infer_and_cast_columns(df)
        assert "int" in str(result["qty"].dtype).lower()

    def test_float_columns_inferred(self):
        import pandas as pd
        df = pd.DataFrame({"price": ["1.5", "2.5", "3.0"]})
        result = _infer_and_cast_columns(df)
        assert "float" in str(result["price"].dtype).lower()

    def test_text_columns_stay_as_object(self):
        import pandas as pd
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
        result = _infer_and_cast_columns(df)
        assert result["name"].dtype == object

    def test_mixed_type_column_stays_as_text(self):
        import pandas as pd
        df = pd.DataFrame({"mixed": ["1", "two", "3"]})
        result = _infer_and_cast_columns(df)
        assert result["mixed"].dtype == object

    def test_null_values_handled(self):
        import pandas as pd
        df = pd.DataFrame({"qty": ["1", None, "3"]})
        result = _infer_and_cast_columns(df)  # must not raise
        assert result is not None
