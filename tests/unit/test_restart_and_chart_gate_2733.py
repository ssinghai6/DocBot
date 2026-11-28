"""DOCBOT-2733: re-upload message after restart, and the step-6.5 chart gate."""
import sqlite3

import pytest

from api.db_service import (
    ConnectionNotFoundError,
    SQLITE_FILE_GONE_MSG,
    _require_sqlite_file,
    _should_run_analysis,
)


def test_missing_sqlite_file_asks_for_reupload(tmp_path):
    with pytest.raises(ConnectionNotFoundError, match="re-upload"):
        _require_sqlite_file({"dialect": "sqlite", "dbname": str(tmp_path / "gone.sqlite")})


def test_empty_sqlite_file_left_by_lazy_connect_asks_for_reupload(tmp_path):
    p = tmp_path / "x.sqlite"
    p.touch()  # what SQLite leaves behind after connecting to a wiped path
    with pytest.raises(ConnectionNotFoundError) as exc:
        _require_sqlite_file({"dialect": "sqlite", "dbname": str(p)})
    assert str(exc.value) == SQLITE_FILE_GONE_MSG


def test_present_sqlite_file_and_other_dialects_pass(tmp_path):
    p = tmp_path / "ok.sqlite"
    sqlite3.connect(p).execute("CREATE TABLE t (a INT)").connection.commit()
    _require_sqlite_file({"dialect": "sqlite", "dbname": str(p)})
    _require_sqlite_file({"dialect": "postgresql", "dbname": "/nope"})


@pytest.mark.parametrize("question,chart_type,n,tieout,expected", [
    ("Does revenue agree with the books?", "auto", 1, True, False),
    ("Does revenue agree with the books?", "auto", 6, True, False),
    ("Total revenue by department", "auto", 1, False, False),
    ("Total revenue by department", "auto", 6, False, True),
    ("Chart revenue by quarter vs the report", "auto", 4, True, True),
    ("Revenue by quarter", "bar", 4, True, True),
    ("anything", "bar", 0, False, False),
])
def test_should_run_analysis(question, chart_type, n, tieout, expected):
    assert _should_run_analysis(question, chart_type, n, tieout) is expected
