"""DOCBOT-1633: regression tests for the bandit B108/B608 remediations."""

from __future__ import annotations

import posixpath
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import update
from sqlalchemy.dialects import postgresql, sqlite

from api.org_service import _count_where, _org_table, _owned_by
from api.utils.pdf_io import open_pdf_bytes
from api.utils.sandbox_paths import SANDBOX_TMP_DIR, sandbox_csv_path


# ---------------------------------------------------------------------------
# B108 — sandbox CSV paths
# ---------------------------------------------------------------------------


class TestSandboxCsvPath:
    def test_plain_name_unchanged(self) -> None:
        assert sandbox_csv_path("sales_2024") == f"{SANDBOX_TMP_DIR}/sales_2024.csv"

    @pytest.mark.parametrize(
        "name",
        ["../../etc/passwd", "a/b", "x'); import os; os.system('id') #", "..", ""],
    )
    def test_hostile_names_stay_in_dir_and_quote_safe(self, name: str) -> None:
        path = sandbox_csv_path(name)
        assert posixpath.dirname(path) == SANDBOX_TMP_DIR
        assert posixpath.normpath(path) == path
        assert "'" not in path and '"' not in path and "/.." not in path

    def test_empty_name_falls_back(self) -> None:
        assert sandbox_csv_path("") == f"{SANDBOX_TMP_DIR}/data.csv"


# ---------------------------------------------------------------------------
# B108 — host temp dirs resolve through tempfile
# ---------------------------------------------------------------------------


def test_upload_dir_uses_tempfile() -> None:
    from api import file_upload_service

    assert file_upload_service._TMP_DIR == Path(tempfile.gettempdir()) / "docbot_uploads"


def test_pdf_opened_from_memory() -> None:
    fitz = pytest.importorskip("fitz")
    src = fitz.open()
    src.new_page().insert_text((72, 72), "DocBot in-memory PDF " * 5)
    content = src.tobytes()
    src.close()

    doc = open_pdf_bytes(content)
    try:
        assert len(doc) == 1
        assert "DocBot in-memory PDF" in doc[0].get_text("text")
    finally:
        doc.close()


def test_pdf_invalid_bytes_raise() -> None:
    fitz = pytest.importorskip("fitz")
    with pytest.raises(fitz.FileDataError):
        open_pdf_bytes(b"not a pdf")


# ---------------------------------------------------------------------------
# B608 — org_service statements are bound, never string-built
# ---------------------------------------------------------------------------

_HOSTILE = "x' OR '1'='1"


@pytest.mark.parametrize("dialect", [postgresql.dialect(), sqlite.dialect()])
def test_owned_count_binds_values(dialect) -> None:
    t = _org_table("db_connections", "user_id", "credentials_blob")
    stmt = _count_where(t, _owned_by(t, _HOSTILE, _HOSTILE), t.c.credentials_blob.like(_HOSTILE))
    compiled = stmt.compile(dialect=dialect)
    assert _HOSTILE not in str(compiled)
    assert list(compiled.params.values()).count(_HOSTILE) == 3


def test_owned_update_binds_values() -> None:
    t = _org_table("sessions", "user_id")
    stmt = update(t).values(org_id=_HOSTILE).where(_owned_by(t, _HOSTILE, "org"))
    compiled = stmt.compile(dialect=postgresql.dialect())
    assert _HOSTILE not in str(compiled)
    assert str(compiled).startswith("UPDATE sessions SET org_id=")
