"""Unit tests for server-side Autopilot data-source resolution.

``/api/autopilot/run`` must derive ``has_docs`` / ``has_db`` / ``has_csv``
from persisted state (db_connections rows + the vector store registry) —
not from client-asserted flags. Covers pure classification, resolver happy
paths, fail-safe fallback on resolution errors, and client-hint precedence.

Pure unit tests only — no network, no real DB, no real Chroma load.
"""

import logging
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import Column, MetaData, String, Table, Text

from api.utils.autopilot_sources import (
    SourceFlags,
    classify_connection,
    docs_available,
    resolve_autopilot_sources,
    _decrypt_creds_tolerant,
)
from api.utils.encryption import encrypt_credentials


def _db_connections_table() -> Table:
    metadata = MetaData()
    return Table(
        "db_connections", metadata,
        Column("id", String, primary_key=True),
        Column("session_id", String),
        Column("dialect", String),
        Column("credentials_blob", Text),
    )


class _FakeAsyncSession:
    """Minimal async context manager mimicking AsyncSession.execute()."""

    def __init__(self, rows, error=None):
        self._rows = rows
        self._error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def execute(self, *_args, **_kwargs):
        if self._error is not None:
            raise self._error
        result = MagicMock()
        result.fetchall.return_value = self._rows
        return result


def _factory_for(rows, error=None):
    """Returns a callable usable as `async_session_factory()`."""
    return lambda: _FakeAsyncSession(rows, error)


def _conn_row(row_id, dialect, blob=None, org_id=None):
    return SimpleNamespace(
        id=row_id, dialect=dialect, credentials_blob=blob, org_id=org_id
    )


@contextmanager
def _stub_disk_recovery(store=None):
    """Patch targets matching the resolver's lazy imports — keep Chroma out."""
    with patch(
        "api.utils.vector_store.load_store", return_value=store
    ), patch(
        "api.utils.embeddings_provider.get_embeddings", return_value=MagicMock()
    ):
        yield


def _csv_blob(profile=None):
    creds = {
        "dialect": "csv",
        "host": "__local_file__",
        "table_name": "fuel_prices",
        "columns": ["date", "price"],
        "csv_content": "ZGF0ZSxwcmljZQ==",
    }
    if profile is not None:
        creds["data_profile"] = profile
    return encrypt_credentials(creds)


_FUEL_PROFILE = {
    "columns": ["date", "price"],
    "dtypes": {"price": "float64"},
    "sample_rows": [{"date": "2021-01-01", "price": 2.85}],
}


async def _resolve(
    rows=None,
    *,
    session_id="sess-1",
    connection_id=None,
    error=None,
    vector_stores=None,
    fallback=None,
    disk_store=None,
    org_keys=None,
):
    with _stub_disk_recovery(disk_store):
        return await resolve_autopilot_sources(
            session_id=session_id,
            connection_id=connection_id,
            db_connections_table=_db_connections_table(),
            async_session_factory=_factory_for(rows if rows is not None else [], error),
            vector_stores=vector_stores,
            org_encryption_keys_table=org_keys,
            fallback=fallback,
        )


@pytest.mark.unit
class TestSourceFlags:
    def test_defaults_are_all_false(self):
        flags = SourceFlags()
        assert flags.has_docs is False
        assert flags.has_db is False
        assert flags.has_csv is False
        assert flags.data_profile is None

    def test_as_kwargs_keys_and_values(self):
        flags = SourceFlags(has_docs=True, has_db=True, has_csv=True)
        kwargs = flags.as_kwargs()
        assert kwargs == {"has_docs": True, "has_db": True, "has_csv": True}

    def test_as_kwargs_keys_match_autopilot_state_contract(self):
        """The resolver output keys must line up with what
        autopilot_service.run_autopilot() copies into AutopilotState."""
        from api.autopilot_service import AutopilotState

        assert {"has_docs", "has_db", "has_csv"}.issubset(AutopilotState.__annotations__)
        assert set(SourceFlags().as_kwargs()) == {"has_docs", "has_db", "has_csv"}


@pytest.mark.unit
class TestClassifyConnection:
    def test_csv_dialect_sets_has_csv_and_surfaces_profile(self):
        creds = {"dialect": "csv", "data_profile": _FUEL_PROFILE}
        flags = classify_connection("csv", creds)
        assert flags == SourceFlags(has_csv=True, data_profile=_FUEL_PROFILE)

    def test_csv_without_profile_has_no_profile(self):
        flags = classify_connection("csv", {"dialect": "csv"})
        assert flags.has_csv is True
        assert flags.data_profile is None

    def test_csv_with_non_dict_profile_ignored(self):
        flags = classify_connection("csv", {"data_profile": ["oops"]})
        assert flags.has_csv is True
        assert flags.data_profile is None

    def test_csv_dialect_is_normalized(self):
        assert classify_connection(" CSV ").has_csv is True

    @pytest.mark.parametrize("dialect", ["sqlite", "postgresql", "mysql", "azure_sql"])
    def test_sql_dialects_set_has_db(self, dialect):
        flags = classify_connection(dialect)
        assert flags.has_db is True
        assert flags.has_csv is False
        assert flags.has_docs is False

    @pytest.mark.parametrize("dialect", ["oracle", "duckdb", "mssql", "postgres"])
    def test_unknown_non_csv_dialect_still_counts_as_db(self, dialect):
        """An unrecognized dialect must never zero out a data source."""
        assert classify_connection(dialect).has_db is True

    @pytest.mark.parametrize("dialect", [None, "", "   "])
    def test_missing_dialect_yields_no_flags(self, dialect):
        assert classify_connection(dialect) == SourceFlags()

    def test_none_creds_are_tolerated(self):
        flags = classify_connection("csv", None)
        assert flags.has_csv is True
        assert flags.data_profile is None


@pytest.mark.unit
class TestDocsAvailable:
    def test_store_registered(self):
        assert docs_available({"sess-1": object()}, "sess-1") is True

    def test_session_absent(self):
        assert docs_available({"other": object()}, "sess-1") is False

    def test_no_registry(self):
        assert docs_available(None, "sess-1") is False

    def test_empty_registry(self):
        assert docs_available({}, "sess-1") is False

    def test_explicit_none_store(self):
        assert docs_available({"sess-1": None}, "sess-1") is False


@pytest.mark.unit
@pytest.mark.asyncio
class TestResolveAutopilotSources:
    async def test_csv_row_resolves_has_csv_and_profile(self):
        flags = await _resolve(
            rows=[_conn_row("c1", "csv", blob=_csv_blob(profile=_FUEL_PROFILE))]
        )
        assert flags.has_csv is True
        assert flags.has_db is False
        assert flags.has_docs is False
        assert flags.data_profile == _FUEL_PROFILE

    async def test_sqlite_row_resolves_has_db(self):
        flags = await _resolve(rows=[_conn_row("c1", "sqlite")])
        assert flags.has_db is True
        assert flags.has_csv is False
        assert flags.data_profile is None

    async def test_csv_and_db_rows_resolve_both_flags(self):
        flags = await _resolve(
            rows=[
                _conn_row("csv-1", "csv", blob=_csv_blob(profile=_FUEL_PROFILE)),
                _conn_row("db-1", "postgresql"),
            ]
        )
        assert flags.has_csv is True
        assert flags.has_db is True
        assert flags.data_profile == _FUEL_PROFILE

    async def test_no_rows_resolves_all_false_without_exception(self):
        flags = await _resolve(rows=[])
        assert flags == SourceFlags()

    async def test_no_rows_but_in_memory_docs_resolves_has_docs(self):
        store = MagicMock(name="vector_store")
        flags = await _resolve(rows=[], vector_stores={"sess-1": store})
        assert flags.has_docs is True
        assert flags.has_db is False
        assert flags.has_csv is False

    async def test_docs_on_disk_are_recovered_and_registered(self):
        store = MagicMock(name="disk_store")
        vector_stores: dict = {}
        flags = await _resolve(
            rows=[], vector_stores=vector_stores, disk_store=store
        )
        assert flags.has_docs is True
        assert vector_stores["sess-1"] is store

    async def test_docs_absent_everywhere_resolves_has_docs_false(self):
        flags = await _resolve(rows=[], vector_stores={})
        assert flags.has_docs is False

    async def test_client_has_docs_hint_used_only_when_disk_recovery_fails(self):
        with (
            patch(
                "api.utils.vector_store.load_store",
                side_effect=RuntimeError("chroma disk hiccup"),
            ),
            patch(
                "api.utils.embeddings_provider.get_embeddings",
                return_value=MagicMock(),
            ),
        ):
            flags = await resolve_autopilot_sources(
                session_id="sess-1",
                db_connections_table=_db_connections_table(),
                async_session_factory=_factory_for([]),
                vector_stores={},
                fallback=SourceFlags(has_docs=True),
            )
        assert flags.has_docs is True

    @pytest.mark.parametrize(
        "bad_blob",
        [
            "not-a-fernet-token",  # legacy scheme, corrupt token
            "ENV1:org-1:not-a-fernet-token",  # envelope tag without valid token
        ],
    )
    async def test_decryption_failure_is_safe_and_dialect_still_trusted(
        self, bad_blob, caplog
    ):
        sentinel_secret = "SECRET_FUEL_BLOB_BYTES"
        with caplog.at_level(logging.WARNING, logger="api.utils.autopilot_sources"):
            flags = await _resolve(
                rows=[_conn_row("c1", "csv", blob=sentinel_secret)]
            )
        assert flags.has_csv is True
        assert flags.has_db is False
        assert flags.data_profile is None
        # Clear, non-fatal warning, and no blob contents in the log.
        assert "could not decrypt" in caplog.text
        assert sentinel_secret not in caplog.text

    async def test_row_lookup_failure_falls_back_to_client_hints(self, caplog):
        with caplog.at_level(logging.WARNING, logger="api.utils.autopilot_sources"):
            flags = await _resolve(
                error=RuntimeError("db down"),
                fallback=SourceFlags(has_db=True, has_csv=True),
            )
        assert flags.has_db is True
        assert flags.has_csv is True
        assert flags.has_docs is False
        assert "db_connections lookup failed" in caplog.text

    async def test_row_lookup_failure_without_hints_resolves_all_false(self):
        flags = await _resolve(error=RuntimeError("db down"))
        assert flags == SourceFlags()

    async def test_server_resolution_overrides_wrong_client_flags(self):
        # Client claimed there is no CSV — the persisted csv row wins.
        flags = await _resolve(
            rows=[_conn_row("csv-1", "csv", blob=_csv_blob(profile=_FUEL_PROFILE))],
            fallback=SourceFlags(has_csv=False),
        )
        assert flags.has_csv is True
        assert flags.data_profile == _FUEL_PROFILE

    async def test_server_truth_beats_stale_client_hints(self):
        # No rows exist — a stale client has_csv=True hint must not survive.
        flags = await _resolve(
            rows=[],
            fallback=SourceFlags(has_csv=True, has_db=True),
        )
        assert flags.has_csv is False
        assert flags.has_db is False

    async def test_explicit_connection_id_includes_that_row(self):
        # Row belongs to another session but is attached by id.
        flags = await _resolve(
            rows=[_conn_row("c1", "csv", blob=_csv_blob(profile=_FUEL_PROFILE))],
            connection_id="c1",
        )
        assert flags.has_csv is True
        assert flags.data_profile == _FUEL_PROFILE

    async def test_malformed_row_never_raises(self):
        flags = await _resolve(rows=[SimpleNamespace(id="c1")])
        assert flags == SourceFlags()

    async def test_envelope_blob_with_unwired_encryption_is_safe_for_dialect(
        self, caplog
    ):
        with caplog.at_level(logging.WARNING, logger="api.utils.autopilot_sources"):
            flags = await _resolve(
                rows=[_conn_row("c1", "csv", blob="ENV1:org-1:dGVzdA==")],
            )
        assert flags.has_csv is True
        assert flags.data_profile is None
        assert "could not decrypt" in caplog.text

    async def test_envelope_blob_routes_through_envelope_decryption(self):
        env_creds = {"dialect": "csv", "data_profile": _FUEL_PROFILE}
        env_decrypt = AsyncMock(return_value=(env_creds, None))
        with _stub_disk_recovery(None), patch(
            "api.utils.autopilot_sources.decrypt_credentials_with_migration",
            env_decrypt,
        ):
            flags = await _resolve(
                rows=[_conn_row("c1", "csv", blob="ENV1:org-1:dGVzdA==")],
                org_keys=object(),
            )
        assert flags.has_csv is True
        assert flags.has_db is False
        assert flags.data_profile == _FUEL_PROFILE
        assert env_decrypt.await_count == 1

    async def test_decrypt_creds_tolerant_raises_for_envelope_when_unwired(self):
        with pytest.raises(RuntimeError, match="not wired"):
            await _decrypt_creds_tolerant(
                "ENV1:org-1:dGVzdA==", "org-1", None, None
            )
