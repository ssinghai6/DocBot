"""Unit tests for edgar_service — ingestion pipeline with mocked deps."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mock_session_factory():
    """Create a mock async session factory that supports context managers."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.begin = MagicMock()
    session.begin.return_value.__aenter__ = AsyncMock()
    session.begin.return_value.__aexit__ = AsyncMock(return_value=False)

    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return factory


def _mock_sessions_table():
    return MagicMock()


def _mock_embeddings():
    return MagicMock()


# ── ingest_edgar_filing tests ────────────────────────────────────────────────

class TestIngestEdgarFiling:
    def test_returns_cached_session_on_duplicate(self):
        """If a filing is already cached, return existing session without downloading."""
        cached = {
            "id": "cache-1",
            "cik": "320193",
            "accession_number": "0000320193-23-000106",
            "filing_type": "10-K",
            "filing_date": "2023-11-03",
            "company_name": "Apple Inc.",
            "ticker": "AAPL",
            "session_id": "existing-session-123",
            "text_hash": "abc",
        }

        with patch("api.connector_store.get_cached_filing", new_callable=AsyncMock, return_value=cached):
            from api.edgar_service import ingest_edgar_filing

            async def run():
                result = await ingest_edgar_filing(
                    cik="320193",
                    accession_number="0000320193-23-000106",
                    primary_document="aapl-20230930.htm",
                    filing_type="10-K",
                    filing_date="2023-11-03",
                    company_name="Apple Inc.",
                    ticker="AAPL",
                    sessions_table=_mock_sessions_table(),
                    async_session_factory=_mock_session_factory(),
                    vector_stores={},
                    get_embeddings_fn=_mock_embeddings,
                    extracted_fields_store={},
                )
                assert result["cached"] is True
                assert result["session_id"] == "existing-session-123"
                assert result["chunks_created"] == 0

            asyncio.run(run())

    def test_creates_session_and_vectors(self):
        """Fresh filing: downloads, chunks, embeds, creates session."""
        mock_text = "UNITED STATES SECURITIES AND EXCHANGE COMMISSION\n" * 50

        mock_chunks = [MagicMock(metadata={}) for _ in range(5)]

        mock_insert = MagicMock()
        mock_insert.return_value.values = MagicMock(return_value=MagicMock())

        with (
            patch("api.connector_store.get_cached_filing", new_callable=AsyncMock, return_value=None),
            patch("api.connector_store.save_filing_cache", new_callable=AsyncMock) as mock_cache_save,
            patch("api.connectors.edgar_connector.EdgarConnector.fetch_filing_text", new_callable=AsyncMock, return_value=mock_text),
            patch("api.utils.chunker.chunk_document", return_value=mock_chunks),
            patch("api.utils.chunker.detect_doc_type", return_value="financial"),
            patch("api.utils.vector_store.create_store") as mock_create_store,
            patch("api.document_extractor.is_extractable_document", return_value=False),
            patch("api.edgar_service.insert", mock_insert),
            patch.dict("os.environ", {"huggingface_api_key": "test-key"}),
        ):
            mock_store = MagicMock()
            mock_create_store.return_value = mock_store

            vector_stores: dict = {}
            from api.edgar_service import ingest_edgar_filing

            async def run():
                result = await ingest_edgar_filing(
                    cik="320193",
                    accession_number="0000320193-23-000106",
                    primary_document="aapl-20230930.htm",
                    filing_type="10-K",
                    filing_date="2023-11-03",
                    company_name="Apple Inc.",
                    ticker="AAPL",
                    sessions_table=_mock_sessions_table(),
                    async_session_factory=_mock_session_factory(),
                    vector_stores=vector_stores,
                    get_embeddings_fn=_mock_embeddings,
                    extracted_fields_store={},
                )
                assert result["cached"] is False
                assert result["chunks_created"] == 5
                assert "session_id" in result
                assert result["session_id"] in vector_stores
                mock_cache_save.assert_called_once()

            asyncio.run(run())

    def test_raises_on_empty_text(self):
        """Should raise ValueError if filing yields no usable text."""
        with (
            patch("api.connector_store.get_cached_filing", new_callable=AsyncMock, return_value=None),
            patch("api.connectors.edgar_connector.EdgarConnector.fetch_filing_text", new_callable=AsyncMock, return_value="   "),
        ):
            from api.edgar_service import ingest_edgar_filing

            async def run():
                with pytest.raises(ValueError, match="no usable text"):
                    await ingest_edgar_filing(
                        cik="320193",
                        accession_number="bad-filing",
                        primary_document="empty.htm",
                        filing_type="10-K",
                        filing_date="2023-01-01",
                        company_name="Test",
                        ticker="TST",
                        sessions_table=_mock_sessions_table(),
                        async_session_factory=_mock_session_factory(),
                        vector_stores={},
                        get_embeddings_fn=_mock_embeddings,
                        extracted_fields_store={},
                    )

            asyncio.run(run())


# ── ingest_multiple_filings tests ────────────────────────────────────────────

class TestBatchIngest:
    def test_ingests_multiple_filings(self):
        filings_list = [
            {"accession_number": "acc-1", "primary_document": "f1.htm", "filing_type": "10-K", "filing_date": "2023-11-03"},
            {"accession_number": "acc-2", "primary_document": "f2.htm", "filing_type": "10-K", "filing_date": "2022-10-28"},
        ]

        ingest_results = [
            {"session_id": "s1", "chunks_created": 10, "cached": False, "filing_date": "2023-11-03"},
            {"session_id": "s2", "chunks_created": 8, "cached": False, "filing_date": "2022-10-28"},
        ]

        with (
            patch("api.connectors.edgar_connector.EdgarConnector.list_filings", new_callable=AsyncMock, return_value=filings_list),
            patch("api.edgar_service.ingest_edgar_filing", new_callable=AsyncMock, side_effect=ingest_results) as mock_ingest,
        ):
            from api.edgar_service import ingest_multiple_filings

            async def run():
                results = await ingest_multiple_filings(
                    cik="320193",
                    ticker="AAPL",
                    company_name="Apple Inc.",
                    filing_type="10-K",
                    count=2,
                    sessions_table=_mock_sessions_table(),
                    async_session_factory=_mock_session_factory(),
                    vector_stores={},
                    get_embeddings_fn=_mock_embeddings,
                    extracted_fields_store={},
                )
                assert len(results) == 2
                assert results[0]["session_id"] == "s1"
                assert results[1]["session_id"] == "s2"

            asyncio.run(run())

    def test_handles_partial_failure(self):
        """One filing succeeds, one fails — both results returned."""
        filings_list = [
            {"accession_number": "acc-1", "primary_document": "f1.htm", "filing_type": "10-K", "filing_date": "2023-11-03"},
            {"accession_number": "acc-2", "primary_document": "f2.htm", "filing_type": "10-K", "filing_date": "2022-10-28"},
        ]

        with (
            patch("api.connectors.edgar_connector.EdgarConnector.list_filings", new_callable=AsyncMock, return_value=filings_list),
            patch("api.edgar_service.ingest_edgar_filing", new_callable=AsyncMock, side_effect=[
                {"session_id": "s1", "chunks_created": 10, "cached": False, "filing_date": "2023-11-03"},
                RuntimeError("download failed"),
            ]),
        ):
            from api.edgar_service import ingest_multiple_filings

            async def run():
                results = await ingest_multiple_filings(
                    cik="320193",
                    ticker="AAPL",
                    company_name="Apple Inc.",
                    filing_type="10-K",
                    count=2,
                    sessions_table=_mock_sessions_table(),
                    async_session_factory=_mock_session_factory(),
                    vector_stores={},
                    get_embeddings_fn=_mock_embeddings,
                    extracted_fields_store={},
                )
                assert len(results) == 2
                assert results[0]["session_id"] == "s1"
                assert "error" in results[1]

            asyncio.run(run())
