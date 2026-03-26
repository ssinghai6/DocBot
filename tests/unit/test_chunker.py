"""Unit tests for api/utils/chunker.py — DOCBOT-1003.

All tests are CI-safe: HuggingFaceEmbeddings and SemanticChunker are mocked.
No network calls, no API keys required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from langchain_core.documents import Document

from api.utils.chunker import chunk_document, detect_doc_type


# ---------------------------------------------------------------------------
# detect_doc_type() helper
# ---------------------------------------------------------------------------


class TestDetectDocType:
    def test_financial_keywords(self):
        text = "This is an annual report for the fiscal year ended December 31."
        assert detect_doc_type(text) == "financial"

    def test_10k_keyword(self):
        text = "Form 10-K filing for the period ending 2023."
        assert detect_doc_type(text) == "financial"

    def test_legal_agreement_keyword(self):
        text = "This agreement is entered into between Party A and Party B."
        assert detect_doc_type(text) == "legal"

    def test_contract_keyword(self):
        text = "This contract sets forth the terms and conditions."
        assert detect_doc_type(text) == "legal"

    def test_general_no_keywords(self):
        text = "The weather in London is often cloudy and rainy."
        assert detect_doc_type(text) == "general"

    def test_only_first_500_chars_examined(self):
        # Financial keyword buried deep — should not affect detection
        prefix = "x" * 600
        text = prefix + " annual report balance sheet"
        assert detect_doc_type(text) == "general"

    def test_case_insensitive(self):
        text = "ANNUAL REPORT Q4 results."
        assert detect_doc_type(text) == "financial"


# ---------------------------------------------------------------------------
# chunk_document() — SemanticChunker path
# ---------------------------------------------------------------------------


class TestSemanticChunkerPath:
    def _make_semantic_chunks(self, text: str) -> list[Document]:
        return [
            Document(page_content=text[:50]),
            Document(page_content=text[50:100] if len(text) > 50 else text),
        ]

    @patch("api.utils.chunker.HuggingFaceEmbeddings")
    @patch("api.utils.chunker.SemanticChunker")
    def test_financial_doc_uses_semantic_chunker(self, mock_chunker_cls, mock_emb_cls):
        sample_text = "Annual report financial statements revenue profit loss " * 10
        mock_instance = MagicMock()
        mock_instance.create_documents.return_value = [
            Document(page_content="chunk1"),
            Document(page_content="chunk2"),
        ]
        mock_chunker_cls.return_value = mock_instance

        result = chunk_document(sample_text, hf_api_key="hf_test", doc_type="financial")

        mock_chunker_cls.assert_called_once()
        mock_instance.create_documents.assert_called_once_with([sample_text])
        assert len(result) == 2
        assert result[0].page_content == "chunk1"

    @patch("api.utils.chunker.HuggingFaceEmbeddings")
    @patch("api.utils.chunker.SemanticChunker")
    def test_legal_doc_uses_semantic_chunker(self, mock_chunker_cls, mock_emb_cls):
        mock_instance = MagicMock()
        mock_instance.create_documents.return_value = [Document(page_content="legal chunk")]
        mock_chunker_cls.return_value = mock_instance

        result = chunk_document("contract agreement terms", hf_api_key="hf_key", doc_type="legal")

        mock_chunker_cls.assert_called_once()
        assert result[0].page_content == "legal chunk"

    @patch("api.utils.chunker.HuggingFaceEmbeddings")
    @patch("api.utils.chunker.SemanticChunker")
    def test_annual_report_uses_semantic_chunker(self, mock_chunker_cls, mock_emb_cls):
        mock_instance = MagicMock()
        mock_instance.create_documents.return_value = [Document(page_content="ar chunk")]
        mock_chunker_cls.return_value = mock_instance

        result = chunk_document("text", hf_api_key="hf_key", doc_type="annual_report")
        mock_chunker_cls.assert_called_once()

    @patch("api.utils.chunker.HuggingFaceEmbeddings")
    @patch("api.utils.chunker.SemanticChunker")
    def test_10k_uses_semantic_chunker(self, mock_chunker_cls, mock_emb_cls):
        mock_instance = MagicMock()
        mock_instance.create_documents.return_value = [Document(page_content="10k chunk")]
        mock_chunker_cls.return_value = mock_instance

        result = chunk_document("text", hf_api_key="hf_key", doc_type="10k")
        mock_chunker_cls.assert_called_once()

    @patch("api.utils.chunker.HuggingFaceEmbeddings")
    @patch("api.utils.chunker.SemanticChunker")
    def test_contract_uses_semantic_chunker(self, mock_chunker_cls, mock_emb_cls):
        mock_instance = MagicMock()
        mock_instance.create_documents.return_value = [Document(page_content="contract chunk")]
        mock_chunker_cls.return_value = mock_instance

        result = chunk_document("text", hf_api_key="hf_key", doc_type="contract")
        mock_chunker_cls.assert_called_once()

    @patch("api.utils.chunker.HuggingFaceEmbeddings")
    @patch("api.utils.chunker.SemanticChunker")
    def test_semantic_chunker_receives_correct_params(self, mock_chunker_cls, mock_emb_cls):
        mock_instance = MagicMock()
        mock_instance.create_documents.return_value = [Document(page_content="x")]
        mock_chunker_cls.return_value = mock_instance

        chunk_document("some text", hf_api_key="hf_key", doc_type="financial")

        # Verify SemanticChunker was constructed with the right threshold params
        call_kwargs = mock_chunker_cls.call_args
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        # Accept positional too
        all_kwargs = {**kwargs}
        if call_kwargs.args and len(call_kwargs.args) > 1:
            # first positional arg is embeddings
            pass
        assert all_kwargs.get("breakpoint_threshold_type") == "percentile"
        assert all_kwargs.get("breakpoint_threshold_amount") == 85


# ---------------------------------------------------------------------------
# chunk_document() — RecursiveCharacterTextSplitter path
# ---------------------------------------------------------------------------


class TestRecursiveSplitterPath:
    @patch("api.utils.chunker.HuggingFaceEmbeddings")
    @patch("api.utils.chunker.SemanticChunker")
    def test_general_doc_uses_recursive_splitter(self, mock_chunker_cls, mock_emb_cls):
        text = "This is a general document about weather. " * 50  # >200 chars
        result = chunk_document(text, hf_api_key="hf_key", doc_type="general")

        # SemanticChunker must NOT be used
        mock_chunker_cls.assert_not_called()
        assert len(result) > 0
        assert all(isinstance(d, Document) for d in result)

    @patch("api.utils.chunker.HuggingFaceEmbeddings")
    @patch("api.utils.chunker.SemanticChunker")
    def test_empty_hf_key_skips_semantic_chunker(self, mock_chunker_cls, mock_emb_cls):
        result = chunk_document("annual report text " * 30, hf_api_key="", doc_type="financial")

        mock_chunker_cls.assert_not_called()
        assert len(result) > 0


# ---------------------------------------------------------------------------
# chunk_document() — fallback on SemanticChunker exception
# ---------------------------------------------------------------------------


class TestSemanticChunkerFallback:
    @patch("api.utils.chunker.HuggingFaceEmbeddings")
    @patch("api.utils.chunker.SemanticChunker")
    def test_falls_back_on_semantic_chunker_exception(self, mock_chunker_cls, mock_emb_cls):
        mock_chunker_cls.side_effect = RuntimeError("embeddings unavailable")

        text = "financial statements balance sheet revenue " * 50
        result = chunk_document(text, hf_api_key="hf_key", doc_type="financial")

        # Should fall back and still return Document objects
        assert len(result) > 0
        assert all(isinstance(d, Document) for d in result)

    @patch("api.utils.chunker.HuggingFaceEmbeddings")
    @patch("api.utils.chunker.SemanticChunker")
    def test_falls_back_on_create_documents_exception(self, mock_chunker_cls, mock_emb_cls):
        mock_instance = MagicMock()
        mock_instance.create_documents.side_effect = Exception("API error")
        mock_chunker_cls.return_value = mock_instance

        text = "10-K filing annual report revenue profit " * 40
        result = chunk_document(text, hf_api_key="hf_key", doc_type="10k")

        assert len(result) > 0
        assert all(isinstance(d, Document) for d in result)
