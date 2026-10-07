"""Unit tests for DOCBOT-1507 prompt versioning in api.document_extractor.

extract_document_fields() is the one LLM call path in the codebase that
bypasses api.utils.llm_provider's chat_completion/call_llm wrappers entirely
(LangExtract drives its own Gemini SDK calls), so it logs via
log_external_llm_call directly. These tests mock `langextract` (not
installed/network-reachable in CI) and assert prompt_version is threaded
through on both the success and failure paths.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _install_fake_langextract(monkeypatch, extractions=None, raise_exc=None):
    """Install a minimal fake `langextract` module so document_extractor's
    `import langextract as lx` succeeds without the real package/network."""
    fake_lx = types.ModuleType("langextract")

    def _extract(**kwargs):
        if raise_exc is not None:
            raise raise_exc
        result = MagicMock()
        result.extractions = extractions or []
        return result

    fake_lx.extract = _extract

    fake_data = types.ModuleType("langextract.data")
    fake_data.Extraction = MagicMock()
    fake_data.ExampleData = MagicMock()
    fake_lx.data = fake_data

    monkeypatch.setitem(sys.modules, "langextract", fake_lx)
    monkeypatch.setitem(sys.modules, "langextract.data", fake_data)


class TestPromptVersionsDict:
    def test_every_prompt_key_has_a_version(self):
        from api.document_extractor import _PROMPTS, _PROMPT_VERSIONS

        assert set(_PROMPTS.keys()) <= set(_PROMPT_VERSIONS.keys())

    def test_versions_start_at_v1(self):
        from api.document_extractor import _PROMPT_VERSIONS

        assert all(v == "v1" for v in _PROMPT_VERSIONS.values())


class TestExtractDocumentFieldsLogging:
    @pytest.mark.asyncio
    async def test_success_logs_prompt_version(self, monkeypatch):
        _install_fake_langextract(monkeypatch, extractions=[])

        with patch("api.utils.llm_provider.log_external_llm_call") as mock_log:
            from api.document_extractor import extract_document_fields

            await extract_document_fields(
                "Total Revenue: $42.1M for fiscal year 2026.",
                session_id="sess-1",
                gemini_api_key="fake-key",
            )

        mock_log.assert_called_once()
        kwargs = mock_log.call_args.kwargs
        assert kwargs["success"] is True
        assert kwargs["caller"] == "document_extraction"
        assert kwargs["prompt_version"] == "v1"

    @pytest.mark.asyncio
    async def test_failure_logs_prompt_version(self, monkeypatch):
        _install_fake_langextract(monkeypatch, raise_exc=RuntimeError("boom"))

        with patch("api.utils.llm_provider.log_external_llm_call") as mock_log:
            from api.document_extractor import extract_document_fields

            result = await extract_document_fields(
                "Total Revenue: $42.1M for fiscal year 2026.",
                session_id="sess-1",
                gemini_api_key="fake-key",
            )

        assert result == []
        mock_log.assert_called_once()
        kwargs = mock_log.call_args.kwargs
        assert kwargs["success"] is False
        assert kwargs["caller"] == "document_extraction"
        assert kwargs["prompt_version"] == "v1"
        assert kwargs["error_class"] == "RuntimeError"
