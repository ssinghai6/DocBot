"""Unit tests for generate_analysis_code — DOCBOT-301.

All tests are fully synchronous from the test runner's perspective; async
functions are driven by pytest-asyncio.  No real Anthropic API calls are made.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure the module is loaded so patch("api.sandbox_service.anthropic") resolves correctly
import api.sandbox_service  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_anthropic_response(text: str) -> MagicMock:
    """Build a minimal mock that looks like an Anthropic Messages response."""
    content_block = MagicMock()
    content_block.text = text

    response = MagicMock()
    response.content = [content_block]
    return response


def _make_anthropic_client(code_text: str = "import pandas as pd\nresult = 'ok'") -> MagicMock:
    """Return a mock AsyncAnthropic client whose messages.create returns code_text."""
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=_make_anthropic_response(code_text))
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestGenerateAnalysisCode:

    async def test_skips_when_fewer_than_5_rows(self):
        """Must return None without calling Anthropic when < 5 rows."""
        with patch("api.sandbox_service.anthropic") as mock_anthropic:
            mock_anthropic.AsyncAnthropic.return_value = _make_anthropic_client()
            from api.sandbox_service import generate_analysis_code

            result = await generate_analysis_code(
                result_dicts=[{"col": "val"}] * 4,
                question="What are the trends?",
                persona_def="Data Analyst",
            )

        assert result is None
        mock_anthropic.AsyncAnthropic.assert_not_called()

    async def test_returns_code_string_for_sufficient_rows(self):
        """Must return a non-empty string when >= 5 rows."""
        expected_code = "import pandas as pd\ndf = pd.DataFrame()\nresult = 'done'"
        client = _make_anthropic_client(expected_code)

        with patch("api.sandbox_service.anthropic") as mock_anthropic:
            mock_anthropic.AsyncAnthropic.return_value = client
            from api.sandbox_service import generate_analysis_code

            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
                result = await generate_analysis_code(
                    result_dicts=[{"col": i} for i in range(10)],
                    question="Show me the distribution",
                    persona_def="Data Analyst",
                )

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_returns_none_on_api_error(self):
        """Must return None gracefully when Anthropic API raises."""
        client = MagicMock()
        client.messages = MagicMock()
        client.messages.create = AsyncMock(side_effect=RuntimeError("API error"))

        with patch("api.sandbox_service.anthropic") as mock_anthropic:
            mock_anthropic.AsyncAnthropic.return_value = client
            from api.sandbox_service import generate_analysis_code

            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
                result = await generate_analysis_code(
                    result_dicts=[{"col": i} for i in range(10)],
                    question="Show me the distribution",
                    persona_def="Data Analyst",
                )

        assert result is None

    async def test_strips_markdown_fences(self):
        """Code returned with ```python fences must be stripped cleanly."""
        raw_code = "```python\nimport pandas as pd\nresult = 'ok'\n```"
        client = _make_anthropic_client(raw_code)

        with patch("api.sandbox_service.anthropic") as mock_anthropic:
            mock_anthropic.AsyncAnthropic.return_value = client
            from api.sandbox_service import generate_analysis_code

            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
                result = await generate_analysis_code(
                    result_dicts=[{"col": i} for i in range(10)],
                    question="Show me the distribution",
                    persona_def="Data Analyst",
                )

        assert result is not None
        assert "```" not in result
        assert "import pandas" in result

    async def test_returns_none_when_api_key_missing(self):
        """Must return None without calling Anthropic when ANTHROPIC_API_KEY is unset."""
        import os
        env_without_key = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

        with patch.dict("os.environ", env_without_key, clear=True):
            with patch("api.sandbox_service.anthropic") as mock_anthropic:
                mock_anthropic.AsyncAnthropic.return_value = _make_anthropic_client()
                from api.sandbox_service import generate_analysis_code

                result = await generate_analysis_code(
                    result_dicts=[{"col": i} for i in range(10)],
                    question="Show me the distribution",
                    persona_def="Data Analyst",
                )

        assert result is None
        mock_anthropic.AsyncAnthropic.assert_not_called()

    async def test_exactly_5_rows_does_not_skip(self):
        """Boundary: exactly 5 rows must proceed to LLM call (not skip)."""
        client = _make_anthropic_client("import pandas as pd\nresult = 'done'")

        with patch("api.sandbox_service.anthropic") as mock_anthropic:
            mock_anthropic.AsyncAnthropic.return_value = client
            from api.sandbox_service import generate_analysis_code

            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
                result = await generate_analysis_code(
                    result_dicts=[{"col": i} for i in range(5)],
                    question="Analyse this",
                    persona_def="Data Analyst",
                )

        # Should have attempted the LLM call (result is not None from 5-row check perspective)
        # but may still be None if the call is made and returns something valid
        assert result is not None
