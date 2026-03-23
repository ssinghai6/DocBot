"""Unit tests for generate_analysis_code — DOCBOT-301.

All tests are fully synchronous from the test runner's perspective; async
functions are driven by pytest-asyncio.  No real Groq API calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Ensure the module is loaded so patch("api.sandbox_service.groq_module") resolves correctly
import api.sandbox_service  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_groq_response(code_text: str) -> MagicMock:
    """Build a minimal mock that looks like a Groq ChatCompletion response."""
    message = MagicMock()
    message.content = code_text

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


def _make_groq_client(code_text: str = "import pandas as pd\nresult = 'ok'") -> MagicMock:
    """Return a mock Groq client whose chat.completions.create returns code_text."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = MagicMock(return_value=_make_groq_response(code_text))
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestGenerateAnalysisCode:

    async def test_skips_when_fewer_than_5_rows(self):
        """Must return None without calling Groq when < 5 rows."""
        with patch("api.sandbox_service.groq_module") as mock_groq:
            mock_groq.Groq.return_value = _make_groq_client()
            from api.sandbox_service import generate_analysis_code

            result = await generate_analysis_code(
                result_dicts=[{"col": "val"}] * 4,
                question="What are the trends?",
                persona_def="Data Analyst",
            )

        assert result is None
        mock_groq.Groq.assert_not_called()

    async def test_returns_code_string_for_sufficient_rows(self):
        """Must return a non-empty string when >= 5 rows."""
        expected_code = "import pandas as pd\ndf = pd.DataFrame()\nresult = 'done'"

        with patch("api.sandbox_service.groq_module") as mock_groq:
            mock_groq.Groq.return_value = _make_groq_client(expected_code)
            from api.sandbox_service import generate_analysis_code

            with patch.dict("os.environ", {"groq_api_key": "test-key"}):
                result = await generate_analysis_code(
                    result_dicts=[{"col": i} for i in range(10)],
                    question="Show me the distribution",
                    persona_def="Data Analyst",
                )

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_returns_none_on_api_error(self):
        """Must return None gracefully when Groq API raises."""
        client = MagicMock()
        client.chat.completions.create = MagicMock(side_effect=RuntimeError("API error"))

        with patch("api.sandbox_service.groq_module") as mock_groq:
            mock_groq.Groq.return_value = client
            from api.sandbox_service import generate_analysis_code

            with patch.dict("os.environ", {"groq_api_key": "test-key"}):
                result = await generate_analysis_code(
                    result_dicts=[{"col": i} for i in range(10)],
                    question="Show me the distribution",
                    persona_def="Data Analyst",
                )

        assert result is None

    async def test_strips_markdown_fences(self):
        """Code returned with ```python fences must be stripped cleanly."""
        raw_code = "```python\nimport pandas as pd\nresult = 'ok'\n```"

        with patch("api.sandbox_service.groq_module") as mock_groq:
            mock_groq.Groq.return_value = _make_groq_client(raw_code)
            from api.sandbox_service import generate_analysis_code

            with patch.dict("os.environ", {"groq_api_key": "test-key"}):
                result = await generate_analysis_code(
                    result_dicts=[{"col": i} for i in range(10)],
                    question="Show me the distribution",
                    persona_def="Data Analyst",
                )

        assert result is not None
        assert "```" not in result
        assert "import pandas" in result

    async def test_returns_none_when_api_key_missing(self):
        """Must return None without calling Groq when groq_api_key is unset."""
        import os
        env_without_key = {k: v for k, v in os.environ.items() if k != "groq_api_key"}

        with patch.dict("os.environ", env_without_key, clear=True):
            with patch("api.sandbox_service.groq_module") as mock_groq:
                mock_groq.Groq.return_value = _make_groq_client()
                from api.sandbox_service import generate_analysis_code

                result = await generate_analysis_code(
                    result_dicts=[{"col": i} for i in range(10)],
                    question="Show me the distribution",
                    persona_def="Data Analyst",
                )

        assert result is None
        mock_groq.Groq.assert_not_called()

    async def test_exactly_5_rows_proceeds_to_llm(self):
        """Boundary: exactly 5 rows must proceed to LLM call (not skip)."""
        with patch("api.sandbox_service.groq_module") as mock_groq:
            mock_groq.Groq.return_value = _make_groq_client("import pandas as pd\nresult = 'done'")
            from api.sandbox_service import generate_analysis_code

            with patch.dict("os.environ", {"groq_api_key": "test-key"}):
                result = await generate_analysis_code(
                    result_dicts=[{"col": i} for i in range(5)],
                    question="Analyse this",
                    persona_def="Data Analyst",
                )

        assert result is not None
        mock_groq.Groq.assert_called_once()
