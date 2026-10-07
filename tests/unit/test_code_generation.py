"""Unit tests for generate_analysis_code — DOCBOT-301.

All tests are fully synchronous from the test runner's perspective; async
functions are driven by pytest-asyncio.  No real Groq API calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Ensure the module is loaded so patches resolve correctly
import api.sandbox_service  # noqa: F401


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestGenerateAnalysisCode:

    async def test_skips_when_empty_result_set(self):
        """Must return None without calling LLM when no rows."""
        from api.sandbox_service import generate_analysis_code

        result = await generate_analysis_code(
            result_dicts=[],
            question="What are the trends?",
            persona_def="Data Analyst",
        )

        assert result is None

    async def test_returns_code_for_small_result_set(self):
        """Small result sets (1-4 rows) are valid for bar/pie charts and
        must proceed to LLM code generation (regression for the old 5-row gate)."""
        expected_code = "import pandas as pd\nresult = 'ok'"

        with patch("api.utils.llm_provider.chat_completion", return_value=expected_code):
            from api.sandbox_service import generate_analysis_code

            with patch.dict("os.environ", {"groq_api_key": "test-key"}):
                result = await generate_analysis_code(
                    result_dicts=[{"quarter": "Q1", "revenue": 10}, {"quarter": "Q2", "revenue": 20}],
                    question="Plot revenue by quarter",
                    persona_def="Data Analyst",
                )

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_returns_code_string_for_sufficient_rows(self):
        """Must return a non-empty string for larger result sets."""
        expected_code = "import pandas as pd\ndf = pd.DataFrame()\nresult = 'done'"

        with patch("api.utils.llm_provider.chat_completion", return_value=expected_code):
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
        """Must return None gracefully when LLM call raises."""
        with patch(
            "api.utils.llm_provider.chat_completion",
            side_effect=RuntimeError("API error"),
        ):
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

        with patch("api.utils.llm_provider.chat_completion", return_value=raw_code):
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
        """Must return None without calling LLM when groq_api_key is unset."""
        import os
        env_without_key = {k: v for k, v in os.environ.items() if k != "groq_api_key"}

        with patch.dict("os.environ", env_without_key, clear=True):
            from api.sandbox_service import generate_analysis_code

            result = await generate_analysis_code(
                result_dicts=[{"col": i} for i in range(10)],
                question="Show me the distribution",
                persona_def="Data Analyst",
            )

        assert result is None

    async def test_exactly_5_rows_proceeds_to_llm(self):
        """Boundary: exactly 5 rows must proceed to LLM call (not skip)."""
        with patch(
            "api.utils.llm_provider.chat_completion",
            return_value="import pandas as pd\nresult = 'done'",
        ) as mock_cc:
            from api.sandbox_service import generate_analysis_code

            with patch.dict("os.environ", {"groq_api_key": "test-key"}):
                result = await generate_analysis_code(
                    result_dicts=[{"col": i} for i in range(5)],
                    question="Analyse this",
                    persona_def="Data Analyst",
                )

        assert result is not None
        mock_cc.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
class TestRetryLadderTokenBudgets:
    """DOCBOT-1404: the fallback model (GROQ_MODEL, a reasoning model since
    DOCBOT-1403) must get at least as much max_tokens headroom as the primary
    attempt gets for the same question, scaled the same way for forecast /
    complex questions — not a flat 4000 regardless of question complexity.
    """

    async def test_analysis_code_fallback_matches_primary_for_forecast(self):
        """Forecast questions give the primary attempt 8000 tokens; the
        fallback attempt (triggered when the primary returns empty) must get
        the same 8000, not the old flat 4000."""
        from api.sandbox_service import generate_analysis_code

        calls = []

        def fake_chat_completion(messages, *, model, temperature, max_tokens, caller=None, prompt_version=None):
            calls.append({"model": model, "max_tokens": max_tokens})
            if len(calls) == 1:
                return ""  # primary attempt returns empty -> triggers fallback
            return "import pandas as pd\nresult = 'done'"

        with patch("api.utils.llm_provider.chat_completion", side_effect=fake_chat_completion):
            with patch.dict("os.environ", {"groq_api_key": "test-key"}):
                result = await generate_analysis_code(
                    result_dicts=[{"date": "2024-01-01", "revenue": 100}],
                    question="forecast next 12 months of revenue",
                    persona_def="Data Analyst",
                )

        assert result is not None
        assert len(calls) == 2
        assert calls[0]["max_tokens"] == 8000
        assert calls[1]["max_tokens"] == 8000

    async def test_analysis_code_fallback_floor_for_non_forecast(self):
        """Non-forecast questions still get a 4000-token floor on the
        fallback attempt (matches the primary's non-forecast budget)."""
        from api.sandbox_service import generate_analysis_code

        calls = []

        def fake_chat_completion(messages, *, model, temperature, max_tokens, caller=None, prompt_version=None):
            calls.append({"model": model, "max_tokens": max_tokens})
            if len(calls) == 1:
                return ""
            return "import pandas as pd\nresult = 'done'"

        with patch("api.utils.llm_provider.chat_completion", side_effect=fake_chat_completion):
            with patch.dict("os.environ", {"groq_api_key": "test-key"}):
                result = await generate_analysis_code(
                    result_dicts=[{"col": i} for i in range(5)],
                    question="show me the distribution",
                    persona_def="Data Analyst",
                )

        assert result is not None
        assert len(calls) == 2
        assert calls[0]["max_tokens"] == 4000
        assert calls[1]["max_tokens"] == 4000

    async def test_csv_analysis_code_fallback_matches_primary_for_complex(self):
        """Complex CSV questions give the primary attempt 8000 tokens; the
        fallback attempt must get the same 8000, not the old flat 4000."""
        from api.sandbox_service import generate_csv_analysis_code

        calls = []

        def fake_chat_completion(messages, *, model, temperature, max_tokens, caller=None, prompt_version=None):
            calls.append({"model": model, "max_tokens": max_tokens})
            if len(calls) == 1:
                return ""
            return "import pandas as pd\nresult = 'done'"

        with patch("api.utils.llm_provider.chat_completion", side_effect=fake_chat_completion):
            with patch.dict("os.environ", {"groq_api_key": "test-key"}):
                result = await generate_csv_analysis_code(
                    csv_path_in_sandbox="/tmp/data.csv",
                    column_names=["date", "revenue"],
                    question="build a forecasting model with hyperparameter tuning and cross-validation",
                    persona_def="Data Analyst",
                )

        assert result is not None
        assert len(calls) == 2
        assert calls[0]["max_tokens"] == 8000
        assert calls[1]["max_tokens"] == 8000

    async def test_csv_analysis_code_fallback_floor_for_simple(self):
        """Simple CSV questions give the primary attempt 2000 tokens, but the
        fallback attempt still gets at least the 4000-token floor."""
        from api.sandbox_service import generate_csv_analysis_code

        calls = []

        def fake_chat_completion(messages, *, model, temperature, max_tokens, caller=None, prompt_version=None):
            calls.append({"model": model, "max_tokens": max_tokens})
            if len(calls) == 1:
                return ""
            return "import pandas as pd\nresult = 'done'"

        with patch("api.utils.llm_provider.chat_completion", side_effect=fake_chat_completion):
            with patch.dict("os.environ", {"groq_api_key": "test-key"}):
                result = await generate_csv_analysis_code(
                    csv_path_in_sandbox="/tmp/data.csv",
                    column_names=["date", "revenue"],
                    question="what columns exist",
                    persona_def="Data Analyst",
                )

        assert result is not None
        assert len(calls) == 2
        assert calls[0]["max_tokens"] == 2000
        assert calls[1]["max_tokens"] == 4000
