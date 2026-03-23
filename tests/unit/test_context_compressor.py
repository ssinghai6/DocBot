"""Unit tests for api/utils/context_compressor.py (DOCBOT-502).

All tests are purely in-memory — no DB, no network, no API calls.
SQLAlchemy select() and session.execute() are patched entirely.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_factory(total_count: int, count_at_last: int = 0, summary: str | None = None,
                  messages: list[dict] | None = None):
    """Build a fully-mocked async_session_factory.

    Patches are applied inside each test via the module-level 'select' patch,
    so we only need to wire session.execute() to return sensible results.
    """
    # Build message row mocks
    msg_rows = []
    for m in (messages or []):
        r = MagicMock()
        r.role = m["role"]
        r.content = m["content"]
        msg_rows.append(r)

    # id rows for total count
    id_rows = [MagicMock() for _ in range(total_count)]

    # Session row
    sess_row = MagicMock()
    sess_row.message_count_at_compression = count_at_last
    sess_row.context_summary = summary

    call_count = [0]

    async def _execute(_query):
        result = MagicMock()
        n = call_count[0]
        call_count[0] += 1
        if n == 0:
            # first call — total count or session summary
            result.fetchall.return_value = id_rows
            result.fetchone.return_value = sess_row
        elif n == 1:
            # second call — session row or message list
            result.fetchall.return_value = msg_rows
            result.fetchone.return_value = sess_row
        else:
            result.fetchall.return_value = msg_rows
            result.fetchone.return_value = sess_row
        return result

    mock_session = AsyncMock()
    mock_session.execute = _execute

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    factory = MagicMock()
    factory.return_value = mock_ctx
    return factory


def _patch_select():
    """Return a patch context that makes select(...) return a benign sentinel."""
    return patch("api.utils.context_compressor.select", return_value=MagicMock())


def _patch_update():
    return patch("api.utils.context_compressor.update", return_value=MagicMock())


# ---------------------------------------------------------------------------
# should_compress
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestShouldCompress:
    @pytest.mark.asyncio
    async def test_fires_when_threshold_reached(self):
        from api.utils.context_compressor import COMPRESSION_THRESHOLD
        factory = _mock_factory(total_count=COMPRESSION_THRESHOLD, count_at_last=0)
        with _patch_select():
            from api.utils.context_compressor import should_compress
            result = await should_compress("s1", MagicMock(), MagicMock(), factory)
        assert result is True

    @pytest.mark.asyncio
    async def test_no_compress_below_threshold(self):
        from api.utils.context_compressor import COMPRESSION_THRESHOLD
        factory = _mock_factory(total_count=COMPRESSION_THRESHOLD - 1, count_at_last=0)
        with _patch_select():
            from api.utils.context_compressor import should_compress
            result = await should_compress("s1", MagicMock(), MagicMock(), factory)
        assert result is False

    @pytest.mark.asyncio
    async def test_respects_count_at_last_compression(self):
        """25 total but 10 already compressed → delta=15 < 20 → no compress."""
        factory = _mock_factory(total_count=25, count_at_last=10)
        with _patch_select():
            from api.utils.context_compressor import should_compress
            result = await should_compress("s1", MagicMock(), MagicMock(), factory)
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_exception(self):
        factory = MagicMock(side_effect=RuntimeError("db down"))
        from api.utils.context_compressor import should_compress
        result = await should_compress("s1", MagicMock(), MagicMock(), factory)
        assert result is False


# ---------------------------------------------------------------------------
# build_compressed_context
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildCompressedContext:
    @pytest.mark.asyncio
    async def test_returns_all_messages_when_no_summary(self):
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        factory = _mock_factory(total_count=2, summary=None, messages=msgs)
        with _patch_select():
            from api.utils.context_compressor import build_compressed_context
            result = await build_compressed_context("s1", MagicMock(), MagicMock(), factory)
        assert len(result) == 2
        assert result[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_prepends_summary_system_message(self):
        from api.utils.context_compressor import RECENT_WINDOW
        msgs = [{"role": "user", "content": f"q{i}"} for i in range(RECENT_WINDOW + 5)]
        factory = _mock_factory(
            total_count=len(msgs),
            summary="• Data source: orders DB\n• Asked about revenue",
            messages=msgs,
        )
        with _patch_select():
            from api.utils.context_compressor import build_compressed_context
            result = await build_compressed_context("s1", MagicMock(), MagicMock(), factory)
        assert result[0]["role"] == "system"
        assert "Earlier conversation summary" in result[0]["content"]
        assert "Data source" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_recent_window_size_enforced(self):
        from api.utils.context_compressor import RECENT_WINDOW
        msgs = [{"role": "user", "content": f"q{i}"} for i in range(RECENT_WINDOW + 5)]
        factory = _mock_factory(
            total_count=len(msgs),
            summary="some summary",
            messages=msgs,
        )
        with _patch_select():
            from api.utils.context_compressor import build_compressed_context
            result = await build_compressed_context("s1", MagicMock(), MagicMock(), factory)
        # result[0] is the system summary msg; rest should be RECENT_WINDOW messages
        assert len(result) == RECENT_WINDOW + 1

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_exception(self):
        factory = MagicMock(side_effect=RuntimeError("boom"))
        from api.utils.context_compressor import build_compressed_context
        result = await build_compressed_context("s1", MagicMock(), MagicMock(), factory)
        assert result == []

    @pytest.mark.asyncio
    async def test_no_compression_for_single_message(self):
        msgs = [{"role": "user", "content": "hello"}]
        factory = _mock_factory(total_count=1, summary=None, messages=msgs)
        with _patch_select():
            from api.utils.context_compressor import build_compressed_context
            result = await build_compressed_context("s1", MagicMock(), MagicMock(), factory)
        assert result == [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestConstants:
    def test_threshold_greater_than_window(self):
        from api.utils.context_compressor import COMPRESSION_THRESHOLD, RECENT_WINDOW
        assert COMPRESSION_THRESHOLD > RECENT_WINDOW, (
            "Threshold must be greater than RECENT_WINDOW so there are messages to summarise"
        )

    def test_recent_window_reasonable(self):
        from api.utils.context_compressor import RECENT_WINDOW
        assert RECENT_WINDOW >= 5, "RECENT_WINDOW must keep enough context for LLM"
