"""E2B sandbox service for isolated Python code execution.

Provides a single async entry-point, run_python(), that:
  - Spins up an E2B code-interpreter sandbox
  - Executes arbitrary Python with a hard 25-second wall-clock timeout
  - Extracts stdout, stderr, and matplotlib charts (as base64 PNGs)
  - Guarantees sandbox teardown in a finally block regardless of outcome
"""

import asyncio
import base64
import logging
import os
import time
from typing import Optional

import groq as groq_module

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Injected before every execution to:
#   1. Force the Agg (non-GUI) backend so matplotlib works in headless sandboxes
#   2. Patch plt.show() to write base64-encoded PNGs to stdout prefixed with
#      CHART_B64: so sandbox_service can extract them without relying on
#      execution.results (which is empty in E2B v2 sync mode)
_MATPLOTLIB_PREAMBLE = """\
import matplotlib as _mpl
_mpl.use('Agg')
import matplotlib.pyplot as _plt
import io as _io, base64 as _b64, sys as _sys

_orig_show = _plt.show
def _patched_show(*_args, **_kwargs):
    buf = _io.BytesIO()
    _plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    _sys.stdout.write('CHART_B64:' + _b64.b64encode(buf.read()).decode() + '\\n')
    _sys.stdout.flush()
    _plt.close('all')
_plt.show = _patched_show
"""

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class SandboxResult(BaseModel):
    """Structured output returned from every sandbox execution."""

    stdout: str
    stderr: str
    # Each entry is a base64-encoded PNG string (no data-URI prefix)
    charts: list[str]
    # Human-readable error message when execution fails; None on success
    error: Optional[str]
    execution_time_ms: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_api_key() -> str:
    """Return the E2B API key from the environment or raise clearly."""
    key = os.getenv("E2B_API_KEY")
    if not key:
        raise EnvironmentError(
            "E2B_API_KEY is not set. Add it to your .env file or Railway environment."
        )
    return key


def _extract_charts(stdout_lines: list[str]) -> tuple[list[str], list[str]]:
    """Separate CHART_B64: lines from regular stdout lines.

    The _MATPLOTLIB_PREAMBLE patches plt.show() to write base64-encoded PNGs
    to stdout prefixed with 'CHART_B64:'. This function splits those out so
    charts end up in SandboxResult.charts and do not pollute SandboxResult.stdout.

    Returns (clean_stdout_lines, chart_b64_strings).
    """
    charts: list[str] = []
    clean: list[str] = []
    prefix = "CHART_B64:"
    for line in stdout_lines:
        if line.startswith(prefix):
            charts.append(line[len(prefix):].strip())
        else:
            clean.append(line)
    return clean, charts


def _run_in_sandbox(code: str) -> SandboxResult:
    """Synchronous execution block — called from a thread via asyncio.

    Kept synchronous because the E2B SDK's Sandbox class is synchronous.
    We offload it to a thread pool so the event loop stays unblocked.
    """
    # Import here to defer the dependency until first use; also makes
    # import errors surfaceable at call-time rather than module-load-time.
    from e2b_code_interpreter import Sandbox

    # v2.x SDK reads E2B_API_KEY from environment automatically; validate it
    # exists before creating the sandbox so the error is clear.
    _get_api_key()
    sandbox: Optional[Sandbox] = None
    start_ms = int(time.monotonic() * 1000)

    try:
        sandbox = Sandbox.create()
        # Prepend preamble so plt.show() auto-encodes charts to stdout
        execution = sandbox.run_code(_MATPLOTLIB_PREAMBLE + code)

        raw_stdout: list[str] = list(execution.logs.stdout or [])
        stderr_lines: list[str] = list(execution.logs.stderr or [])

        clean_stdout, charts = _extract_charts(raw_stdout)
        stdout = "\n".join(clean_stdout)
        stderr = "\n".join(stderr_lines)

        error_msg: Optional[str] = None
        if execution.error is not None:
            # ExecutionError has .name and .value attributes
            name = getattr(execution.error, "name", "ExecutionError")
            value = getattr(execution.error, "value", str(execution.error))
            error_msg = f"{name}: {value}"
            logger.warning("Sandbox execution error: %s", name)

        elapsed_ms = int(time.monotonic() * 1000) - start_ms
        logger.info(
            "Sandbox execution finished in %d ms. charts=%d stderr_bytes=%d",
            elapsed_ms,
            len(charts),
            len(stderr),
        )

        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            charts=charts,
            error=error_msg,
            execution_time_ms=elapsed_ms,
        )

    finally:
        if sandbox is not None:
            try:
                sandbox.kill()
            except Exception as teardown_err:
                logger.warning("Failed to close sandbox cleanly: %s", teardown_err)


# ---------------------------------------------------------------------------
# DOCBOT-301: Python code generation
# ---------------------------------------------------------------------------


async def generate_analysis_code(
    result_dicts: list[dict],
    question: str,
    persona_def: str,
) -> Optional[str]:
    """Generate Python analysis code for a SQL result set using Qwen 2.5 Coder via Groq.

    Returns a Python code string suitable for run_python(), or None when
    fewer than 5 rows are available (not worth analysing) or on any error.
    """
    if len(result_dicts) < 5:
        return None

    api_key = os.getenv("groq_api_key")
    if not api_key:
        logger.warning("generate_analysis_code: groq_api_key not set, skipping")
        return None

    system_prompt = (
        "You are a Python data analyst. Given a question and a dataset, write Python code that:\n"
        "1. Imports pandas, numpy, and matplotlib.pyplot as plt\n"
        "2. Creates a DataFrame `df` from the provided data\n"
        "3. Performs relevant analysis to answer the question\n"
        "4. Creates at least one matplotlib chart and saves it with plt.savefig('chart_0.png')\n"
        "5. Assigns a brief text summary to a variable named `result`\n\n"
        "Rules:\n"
        "- Output ONLY raw Python code, no markdown fences, no explanations\n"
        "- Do not use plt.show()\n"
        "- The chart filename must be 'chart_0.png'\n"
        "- Keep code under 60 lines"
    )

    import json as _json
    sample = result_dicts[:50]
    user_message = (
        f"Question: {question}\n\n"
        f"Data ({len(sample)} rows):\n{_json.dumps(sample, default=str)}"
    )

    try:
        client = groq_module.Groq(api_key=api_key)
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="qwen-2.5-coder-32b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=1500,
                temperature=0,
            ),
        )
        code = response.choices[0].message.content.strip()

        # Strip markdown fences if the model adds them anyway
        lines = code.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    except Exception as exc:
        logger.warning("generate_analysis_code failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------


async def run_python(
    code: str,
    timeout_seconds: int = 25,
) -> SandboxResult:
    """Execute Python code in an isolated E2B sandbox.

    Parameters
    ----------
    code:
        Python source to execute.  Never logged to avoid leaking sensitive
        computation details — only timing and structural metrics are logged.
    timeout_seconds:
        Hard wall-clock limit.  Defaults to 25 s to stay comfortably inside
        the Vercel / Railway 30-second response deadline.

    Returns
    -------
    SandboxResult
        Always returns a result — never raises on execution errors.  Only
        infrastructure failures (missing API key, network errors) propagate
        as exceptions.
    """
    loop = asyncio.get_running_loop()

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _run_in_sandbox, code),
            timeout=float(timeout_seconds),
        )
    except asyncio.TimeoutError:
        logger.error(
            "Sandbox execution timed out after %d seconds.", timeout_seconds
        )
        return SandboxResult(
            stdout="",
            stderr="",
            charts=[],
            error=f"Execution timed out after {timeout_seconds} seconds.",
            execution_time_ms=timeout_seconds * 1000,
        )

    return result
