"""E2B sandbox service for isolated Python code execution.

Provides a single async entry-point, run_python(), that:
  - Spins up an E2B code-interpreter sandbox
  - Executes arbitrary Python with a hard 25-second wall-clock timeout
  - Extracts stdout, stderr, and matplotlib charts (as base64 PNGs)
  - Guarantees sandbox teardown in a finally block regardless of outcome
"""

import asyncio
import base64
import json as _stdlib_json
import logging
import os
import time
from typing import AsyncGenerator, List, Optional

import groq as groq_module

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# DOCBOT-305: supported chart types for validate + prompt routing
VALID_CHART_TYPES = {"auto", "bar", "line", "scatter", "heatmap", "box", "multi"}

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

def _flush_open_figures():
    \"\"\"Capture any matplotlib figures that are still open and write to stdout.\"\"\"
    for _fnum in _plt.get_fignums():
        _fig = _plt.figure(_fnum)
        _buf = _io.BytesIO()
        _fig.savefig(_buf, format='png', bbox_inches='tight', dpi=150)
        _buf.seek(0)
        _sys.stdout.write('CHART_B64:' + _b64.b64encode(_buf.read()).decode() + '\\n')
        _sys.stdout.flush()
    _plt.close('all')

_orig_show = _plt.show
def _patched_show(*_args, **_kwargs):
    _flush_open_figures()
_plt.show = _patched_show
"""

# Appended after every execution — captures figures that were created but
# never shown (e.g. code that calls savefig() to a file, or forgets plt.show())
_MATPLOTLIB_SUFFIX = "\n# --- auto-capture any remaining open figures ---\n_flush_open_figures()\n"

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class ChartMetadata(BaseModel):
    """Metadata about a generated chart — extracted from sandbox stdout."""

    type: str = "auto"
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    series_count: int = 1


class SandboxResult(BaseModel):
    """Structured output returned from every sandbox execution."""

    stdout: str
    stderr: str
    # Each entry is a base64-encoded PNG string (no data-URI prefix)
    charts: list[str]
    # Human-readable error message when execution fails; None on success
    error: Optional[str]
    execution_time_ms: int
    # DOCBOT-305: chart metadata extracted from CHART_META: stdout lines
    chart_metadata: list[ChartMetadata] = []


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


def _extract_charts(
    stdout_lines: list[str],
) -> tuple[list[str], list[str], list[ChartMetadata]]:
    """Separate CHART_B64: and CHART_META: lines from regular stdout lines.

    The _MATPLOTLIB_PREAMBLE patches plt.show() to write base64-encoded PNGs
    to stdout prefixed with 'CHART_B64:'. DOCBOT-305 generated code also writes
    'CHART_META:{...json...}' lines so callers receive structured chart info.

    Returns (clean_stdout_lines, chart_b64_strings, chart_metadata_list).
    """
    import json as _json

    charts: list[str] = []
    metadata: list[ChartMetadata] = []
    clean: list[str] = []

    for line in stdout_lines:
        if line.startswith("CHART_B64:"):
            charts.append(line[len("CHART_B64:"):].strip())
        elif line.startswith("CHART_META:"):
            try:
                raw = _json.loads(line[len("CHART_META:"):].strip())
                metadata.append(ChartMetadata(**raw))
            except Exception:
                pass  # malformed metadata — silently skip
        else:
            clean.append(line)

    return clean, charts, metadata


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
        # Prepend preamble + append suffix so all figures are captured
        full_code = _MATPLOTLIB_PREAMBLE + code + _MATPLOTLIB_SUFFIX
        execution = sandbox.run_code(full_code)

        # Join all stdout chunks then split by newline — handles SDKs that
        # buffer output in chunks rather than line-by-line
        raw_stdout_joined = "".join(execution.logs.stdout or [])
        raw_stdout: list[str] = raw_stdout_joined.splitlines()
        stderr_lines: list[str] = list(execution.logs.stderr or [])

        clean_stdout, charts, chart_meta = _extract_charts(raw_stdout)
        stdout = "\n".join(clean_stdout)
        stderr = "\n".join(stderr_lines)

        error_msg: Optional[str] = None
        if execution.error is not None:
            # ExecutionError has .name and .value attributes
            name = getattr(execution.error, "name", "ExecutionError")
            value = getattr(execution.error, "value", str(execution.error))
            error_msg = f"{name}: {value}"
            logger.warning("Sandbox execution error: %s: %s", name, value[:300])

        logger.info(
            "Sandbox stdout_lines=%d chart_b64_lines=%d stderr_bytes=%d error=%s",
            len(raw_stdout), len(charts), len(stderr), error_msg,
        )

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
            chart_metadata=chart_meta,
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


def _chart_type_instructions(chart_type: str) -> str:
    """Return chart-type-specific instructions for the code-gen prompt."""
    instructions = {
        "bar": (
            "Create a bar chart. Use vertical bars unless there are many categories, "
            "in which case use horizontal bars for readability."
        ),
        "line": (
            "Create a line chart. Use markers if fewer than 20 data points."
        ),
        "scatter": (
            "Create a scatter plot. Add a linear regression line (numpy polyfit) "
            "if both axes are numeric."
        ),
        "heatmap": (
            "Create a heatmap. If the data has numeric columns, compute a correlation "
            "matrix with df.corr() and plot it using matplotlib imshow or seaborn heatmap. "
            "Annotate cells with values."
        ),
        "box": (
            "Create a box plot showing the distribution of numeric columns. "
            "Use plt.boxplot or df.boxplot."
        ),
        "multi": (
            "Create a 2x2 subplot grid (fig, axes = plt.subplots(2, 2, figsize=(12, 10))). "
            "Fill each panel with the most informative chart for the data: "
            "e.g. bar, line, scatter, box. Ensure tight_layout() is called."
        ),
    }
    return instructions.get(chart_type, "Choose the most appropriate chart type for the data and question.")


async def generate_analysis_code(
    result_dicts: list[dict],
    question: str,
    persona_def: str,
    chart_type: str = "auto",
) -> Optional[str]:
    """Generate Python analysis code for a SQL result set using Qwen 2.5 Coder via Groq.

    Returns a Python code string suitable for run_python(), or None when
    fewer than 5 rows are available (not worth analysing) or on any error.

    Parameters
    ----------
    chart_type : one of VALID_CHART_TYPES — controls which chart the LLM produces.
                 "auto" lets the LLM choose the best chart for the data.
    """
    if len(result_dicts) < 5:
        return None

    # Normalise and validate chart_type
    chart_type = chart_type.lower() if chart_type else "auto"
    if chart_type not in VALID_CHART_TYPES:
        chart_type = "auto"

    api_key = os.getenv("groq_api_key")
    if not api_key:
        logger.warning("generate_analysis_code: groq_api_key not set, skipping")
        return None

    chart_instructions = _chart_type_instructions(chart_type)

    system_prompt = (
        "You are a Python data analyst. Given a question and a dataset, write Python code that:\n"
        "1. Imports pandas, numpy, matplotlib.pyplot as plt, and seaborn as sns if needed\n"
        "2. Creates a DataFrame `df` from the provided data\n"
        "3. Performs relevant analysis to answer the question\n"
        f"4. CHART TYPE REQUIREMENT: {chart_instructions}\n"
        "5. Calls plt.show() exactly once after creating the chart\n"
        "6. Assigns a brief text summary to a variable named `result`\n"
        "7. AFTER plt.show(), prints chart metadata in EXACTLY this format (one line):\n"
        '   import json; print(\'CHART_META:\' + json.dumps({\'type\': \'<chart_type>\', '
        "'title': '<chart_title>', 'x_label': '<x_axis_label>', "
        "'y_label': '<y_axis_label>', 'series_count': <int>}))\n\n"
        "Rules:\n"
        "- Output ONLY raw Python code, no markdown fences, no explanations\n"
        "- Do NOT call plt.savefig() — use plt.show() only\n"
        "- Keep code under 80 lines"
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
                model="qwen/qwen3-32b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=1500,
                temperature=0,
            ),
        )
        code = response.choices[0].message.content.strip()

        # Strip <think>...</think> reasoning blocks emitted by Qwen 3
        import re as _re
        code = _re.sub(r"<think>.*?</think>", "", code, flags=_re.DOTALL).strip()

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


# ---------------------------------------------------------------------------
# CSV → E2B pandas pipeline (DOCBOT-207 replacement)
# ---------------------------------------------------------------------------


async def generate_csv_analysis_code(
    csv_path_in_sandbox: str,
    column_names: List[str],
    question: str,
    persona_def: str,
    chart_type: str = "auto",
) -> Optional[str]:
    """Generate Python/pandas code to answer a question about a CSV file on E2B.

    Returns a Python code string, or None on error / missing API key.
    """
    api_key = os.getenv("groq_api_key")
    if not api_key:
        logger.warning("generate_csv_analysis_code: groq_api_key not set, using fallback")
        return None

    chart_instructions = _chart_type_instructions(chart_type)
    cols_preview = ", ".join(column_names[:20])
    if len(column_names) > 20:
        cols_preview += f" … (+{len(column_names) - 20} more)"

    system_prompt = (
        "You are a Python data analyst. A CSV file has already been uploaded to the sandbox. "
        "Write Python code that:\n"
        f"1. Loads the CSV: df = pd.read_csv('{csv_path_in_sandbox}')\n"
        "2. Answers the user's question using pandas operations\n"
        f"3. CHART TYPE REQUIREMENT: {chart_instructions}\n"
        "4. If the question asks about schema / columns / structure: print df.dtypes and df.shape, skip charting\n"
        "5. Otherwise calls plt.show() exactly once after creating the chart\n"
        "6. Prints a brief text summary of findings\n"
        "7. AFTER plt.show() (if chart was made), prints chart metadata on ONE line:\n"
        "   import json; print('CHART_META:' + json.dumps({'type': '<chart_type>', "
        "'title': '<chart_title>', 'x_label': '<x_label>', 'y_label': '<y_label>', 'series_count': <int>}))\n\n"
        "Rules:\n"
        "- Output ONLY raw Python code, no markdown fences, no explanations\n"
        "- Always import pandas as pd and import matplotlib.pyplot as plt\n"
        "- Keep code under 70 lines"
    )

    user_message = (
        f"CSV columns: {cols_preview}\n\n"
        f"Question: {question}"
    )

    try:
        client = groq_module.Groq(api_key=api_key)
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=2000,
                temperature=0,
            ),
        )
        code = response.choices[0].message.content.strip()

        # Strip <think>…</think> reasoning blocks
        import re as _re
        code = _re.sub(r"<think>.*?</think>", "", code, flags=_re.DOTALL).strip()

        # Strip markdown fences
        lines = code.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines).strip()

        # Validate syntax before returning — a truncated response from the LLM
        # can produce unterminated string literals that crash E2B at runtime.
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as syn_err:
            logger.warning(
                "generate_csv_analysis_code: generated code has syntax error (%s) — "
                "returning None so caller uses safe fallback",
                syn_err,
            )
            return None

        return code

    except Exception as exc:
        logger.warning("generate_csv_analysis_code failed: %s", exc)
        return None


def _run_csv_in_sandbox_sync(code: str, csv_path: str, csv_bytes: bytes) -> SandboxResult:
    """Synchronous: upload CSV bytes to E2B sandbox, run code, return result."""
    from e2b_code_interpreter import Sandbox

    _get_api_key()
    sandbox: Optional[Sandbox] = None
    start_ms = int(time.monotonic() * 1000)

    try:
        sandbox = Sandbox.create()
        # Upload CSV bytes to the sandbox filesystem
        sandbox.files.write(csv_path, csv_bytes)

        full_code = _MATPLOTLIB_PREAMBLE + code + _MATPLOTLIB_SUFFIX
        execution = sandbox.run_code(full_code)

        raw_stdout_joined = "".join(execution.logs.stdout or [])
        raw_stdout: list[str] = raw_stdout_joined.splitlines()
        stderr_lines: list[str] = list(execution.logs.stderr or [])

        clean_stdout, charts, chart_meta = _extract_charts(raw_stdout)
        stdout = "\n".join(clean_stdout)
        stderr = "\n".join(stderr_lines)

        error_msg: Optional[str] = None
        if execution.error is not None:
            name = getattr(execution.error, "name", "ExecutionError")
            value = getattr(execution.error, "value", str(execution.error))
            error_msg = f"{name}: {value}"
            logger.warning("CSV sandbox execution error: %s: %s", name, value[:300])

        elapsed_ms = int(time.monotonic() * 1000) - start_ms
        logger.info(
            "CSV sandbox finished in %d ms. charts=%d stderr_bytes=%d",
            elapsed_ms, len(charts), len(stderr),
        )
        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            charts=charts,
            error=error_msg,
            execution_time_ms=elapsed_ms,
            chart_metadata=chart_meta,
        )

    finally:
        if sandbox is not None:
            try:
                sandbox.kill()
            except Exception as teardown_err:
                logger.warning("Failed to close CSV sandbox cleanly: %s", teardown_err)


async def _run_csv_in_sandbox(code: str, csv_path: str, csv_bytes: bytes) -> SandboxResult:
    """Async wrapper around _run_csv_in_sandbox_sync with 30-second timeout."""
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _run_csv_in_sandbox_sync, code, csv_path, csv_bytes),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        logger.error("CSV sandbox timed out after 30 seconds.")
        return SandboxResult(
            stdout="", stderr="", charts=[],
            error="CSV analysis timed out after 30 seconds.",
            execution_time_ms=30000,
        )


async def run_csv_query_on_e2b(
    csv_content_b64: str,
    question: str,
    persona: str,
    table_name: str,
    column_names: List[str],
    chart_type: str = "auto",
    expert_personas: Optional[dict] = None,
) -> AsyncGenerator[str, None]:
    """Process a user question about a CSV file using pandas on E2B.

    Yields SSE-formatted strings matching the SQL pipeline format so the
    frontend needs no changes.
    """
    persona_def = (
        (expert_personas or {})
        .get(persona, (expert_personas or {}).get("Generalist", {}))
        .get("persona_def", "You are a helpful data analyst.")
    )

    csv_path_in_sandbox = f"/tmp/{table_name}.csv"

    # Generate pandas code via LLM
    code = await generate_csv_analysis_code(
        csv_path_in_sandbox=csv_path_in_sandbox,
        column_names=column_names,
        question=question,
        persona_def=persona_def,
        chart_type=chart_type,
    )

    # Fallback code if LLM is unavailable or generation produced invalid syntax.
    # Uses only safe, deterministic pandas operations — no f-strings or
    # dynamic string building that could re-introduce a syntax error.
    if not code:
        code = (
            "import pandas as pd\n"
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            f"df = pd.read_csv('{csv_path_in_sandbox}')\n"
            "print('Shape:', df.shape)\n"
            "print('Columns:', list(df.columns))\n"
            "print()\n"
            "print('--- Summary Statistics ---')\n"
            "print(df.describe(include='all').to_string())\n"
            "print()\n"
            "print('--- First 10 rows ---')\n"
            "print(df.head(10).to_string())\n"
        )

    # Decode CSV and run on E2B
    try:
        csv_bytes = base64.b64decode(csv_content_b64)
    except Exception as exc:
        error_chunk = _stdlib_json.dumps({"type": "error", "error_type": "InternalError", "detail": f"Failed to decode CSV content: {exc}"})
        yield f"data: {error_chunk}\n\n"
        return

    result = await _run_csv_in_sandbox(code, csv_path_in_sandbox, csv_bytes)

    # --- SSE: metadata chunk ---
    cols_summary = ", ".join(column_names[:6])
    if len(column_names) > 6:
        cols_summary += f" (+{len(column_names) - 6} more)"
    meta_chunk = {
        "type": "metadata",
        "sql_query": None,
        "explanation": f"Analyzed {table_name} ({len(csv_bytes) // 1024 or 1} KB) using pandas — columns: {cols_summary}",
        "result_preview": [],
        "row_count": 0,
        "execution_time_ms": result.execution_time_ms,
        "sources": [table_name],
    }
    yield f"data: {_stdlib_json.dumps(meta_chunk)}\n\n"

    # --- SSE: charts ---
    if result.charts:
        yield f"data: {_stdlib_json.dumps({'type': 'analysis_code', 'code': code})}\n\n"
        for idx, chart_b64 in enumerate(result.charts):
            meta = result.chart_metadata[idx] if idx < len(result.chart_metadata) else None
            yield f"data: {_stdlib_json.dumps({'type': 'chart', 'base64': chart_b64, 'index': idx, 'metadata': meta.model_dump() if meta else None})}\n\n"

    # --- SSE: answer tokens (stdout is the pandas output) ---
    answer = result.stdout.strip()
    if result.error and not answer:
        answer = f"Execution error: {result.error}"
    elif result.error:
        answer = f"{answer}\n\n⚠️ {result.error}"

    if answer:
        yield f"data: {_stdlib_json.dumps({'type': 'token', 'content': answer})}\n\n"

    yield f"data: {_stdlib_json.dumps({'type': 'done'})}\n\n"
