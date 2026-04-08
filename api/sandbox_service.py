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
import re as _re_module
import time
from typing import AsyncGenerator, List, Optional

import groq as groq_module
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# DOCBOT-305: supported chart types for validate + prompt routing
VALID_CHART_TYPES = {"auto", "bar", "line", "scatter", "heatmap", "box", "multi"}

# Regex to detect complex analytical queries that need more code/time
_COMPLEX_QUERY_RE = _re_module.compile(
    r'\b(predict|forecast|trend|seasonalit|correlat|regress|cluster|anomal'
    r'|outli|model|classif|time.?series|arima|prophet|xgboost|machine.?learn)\b',
    _re_module.IGNORECASE,
)

_SANDBOX_EXTRA_PACKAGES = [
    "statsmodels",
    "scikit-learn",
    "scipy",
    "seaborn",
    "plotly",
    "xgboost",
    "lightgbm",
]

# Map top-level import module name -> pip package name.
# Core packages (pandas, numpy, matplotlib) are pre-installed in E2B's
# code-interpreter image, so they're intentionally omitted here.
_IMPORT_TO_PACKAGE: dict[str, str] = {
    "statsmodels": "statsmodels",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "seaborn": "seaborn",
    "plotly": "plotly",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
}

# Match both "import foo" and "from foo import bar", including dotted forms
# like "import sklearn.linear_model" (we capture just the top-level name).
_IMPORT_RE = _re_module.compile(
    r"^\s*(?:from|import)\s+([a-zA-Z_][\w]*)",
    _re_module.MULTILINE,
)


def _detect_needed_packages(code: str) -> list[str]:
    """Scan generated code for imports and return the list of non-default
    packages that must be pip-installed in the sandbox.

    Returns a stable, de-duplicated list preserving first-seen order. Core
    packages already baked into the E2B image (pandas, numpy, matplotlib)
    are never returned.
    """
    if not code:
        return []
    imports = _IMPORT_RE.findall(code)
    needed: list[str] = []
    seen: set[str] = set()
    for imp in imports:
        pkg = _IMPORT_TO_PACKAGE.get(imp)
        if pkg and pkg not in seen:
            seen.add(pkg)
            needed.append(pkg)
    return needed


def _install_sandbox_packages(sandbox, packages: Optional[list[str]] = None) -> None:
    """Install only the specified extra packages in the E2B sandbox.

    No-op when ``packages`` is empty or None — this is the common case for
    simple pandas/matplotlib analyses and lets the sandbox boot in ~2s
    instead of the 20-40s it took to eagerly install the full ML stack.
    """
    if not packages:
        return
    quoted = ", ".join(repr(p) for p in packages)
    install_result = sandbox.run_code(
        f"import subprocess; subprocess.check_call(['pip', 'install', '-q', {quoted}])"
    )
    if install_result.error is not None:
        name = getattr(install_result.error, "name", "InstallError")
        value = getattr(install_result.error, "value", str(install_result.error))
        logger.warning("Sandbox package install warning: %s: %s", name, value[:300])

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


_MODULE_NOT_FOUND_RE = _re_module.compile(
    r"No module named ['\"]([a-zA-Z0-9_]+)['\"]"
)

_KNOWN_IMPORT_TO_PYPI = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "bs4": "beautifulsoup4",
    "yaml": "pyyaml",
    "attr": "attrs",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "gi": "PyGObject",
    "lxml": "lxml",
}


def _parse_missing_module(error_str: str) -> Optional[str]:
    match = _MODULE_NOT_FOUND_RE.search(error_str)
    if not match:
        return None
    module_name = match.group(1)
    return _KNOWN_IMPORT_TO_PYPI.get(module_name, module_name)


def _pip_install_in_sandbox(sandbox, package_name: str) -> bool:
    install_result = sandbox.run_code(
        f"import subprocess; subprocess.check_call(['pip', 'install', '-q', {package_name!r}])"
    )
    if install_result.error is not None:
        name = getattr(install_result.error, "name", "InstallError")
        value = getattr(install_result.error, "value", str(install_result.error))
        logger.warning("Sandbox pip install %s failed: %s: %s", package_name, name, value[:300])
        return False
    logger.info("Sandbox pip install %s succeeded", package_name)
    return True


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

    # Determine which heavy packages (if any) the generated code actually
    # imports. Doing this BEFORE sandbox creation lets us skip the pip
    # install entirely for the common pandas/matplotlib-only case.
    needed_packages = _detect_needed_packages(code)

    try:
        sandbox = Sandbox.create()
        _install_sandbox_packages(sandbox, needed_packages)
        # Prepend preamble + append suffix so all figures are captured
        full_code = _MATPLOTLIB_PREAMBLE + code + _MATPLOTLIB_SUFFIX
        execution = sandbox.run_code(full_code)

        if (
            execution.error is not None
            and getattr(execution.error, "name", "") == "ModuleNotFoundError"
        ):
            pkg = _parse_missing_module(
                getattr(execution.error, "value", str(execution.error))
            )
            if pkg and _pip_install_in_sandbox(sandbox, pkg):
                logger.info("Retrying sandbox execution after installing %s", pkg)
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
    the result set is empty or on any error.

    Small result sets (1–4 rows) are perfectly valid for bar/pie charts
    (e.g. "revenue by quarter" with 3 quarters) — the previous 5-row gate
    was too conservative and killed legitimate visualisations.

    Parameters
    ----------
    chart_type : one of VALID_CHART_TYPES — controls which chart the LLM produces.
                 "auto" lets the LLM choose the best chart for the data.
    """
    if len(result_dicts) < 1:
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
        "1. Imports pandas, numpy, json, matplotlib.pyplot as plt, and seaborn as sns if needed\n"
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
        from api.utils.llm_provider import chat_completion, GROQ_CODE_MODEL

        code = chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            model=GROQ_CODE_MODEL,
            temperature=0,
            max_tokens=1500,
        )

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


def _inspect_csv_profile(csv_bytes: bytes) -> tuple:
    """Extract dtypes and sample rows from CSV bytes for LLM context.

    Handles multi-table CSVs (e.g. Excel exports with section headers as rows)
    by detecting when real headers are in a data row rather than the CSV header.

    Returns (dtypes_info, sample_rows) as formatted strings, or (None, None)
    on error.
    """
    import io
    import re

    import pandas as pd

    try:
        df = None
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(csv_bytes), encoding=encoding, on_bad_lines="skip", nrows=20)
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue

        if df is None or df.empty:
            return None, None

        # Normalize column names
        df.columns = [
            re.sub(r"[^a-z0-9_]", "_", c.strip().lower().replace(" ", "_"))
            for c in df.columns
        ]
        df = df.dropna(how="all").dropna(axis=1, how="all")

        # Detect multi-table CSV: if most columns are "unnamed__*", the real
        # headers are likely in the first non-empty data row.
        unnamed_count = sum(1 for c in df.columns if c.startswith("unnamed_"))
        is_multi_table = unnamed_count > len(df.columns) * 0.5

        profile_parts = []
        if is_multi_table:
            profile_parts.append(
                "WARNING: This CSV appears to be a multi-section Excel export. "
                "The first row of the CSV is a section title, not a header. "
                "The real column headers are likely in row index 0 or 1 of the DataFrame. "
                "You MUST detect the actual header row and use df.columns = df.iloc[header_row] "
                "then df = df.iloc[header_row+1:].reset_index(drop=True) before analysis."
            )
            # Show raw first 5 rows so LLM can find the real headers
            profile_parts.append(f"\nRaw first 5 rows (inspect to find real headers):\n{df.head(5).to_string(max_colwidth=50)}")

            # Also scan for section markers (EXHIBIT, Sheet, Table)
            first_col = df.iloc[:, 0].astype(str)
            sections = []
            for idx, val in first_col.items():
                if any(kw in val.upper() for kw in ("EXHIBIT", "SHEET", "TABLE")):
                    sections.append(f"  Row {idx}: {val[:80]}")
            if sections:
                profile_parts.append(f"\nDetected section headers:\n" + "\n".join(sections))
        else:
            profile_parts.append(f"Column dtypes:\n{df.dtypes.to_string()}")
            profile_parts.append(f"\nFirst 3 rows:\n{df.head(3).to_string(max_colwidth=40)}")

        dtypes_info = "\n".join(profile_parts)
        # For multi-table CSVs, sample_rows is redundant (already in dtypes_info)
        sample_rows = None if is_multi_table else df.head(3).to_string(max_colwidth=40)

        if len(dtypes_info) > 2500:
            dtypes_info = dtypes_info[:2500] + "\n... (truncated)"
        if sample_rows and len(sample_rows) > 2000:
            sample_rows = sample_rows[:2000] + "\n... (truncated)"

        return dtypes_info, sample_rows

    except Exception as exc:
        logger.warning("_inspect_csv_profile failed: %s", exc)
        return None, None


def _rephrase_for_csv(
    question: str,
    chat_history: list[dict],
) -> tuple[str, str]:
    """Rephrase a follow-up question into a standalone query using chat history.

    Returns (rephrased_question, previous_context_summary). Falls back to the
    original question on any error.
    """
    try:
        from api.utils.llm_provider import chat_completion, GROQ_CODE_MODEL

        # Build a compact history summary (last 4 messages max)
        recent = chat_history[-4:]
        history_text = "\n".join(
            f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')[:200]}"
            for m in recent
        )

        rephrase_prompt = (
            "Given this chat history and the latest question, rewrite the question "
            "as a standalone data analysis question. Return ONLY the rephrased question.\n\n"
            f"Chat history:\n{history_text}\n\n"
            f"Latest question: {question}"
        )
        rephrased = chat_completion(
            [{"role": "user", "content": rephrase_prompt}],
            model=GROQ_CODE_MODEL,
            temperature=0,
            max_tokens=200,
        ).strip()

        # Strip <think>...</think> blocks from Qwen 3
        import re as _re
        rephrased = _re.sub(r"<think>.*?</think>", "", rephrased, flags=_re.DOTALL).strip()

        # Extract last assistant message for context
        previous_context = ""
        for m in reversed(chat_history):
            if m.get("role") == "assistant":
                previous_context = f"Previous analysis result: {m['content'][:300]}"
                break

        logger.info("CSV rephrase: '%s' -> '%s'", question[:80], rephrased[:80])
        return rephrased or question, previous_context

    except Exception as exc:
        logger.warning("_rephrase_for_csv failed: %s, using original question", exc)
        return question, ""


async def generate_csv_analysis_code(
    csv_path_in_sandbox: str,
    column_names: List[str],
    question: str,
    persona_def: str,
    chart_type: str = "auto",
    section_manifest: Optional[str] = None,
    is_multi_section: bool = False,
    data_profile: Optional[str] = None,
    chat_history: Optional[list[dict]] = None,
    error_context: Optional[str] = None,
) -> Optional[str]:
    """Generate Python/pandas code to answer a question about a CSV file on E2B.

    Returns a Python code string, or None on error / missing API key.

    Parameters
    ----------
    data_profile : optional serialized DataProfile string for LLM context
    chat_history : optional list of {role, content} dicts for conversational memory
    error_context : optional error message from a failed previous attempt (retry)
    """
    api_key = os.getenv("groq_api_key")
    if not api_key:
        logger.warning("generate_csv_analysis_code: groq_api_key not set, using fallback")
        return None

    # Detect complex queries for adaptive limits
    is_complex = bool(_COMPLEX_QUERY_RE.search(question))
    max_code_lines = 150 if is_complex else 70
    code_gen_max_tokens = 4000 if is_complex else 2000

    # --- Conversational rephrase (Task 4) ---
    effective_question = question
    previous_context = ""
    if chat_history:
        effective_question, previous_context = _rephrase_for_csv(
            question, chat_history
        )

    chart_instructions = _chart_type_instructions(chart_type)

    # Base system prompt
    system_prompt = (
        "You are a Python data analyst. A CSV file is pre-loaded as `df` by a preamble.\n\n"
        "PREAMBLE (already done — DO NOT repeat):\n"
        "- `import pandas as pd`, `import re`, `import numpy as np`\n"
        "- CSV loaded and column names normalized to lowercase_underscores\n"
        "- Empty rows/columns dropped, numeric columns auto-converted\n"
    )

    if is_multi_section:
        system_prompt += (
            "\nMULTI-SECTION CSV: This file contains multiple data sections (exhibits/sheets).\n"
            "A helper `load_section(idx)` is available. `df` defaults to section 0.\n"
            "- Call `df = load_section(N)` to switch sections. `_SECTIONS` dict has metadata.\n"
            "- For cross-section queries: load each section separately, then merge/compare.\n"
            "- DO NOT use pd.read_csv(). DO NOT reference _RAW directly.\n\n"
        )
    else:
        system_prompt += "- `df` is ready to use directly.\n\n"

    system_prompt += (
        "YOUR CODE RULES:\n"
        "1. Do NOT import pandas, numpy, or json — already imported in preamble\n"
        "2. Import matplotlib.pyplot as plt, seaborn as sns if needed for charts\n"
        "3. Output ONLY raw Python code — no markdown fences, no explanations\n"
        "4. ALWAYS .dropna() or .fillna(0) before boolean masks or aggregation\n\n"
        "DATA CLEANING PATTERNS (use as needed):\n"
        "- Currency: col.str.replace(r'[$,]', '', regex=True).pipe(pd.to_numeric, errors='coerce')\n"
        "- Percentages: col.str.rstrip('%').pipe(pd.to_numeric, errors='coerce').div(100)\n"
        "- Dates: pd.to_datetime(col, format='mixed', dayfirst=False)\n\n"
        f"CHARTING: {chart_instructions}\n"
        "- ONLY skip charting if the user explicitly asks about column names, data types, or table structure "
        "(e.g. 'what columns exist', 'show schema', 'list fields'). "
        "For ALL analytical, forecasting, comparison, trend, or statistical questions: ALWAYS generate a chart.\n"
        "- Otherwise call plt.show() once, then print chart metadata:\n"
        "  import json; print('CHART_META:' + json.dumps({'type': '<type>', "
        "'title': '<title>', 'x_label': '<x>', 'y_label': '<y>', 'series_count': <int>}))\n\n"
        "Print a brief human-readable narrative summary of findings (2-3 sentences). "
        "Do NOT print raw DataFrame output — if showing tabular data, use df.to_markdown(). "
        f"Keep code under {max_code_lines} lines."
    )

    # Build user message with data profile, section manifest, and column list
    user_parts: list[str] = []

    if data_profile:
        user_parts.append(f"DATA PROFILE:\n{data_profile}")

    if previous_context:
        user_parts.append(f"PREVIOUS ANALYSIS CONTEXT:\n{previous_context}")

    if is_multi_section and section_manifest:
        user_parts.append(
            "MULTI-SECTION CSV — use load_section(idx) to access the right section.\n\n"
            f"SECTION MANIFEST:\n{section_manifest}"
        )
    else:
        cols_preview = ", ".join(column_names[:20])
        if len(column_names) > 20:
            cols_preview += f" ... (+{len(column_names) - 20} more)"
        user_parts.append(f"CSV columns: {cols_preview}")

    user_parts.append(f"Question: {effective_question}")

    if error_context:
        user_parts.append(error_context)

    user_message = "\n\n".join(user_parts)

    try:
        from api.utils.llm_provider import chat_completion, GROQ_CODE_MODEL

        response_text = chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            model=GROQ_CODE_MODEL,
            temperature=0,
            max_tokens=code_gen_max_tokens,
        )
        code = response_text

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

    # Conditionally install only the packages the generated code imports.
    # Empty list → no install at all → sandbox boots ~20x faster.
    needed_packages = _detect_needed_packages(code)

    try:
        sandbox = Sandbox.create()
        _install_sandbox_packages(sandbox, needed_packages)
        # Upload CSV bytes to the sandbox filesystem
        sandbox.files.write(csv_path, csv_bytes)

        full_code = _MATPLOTLIB_PREAMBLE + code + _MATPLOTLIB_SUFFIX
        execution = sandbox.run_code(full_code)

        if (
            execution.error is not None
            and getattr(execution.error, "name", "") == "ModuleNotFoundError"
        ):
            pkg = _parse_missing_module(
                getattr(execution.error, "value", str(execution.error))
            )
            if pkg and _pip_install_in_sandbox(sandbox, pkg):
                logger.info("Retrying CSV sandbox execution after installing %s", pkg)
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


async def _run_csv_in_sandbox(
    code: str,
    csv_path: str,
    csv_bytes: bytes,
    timeout_seconds: int = 30,
) -> SandboxResult:
    """Async wrapper around _run_csv_in_sandbox_sync with configurable timeout."""
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _run_csv_in_sandbox_sync, code, csv_path, csv_bytes),
            timeout=float(timeout_seconds),
        )
    except asyncio.TimeoutError:
        logger.error("CSV sandbox timed out after %d seconds.", timeout_seconds)
        return SandboxResult(
            stdout="", stderr="", charts=[],
            error=f"CSV analysis timed out after {timeout_seconds} seconds.",
            execution_time_ms=timeout_seconds * 1000,
        )


def _format_stdout_as_markdown(text: str) -> str:
    """Best-effort formatting of pandas stdout output for markdown rendering.

    Handles:
    - Markdown pipe tables (pass-through)
    - Section headers (--- Something ---)
    - Raw pandas .to_string() / .describe() output → wrapped in code blocks
    - CHART_META: lines (strip from visible output)
    """
    if not text:
        return text

    # Strip CHART_META lines — they're machine-readable, not for humans
    lines = [l for l in text.split('\n') if not l.strip().startswith('CHART_META:')]

    result_lines = []
    in_raw_table = False
    raw_table_lines: list[str] = []

    def _flush_raw_table():
        """Convert accumulated raw table lines to a code block."""
        nonlocal raw_table_lines
        if raw_table_lines:
            result_lines.append('\n```')
            result_lines.extend(raw_table_lines)
            result_lines.append('```\n')
            raw_table_lines = []

    def _looks_like_raw_table(line: str) -> bool:
        """Detect pandas .to_string() output: multiple whitespace-separated columns."""
        stripped = line.strip()
        if not stripped:
            return False
        # Lines with 3+ whitespace-separated tokens and at least 2 multi-space gaps
        parts = stripped.split()
        gaps = len([1 for i in range(len(stripped) - 1) if stripped[i:i+2] == '  '])
        return len(parts) >= 3 and gaps >= 2

    for line in lines:
        # Already markdown formatted (has pipes) — keep as-is
        if '|' in line and not in_raw_table:
            _flush_raw_table()
            result_lines.append(line)
            continue

        # Section headers (--- Something ---) — convert to bold
        if line.strip().startswith('---') and line.strip().endswith('---'):
            _flush_raw_table()
            header = line.strip().strip('-').strip()
            if header:
                result_lines.append(f'\n**{header}**\n')
                continue

        # Bold section headers like **Summary Statistics**
        if line.strip().startswith('**') and line.strip().endswith('**'):
            _flush_raw_table()
            result_lines.append(line)
            in_raw_table = False
            continue

        # Detect raw table output (pandas .to_string())
        if _looks_like_raw_table(line):
            in_raw_table = True
            raw_table_lines.append(line)
            continue

        # Non-table line
        if in_raw_table:
            # Blank line might be separator between tables
            if not line.strip():
                raw_table_lines.append(line)
                continue
            _flush_raw_table()
            in_raw_table = False

        result_lines.append(line)

    _flush_raw_table()
    return '\n'.join(result_lines)


async def run_csv_query_on_e2b(
    csv_content_b64: str,
    question: str,
    persona: str,
    table_name: str,
    column_names: List[str],
    chart_type: str = "auto",
    expert_personas: Optional[dict] = None,
    sections: Optional[List[dict]] = None,
    section_manifest: Optional[str] = None,
    data_profile_str: Optional[str] = None,
    chat_history: Optional[list[dict]] = None,
) -> AsyncGenerator[str, None]:
    """Process a user question about a CSV file using pandas on E2B.

    Yields SSE-formatted strings matching the SQL pipeline format so the
    frontend needs no changes.  Supports multi-section CSVs via
    csv_preprocessor section metadata.

    Parameters
    ----------
    data_profile_str : optional serialized DataProfile for LLM context enrichment
    chat_history : optional list of {role, content} dicts for conversational memory
    """
    persona_def = (
        (expert_personas or {})
        .get(persona, (expert_personas or {}).get("Generalist", {}))
        .get("persona_def", "You are a helpful data analyst.")
    )

    csv_path_in_sandbox = f"/tmp/{table_name}.csv"

    # Detect complex queries for adaptive timeout
    is_complex = bool(_COMPLEX_QUERY_RE.search(question))
    sandbox_timeout = 60 if is_complex else 30

    # Build section-aware preamble if section metadata is available
    is_multi_section = sections is not None and len(sections) > 1
    if sections:
        from api.utils.csv_preprocessor import dicts_to_sections, generate_section_preamble
        csv_sections = dicts_to_sections(sections)
        preamble = generate_section_preamble(csv_sections, csv_path_in_sandbox)
    else:
        # Fallback: inspect CSV on-the-fly (for connections created before this update)
        try:
            csv_bytes_for_profile = base64.b64decode(csv_content_b64)
        except Exception:
            csv_bytes_for_profile = b""
        dtypes_info, sample_rows = await asyncio.get_running_loop().run_in_executor(
            None, _inspect_csv_profile, csv_bytes_for_profile,
        )
        preamble = (
            "import pandas as pd\n"
            "import re as _re\n"
            "import numpy as np\n"
            "\n"
            f"df = pd.read_csv('{csv_path_in_sandbox}')\n"
            "df.columns = [_re.sub(r'[^a-z0-9_]', '_', str(c).strip().lower().replace(' ', '_')) for c in df.columns]\n"
            "df = df.dropna(how='all').dropna(axis=1, how='all')\n"
            "for _c in df.columns:\n"
            "    _v = pd.to_numeric(df[_c], errors='coerce')\n"
            "    if _v.notna().sum() / max(df[_c].notna().sum(), 1) > 0.5:\n"
            "        df[_c] = _v\n"
            "# --- end preamble ---\n\n"
        )

    # Generate pandas code via LLM
    code = await generate_csv_analysis_code(
        csv_path_in_sandbox=csv_path_in_sandbox,
        column_names=column_names,
        question=question,
        persona_def=persona_def,
        chart_type=chart_type,
        section_manifest=section_manifest,
        is_multi_section=is_multi_section,
        data_profile=data_profile_str,
        chat_history=chat_history,
    )

    # Fallback code if LLM is unavailable or generation produced invalid syntax.
    if not code:
        code = (
            preamble
            + "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "print('**Dataset Overview**')\n"
            "print(f'Shape: {df.shape[0]} rows x {df.shape[1]} columns')\n"
            "print(f'Columns: {\", \".join(df.columns.tolist())}')\n"
            "print()\n"
            "numeric_cols = df.select_dtypes(include='number').columns.tolist()\n"
            "date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]\n"
            "print(f'Numeric columns: {\", \".join(numeric_cols) if numeric_cols else \"None\"}')\n"
            "print(f'Date columns: {\", \".join(date_cols) if date_cols else \"None\"}')\n"
            "print()\n"
            "if numeric_cols:\n"
            "    print('**Summary Statistics**')\n"
            "    try:\n"
            "        print(df[numeric_cols].describe().to_markdown())\n"
            "    except Exception:\n"
            "        print(df[numeric_cols].describe().to_string())\n"
            "    print()\n"
            "print('**Sample Data (first 5 rows)**')\n"
            "try:\n"
            "    print(df.head(5).to_markdown(index=False))\n"
            "except Exception:\n"
            "    print(df.head(5).to_string(index=False))\n"
            "\n"
            "# Auto-chart: line plot for time series, histogram otherwise\n"
            "if numeric_cols:\n"
            "    col = numeric_cols[0]\n"
            "    fig, ax = plt.subplots(figsize=(10, 6))\n"
            "    if date_cols:\n"
            "        _dates = pd.to_datetime(df[date_cols[0]], errors='coerce')\n"
            "        ax.plot(_dates, df[col], color='#667eea', linewidth=1.5)\n"
            "        ax.set_title(f'{col} over time', fontsize=14)\n"
            "        ax.set_xlabel(date_cols[0])\n"
            "        fig.autofmt_xdate()\n"
            "    else:\n"
            "        df[col].dropna().hist(bins=30, ax=ax, color='#667eea', edgecolor='white')\n"
            "        ax.set_title(f'Distribution of {col}', fontsize=14)\n"
            "        ax.set_xlabel(col)\n"
            "    ax.set_ylabel(col)\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
            "    _chart_type = 'line' if date_cols else 'histogram'\n"
            "    _chart_title = f'{col} over time' if date_cols else f'Distribution of {col}'\n"
            "    print('CHART_META:' + json.dumps({'type': _chart_type, 'title': _chart_title, "
            "'x_label': date_cols[0] if date_cols else col, 'y_label': col, 'series_count': 1}))\n"
        )
    else:
        # Prepend preamble to LLM-generated code, stripping any duplicate
        # pd.read_csv() / import pandas the LLM may have included
        import re as _re
        code = _re.sub(
            r"^.*pd\.read_csv\s*\(.*\).*$",
            "# (read_csv handled by preamble)",
            code,
            count=1,
            flags=_re.MULTILINE,
        )
        code = _re.sub(r"^import pandas as pd\s*$", "", code, count=1, flags=_re.MULTILINE)
        code = preamble + code

    # Decode CSV and run on E2B
    try:
        csv_bytes = base64.b64decode(csv_content_b64)
    except Exception as exc:
        error_chunk = _stdlib_json.dumps({"type": "error", "error_type": "InternalError", "detail": f"Failed to decode CSV content: {exc}"})
        yield f"data: {error_chunk}\n\n"
        return

    result = await _run_csv_in_sandbox(code, csv_path_in_sandbox, csv_bytes, timeout_seconds=sandbox_timeout)

    # --- Task 3: Error retry with feedback ---
    _is_timeout = result.error and "timed out" in result.error.lower()
    if result.error and not _is_timeout:
        logger.info("CSV sandbox failed, attempting one retry with error feedback")
        yield f"data: {_stdlib_json.dumps({'type': 'status', 'content': 'Retrying with corrected code...'})}\n\n"

        error_feedback = (
            f"PREVIOUS ATTEMPT FAILED with error:\n"
            f"{result.error}\n"
            f"{result.stderr[:500] if result.stderr else ''}\n\n"
            "Fix the code and try again. Common fixes:\n"
            "- Wrong column name -> check the column list above\n"
            "- Date parsing -> try pd.to_datetime(col, format='mixed', errors='coerce')\n"
            "- Type error -> ensure numeric conversion before math operations"
        )
        retry_code = await generate_csv_analysis_code(
            csv_path_in_sandbox=csv_path_in_sandbox,
            column_names=column_names,
            question=question,
            persona_def=persona_def,
            chart_type=chart_type,
            section_manifest=section_manifest,
            is_multi_section=is_multi_section,
            data_profile=data_profile_str,
            chat_history=chat_history,
            error_context=error_feedback,
        )
        if retry_code:
            # Prepend preamble to retry code same as original
            import re as _re
            retry_code = _re.sub(
                r"^.*pd\.read_csv\s*\(.*\).*$",
                "# (read_csv handled by preamble)",
                retry_code,
                count=1,
                flags=_re.MULTILINE,
            )
            retry_code = _re.sub(r"^import pandas as pd\s*$", "", retry_code, count=1, flags=_re.MULTILINE)
            retry_code = preamble + retry_code

            retry_result = await _run_csv_in_sandbox(
                retry_code, csv_path_in_sandbox, csv_bytes, timeout_seconds=sandbox_timeout
            )
            if retry_result.error is None:
                logger.info("CSV retry succeeded")
                result = retry_result
                code = retry_code
            else:
                logger.warning("CSV retry also failed: %s, using original error", retry_result.error[:200])
                # Keep original result (don't show retry error)

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
    answer = _format_stdout_as_markdown(result.stdout.strip())
    if result.error and not answer:
        answer = f"Execution error: {result.error}"
    elif result.error:
        answer = f"{answer}\n\n⚠️ {result.error}"

    if answer:
        yield f"data: {_stdlib_json.dumps({'type': 'token', 'content': answer})}\n\n"

    yield f"data: {_stdlib_json.dumps({'type': 'done'})}\n\n"
