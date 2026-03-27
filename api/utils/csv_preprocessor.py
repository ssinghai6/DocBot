"""CSV section splitter for multi-table Excel exports.

Detects section boundaries (EXHIBIT, SHEET, TABLE markers) in concatenated
CSV files, extracts per-section metadata, and generates E2B sandbox preamble
code that lets the LLM query any section by index.

Single-table CSVs are handled transparently (one section = whole file).
"""

import io
import logging
import re
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_SECTION_KW = re.compile(
    r"^\s*(EXHIBIT|SHEET|TABLE|SECTION|APPENDIX|SCHEDULE|PART)\s*\d*\s*[:\-.\s]",
    re.IGNORECASE,
)

_COL_CLEAN = re.compile(r"[^a-z0-9_]")


@dataclass
class DataProfile:
    """Deterministic profile of a CSV dataset, computed once on upload."""

    row_count: int
    column_count: int
    dtypes: dict[str, str]  # col_name -> dtype string
    sample_rows: str  # first 5 rows as .to_string()
    numeric_summary: str  # .describe() output for numeric cols
    datetime_columns: list[str]  # columns detected as datetime
    null_percentages: dict[str, float]  # col_name -> null %
    low_cardinality: dict[str, list[str]]  # col_name -> unique values (<=15 unique)
    memory_note: str  # e.g. "1247 rows x 6 columns, daily frequency detected"


@dataclass
class CSVSection:
    """A single logical section within a multi-section CSV file."""

    index: int
    name: str
    title_row_idx: int
    header_row_idx: int
    data_start_idx: int
    data_end_idx: int  # exclusive
    columns: List[str]
    row_count: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clean_col(name: str) -> str:
    return _COL_CLEAN.sub("_", name.strip().lower().replace(" ", "_"))


def _is_boundary(row: pd.Series) -> bool:
    """Return True if *row* looks like a section title.

    Only matches rows with explicit section keywords (EXHIBIT, SHEET, TABLE,
    etc.) where most other cells are empty.  Avoids false positives on
    data rows that happen to start with a capitalized name.
    """
    cell0 = str(row.iloc[0]).strip()
    if not cell0 or cell0.lower() == "nan":
        return False
    if _SECTION_KW.match(cell0):
        # Keyword match — also check most other cells empty
        non_null = row.dropna()
        return len(non_null) <= max(3, len(row) * 0.3)
    return False


def _find_header(raw: pd.DataFrame, start: int, limit: int = 5) -> int:
    """Scan forward from *start* to find the header row.

    A header row has >=40 % non-null cells and at least 2 of the first 5
    non-null values are non-numeric strings.
    """
    for offset in range(limit):
        idx = start + offset
        if idx >= len(raw):
            return start
        row = raw.iloc[idx].dropna()
        if len(row) < max(2, len(raw.columns) * 0.3):
            continue
        vals = [str(v).strip() for v in row.values[:5] if str(v).strip()]
        alpha_count = sum(1 for v in vals if not _is_numeric(v))
        if alpha_count >= 2:
            return idx
    return start


def _is_numeric(s: str) -> bool:
    try:
        float(s.replace(",", "").replace("$", "").replace("%", ""))
        return True
    except ValueError:
        return False


def _extract_columns(raw: pd.DataFrame, header_idx: int) -> List[str]:
    """Clean column names from a header row."""
    row = raw.iloc[header_idx]
    cols = []
    for v in row:
        s = str(v).strip() if pd.notna(v) else ""
        cleaned = _clean_col(s) if s and s.lower() != "nan" else ""
        cols.append(cleaned)
    # Drop trailing empty columns
    while cols and not cols[-1]:
        cols.pop()
    return cols if cols else [f"col_{i}" for i in range(len(row))]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split_csv_sections(csv_bytes: bytes) -> List[CSVSection]:
    """Detect section boundaries in a CSV file.

    Returns a list of CSVSection objects. Guaranteed non-empty: single-section
    fallback for normal CSVs.
    """
    raw = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            raw = pd.read_csv(
                io.BytesIO(csv_bytes),
                header=None,
                dtype=str,
                on_bad_lines="skip",
                encoding=enc,
            )
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    if raw is None or raw.empty:
        return [CSVSection(0, "data", 0, 0, 1, 1, ["col_0"], 0)]

    # Drop fully-empty rows/cols for scanning
    raw = raw.dropna(how="all").reset_index(drop=True)
    raw = raw.dropna(axis=1, how="all")

    # Find boundary rows
    boundaries: List[Tuple[int, str]] = []
    for i in range(len(raw)):
        if _is_boundary(raw.iloc[i]):
            title = str(raw.iloc[i, 0]).strip()[:120]
            boundaries.append((i, title))

    # Single-table fallback
    if len(boundaries) < 1:
        header_idx = _find_header(raw, 0)
        cols = _extract_columns(raw, header_idx)
        n_rows = len(raw) - header_idx - 1
        return [
            CSVSection(
                index=0,
                name="data",
                title_row_idx=0,
                header_row_idx=header_idx,
                data_start_idx=header_idx + 1,
                data_end_idx=len(raw),
                columns=cols,
                row_count=max(n_rows, 0),
            )
        ]

    sections: List[CSVSection] = []
    for i, (title_idx, title) in enumerate(boundaries):
        end_idx = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(raw)
        scan_limit = min(4, end_idx - title_idx - 1)
        header_idx = _find_header(raw, title_idx + 1, limit=max(scan_limit, 1))
        if header_idx >= end_idx:
            header_idx = title_idx + 1 if title_idx + 1 < end_idx else title_idx
        cols = _extract_columns(raw, min(header_idx, len(raw) - 1))
        data_start = header_idx + 1
        n_rows = max(end_idx - data_start, 0)
        sections.append(
            CSVSection(
                index=i,
                name=title,
                title_row_idx=title_idx,
                header_row_idx=header_idx,
                data_start_idx=data_start,
                data_end_idx=end_idx,
                columns=cols,
                row_count=n_rows,
            )
        )

    return sections


def build_section_manifest(sections: List[CSVSection]) -> str:
    """Create a compact text manifest for LLM context."""
    if len(sections) == 1:
        s = sections[0]
        cols = ", ".join(s.columns[:10])
        if len(s.columns) > 10:
            cols += f" (+{len(s.columns) - 10} more)"
        return f"Single-table CSV — columns: {cols} ({s.row_count} rows)"

    lines = []
    for s in sections:
        cols = ", ".join(s.columns[:8])
        if len(s.columns) > 8:
            cols += f" (+{len(s.columns) - 8} more)"
        lines.append(
            f"Section {s.index} (row {s.title_row_idx}): \"{s.name}\"\n"
            f"  Columns: {cols}\n"
            f"  Data rows: {s.data_start_idx}-{s.data_end_idx - 1} ({s.row_count} rows)"
        )

    manifest = "\n\n".join(lines)
    return manifest[:2500] if len(manifest) > 2500 else manifest


def generate_section_preamble(
    sections: List[CSVSection], csv_path: str
) -> str:
    """Generate Python preamble code for E2B sandbox.

    For single-section CSVs: standard cleaning preamble.
    For multi-section: defines ``load_section(idx)`` helper + ``_SECTIONS`` dict.
    """
    if len(sections) == 1:
        s = sections[0]
        return (
            "import pandas as pd\n"
            "import re as _re\n"
            "import numpy as np\n"
            "import json\n"
            "\n"
            f"df = pd.read_csv('{csv_path}', header=None, dtype=str)\n"
            "df = df.dropna(how='all').dropna(axis=1, how='all')\n"
            f"# Promote header row {s.header_row_idx}\n"
            f"df.columns = [_re.sub(r'[^a-z0-9_]', '_', str(v).strip().lower().replace(' ', '_')) for v in df.iloc[{s.header_row_idx}]]\n"
            f"df = df.iloc[{s.data_start_idx}:].reset_index(drop=True)\n"
            "df = df.loc[:, ~df.columns.isin(['nan', 'none', ''])]\n"
            "df = df.dropna(how='all').dropna(axis=1, how='all')\n"
            "for _c in df.columns:\n"
            "    _v = pd.to_numeric(df[_c], errors='coerce')\n"
            "    if _v.notna().sum() / max(df[_c].notna().sum(), 1) > 0.5:\n"
            "        df[_c] = _v\n"
            "# --- end preamble ---\n\n"
        )

    # Multi-section: build _SECTIONS dict and load_section() helper
    sec_entries = []
    for s in sections:
        cols_repr = repr(s.columns)
        sec_entries.append(
            f"    {s.index}: {{'name': {repr(s.name)}, "
            f"'header_row': {s.header_row_idx}, "
            f"'data_start': {s.data_start_idx}, "
            f"'data_end': {s.data_end_idx}, "
            f"'columns': {cols_repr}}}"
        )
    sections_dict = "{\n" + ",\n".join(sec_entries) + "\n}"

    return (
        "import pandas as pd\n"
        "import re as _re\n"
        "import numpy as np\n"
        "import json\n"
        "\n"
        f"_RAW = pd.read_csv('{csv_path}', header=None, dtype=str)\n"
        "_RAW = _RAW.dropna(how='all').dropna(axis=1, how='all')\n"
        "\n"
        f"_SECTIONS = {sections_dict}\n"
        "\n"
        "def load_section(idx: int) -> pd.DataFrame:\n"
        "    '''Load a CSV section by index. Returns cleaned DataFrame.'''\n"
        "    sec = _SECTIONS[idx]\n"
        "    s, e = sec['data_start'], sec['data_end']\n"
        "    chunk = _RAW.iloc[s:e].copy()\n"
        "    # Promote header\n"
        "    hdr = _RAW.iloc[sec['header_row']]\n"
        "    new_cols = [_re.sub(r'[^a-z0-9_]', '_', str(v).strip().lower().replace(' ', '_')) for v in hdr]\n"
        "    # Trim to match chunk width\n"
        "    chunk.columns = new_cols[:len(chunk.columns)] if len(new_cols) >= len(chunk.columns) else new_cols + [f'col_{i}' for i in range(len(chunk.columns) - len(new_cols))]\n"
        "    chunk = chunk.loc[:, ~chunk.columns.isin(['nan', 'none', ''])]\n"
        "    chunk = chunk.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)\n"
        "    for _c in chunk.columns:\n"
        "        _v = pd.to_numeric(chunk[_c], errors='coerce')\n"
        "        if _v.notna().sum() / max(chunk[_c].notna().sum(), 1) > 0.5:\n"
        "            chunk[_c] = _v\n"
        "    return chunk\n"
        "\n"
        "# Default: load first section\n"
        "df = load_section(0)\n"
        "# --- end preamble ---\n\n"
    )


def sections_to_dicts(sections: List[CSVSection]) -> List[dict]:
    """Serialize sections for JSON storage in credentials blob."""
    return [asdict(s) for s in sections]


def dicts_to_sections(data: List[dict]) -> List[CSVSection]:
    """Deserialize sections from credentials blob."""
    return [CSVSection(**d) for d in data]


# ---------------------------------------------------------------------------
# Data Profiling (Task 1)
# ---------------------------------------------------------------------------


def build_data_profile(csv_bytes: bytes) -> Optional[DataProfile]:
    """Build a deterministic DataProfile from raw CSV bytes.

    Computes dtypes, sample rows, numeric summary, datetime detection,
    null percentages, and low-cardinality column values. Returns None on
    any parsing error.
    """
    try:
        df = None
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                df = pd.read_csv(
                    io.BytesIO(csv_bytes), encoding=enc, on_bad_lines="skip"
                )
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue

        if df is None or df.empty:
            return None

        # Normalize column names (same as existing pipeline)
        df.columns = [_clean_col(str(c)) for c in df.columns]
        df = df.dropna(how="all").dropna(axis=1, how="all")

        row_count = len(df)
        column_count = len(df.columns)

        # Dtypes
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Sample rows (first 5, truncated)
        sample_rows = df.head(5).to_string(max_colwidth=60)
        if len(sample_rows) > 2000:
            sample_rows = sample_rows[:2000] + "\n... (truncated)"

        # Numeric summary
        numeric_df = df.select_dtypes(include="number")
        if not numeric_df.empty:
            numeric_summary = numeric_df.describe().to_string()
            if len(numeric_summary) > 1500:
                numeric_summary = numeric_summary[:1500] + "\n... (truncated)"
        else:
            numeric_summary = "No numeric columns."

        # Datetime detection
        datetime_columns: list[str] = []
        freq_notes: list[str] = []
        for col in df.columns:
            is_datetime = False
            # Check by name
            if any(kw in col for kw in ("date", "time", "timestamp")):
                is_datetime = True
            # Check by parsing object columns
            if not is_datetime and df[col].dtype == object:
                parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
                valid_ratio = parsed.notna().sum() / max(df[col].notna().sum(), 1)
                if valid_ratio > 0.7:
                    is_datetime = True
            if is_datetime:
                datetime_columns.append(col)
                # Try to infer frequency
                try:
                    dt_series = pd.to_datetime(df[col], errors="coerce").dropna().sort_values()
                    if len(dt_series) >= 3:
                        freq = pd.infer_freq(dt_series)
                        if freq:
                            freq_notes.append(f"{col}: {freq}")
                except (ValueError, TypeError):
                    pass

        # Null percentages
        null_percentages: dict[str, float] = {}
        for col in df.columns:
            pct = round(df[col].isna().sum() / max(row_count, 1) * 100, 1)
            if pct > 0:
                null_percentages[col] = pct

        # Low cardinality (<=15 unique values)
        low_cardinality: dict[str, list[str]] = {}
        for col in df.columns:
            nunique = df[col].nunique(dropna=True)
            if 0 < nunique <= 15:
                vals = df[col].dropna().unique().tolist()
                low_cardinality[col] = [str(v) for v in vals[:15]]

        # Memory note
        freq_desc = ""
        if freq_notes:
            freq_desc = ", " + ", ".join(freq_notes[:2]) + " frequency detected"
        memory_note = f"{row_count} rows x {column_count} columns{freq_desc}"

        return DataProfile(
            row_count=row_count,
            column_count=column_count,
            dtypes=dtypes,
            sample_rows=sample_rows,
            numeric_summary=numeric_summary,
            datetime_columns=datetime_columns,
            null_percentages=null_percentages,
            low_cardinality=low_cardinality,
            memory_note=memory_note,
        )

    except Exception as exc:
        logger.warning("build_data_profile failed: %s", exc)
        return None


def profile_to_dict(profile: DataProfile) -> dict:
    """Serialize a DataProfile to a JSON-safe dict."""
    return asdict(profile)


def dict_to_profile(data: dict) -> DataProfile:
    """Deserialize a DataProfile from a dict."""
    return DataProfile(**data)
