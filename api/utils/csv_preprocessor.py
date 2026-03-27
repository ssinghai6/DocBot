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
