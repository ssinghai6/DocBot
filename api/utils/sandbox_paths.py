"""Filesystem paths inside the remote E2B sandbox VM (DOCBOT-1633).

These paths never touch the API host: CSV bytes are uploaded with
``sandbox.files.write(path, ...)`` into a fresh, single-use E2B VM and read
back by generated pandas code running in that same VM. Host-side temporary
files must use :mod:`tempfile` instead.
"""

from __future__ import annotations

import posixpath
import re

# Scratch directory inside the isolated, per-request E2B VM (not the API host,
# so tempfile/gettempdir would point at the wrong machine).
SANDBOX_TMP_DIR = "/tmp"  # remote single-use E2B VM path, not host tmp  # nosec B108

_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
_FALLBACK_NAME = "data"
_MAX_NAME_LEN = 100


def _safe_basename(name: str) -> str:
    cleaned = _UNSAFE_NAME_CHARS.sub("_", name or "").strip("._")[:_MAX_NAME_LEN]
    return cleaned or _FALLBACK_NAME


def sandbox_csv_path(table_name: str) -> str:
    """Return the in-sandbox path for a CSV table's bytes.

    The name is reduced to ``[A-Za-z0-9_.-]`` so it cannot traverse out of
    ``SANDBOX_TMP_DIR`` or break out of the ``pd.read_csv('<path>')`` string
    literal embedded in generated sandbox code.
    """
    return posixpath.join(SANDBOX_TMP_DIR, f"{_safe_basename(table_name)}.csv")
