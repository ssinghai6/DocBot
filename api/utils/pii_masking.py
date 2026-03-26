"""PII auto-detection and masking — DOCBOT-604.

Detects and masks PII in query result rows before they are passed to the LLM
or returned to the frontend. All detection is regex-based (no LLM calls,
no external dependencies). Adds < 5ms per 500 rows.

Supported PII types:
    - Email addresses         →  j***@example.com
    - US phone numbers        →  ***-***-4567
    - UK phone numbers        →  ***-***-4567  (+44 7xxx xxxxxx, 07xxx xxxxxx)
    - India phone numbers     →  ***-***-4567  (+91 xxxxx xxxxx)
    - EU generic phones       →  ***-***-4567  (+3x/+4x prefix, 9-12 digits)
    - US Social Security Nos  →  ***-**-1234
    - Credit card numbers     →  ****-****-****-1234

Usage:
    from api.utils.pii_masking import mask_rows, detect_pii_summary, mask_pii, mask_pii_dataframe_output

    safe_rows = mask_rows(result_dicts)         # returns new list, never mutates input
    safe_text = mask_pii(answer_text)           # mask PII in a plain string
    safe_output = mask_pii_dataframe_output(t)  # mask PII in tabular/CSV-like E2B output
"""

from __future__ import annotations

import copy
import re
from typing import Any

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

_PHONE_RE = re.compile(
    r"""
    (?<!\d)                        # not preceded by digit
    (?:\+?1[\s\-.]?)?              # optional country code +1
    (?:\(?\d{3}\)?[\s\-.]?)        # area code (with optional parens)
    \d{3}                          # exchange
    [\s\-.]?                       # separator
    \d{4}                          # subscriber
    (?!\d)                         # not followed by digit
    """,
    re.VERBOSE,
)

# UK: +44 7xxx xxxxxx  or  07xxx xxxxxx  (mobile pattern)
_PHONE_UK_RE = re.compile(
    r"""
    (?<!\d)
    (?:
        \+44[\s\-.]?7\d{3}[\s\-.]?\d{6}   # +44 7xxx xxxxxx
        |
        07\d{3}[\s\-.]?\d{6}               # 07xxx xxxxxx
    )
    (?!\d)
    """,
    re.VERBOSE,
)

# India: +91 followed by 10 digits (spaces or dashes as separators)
_PHONE_IN_RE = re.compile(
    r"""
    (?<!\d)
    \+91[\s\-]?\d{5}[\s\-]?\d{5}
    (?!\d)
    """,
    re.VERBOSE,
)

# EU generic: +3x or +4x country codes (excluding +44 handled above) with 9-12 digits total
_PHONE_EU_RE = re.compile(
    r"""
    (?<!\d)
    \+(?:3[0-9]|4[0-35-9])  # +3x or +4x but NOT +44
    [\s\-.]?
    \d[\d\s\-.]{7,11}       # 9-12 digits with optional separators
    (?!\d)
    """,
    re.VERBOSE,
)

_SSN_RE = re.compile(
    r"""
    (?<!\d)
    (?!000|666|9\d{2})             # invalid area numbers
    \d{3}
    [-\s]
    (?!00)
    \d{2}
    [-\s]
    (?!0000)
    \d{4}
    (?!\d)
    """,
    re.VERBOSE,
)

# Matches 13–19 digit sequences (with optional spaces/dashes between groups)
_CC_RE = re.compile(
    r"""
    (?<!\d)
    (?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6011|(?:2131|1800|35\d{3}))  # BIN prefix
    [\s\-]?
    \d{4}
    [\s\-]?
    \d{4}
    [\s\-]?
    \d{1,4}
    (?!\d)
    """,
    re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------


def _mask_email(m: re.Match) -> str:
    raw = m.group()
    local, domain = raw.split("@", 1)
    masked_local = local[0] + "***" if len(local) > 1 else "***"
    return f"{masked_local}@{domain}"


def _mask_phone(m: re.Match) -> str:
    # Keep only last 4 digits
    digits = re.sub(r"\D", "", m.group())
    return f"***-***-{digits[-4:]}"


def _mask_ssn(m: re.Match) -> str:
    raw = m.group()
    sep = "-" if "-" in raw else " "
    digits = re.sub(r"\D", "", raw)
    return f"***{sep}**{sep}{digits[-4:]}"


def _mask_cc(m: re.Match) -> str:
    digits = re.sub(r"\D", "", m.group())
    return f"****-****-****-{digits[-4:]}"


def _mask_string(value: str) -> str:
    """Apply all PII masks to a single string value."""
    value = _CC_RE.sub(_mask_cc, value)
    value = _SSN_RE.sub(_mask_ssn, value)
    # International phone patterns applied before US to avoid partial overlap
    value = _PHONE_UK_RE.sub(_mask_phone, value)
    value = _PHONE_IN_RE.sub(_mask_phone, value)
    value = _PHONE_EU_RE.sub(_mask_phone, value)
    value = _PHONE_RE.sub(_mask_phone, value)
    value = _EMAIL_RE.sub(_mask_email, value)
    return value


def _mask_value(value: Any) -> Any:
    """Mask PII in any scalar value. Non-strings are returned unchanged."""
    if isinstance(value, str):
        return _mask_string(value)
    return value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mask_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a new list of rows with PII masked in all string values.

    The original list and dicts are never mutated.
    Non-string values (int, float, None, bool) pass through unchanged.
    """
    if not rows:
        return rows
    masked = []
    for row in rows:
        masked.append({k: _mask_value(v) for k, v in row.items()})
    return masked


def mask_pii(text: str) -> str:
    """Mask PII in a plain text string (LLM answer text, log lines, etc.).

    Applies all PII patterns: credit cards, SSNs, phone numbers (US, UK,
    India, EU), and email addresses.

    Returns the masked string. Input is never mutated.
    """
    return _mask_string(text)


def mask_pii_dataframe_output(text: str) -> str:
    """Mask PII in tabular / CSV-like text produced by E2B pandas sandboxes.

    Handles the common case where emails and phone numbers appear as values
    in pipe- or comma-separated table output, e.g.::

        name       | email              | phone
        Alice      | alice@example.com  | +44 7700 900123
        Bob        | bob@corp.io        | 07912 345678

    Delegates to ``mask_pii()`` after normalising multi-space padding so that
    values embedded in columnar whitespace are still matched by the regexes.
    """
    # Collapse runs of spaces around pipe separators so regex word boundaries
    # behave consistently for values padded for column alignment.
    normalised = re.sub(r" {2,}", " ", text)
    return mask_pii(normalised)


def detect_pii_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Scan rows and return counts of each PII type found (before masking).

    Useful for logging / audit purposes.
    Returns: {"email": N, "phone": N, "ssn": N, "credit_card": N}
    """
    counts: dict[str, int] = {"email": 0, "phone": 0, "ssn": 0, "credit_card": 0}
    for row in rows:
        for value in row.values():
            if not isinstance(value, str):
                continue
            counts["email"] += len(_EMAIL_RE.findall(value))
            counts["phone"] += len(_PHONE_RE.findall(value))
            counts["ssn"] += len(_SSN_RE.findall(value))
            counts["credit_card"] += len(_CC_RE.findall(value))
    return counts
