"""PII auto-detection and masking — DOCBOT-604, hardened in DOCBOT-2003.

Detects and masks PII in query result rows before they are passed to the LLM
or returned to the frontend. All detection is regex-based (no LLM calls,
no external dependencies). Adds < 5ms per 500 rows.

Supported PII types:
    - Email addresses         →  j***@example.com
    - US phone numbers        →  ***-***-4567  (structured, or bare after a phone keyword)
    - UK phone numbers        →  ***-***-4567  (+44 7xxx xxxxxx, 07xxx xxxxxx)
    - India phone numbers     →  ***-***-4567  (+91 xxxxx xxxxx)
    - EU generic phones       →  ***-***-4567  (+3x/+4x prefix, 9-12 digits)
    - US Social Security Nos  →  ***-**-1234   (dashed, or spaced after "SSN")
    - Credit card numbers     →  ****-****-****-1234  (Luhn-valid only)

Financial figures are never touched: bare digit runs (4523000000), comma or
decimal figures ($4,523,000,000 / 4523000000.00), years and percentages are
left exactly as they are (DOCBOT-2003).

Usage:
    from api.utils.pii_masking import mask_rows, detect_pii_summary, mask_pii, mask_pii_dataframe_output

    safe_rows = mask_rows(result_dicts)         # returns new list, never mutates input
    safe_text = mask_pii(answer_text)           # mask PII in a plain string
    safe_output = mask_pii_dataframe_output(t)  # mask PII in tabular/CSV-like E2B output

    masker = StreamingPIIMasker()               # for token streams
    for tok in stream:
        out = masker.feed(tok)
        if out:
            yield out
    tail = masker.flush()
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
#
# DOCBOT-2003: every numeric pattern below is anchored so it can never match
# inside a larger number. Financial answers are full of bare digit runs
# (4523000000), comma/decimal figures ($4,523,000,000 / 4523000000.00), years
# and percentages; the old phone regex turned ``4523000000`` into
# ``***-***-0000``. Rules now:
#
#   * US phones need structure: a parenthesised area code, a consistent
#     ``-``/``.`` separator, or a ``+1`` country code. A bare (or
#     space-separated) 10-digit run is only a phone when preceded by a phone
#     context word ("phone", "tel", "call", "mobile", "fax", ...).
#   * SSNs need dashes (``123-45-6789``); a space-separated or bare SSN is only
#     masked when preceded by "SSN" / "social security".
#   * Credit cards must pass the Luhn checksum.
#   * No numeric match may start right after a digit or ``digit + [.,]``, or
#     end right before a digit or ``[.,] + digit``.

# Shared guards: not inside a bigger number / decimal / comma-grouped figure.
_NUM_START = r"(?<!\d)(?<!\d[.,])"
_NUM_END = r"(?!\d)(?![.,]\d)"

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

# US phone — structured forms only (never a bare digit run).
_PHONE_RE = re.compile(
    r"(?<!\+)" + _NUM_START
    + r"""
    (?:
        \+1[\s\-.]?                                         # +1 country code
        (?:\(\d{3}\)|\d{3})[\s\-.]?\d{3}[\s\-.]?\d{4}
      |
        (?:1[\s\-.])?\(\d{3}\)[\s\-.]?\d{3}[\s\-.]\d{4}     # (555) 123-4567
      |
        (?:1[\-.])?\d{3}(?P<sep>[\-.])\d{3}(?P=sep)\d{4}    # 555-123-4567 / 555.123.4567
    )
    """
    + _NUM_END,
    re.VERBOSE,
)

# Bare / space-separated 10-digit number, only after a phone context word.
# The gap between keyword and number may not contain digits, currency symbols
# or "%", so "call options worth $4523000000" is left alone.
_PHONE_CONTEXT_RE = re.compile(
    r"""
    (?P<ctx>
        \b(?:phone|ph|tel|telephone|call|mobile|cell|cellphone|fax|whatsapp|contact)\b
        [^\d\n$€£¥%]{0,20}?
    )
    """
    + _NUM_START
    + r"""
    (?P<num>(?:\+?1[\s\-.]?)?\d{3}[\s\-.]?\d{3}[\s\-.]?\d{4})
    """
    + _NUM_END,
    re.VERBOSE | re.IGNORECASE,
)

# UK: +44 7xxx xxxxxx  or  07xxx xxxxxx  (mobile pattern)
_PHONE_UK_RE = re.compile(
    _NUM_START
    + r"""
    (?:
        \+44[\s\-.]?7\d{3}[\s\-.]?\d{6}   # +44 7xxx xxxxxx
        |
        07\d{3}[\s\-.]?\d{6}               # 07xxx xxxxxx
    )
    """
    + _NUM_END,
    re.VERBOSE,
)

# India: +91 followed by 10 digits (spaces or dashes as separators)
_PHONE_IN_RE = re.compile(
    r"""
    (?<![\d+])
    \+91[\s\-]?\d{5}[\s\-]?\d{5}
    """
    + _NUM_END,
    re.VERBOSE,
)

# EU generic: +3x or +4x country codes (excluding +44 handled above) with
# 9-12 digits. Dots are not accepted as separators so signed decimals such as
# "+35.2" in a percentage-change table are never matched.
_PHONE_EU_RE = re.compile(
    r"""
    (?<![\d+])
    \+(?:3[0-9]|4[0-35-9])  # +3x or +4x but NOT +44
    [\s\-]?
    \d[\d\s\-]{7,11}\d      # digits with optional space/dash separators
    """
    + _NUM_END,
    re.VERBOSE,
)

# SSN: dashed form only. Space-separated / bare SSNs need context (below).
_SSN_RE = re.compile(
    _NUM_START
    + r"""
    (?!000|666|9\d{2})             # invalid area numbers
    \d{3}
    -
    (?!00)
    \d{2}
    -
    (?!0000)
    \d{4}
    """
    + _NUM_END,
    re.VERBOSE,
)

_SSN_CONTEXT_RE = re.compile(
    r"""
    (?P<ctx>\b(?:ssn|social\s+security(?:\s+(?:number|no\.?))?)\b[^\d\n]{0,10}?)
    """
    + _NUM_START
    + r"""
    (?P<num>(?!000|666|9\d{2})\d{3}[\s\-]?(?!00)\d{2}[\s\-]?(?!0000)\d{4})
    """
    + _NUM_END,
    re.VERBOSE | re.IGNORECASE,
)

# Credit card candidates: BIN prefix + 13-19 digits, either a bare run or
# 4-digit groups with a consistent space/dash separator (or Amex 4-6-5).
# Every candidate must additionally pass the Luhn checksum (see _mask_cc).
_CC_RE = re.compile(
    _NUM_START
    + r"""
    (?:
        (?:4\d{3}|5[1-5]\d{2}|2[2-7]\d{2}|3[47]\d{2}|6011|35\d{2})
        (?P<sep>[\s\-])\d{4}(?P=sep)\d{4}(?P=sep)\d{1,7}   # grouped 4-4-4-x
      |
        3[47]\d{2}(?P<asep>[\s\-])\d{6}(?P=asep)\d{5}        # Amex 4-6-5
      |
        (?:4|5[1-5]|2[2-7]|3[47]|6011|35)\d{11,17}          # bare run
    )
    """
    + _NUM_END,
    re.VERBOSE,
)

_PHONE_PATTERNS = (_PHONE_UK_RE, _PHONE_IN_RE, _PHONE_EU_RE, _PHONE_RE, _PHONE_CONTEXT_RE)
_ALL_PATTERNS = (_CC_RE, _SSN_CONTEXT_RE, _SSN_RE, *_PHONE_PATTERNS, _EMAIL_RE)


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------


def _digits(raw: str) -> str:
    return re.sub(r"\D", "", raw)


def _luhn_valid(digits: str) -> bool:
    """Return True if ``digits`` (13-19 chars) passes the Luhn mod-10 checksum."""
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _mask_email(m: re.Match) -> str:
    raw = m.group()
    local, domain = raw.split("@", 1)
    masked_local = local[0] + "***" if len(local) > 1 else "***"
    return f"{masked_local}@{domain}"


def _phone_mask_for(raw: str) -> str:
    # Keep only last 4 digits
    return f"***-***-{_digits(raw)[-4:]}"


def _mask_phone(m: re.Match) -> str:
    return _phone_mask_for(m.group())


def _mask_phone_context(m: re.Match) -> str:
    return m.group("ctx") + _phone_mask_for(m.group("num"))


def _ssn_mask_for(raw: str) -> str:
    sep = "-" if "-" in raw else " "
    return f"***{sep}**{sep}{_digits(raw)[-4:]}"


def _mask_ssn(m: re.Match) -> str:
    return _ssn_mask_for(m.group())


def _mask_ssn_context(m: re.Match) -> str:
    return m.group("ctx") + _ssn_mask_for(m.group("num"))


def _mask_cc(m: re.Match) -> str:
    raw = m.group()
    digits = _digits(raw)
    if not _luhn_valid(digits):
        return raw
    return f"****-****-****-{digits[-4:]}"


def _mask_string(value: str) -> str:
    """Apply all PII masks to a single string value."""
    value = _CC_RE.sub(_mask_cc, value)
    value = _SSN_CONTEXT_RE.sub(_mask_ssn_context, value)
    value = _SSN_RE.sub(_mask_ssn, value)
    # International phone patterns applied before US to avoid partial overlap
    value = _PHONE_UK_RE.sub(_mask_phone, value)
    value = _PHONE_IN_RE.sub(_mask_phone, value)
    value = _PHONE_EU_RE.sub(_mask_phone, value)
    value = _PHONE_RE.sub(_mask_phone, value)
    value = _PHONE_CONTEXT_RE.sub(_mask_phone_context, value)
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
    India, EU), and email addresses. Financial figures are left unchanged.

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
            counts["phone"] += sum(
                1 for pat in _PHONE_PATTERNS for _ in pat.finditer(value)
            )
            counts["ssn"] += sum(
                1 for pat in (_SSN_RE, _SSN_CONTEXT_RE) for _ in pat.finditer(value)
            )
            counts["credit_card"] += sum(
                1 for m in _CC_RE.finditer(value) if _luhn_valid(_digits(m.group()))
            )
    return counts


# ---------------------------------------------------------------------------
# Streaming helper — DOCBOT-2003
# ---------------------------------------------------------------------------


class StreamingPIIMasker:
    """Line-buffered PII masker for token/chunk streams.

    Masking each streamed token independently is unsafe: a phone number split
    across tokens (``"555-86"`` + ``"7-5309"``) escapes every pattern, and a
    fragment of a number is judged without its neighbouring digits. This
    helper buffers text and releases only *completed lines*, masked as a
    whole, so every pattern sees full values.

    A long line without a newline is soft-released once the buffer exceeds
    ``max_buffer_chars``: it is cut just after a whitespace char that sits at
    least ``hold_chars`` before the end of the buffer and is not inside any
    PII match, so a value is never split across releases. ``hold_chars`` must
    exceed the longest possible match (context keyword + number < 50 chars).

    Usage::

        masker = StreamingPIIMasker()
        for token in llm_stream:
            out = masker.feed(token)
            if out:
                yield out
        tail = masker.flush()
        if tail:
            yield tail
    """

    def __init__(self, max_buffer_chars: int = 400, hold_chars: int = 64) -> None:
        if hold_chars < 1 or max_buffer_chars <= hold_chars:
            raise ValueError("max_buffer_chars must be greater than hold_chars >= 1")
        self._buf = ""
        self._max = max_buffer_chars
        self._hold = hold_chars

    def feed(self, chunk: str) -> str:
        """Add ``chunk``; return masked text that is now safe to emit (may be "")."""
        if not chunk:
            return ""
        self._buf += chunk
        nl = self._buf.rfind("\n")
        if nl != -1:
            ready, self._buf = self._buf[: nl + 1], self._buf[nl + 1:]
            return mask_pii(ready)
        if len(self._buf) > self._max:
            cut = self._safe_cut(self._buf)
            if cut > 0:
                ready, self._buf = self._buf[:cut], self._buf[cut:]
                return mask_pii(ready)
        return ""

    def flush(self) -> str:
        """Return the masked remainder of the buffer and reset it."""
        ready, self._buf = self._buf, ""
        return mask_pii(ready) if ready else ""

    def _safe_cut(self, text: str) -> int:
        """Index just after a whitespace char, outside every PII match, that
        leaves at least ``hold_chars`` buffered. Returns 0 if none exists."""
        spans = [m.span() for pat in _ALL_PATTERNS for m in pat.finditer(text)]
        for i in range(len(text) - self._hold - 1, -1, -1):
            if not text[i].isspace():
                continue
            cut = i + 1
            if all(not (start < cut < end) for start, end in spans):
                return cut
        return 0
