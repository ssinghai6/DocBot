"""Value fingerprints for cross-source containment matching (DOCBOT-2911).

Column values are normalized and hashed so two sources can be compared for
value overlap without storing raw values (the ontology table is plaintext).
Normalization: str, strip, lowercase; numbers canonicalized so ``1``,
``1.0`` and ``"1"`` produce the same fingerprint.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
from decimal import Decimal, InvalidOperation
from typing import Iterable, Optional

MAX_FINGERPRINTS = 200
_HASH_HEX_CHARS = 16


def _normalize(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    s = str(value).strip().lower()
    if not s or s in ("nan", "none", "null"):
        return None
    try:
        num = Decimal(s)
    except InvalidOperation:
        return s
    if not num.is_finite():
        return s
    if abs(num.adjusted()) > 30:  # "1e999999999" must not expand into a billion-digit string
        return s
    if num == num.to_integral_value():
        return str(int(num))
    return format(num.normalize(), "f")


def _fingerprint_key() -> Optional[bytes]:
    # Keyed (HMAC) so someone reading the plaintext ontology table can't confirm guessed emails/ids.
    secret = os.getenv("DB_ENCRYPTION_KEY")
    if not secret:
        return None
    return hashlib.sha256(b"docbot-ontology-fingerprint:" + secret.encode("utf-8")).digest()


def fingerprint_values(values: Iterable[object]) -> list[str]:
    """Distinct HMAC-sha256[:16] fingerprints of normalized values, max 200, first-seen order.

    Returns [] when no key is configured (matching then falls back to name/type evidence).
    """
    key = _fingerprint_key()
    if key is None:
        return []
    seen: dict[str, None] = {}
    for v in values:
        norm = _normalize(v)
        if norm is None:
            continue
        seen.setdefault(hmac.new(key, norm.encode("utf-8"), hashlib.sha256).hexdigest()[:_HASH_HEX_CHARS], None)
        if len(seen) >= MAX_FINGERPRINTS:
            break
    return list(seen)
