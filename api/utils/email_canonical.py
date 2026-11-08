"""Canonical email form for one-account-per-mailbox dedupe (DOCBOT-2332).

``A.B+promo@GoogleMail.com`` and ``ab@gmail.com`` reach the same inbox, so
they must map to one account (otherwise each alias farms a free credit grant).
"""

from __future__ import annotations

_GMAIL_DOMAINS = frozenset({"gmail.com", "googlemail.com"})


def canonical_email(email: str) -> str:
    """Lowercase; drop ``+tag`` for every domain; drop dots for Gmail.

    ``googlemail.com`` is mapped to ``gmail.com``. Other domains keep their
    dots (``first.last@corp.com`` and ``firstlast@corp.com`` can differ).
    """
    local, sep, domain = email.strip().lower().rpartition("@")
    if not sep:
        return email.strip().lower()
    local = local.split("+", 1)[0]
    if domain in _GMAIL_DOMAINS:
        local = local.replace(".", "")
        domain = "gmail.com"
    return f"{local}@{domain}"
