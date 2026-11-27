"""Value-containment overlap between two columns' value fingerprints (DOCBOT-2911).

Idea ported from NVIDIA/auto-ontology ``semantic/fk_values.py`` (containment
join verification); written fresh here, no code copied.
"""

from typing import Iterable, Optional


def containment(a: Iterable[str], b: Iterable[str]) -> Optional[float]:
    """|A ∩ B| / min(|A|, |B|) over distinct hashes; None if either side is empty.

    Using min() (not union) means a small dimension column fully contained in a
    large fact column scores 1.0 — the FK / shared-key case.
    """
    sa, sb = set(a or ()), set(b or ())
    if not sa or not sb:
        return None
    return len(sa & sb) / min(len(sa), len(sb))
