"""
DocBot Persona Router — DOCBOT-802 (real implementation)

Routes a user question to the best-fit expert persona by scoring it against
each persona's detection_keywords at query time, not upload time.

Primary keywords score 2 points each; secondary keywords score 1 point each.
The top-scoring persona wins if it clears the minimum threshold.
Ties break in favour of the persona with the higher primary hit count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_PRIMARY_WEIGHT = 2
_SECONDARY_WEIGHT = 1
_MIN_SCORE = 2          # require at least 2 points to override Generalist


@dataclass
class RoutingDecision:
    persona: str
    score: int
    primary_hits: int
    secondary_hits: int
    was_routed: bool    # False when Generalist was chosen by default


def route_persona(question: str, expert_personas: dict[str, Any]) -> RoutingDecision:
    """
    Score the question against each persona's detection_keywords and return the
    best-match RoutingDecision.

    Only considers personas that have at least one primary keyword defined.
    Generalist is always the fallback (empty detection_keywords).
    """
    q = question.lower()
    best_persona = "Generalist"
    best_score = 0
    best_primary = 0
    best_secondary = 0

    for name, data in expert_personas.items():
        if name == "Generalist":
            continue
        kw = data.get("detection_keywords", {})
        primary = kw.get("primary", [])
        secondary = kw.get("secondary", [])
        if not primary:
            continue

        p_hits = sum(1 for kw_item in primary if kw_item.lower() in q)
        s_hits = sum(1 for kw_item in secondary if kw_item.lower() in q)
        score = p_hits * _PRIMARY_WEIGHT + s_hits * _SECONDARY_WEIGHT

        if score > best_score or (
            score == best_score and score > 0 and p_hits > best_primary
        ):
            best_score = score
            best_persona = name
            best_primary = p_hits
            best_secondary = s_hits

    if best_score < _MIN_SCORE:
        return RoutingDecision(
            persona="Generalist",
            score=0,
            primary_hits=0,
            secondary_hits=0,
            was_routed=False,
        )

    return RoutingDecision(
        persona=best_persona,
        score=best_score,
        primary_hits=best_primary,
        secondary_hits=best_secondary,
        was_routed=True,
    )
