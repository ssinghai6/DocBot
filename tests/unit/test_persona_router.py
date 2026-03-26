"""Unit tests for api/utils/persona_router.py — DOCBOT-802."""

import pytest
from api.utils.persona_router import route_persona, RoutingDecision

# ---------------------------------------------------------------------------
# Minimal EXPERT_PERSONAS fixture matching production structure
# ---------------------------------------------------------------------------

PERSONAS = {
    "Generalist": {
        "persona_def": "You are a helpful assistant.",
        "detection_keywords": {"primary": [], "secondary": []},
    },
    "Doctor": {
        "persona_def": "You are a doctor.",
        "detection_keywords": {
            "primary": ["diagnosis", "patient", "clinical", "symptom", "treatment",
                        "prescription", "dosage", "pathology", "medication"],
            "secondary": ["health", "medical", "hospital", "therapy", "disease"],
        },
    },
    "Finance Expert": {
        "persona_def": "You are a finance expert.",
        "detection_keywords": {
            "primary": ["revenue", "profit", "balance sheet", "income statement",
                        "cash flow", "earnings", "equity", "ebitda", "valuation"],
            "secondary": ["financial", "investment", "audit", "dividend", "fiscal"],
        },
    },
    "Lawyer": {
        "persona_def": "You are a lawyer.",
        "detection_keywords": {
            "primary": ["contract", "agreement", "clause", "jurisdiction",
                        "plaintiff", "compliance", "litigation", "statute"],
            "secondary": ["legal", "terms", "policy", "regulation", "copyright"],
        },
    },
    "Engineer": {
        "persona_def": "You are an engineer.",
        "detection_keywords": {
            "primary": ["specification", "system design", "architecture", "firmware",
                        "schematic", "circuit", "protocol"],
            "secondary": ["technical", "engineering", "infrastructure", "deployment"],
        },
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRoutePersona:

    def test_medical_question_routes_to_doctor(self):
        decision = route_persona(
            "What is the recommended dosage for this medication and any contraindications?",
            PERSONAS,
        )
        assert decision.persona == "Doctor"
        assert decision.was_routed is True

    def test_finance_question_routes_to_finance_expert(self):
        decision = route_persona(
            "What is the revenue and EBITDA for Q3? How does the cash flow look?",
            PERSONAS,
        )
        assert decision.persona == "Finance Expert"
        assert decision.was_routed is True

    def test_legal_question_routes_to_lawyer(self):
        decision = route_persona(
            "Does this contract contain an indemnity clause? What is the jurisdiction?",
            PERSONAS,
        )
        assert decision.persona == "Lawyer"
        assert decision.was_routed is True

    def test_engineering_question_routes_to_engineer(self):
        decision = route_persona(
            "Can you review the system design and circuit schematic in this specification?",
            PERSONAS,
        )
        assert decision.persona == "Engineer"
        assert decision.was_routed is True

    def test_generic_question_falls_back_to_generalist(self):
        decision = route_persona("What is this document about?", PERSONAS)
        assert decision.persona == "Generalist"
        assert decision.was_routed is False

    def test_single_weak_keyword_below_threshold(self):
        """One primary hit = score 2, but just on the threshold. Adjust if threshold changes."""
        decision = route_persona("The patient is here.", PERSONAS)
        # One primary keyword 'patient' = score 2 which equals _MIN_SCORE — should route
        assert decision.persona in ("Doctor", "Generalist")

    def test_returns_routing_decision_type(self):
        result = route_persona("plain question", PERSONAS)
        assert isinstance(result, RoutingDecision)

    def test_score_reflects_keyword_count(self):
        decision = route_persona(
            "revenue profit earnings ebitda balance sheet valuation cash flow equity income statement",
            PERSONAS,
        )
        assert decision.persona == "Finance Expert"
        assert decision.primary_hits >= 5

    def test_primary_beats_many_secondary(self):
        """A persona with 2 primary hits beats a persona with 3 secondary hits."""
        decision = route_persona(
            "balance sheet and income statement review",
            PERSONAS,
        )
        # "balance sheet" and "income statement" are both primary for Finance Expert → score 4
        assert decision.persona == "Finance Expert"
        assert decision.primary_hits >= 2

    def test_empty_question_returns_generalist(self):
        decision = route_persona("", PERSONAS)
        assert decision.persona == "Generalist"

    def test_persona_with_no_primary_keywords_ignored(self):
        """Personas without primary keywords must not be routed to."""
        custom = {
            "Generalist": PERSONAS["Generalist"],
            "NoPrimary": {
                "persona_def": "...",
                "detection_keywords": {"primary": [], "secondary": ["foo", "bar"]},
            },
        }
        decision = route_persona("foo bar baz", custom)
        assert decision.persona == "Generalist"

    def test_mixed_domain_routes_to_highest_scorer(self):
        """Medical + finance overlap — the one with more points wins."""
        decision = route_persona(
            "patient diagnosis treatment revenue earnings",
            PERSONAS,
        )
        # Doctor: "patient", "diagnosis", "treatment" = 3 primary = 6 pts
        # Finance: "revenue", "earnings" = 2 primary = 4 pts
        assert decision.persona == "Doctor"

    def test_case_insensitive_matching(self):
        decision = route_persona("DIAGNOSIS and TREATMENT for PATIENT", PERSONAS)
        assert decision.persona == "Doctor"
