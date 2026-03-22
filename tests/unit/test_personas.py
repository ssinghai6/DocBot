"""Unit tests for EXPERT_PERSONAS dict — DOCBOT-304."""

import pytest
from api.index import EXPERT_PERSONAS


@pytest.mark.unit
class TestDataAnalystPersona:

    def test_data_analyst_key_exists(self):
        assert "Data Analyst" in EXPERT_PERSONAS, (
            "'Data Analyst' key is missing from EXPERT_PERSONAS"
        )

    def test_data_analyst_has_persona_def(self):
        persona = EXPERT_PERSONAS["Data Analyst"]
        assert "persona_def" in persona
        assert isinstance(persona["persona_def"], str)
        assert len(persona["persona_def"]) > 0

    def test_data_analyst_has_expertise_areas(self):
        persona = EXPERT_PERSONAS["Data Analyst"]
        assert "expertise_areas" in persona
        assert isinstance(persona["expertise_areas"], list)
        assert len(persona["expertise_areas"]) > 0

    def test_data_analyst_has_response_style(self):
        persona = EXPERT_PERSONAS["Data Analyst"]
        assert "response_style" in persona
        assert isinstance(persona["response_style"], str)
        assert len(persona["response_style"]) > 0

    def test_data_analyst_disclaimer_is_none(self):
        persona = EXPERT_PERSONAS["Data Analyst"]
        assert "disclaimer" in persona
        assert persona["disclaimer"] is None

    def test_data_analyst_persona_def_mentions_databot(self):
        persona_def = EXPERT_PERSONAS["Data Analyst"]["persona_def"]
        assert "DataBot" in persona_def

    def test_data_analyst_persona_def_mentions_sql_transparency(self):
        persona_def = EXPERT_PERSONAS["Data Analyst"]["persona_def"]
        assert "SQL" in persona_def

    def test_data_analyst_expertise_areas_content(self):
        expected_areas = [
            "SQL query analysis",
            "Statistical summaries",
            "Data quality assessment",
            "Trend analysis",
            "Business metrics",
            "Exploratory data analysis",
        ]
        actual_areas = EXPERT_PERSONAS["Data Analyst"]["expertise_areas"]
        for area in expected_areas:
            assert area in actual_areas, (
                f"Expected expertise area '{area}' not found in Data Analyst persona"
            )


@pytest.mark.unit
class TestAllPersonasStructure:
    """Regression guard — every persona must satisfy the required schema."""

    REQUIRED_KEYS = {"persona_def", "expertise_areas", "response_style", "disclaimer"}

    def test_all_personas_have_required_keys(self):
        for name, data in EXPERT_PERSONAS.items():
            missing = self.REQUIRED_KEYS - data.keys()
            assert not missing, (
                f"Persona '{name}' is missing keys: {missing}"
            )

    def test_persona_count_includes_data_analyst(self):
        assert len(EXPERT_PERSONAS) >= 8, (
            f"Expected at least 8 personas, found {len(EXPERT_PERSONAS)}"
        )
