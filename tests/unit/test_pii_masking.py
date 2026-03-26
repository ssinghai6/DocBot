"""Unit tests for api/utils/pii_masking.py — DOCBOT-604 + international extensions.

All tests are CI-safe: no network calls, no API keys, no external services.
"""

import pytest

from api.utils.pii_masking import (
    detect_pii_summary,
    mask_pii,
    mask_pii_dataframe_output,
    mask_rows,
)


# ---------------------------------------------------------------------------
# Email masking
# ---------------------------------------------------------------------------


class TestEmailMasking:
    def test_standard_email_masked(self):
        rows = [{"email": "john.doe@example.com"}]
        result = mask_rows(rows)
        assert "@example.com" in result[0]["email"]
        assert "john.doe" not in result[0]["email"]
        assert result[0]["email"].startswith("j***")

    def test_single_char_local_masked(self):
        rows = [{"email": "a@b.com"}]
        result = mask_rows(rows)
        assert "***@b.com" in result[0]["email"]

    def test_email_in_sentence(self):
        rows = [{"note": "Contact us at support@company.org for help"}]
        result = mask_rows(rows)
        assert "support@company.org" not in result[0]["note"]
        assert "s***@company.org" in result[0]["note"]

    def test_no_email_unchanged(self):
        rows = [{"name": "John Doe"}]
        result = mask_rows(rows)
        assert result[0]["name"] == "John Doe"


# ---------------------------------------------------------------------------
# Phone masking
# ---------------------------------------------------------------------------


class TestPhoneMasking:
    def test_dashed_phone(self):
        rows = [{"phone": "555-867-5309"}]
        result = mask_rows(rows)
        assert "5309" in result[0]["phone"]
        assert "555" not in result[0]["phone"]

    def test_dotted_phone(self):
        rows = [{"phone": "555.867.5309"}]
        result = mask_rows(rows)
        assert "5309" in result[0]["phone"]

    def test_country_code_phone(self):
        rows = [{"phone": "+1 555-867-5309"}]
        result = mask_rows(rows)
        assert "5309" in result[0]["phone"]
        assert "555" not in result[0]["phone"]

    def test_phone_in_sentence(self):
        rows = [{"info": "Call me at 555-867-5309 tomorrow"}]
        result = mask_rows(rows)
        assert "555-867-5309" not in result[0]["info"]


# ---------------------------------------------------------------------------
# SSN masking
# ---------------------------------------------------------------------------


class TestSsnMasking:
    def test_dashed_ssn(self):
        rows = [{"ssn": "123-45-6789"}]
        result = mask_rows(rows)
        assert "6789" in result[0]["ssn"]
        assert "123" not in result[0]["ssn"]
        assert "45" not in result[0]["ssn"]

    def test_spaced_ssn(self):
        rows = [{"ssn": "123 45 6789"}]
        result = mask_rows(rows)
        assert "6789" in result[0]["ssn"]
        assert "123" not in result[0]["ssn"]

    def test_invalid_ssn_not_masked(self):
        # 000 area is invalid — should NOT be detected as SSN
        rows = [{"val": "000-45-6789"}]
        result = mask_rows(rows)
        assert result[0]["val"] == "000-45-6789"


# ---------------------------------------------------------------------------
# Credit card masking
# ---------------------------------------------------------------------------


class TestCreditCardMasking:
    def test_visa_card(self):
        rows = [{"card": "4111 1111 1111 1111"}]
        result = mask_rows(rows)
        assert "1111" in result[0]["card"]
        assert "4111 1111 1111" not in result[0]["card"]

    def test_mastercard(self):
        rows = [{"card": "5500-0000-0000-0004"}]
        result = mask_rows(rows)
        assert "0004" in result[0]["card"]
        assert "5500" not in result[0]["card"]


# ---------------------------------------------------------------------------
# mask_rows — general behaviour
# ---------------------------------------------------------------------------


class TestMaskRows:
    def test_empty_list_returns_empty(self):
        assert mask_rows([]) == []

    def test_non_string_values_unchanged(self):
        rows = [{"count": 42, "amount": 3.14, "active": True, "nullable": None}]
        result = mask_rows(rows)
        assert result[0]["count"] == 42
        assert result[0]["amount"] == 3.14
        assert result[0]["active"] is True
        assert result[0]["nullable"] is None

    def test_original_not_mutated(self):
        original = [{"email": "jane@example.com"}]
        mask_rows(original)
        assert original[0]["email"] == "jane@example.com"

    def test_multiple_pii_in_one_field(self):
        rows = [{"text": "Call 555-867-5309 or email bob@test.com"}]
        result = mask_rows(rows)
        assert "555-867-5309" not in result[0]["text"]
        assert "bob@test.com" not in result[0]["text"]

    def test_multiple_rows(self):
        rows = [
            {"email": "alice@example.com"},
            {"email": "bob@example.com"},
        ]
        result = mask_rows(rows)
        assert "alice@example.com" not in result[0]["email"]
        assert "bob@example.com" not in result[1]["email"]


# ---------------------------------------------------------------------------
# detect_pii_summary
# ---------------------------------------------------------------------------


class TestDetectPiiSummary:
    def test_counts_email(self):
        rows = [{"a": "foo@bar.com"}, {"b": "baz@qux.net"}]
        summary = detect_pii_summary(rows)
        assert summary["email"] == 2

    def test_counts_ssn(self):
        rows = [{"ssn": "123-45-6789"}]
        summary = detect_pii_summary(rows)
        assert summary["ssn"] == 1

    def test_empty_rows_all_zero(self):
        summary = detect_pii_summary([])
        assert all(v == 0 for v in summary.values())

    def test_no_pii_all_zero(self):
        rows = [{"name": "Acme Corp", "revenue": "1000000"}]
        summary = detect_pii_summary(rows)
        assert all(v == 0 for v in summary.values())

    def test_returns_all_keys(self):
        summary = detect_pii_summary([])
        assert set(summary.keys()) == {"email", "phone", "ssn", "credit_card"}


# ---------------------------------------------------------------------------
# International phone — UK
# ---------------------------------------------------------------------------


class TestUKPhoneFormats:
    def test_uk_plus44_format(self):
        text = "Call me at +44 7700 900123 please."
        result = mask_pii(text)
        assert "+44 7700 900123" not in result
        assert "***" in result

    def test_uk_local_07_format(self):
        text = "My number is 07700 900456."
        result = mask_pii(text)
        assert "07700 900456" not in result
        assert "***" in result

    def test_uk_plus44_compact(self):
        text = "Reach us at +447911123456."
        result = mask_pii(text)
        assert "+447911123456" not in result

    def test_uk_last_four_preserved(self):
        text = "+44 7700 900123"
        result = mask_pii(text)
        # Last 4 digits of the full number should appear in masked output
        assert "0123" in result


# ---------------------------------------------------------------------------
# International phone — India
# ---------------------------------------------------------------------------


class TestIndiaPhoneFormats:
    def test_india_plus91_space(self):
        text = "Contact: +91 98765 43210"
        result = mask_pii(text)
        assert "+91 98765 43210" not in result
        assert "***" in result

    def test_india_plus91_dash(self):
        text = "Helpline: +91-98765-43210"
        result = mask_pii(text)
        assert "+91-98765-43210" not in result
        assert "***" in result

    def test_india_last_four_preserved(self):
        text = "+91 98765 43210"
        result = mask_pii(text)
        assert "3210" in result


# ---------------------------------------------------------------------------
# mask_pii() — public API for LLM answer text
# ---------------------------------------------------------------------------


class TestMaskPii:
    def test_callable_as_mask_pii(self):
        """mask_pii(text) is the documented public API — must remain callable."""
        result = mask_pii("Email me at bob@corp.com.")
        assert "bob@corp.com" not in result
        assert "b***@corp.com" in result

    def test_plain_text_email(self):
        text = "Reach alice@example.org for details."
        result = mask_pii(text)
        assert "alice" not in result

    def test_plain_text_us_phone(self):
        text = "Call (800) 555-0199 now."
        result = mask_pii(text)
        assert "(800) 555-0199" not in result

    def test_no_pii_unchanged(self):
        text = "The revenue for Q3 was $1.2M."
        assert mask_pii(text) == text

    def test_mixed_pii_all_masked(self):
        text = "Contact bob@example.com or call +44 7700 900111."
        result = mask_pii(text)
        assert "bob@example.com" not in result
        assert "+44 7700 900111" not in result


# ---------------------------------------------------------------------------
# mask_pii_dataframe_output()
# ---------------------------------------------------------------------------


class TestMaskPiiDataframeOutput:
    def test_pipe_separated_email_masked(self):
        table = (
            "name       | email               | dept\n"
            "Alice      | alice@example.com   | Eng\n"
            "Bob        | bob@corp.io         | Sales\n"
        )
        result = mask_pii_dataframe_output(table)
        assert "alice@example.com" not in result
        assert "bob@corp.io" not in result

    def test_comma_separated_email_masked(self):
        csv_line = "Alice,alice@example.com,Engineering"
        result = mask_pii_dataframe_output(csv_line)
        assert "alice@example.com" not in result

    def test_pipe_separated_phone_masked(self):
        table = "name  | phone\nAlice | +44 7700 900123\n"
        result = mask_pii_dataframe_output(table)
        assert "+44 7700 900123" not in result

    def test_non_pii_table_unchanged(self):
        table = "product | revenue\nWidget  | 1200\n"
        result = mask_pii_dataframe_output(table)
        assert "Widget" in result
        assert "1200" in result

    def test_delegates_to_mask_pii(self):
        """mask_pii_dataframe_output must apply all PII patterns, not just email."""
        text = "Row: 4111111111111111, bob@test.com, +91 99999 88888"
        result = mask_pii_dataframe_output(text)
        assert "bob@test.com" not in result
        assert "4111111111111111" not in result
        assert "+91 99999 88888" not in result
