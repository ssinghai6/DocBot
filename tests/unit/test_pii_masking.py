"""Unit tests for api/utils/pii_masking.py — DOCBOT-604 + international extensions.

All tests are CI-safe: no network calls, no API keys, no external services.
"""

import pytest

from api.utils.pii_masking import (
    StreamingPIIMasker,
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

    def test_spaced_ssn_with_context(self):
        # DOCBOT-2003: spaced SSNs are only masked when labelled as an SSN.
        rows = [{"ssn": "SSN: 123 45 6789"}]
        result = mask_rows(rows)
        assert "6789" in result[0]["ssn"]
        assert "123" not in result[0]["ssn"]

    def test_spaced_digits_without_context_unchanged(self):
        # Whitespace-aligned table cells must not be read as an SSN.
        rows = [{"val": "123 45 6789"}]
        result = mask_rows(rows)
        assert result[0]["val"] == "123 45 6789"

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


# ---------------------------------------------------------------------------
# DOCBOT-2003 — financial numbers must never be corrupted
# ---------------------------------------------------------------------------

# Strings that must pass through mask_pii() byte-for-byte unchanged.
FINANCIAL_FIXTURES = [
    "4523000000",                                  # bare 10-digit total (old bug)
    "$4,523,000,000",                              # currency + commas
    "4,523,000,000",
    "1,234,567,890",
    "4523000000.00",                               # decimal
    "$4523000000.50",
    "Revenue was 4523000000 in FY2024.",
    "| Revenue | 1234567890 |",                    # markdown table cell
    "Total,1234567890,2024",                       # CSV row
    "Account total: 9876543210",
    "Net income 5551234567 USD",
    "2023",                                        # year
    "FY2023 vs FY2024",
    "2023-2024",
    "Q4 2024: 12.5%",                              # percentages
    "+35.2% YoY",
    "+35.2 12.4 13.5 14.6",                        # signed % change row
    "-4523000000",
    "100 200 3000",                                # whitespace-aligned columns
    "123 45 6789",                                 # spaced digits, no SSN label
    "call options worth $4523000000",              # phone keyword + money
    "4000000000000001",                            # Luhn-invalid 16 digits
    "4111111111111112",                            # Luhn-invalid
    "5500000000000001",
    "45230000000000000",                           # 17-digit figure
    "1234567890123",
    "Shares outstanding: 4,111,111,111,111,111",
    "EPS 4.111111111111111",
]

# (input, raw fragment that must disappear, fragment that must remain)
REAL_PII_FIXTURES = [
    ("(555) 123-4567", "123-4567", "4567"),
    ("555-123-4567", "555-123", "4567"),
    ("555.123.4567", "555.123", "4567"),
    ("+1 555 123 4567", "555 123", "4567"),
    ("+1-555-123-4567", "555-123", "4567"),
    ("Phone: 5551234567", "5551234567", "4567"),
    ("tel 555 123 4567", "555 123", "4567"),
    ("Call 5551234567 today", "5551234567", "4567"),
    ("mobile: 555 123 4567", "555 123", "4567"),
    ("fax: +15551234567", "5551234567", "4567"),
    ("Alice,555-867-5309,Eng", "555-867", "5309"),
    ("+44 7700 900123", "7700 900", "0123"),
    ("+91 98765 43210", "98765", "3210"),
    ("123-45-6789", "123-45", "6789"),
    ("SSN 123 45 6789", "123 45", "6789"),
    ("social security number: 123456789", "123456789", "6789"),
    ("4111 1111 1111 1111", "4111 1111", "1111"),
    ("4111-1111-1111-1111", "4111-1111", "1111"),
    ("4111111111111111", "411111111111", "1111"),
    ("378282246310005", "37828224631", "0005"),
    ("john.doe@example.com", "john.doe", "@example.com"),
]


class TestFinancialNumbersUnchanged:
    @pytest.mark.parametrize("text", FINANCIAL_FIXTURES)
    def test_mask_pii_leaves_number_unchanged(self, text):
        assert mask_pii(text) == text

    @pytest.mark.parametrize("text", FINANCIAL_FIXTURES)
    def test_mask_rows_leaves_number_unchanged(self, text):
        assert mask_rows([{"v": text}])[0]["v"] == text

    def test_old_bug_regression(self):
        assert mask_pii("4523000000") == "4523000000"

    def test_no_false_positive_counts(self):
        summary = detect_pii_summary([{"v": t} for t in FINANCIAL_FIXTURES])
        assert summary == {"email": 0, "phone": 0, "ssn": 0, "credit_card": 0}

    def test_financial_table_unchanged(self):
        table = (
            "| Year | Revenue | Net Income | Margin |\n"
            "|------|---------|------------|--------|\n"
            "| 2023 | 4523000000 | 1234567890 | 27.3% |\n"
            "| 2024 | $4,890,000,000 | $1,301,000,000.00 | 26.6% |\n"
        )
        assert mask_pii(table) == table


class TestRealPiiStillMasked:
    @pytest.mark.parametrize("text,gone,kept", REAL_PII_FIXTURES)
    def test_masked(self, text, gone, kept):
        result = mask_pii(text)
        assert gone not in result
        assert kept in result
        assert "*" in result

    def test_phone_next_to_financial_figure(self):
        text = "Revenue 4523000000; call 555-123-4567."
        result = mask_pii(text)
        assert "4523000000" in result
        assert "555-123-4567" not in result

    def test_real_pii_counted(self):
        summary = detect_pii_summary([{"v": t} for t, _, _ in REAL_PII_FIXTURES])
        assert summary["phone"] >= 11
        assert summary["ssn"] == 3
        assert summary["credit_card"] == 4
        assert summary["email"] == 1


# ---------------------------------------------------------------------------
# StreamingPIIMasker
# ---------------------------------------------------------------------------


def _stream(chunks, **kwargs):
    masker = StreamingPIIMasker(**kwargs)
    out = [masker.feed(c) for c in chunks]
    out.append(masker.flush())
    return out


class TestStreamingPIIMasker:
    def test_phone_split_across_chunks_masked(self):
        joined = "".join(_stream(["Call me at 555-86", "7-53", "09 tomorrow.\n"]))
        assert "555-867-5309" not in joined
        assert "***-***-5309" in joined

    def test_phone_split_without_trailing_newline_masked_on_flush(self):
        out = _stream(["Phone: 555", "12345", "67"])
        assert out[:-1] == ["", "", ""]
        assert out[-1] == "Phone: ***-***-4567"

    def test_email_split_across_chunks_masked(self):
        joined = "".join(_stream(["contact bob@ex", "ample.com\n"]))
        assert "bob@example.com" not in joined
        assert "b***@example.com" in joined

    def test_number_split_across_chunks_unchanged(self):
        chunks = ["Revenue was 45", "2300", "0000 and $4,52", "3,000,000.", "00.\n"]
        joined = "".join(_stream(chunks))
        assert joined == "Revenue was 4523000000 and $4,523,000,000.00.\n"

    def test_only_completed_lines_emitted(self):
        masker = StreamingPIIMasker()
        assert masker.feed("line one\nline tw") == "line one\n"
        assert masker.feed("o\n") == "line two\n"
        assert masker.feed("partial") == ""
        assert masker.flush() == "partial"

    def test_flush_emits_remainder_and_resets(self):
        masker = StreamingPIIMasker()
        masker.feed("tail 123-45-6789")
        assert masker.flush() == "tail ***-**-6789"
        assert masker.flush() == ""

    def test_empty_chunk_is_noop(self):
        masker = StreamingPIIMasker()
        assert masker.feed("") == ""
        assert masker.flush() == ""

    def test_lossless_for_non_pii_text(self):
        text = "Line A 2024\nLine B $1.2M\n| x | 4523000000 |\nend"
        chunks = [text[i:i + 3] for i in range(0, len(text), 3)]
        assert "".join(_stream(chunks)) == text

    def test_long_line_soft_released_without_splitting_pii(self):
        text = "word " * 30 + "call 555-123-4567 " + "more " * 30
        chunks = [text[i:i + 7] for i in range(0, len(text), 7)]
        out = _stream(chunks, max_buffer_chars=100, hold_chars=40)
        assert any(o for o in out[:-1])  # released before final flush
        joined = "".join(out)
        assert "555-123-4567" not in joined
        assert "***-***-4567" in joined
        assert joined.replace("***-***-4567", "555-123-4567") == text

    def test_invalid_config_rejected(self):
        with pytest.raises(ValueError):
            StreamingPIIMasker(max_buffer_chars=10, hold_chars=10)
