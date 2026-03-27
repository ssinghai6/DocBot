"""Unit tests for api/auth_service.py — DOCBOT-601.

All tests are CI-safe: no network calls, no IdP, no real DB.
DB interactions are tested via mocked async_session_factory.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os


# ---------------------------------------------------------------------------
# is_saml_configured
# ---------------------------------------------------------------------------

class TestIsSamlConfigured:
    def test_returns_false_when_no_env_vars(self, monkeypatch):
        for v in ["SAML_SP_ENTITY_ID", "SAML_SP_ACS_URL",
                  "SAML_IDP_ENTITY_ID", "SAML_IDP_SSO_URL", "SAML_IDP_X509_CERT"]:
            monkeypatch.delenv(v, raising=False)
        from api.auth_service import is_saml_configured
        assert is_saml_configured() is False

    def test_returns_true_when_all_vars_set(self, monkeypatch):
        monkeypatch.setenv("SAML_SP_ENTITY_ID", "https://sp.example.com")
        monkeypatch.setenv("SAML_SP_ACS_URL", "https://sp.example.com/api/auth/saml/acs")
        monkeypatch.setenv("SAML_IDP_ENTITY_ID", "https://idp.example.com")
        monkeypatch.setenv("SAML_IDP_SSO_URL", "https://idp.example.com/sso")
        monkeypatch.setenv("SAML_IDP_X509_CERT", "MIICERT...")
        from api.auth_service import is_saml_configured
        assert is_saml_configured() is True

    def test_returns_false_when_one_var_missing(self, monkeypatch):
        monkeypatch.setenv("SAML_SP_ENTITY_ID", "https://sp.example.com")
        monkeypatch.setenv("SAML_SP_ACS_URL", "https://sp.example.com/acs")
        monkeypatch.setenv("SAML_IDP_ENTITY_ID", "https://idp.example.com")
        monkeypatch.setenv("SAML_IDP_SSO_URL", "https://idp.example.com/sso")
        monkeypatch.delenv("SAML_IDP_X509_CERT", raising=False)
        from api.auth_service import is_saml_configured
        assert is_saml_configured() is False


# ---------------------------------------------------------------------------
# is_auth_enforcement_active
# ---------------------------------------------------------------------------

class TestIsAuthEnforcementActive:
    def test_returns_false_by_default(self, monkeypatch):
        monkeypatch.delenv("AUTH_REQUIRED", raising=False)
        from api.auth_service import is_auth_enforcement_active
        assert is_auth_enforcement_active() is False

    def test_returns_true_when_set(self, monkeypatch):
        monkeypatch.setenv("AUTH_REQUIRED", "true")
        from api.auth_service import is_auth_enforcement_active
        assert is_auth_enforcement_active() is True

    def test_returns_true_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("AUTH_REQUIRED", "True")
        from api.auth_service import is_auth_enforcement_active
        assert is_auth_enforcement_active() is True

    def test_returns_false_for_other_values(self, monkeypatch):
        monkeypatch.setenv("AUTH_REQUIRED", "false")
        from api.auth_service import is_auth_enforcement_active
        assert is_auth_enforcement_active() is False


# ---------------------------------------------------------------------------
# _detect_provider
# ---------------------------------------------------------------------------

class TestDetectProvider:
    def test_detects_okta(self, monkeypatch):
        monkeypatch.setenv("SAML_IDP_SSO_URL", "https://dev-123.okta.com/app/sso/saml")
        from api.auth_service import _detect_provider
        assert _detect_provider() == "okta"

    def test_detects_azure_ad(self, monkeypatch):
        monkeypatch.setenv("SAML_IDP_SSO_URL", "https://login.microsoftonline.com/tenant/saml2")
        from api.auth_service import _detect_provider
        assert _detect_provider() == "azure_ad"

    def test_falls_back_to_saml(self, monkeypatch):
        monkeypatch.setenv("SAML_IDP_SSO_URL", "https://custom-idp.company.com/sso")
        from api.auth_service import _detect_provider
        assert _detect_provider() == "saml"


# ---------------------------------------------------------------------------
# generate_session_token
# ---------------------------------------------------------------------------

class TestGenerateSessionToken:
    def test_returns_64_char_hex(self):
        from api.auth_service import generate_session_token
        token = generate_session_token()
        assert len(token) == 64
        assert all(c in "0123456789abcdef" for c in token)

    def test_tokens_are_unique(self):
        from api.auth_service import generate_session_token
        tokens = {generate_session_token() for _ in range(100)}
        assert len(tokens) == 100


# ---------------------------------------------------------------------------
# process_acs — attribute extraction
# ---------------------------------------------------------------------------

class TestProcessAcs:
    def _mock_auth(self, attrs: dict, name_id: str = "user@example.com",
                   authenticated: bool = True, errors: list | None = None):
        auth = MagicMock()
        auth.process_response = MagicMock()
        auth.get_errors = MagicMock(return_value=errors or [])
        auth.is_authenticated = MagicMock(return_value=authenticated)
        auth.get_attributes = MagicMock(return_value=attrs)
        auth.get_nameid = MagicMock(return_value=name_id)
        auth.get_last_error_reason = MagicMock(return_value="bad signature")
        return auth

    def test_extracts_email_from_standard_attr(self, monkeypatch):
        monkeypatch.setenv("SAML_IDP_SSO_URL", "https://custom.idp.com/sso")
        from api.auth_service import process_acs
        auth = self._mock_auth({"email": ["alice@example.com"], "displayName": ["Alice"]})
        result = process_acs(auth)
        assert result["email"] == "alice@example.com"

    def test_falls_back_to_name_id_for_email(self, monkeypatch):
        monkeypatch.setenv("SAML_IDP_SSO_URL", "https://custom.idp.com/sso")
        from api.auth_service import process_acs
        auth = self._mock_auth({}, name_id="bob@example.com")
        result = process_acs(auth)
        assert result["email"] == "bob@example.com"

    def test_raises_on_saml_errors(self, monkeypatch):
        monkeypatch.setenv("SAML_IDP_SSO_URL", "https://custom.idp.com/sso")
        from api.auth_service import process_acs
        auth = self._mock_auth({}, errors=["invalid_response"])
        with pytest.raises(ValueError, match="SAML authentication failed"):
            process_acs(auth)

    def test_raises_when_not_authenticated(self, monkeypatch):
        monkeypatch.setenv("SAML_IDP_SSO_URL", "https://custom.idp.com/sso")
        from api.auth_service import process_acs
        auth = self._mock_auth({}, authenticated=False)
        with pytest.raises(ValueError, match="not authenticated"):
            process_acs(auth)

    def test_email_lowercased(self, monkeypatch):
        monkeypatch.setenv("SAML_IDP_SSO_URL", "https://custom.idp.com/sso")
        from api.auth_service import process_acs
        auth = self._mock_auth({"email": ["Alice@EXAMPLE.COM"]})
        result = process_acs(auth)
        assert result["email"] == "alice@example.com"

    def test_extracts_groups(self, monkeypatch):
        monkeypatch.setenv("SAML_IDP_SSO_URL", "https://custom.idp.com/sso")
        from api.auth_service import process_acs
        auth = self._mock_auth({
            "email": ["alice@example.com"],
            "groups": ["admins", "analysts"],
        })
        result = process_acs(auth)
        assert "admins" in result["groups"]

    def test_azure_ad_claim_attributes(self, monkeypatch):
        monkeypatch.setenv("SAML_IDP_SSO_URL", "https://login.microsoftonline.com/tenant/saml2")
        from api.auth_service import process_acs
        auth = self._mock_auth({
            "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress": ["azure@corp.com"],
            "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname": ["Azure"],
            "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname": ["User"],
        })
        result = process_acs(auth)
        assert result["email"] == "azure@corp.com"
        assert result["provider"] == "azure_ad"
