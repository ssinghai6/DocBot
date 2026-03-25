"""Unit tests for api/oauth_service.py — DOCBOT-701.

All tests are CI-safe: no network calls, no real DB.
"""

import os
import time
import pytest

from api.oauth_service import (
    is_github_configured,
    is_google_configured,
    get_available_auth_methods,
    generate_oauth_state,
    validate_oauth_state,
    hash_password,
    verify_password,
    validate_password_strength,
    github_authorize_url,
    google_authorize_url,
)


# ---------------------------------------------------------------------------
# Provider configuration checks
# ---------------------------------------------------------------------------

class TestProviderConfig:
    def test_github_not_configured_by_default(self, monkeypatch):
        monkeypatch.delenv("GITHUB_CLIENT_ID", raising=False)
        monkeypatch.delenv("GITHUB_CLIENT_SECRET", raising=False)
        assert is_github_configured() is False

    def test_github_configured_when_both_vars_set(self, monkeypatch):
        monkeypatch.setenv("GITHUB_CLIENT_ID", "gh-id")
        monkeypatch.setenv("GITHUB_CLIENT_SECRET", "gh-secret")
        assert is_github_configured() is True

    def test_google_not_configured_by_default(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
        monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)
        assert is_google_configured() is False

    def test_google_configured_when_both_vars_set(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "gc-id")
        monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "gc-secret")
        assert is_google_configured() is True

    def test_email_always_in_available_methods(self, monkeypatch):
        monkeypatch.delenv("GITHUB_CLIENT_ID", raising=False)
        monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
        methods = get_available_auth_methods()
        assert "email" in methods

    def test_github_in_methods_when_configured(self, monkeypatch):
        monkeypatch.setenv("GITHUB_CLIENT_ID", "x")
        monkeypatch.setenv("GITHUB_CLIENT_SECRET", "y")
        methods = get_available_auth_methods()
        assert "github" in methods

    def test_google_in_methods_when_configured(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "x")
        monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "y")
        methods = get_available_auth_methods()
        assert "google" in methods


# ---------------------------------------------------------------------------
# OAuth state CSRF protection
# ---------------------------------------------------------------------------

class TestOAuthState:
    def test_generated_state_validates(self):
        state = generate_oauth_state()
        assert validate_oauth_state(state) is True

    def test_state_can_only_be_used_once(self):
        state = generate_oauth_state()
        validate_oauth_state(state)  # consume it
        assert validate_oauth_state(state) is False

    def test_unknown_state_rejected(self):
        assert validate_oauth_state("totally-fake-state") is False

    def test_expired_state_rejected(self, monkeypatch):
        import api.oauth_service as svc
        state = generate_oauth_state()
        # Manually set expiry to the past
        svc._pending_states[state] = time.time() - 1
        assert validate_oauth_state(state) is False


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

class TestPasswordHash:
    def test_hash_and_verify(self):
        h = hash_password("correct-horse-battery-staple")
        assert verify_password("correct-horse-battery-staple", h) is True

    def test_wrong_password_fails(self):
        h = hash_password("correct-horse-battery-staple")
        assert verify_password("wrong-password", h) is False

    def test_hash_is_not_plaintext(self):
        pw = "mysecret"
        assert hash_password(pw) != pw

    def test_same_password_produces_different_hashes(self):
        pw = "same-password"
        assert hash_password(pw) != hash_password(pw)  # bcrypt uses random salt


class TestPasswordStrength:
    def test_short_password_rejected(self):
        assert validate_password_strength("short") is not None

    def test_eight_char_password_accepted(self):
        assert validate_password_strength("12345678") is None

    def test_long_password_accepted(self):
        assert validate_password_strength("a" * 20) is None


# ---------------------------------------------------------------------------
# OAuth URL building
# ---------------------------------------------------------------------------

class TestOAuthUrls:
    def test_github_url_contains_client_id(self, monkeypatch):
        monkeypatch.setenv("GITHUB_CLIENT_ID", "my-github-id")
        url = github_authorize_url("test-state")
        assert "my-github-id" in url
        assert "test-state" in url
        assert "github.com/login/oauth/authorize" in url

    def test_google_url_contains_client_id(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "my-google-id")
        url = google_authorize_url("test-state")
        assert "my-google-id" in url
        assert "test-state" in url
        assert "accounts.google.com" in url

    def test_github_url_includes_email_scope(self, monkeypatch):
        monkeypatch.setenv("GITHUB_CLIENT_ID", "id")
        url = github_authorize_url("s")
        assert "user%3Aemail" in url or "user:email" in url

    def test_google_url_includes_openid_scope(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "id")
        url = google_authorize_url("s")
        assert "openid" in url
