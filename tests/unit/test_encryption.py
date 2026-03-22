"""Unit tests for Fernet credential encryption — DOCBOT-201."""

import pytest
from cryptography.fernet import Fernet, InvalidToken
from api.utils.encryption import encrypt_credentials, decrypt_credentials


@pytest.mark.unit
class TestEncryption:

    def test_round_trip(self):
        data = {"dialect": "postgresql", "host": "db.example.com", "port": 5432,
                "dbname": "mydb", "user": "admin", "password": "super_secret_123"}
        blob = encrypt_credentials(data)
        recovered = decrypt_credentials(blob)
        assert recovered == data

    def test_password_not_visible_in_blob(self):
        data = {"password": "super_secret_password"}
        blob = encrypt_credentials(data)
        assert "super_secret_password" not in blob

    def test_blob_is_string(self):
        blob = encrypt_credentials({"key": "value"})
        assert isinstance(blob, str)
        assert len(blob) > 50

    def test_tampered_blob_raises_invalid_token(self):
        blob = encrypt_credentials({"key": "value"})
        tampered = blob[:-5] + "XXXXX"
        with pytest.raises(InvalidToken):
            decrypt_credentials(tampered)

    def test_empty_dict_round_trip(self):
        assert decrypt_credentials(encrypt_credentials({})) == {}

    def test_nested_values_round_trip(self):
        data = {"port": 5432, "ssl": True, "tags": ["a", "b"]}
        assert decrypt_credentials(encrypt_credentials(data)) == data

    def test_missing_key_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("DB_ENCRYPTION_KEY", raising=False)
        with pytest.raises(RuntimeError, match="DB_ENCRYPTION_KEY"):
            encrypt_credentials({"x": 1})
