"""Fernet-based symmetric encryption for DB credentials."""

import json
import os
from cryptography.fernet import Fernet, InvalidToken


def _get_fernet() -> Fernet:
    key = os.getenv("DB_ENCRYPTION_KEY")
    if not key:
        raise RuntimeError("DB_ENCRYPTION_KEY environment variable is not set.")
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt_credentials(data: dict) -> str:
    """Serialize *data* to JSON and return a Fernet-encrypted string."""
    fernet = _get_fernet()
    plaintext = json.dumps(data).encode()
    return fernet.encrypt(plaintext).decode()


def decrypt_credentials(blob: str) -> dict:
    """Decrypt a Fernet-encrypted blob and return the original dict.

    Raises:
        InvalidToken: if the blob is tampered or the key is wrong.
    """
    fernet = _get_fernet()
    try:
        plaintext = fernet.decrypt(blob.encode() if isinstance(blob, str) else blob)
    except InvalidToken as exc:
        raise InvalidToken("Credential decryption failed — invalid token or key mismatch.") from exc
    return json.loads(plaintext.decode())
