"""DOCBOT-2334: the IP limiter skips exactly the Stripe webhook path, nothing else."""

from __future__ import annotations

import os

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

os.environ.setdefault("DB_ENCRYPTION_KEY", Fernet.generate_key().decode())
os.environ.setdefault("DATABASE_URL", "postgresql://fake:fake@localhost:5432/testdb")

import api.index as idx  # noqa: E402
from api.utils import request_guard  # noqa: E402


@pytest.fixture(autouse=True)
def _guards(monkeypatch):
    monkeypatch.setattr(request_guard, "RATE_LIMIT_PER_MIN", 1)
    monkeypatch.setattr(request_guard, "OWNER_KEY", "")
    monkeypatch.setattr(request_guard, "PUBLIC_DEMO_MODE", False)
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    request_guard.reset_state()
    yield
    request_guard.reset_state()


def test_webhook_exempt_but_neighbour_path_limited():
    client = TestClient(idx.app, raise_server_exceptions=False)
    for _ in range(3):
        # unset secret -> 503 from the route itself, never the limiter's 429
        assert client.post("/api/billing/webhook", content=b"{}").status_code == 503
    client.post("/api/billing/webhook/")
    assert client.post("/api/billing/webhook/").status_code == 429
