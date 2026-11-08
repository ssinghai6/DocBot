"""DOCBOT-2333: one-time X-Forwarded-For shape log never contains an address."""
import logging
from types import SimpleNamespace

from api.utils import request_guard


def _req(xff: str, peer: str = "10.0.0.5"):
    return SimpleNamespace(
        headers={"x-forwarded-for": xff} if xff else {},
        client=SimpleNamespace(host=peer),
    )


def test_logs_shape_once_without_addresses(monkeypatch, caplog):
    monkeypatch.setattr(request_guard, "_xff_shape_logged", False)
    monkeypatch.setattr(request_guard, "TRUSTED_PROXY_HOPS", 2)
    caplog.set_level(logging.INFO, logger=request_guard.__name__)

    ip = request_guard.client_ip(_req("9.9.9.9, 1.1.1.1, 10.1.2.3"))
    request_guard.client_ip(_req("8.8.8.8"))

    lines = [r.getMessage() for r in caplog.records if "xff-shape" in r.getMessage()]
    assert len(lines) == 1
    line = lines[0]
    assert "entries=3" in line and "chosen_index=1" in line
    assert "['public', 'public', 'private']" in line
    for addr in ("9.9.9.9", "1.1.1.1", "10.1.2.3", "10.0.0.5"):
        assert addr not in line
    assert ip == "1.1.1.1"  # behaviour unchanged


def test_zero_hops_logs_socket_peer(monkeypatch, caplog):
    monkeypatch.setattr(request_guard, "_xff_shape_logged", False)
    monkeypatch.setattr(request_guard, "TRUSTED_PROXY_HOPS", 0)
    caplog.set_level(logging.INFO, logger=request_guard.__name__)
    assert request_guard.client_ip(_req("", peer="1.1.1.1")) == "1.1.1.1"
    assert any("chosen_index=socket-peer" in r.getMessage() for r in caplog.records)
