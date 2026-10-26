"""DOCBOT-1618: Sentry events must never carry raw secrets."""

import json
from unittest.mock import patch

from api.utils import sentry_setup
from api.utils.sentry_setup import before_breadcrumb, before_send, init_sentry

def _fake(prefix: str, n: int, suffix: str = "") -> str:
    """Build fake secrets at runtime so no literal in this file trips gitleaks."""
    return prefix + ("Ab1Cd2Ef3Gh4" * 8)[:n] + suffix


SECRETS = {
    "groq": _fake("gsk" + "_", 32),
    "openai": _fake("sk" + "-proj-", 24),
    "google": _fake("AI" + "za", 35),
    "e2b": _fake("e2b" + "_", 32),
    "resend": _fake("re" + "_", 24),
    "stripe_live": _fake("sk_" + "live_", 24),
    "stripe_test": _fake("sk_" + "test_", 24),
    "stripe_whsec": _fake("wh" + "sec_", 24),
    "fernet_key": _fake("", 43, "="),
    "fernet_token": _fake("gAAAA" + "A", 40, "=="),
    "db_pass": _fake("Db", 12),
    "password_field": _fake("Pw", 10),
    "bearer": _fake("ey" + "J", 20) + ".payload.sig",
    "cookie": "session=" + _fake("ck", 12),
    "owner_key": _fake("ok-", 12),
}


def _canary_event() -> dict:
    blob = " ".join(f"{k}={v}" for k, v in SECRETS.items() if k not in ("password_field", "cookie", "db_pass", "bearer"))
    db_url = f"postgresql://admin:{SECRETS['db_pass']}@db.internal:5432/app"
    return {
        "message": f"Failed calling Groq with {SECRETS['groq']} and DB {db_url} — keep this text",
        "logentry": {"message": f"key leak {SECRETS['openai']}", "params": [SECRETS["google"]]},
        "exception": {
            "values": [
                {
                    "type": "ValueError",
                    "value": f"bad creds {blob} password={SECRETS['password_field']}",
                    "stacktrace": {"frames": [{"function": "query", "vars": {"api_key": SECRETS["e2b"]}}]},
                }
            ]
        },
        "extra": {
            "token": SECRETS["bearer"],
            "nested": {"conn": db_url, "list": [SECRETS["resend"], {"secret": SECRETS["stripe_whsec"]}]},
            "stripe": (SECRETS["stripe_live"], SECRETS["stripe_test"]),
            "fernet": [SECRETS["fernet_key"], SECRETS["fernet_token"]],
            "row_count": 42,
            "note": "plain diagnostic text",
        },
        "request": {
            "url": "https://api.docbot.app/api/chat",
            "headers": {
                "Authorization": f"Bearer {SECRETS['bearer']}",
                "Cookie": SECRETS["cookie"],
                "X-Owner-Key": SECRETS["owner_key"],
                "User-Agent": "pytest-agent",
            },
            "cookies": {"session": SECRETS["cookie"]},
            "data": {"password": SECRETS["password_field"]},
        },
        "breadcrumbs": {
            "values": [
                {"category": "httplib", "data": {"url": f"https://x.test/?api_key={SECRETS['groq']}"}},
                {"message": f"Authorization: Bearer {SECRETS['bearer']}"},
            ]
        },
    }


def test_before_send_removes_every_secret():
    out = json.dumps(before_send(_canary_event(), None))
    for name, secret in SECRETS.items():
        assert secret not in out, f"{name} leaked"


def test_before_send_preserves_non_secret_text():
    event = before_send(_canary_event(), None)
    assert "keep this text" in event["message"]
    assert "postgresql://admin:***@db.internal:5432/app" in event["message"]
    assert event["extra"]["row_count"] == 42
    assert event["extra"]["note"] == "plain diagnostic text"
    assert event["request"]["headers"] == {"User-Agent": "pytest-agent"}
    assert "cookies" not in event["request"]
    assert event["request"]["url"] == "https://api.docbot.app/api/chat"
    assert event["exception"]["values"][0]["type"] == "ValueError"


def test_before_breadcrumb_scrubs():
    crumb = {"category": "log", "message": f"connecting mysql://root:{SECRETS['db_pass']}@h/db with {SECRETS['e2b']}"}
    out = json.dumps(before_breadcrumb(crumb, None))
    assert SECRETS["db_pass"] not in out and SECRETS["e2b"] not in out
    assert "mysql://root:***@h/db" in out


def test_init_noop_without_dsn(monkeypatch):
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    with patch("sentry_sdk.init") as init:
        assert init_sentry() is False
    init.assert_not_called()


def test_init_kwargs(monkeypatch):
    monkeypatch.setenv("SENTRY_DSN", "https://public@o0.ingest.sentry.io/0")
    with patch("sentry_sdk.init") as init:
        assert init_sentry() is True
    kwargs = init.call_args.kwargs
    assert kwargs["include_local_variables"] is False
    assert kwargs["max_request_body_size"] == "never"
    assert kwargs["send_default_pii"] is False
    assert kwargs["traces_sample_rate"] == 0.0
    assert kwargs["before_send"] is sentry_setup.before_send
    assert kwargs["before_breadcrumb"] is sentry_setup.before_breadcrumb
