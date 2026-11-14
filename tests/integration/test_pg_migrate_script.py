"""DOCBOT-2515: infra/aws/scripts/pg_migrate.sh against two throwaway local clusters.

Needs local initdb/pg_ctl (marked postgres, skipped in CI). Builds a source DB
with DocBot's immutability trigger + a serial column, a target DB that the app
already initialised (empty tables), runs the script and checks it verifies,
never prints the connection strings, and refuses a non-RDS target by default.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess  # nosec B404 - runs local postgres binaries and a repo script, fixed argv
from pathlib import Path

import pytest

pytestmark = pytest.mark.postgres

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "infra" / "aws" / "scripts" / "pg_migrate.sh"
PASSWORD = ":".join(["local", "only@/x"])  # throwaway cluster; URL-reserved chars on purpose

SCHEMA = """
CREATE TABLE audit_log (id SERIAL PRIMARY KEY, detail TEXT, ts TIMESTAMPTZ DEFAULT now());
CREATE FUNCTION audit_log_immutable() RETURNS TRIGGER LANGUAGE plpgsql AS
  $$ BEGIN RAISE EXCEPTION 'audit_log rows are immutable'; END; $$;
CREATE TRIGGER audit_log_no_mutate BEFORE UPDATE OR DELETE ON audit_log
  FOR EACH ROW EXECUTE FUNCTION audit_log_immutable();
CREATE TABLE sessions (id TEXT PRIMARY KEY, blob BYTEA, meta JSONB);
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(*argv: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, env=env, timeout=120)  # nosec B603


@pytest.fixture(scope="module")
def clusters(tmp_path_factory):
    if not (shutil.which("initdb") and shutil.which("pg_ctl")):
        pytest.skip("local initdb/pg_ctl not available")
    urls = []
    dirs = []
    for name in ("src", "tgt"):
        d = tmp_path_factory.mktemp(name)
        pw = d / "pw"
        pw.write_text(PASSWORD)
        _run("initdb", "-D", str(d / "data"), "-U", "mig", "--auth=md5", f"--pwfile={pw}").check_returncode()
        port = _free_port()
        _run("pg_ctl", "-D", str(d / "data"), "-l", str(d / "log"), "-w",
             "-o", f"-p {port} -c unix_socket_directories='' -c listen_addresses=127.0.0.1", "start").check_returncode()
        dirs.append(d / "data")
        from urllib.parse import quote
        urls.append(f"postgresql://mig:{quote(PASSWORD, safe='')}@127.0.0.1:{port}/postgres")
    env = {**os.environ, "PGPASSWORD": PASSWORD}
    psql = lambda url, sql: _run("psql", "-X", "-v", "ON_ERROR_STOP=1", "-d", url, "-c", sql, env=env)  # noqa: E731
    psql(urls[0], SCHEMA).check_returncode()
    psql(urls[0], "INSERT INTO audit_log (detail) SELECT 'e' || g FROM generate_series(1, 50) g;"
                  "INSERT INTO sessions VALUES ('a', '\\x00ff', '{\"k\": 1}'), ('b', NULL, NULL);").check_returncode()
    psql(urls[1], SCHEMA).check_returncode()  # target app already ran init_db (empty tables)
    yield urls
    for d in dirs:
        _run("pg_ctl", "-D", str(d), "-m", "immediate", "stop")


def _migrate(urls, **extra: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PG")}
    env.update(SOURCE_DATABASE_URL=urls[0], TARGET_DATABASE_URL=urls[1], **extra)
    return _run("bash", str(SCRIPT), env=env)


def test_refuses_non_rds_target(clusters):
    r = _migrate(clusters)
    assert r.returncode == 2
    assert "not an RDS instance" in r.stderr


def test_migrates_and_verifies(clusters):
    r = _migrate(clusters, ALLOW_ANY_TARGET="1")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "RESULT: VERIFIED" in out
    assert "audit_log_no_mutate blocks DELETE" in out
    assert "audit_log_id_seq" in out
    assert PASSWORD not in out and "postgresql://" not in out
    # Re-running is idempotent (--clean --if-exists) and still verifies.
    again = _migrate(clusters, ALLOW_ANY_TARGET="1")
    assert again.returncode == 0, again.stdout + again.stderr
