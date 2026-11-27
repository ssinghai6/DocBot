# DOCBOT-3510 RB-10 SQL-through-NL attack set: validator unit tests + DB checksum + cross-tenant read
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-35 | 1 | S1 | P0 | 3 | senior-dev | - | New (scheduled from parking lot) |

## User story
As a fractional CFO connecting a client's live accounting database, I want proof that no question, document text or data value can make DocBot change, lock, read files from, or reach outside that database, so that I can grant DocBot a connection without risking the client's books.

## Why now
Design partners connect real databases in S1. The 3-layer read-only rule is a `CLAUDE.md` invariant but has no attack-shaped test suite, and a local check against the current validator finds gaps (below).

## Scope
In:
- An owned attack case file (P2SQL-style) covering: write/DDL, stacked queries, writable CTEs, comment smuggling, `ATTACH`, `PRAGMA`, `COPY ... PROGRAM`, `INTO OUTFILE`, side-effect and file-read functions inside a SELECT (`pg_read_file`, `lo_import`, `dblink`, `pg_sleep`, `set_config`, sqlite `load_extension`, MySQL `LOAD_FILE`/`SLEEP`), row locks (`FOR UPDATE`), system catalogs, unicode lookalikes, and instructions planted in row values and column comments.
- Validator hardening so every side-effect case is rejected: function and catalog denylist on the sqlglot AST, reject `exp.Lock`.
- Layer-3 gap fixes in `_execute_query` for SQLite.
- End-to-end checksum test and cross-tenant read tests.
Out:
- Sandbox exfiltration (3511); prompt-injection eval on documents (later RB items).
- Azure SQL read-only enforcement beyond documenting it (see open questions).

## Acceptance criteria
- [ ] Given the attack file (>= 40 cases across sqlite, postgres, mysql, tsql), when `pytest tests/unit/test_sql_attack_set.py` runs offline in CI, then `validate_and_sanitize_sql` rejects 100% of side-effect cases and accepts 100% of a control set of >= 15 legitimate tie-out SELECTs (UNION ALL, CTEs, window functions).
- [ ] Given a seeded SQLite DB and a mocked SQL generator that returns each attack, when every case is pushed through `run_sql_pipeline`, then a sha256 of a full ordered dump of every table is identical before and after.
- [ ] Given the validator bypassed in a test, when `_execute_query` runs `DELETE` on SQLite, then it fails with a read-only error (layer 3 holds on its own). The PostgreSQL equivalent (`SET TRANSACTION READ ONLY`) runs under `@pytest.mark.postgres`.
- [ ] Given two owners A and B, when A asks "show connections for all users" or calls the query route with B's `connection_id`, then the route returns 404 and no B rows appear.
- [ ] Given the rejection path, when a case is rejected, then the error message never contains the SQL text (existing `QueryValidationError` contract).

## Deliverable and approach
- Local check against `api/utils/sql_validator.py` with sqlglot 30.31.0: these are accepted today: `SELECT pg_read_file('/etc/passwd')`, `SELECT lo_import(...)`, `SELECT * FROM dblink('host=x','DELETE FROM t') AS t(a int)`, `SELECT pg_sleep(30)`, `SELECT set_config(...)`, sqlite `SELECT load_extension('x')`, MySQL `SELECT LOAD_FILE(...)` and `SELECT SLEEP(30)`, `SELECT * FROM pg_shadow`, `SELECT * FROM sqlite_master`, `SELECT ... FOR UPDATE`. Rejected correctly: stacked statements, writable CTE, `INTO OUTFILE`, `ATTACH`, `PRAGMA`, `COPY ... TO PROGRAM`. `dblink` matters most: it opens a new connection outside the `READ ONLY` transaction, so it bypasses layer 3.
- `api/utils/sql_validator.py`: add a `_DENY_FUNCTIONS` set (checked on `exp.Anonymous` names and known `exp.Func` classes, case-insensitive) and `_DENY_TABLE_PREFIXES` (`pg_`, `information_schema` unless the question is a schema question, `sqlite_`, `mysql.`, `sys.`); reject any `exp.Lock`. AST only, no regex (`CLAUDE.md`).
- `api/db_service.py:_execute_query` (lines 2054-2110): SQLite has no layer 3 today (comment says "no write ops possible via SELECT-only validated SQL"). Add `PRAGMA query_only = ON` per connection (or open with `mode=ro`). Azure SQL uses `SET TRANSACTION ISOLATION LEVEL SNAPSHOT`, which is not read-only.
- Cases in `tests/unit/sql_attack_cases.py` (data module) so the robustness suite can reuse them; OWASP LLM05/LLM06 ids per case.
- Cross-tenant: extend `tests/integration/test_connection_scoping.py`.
- Branch `feature/DOCBOT-3510-sql-attack-set`.

## Test / verification
- `tests/unit/test_sql_attack_set.py` (offline, CI).
- `tests/integration/test_sql_readonly_checksum.py` (temp SQLite, mocked LLM, CI).
- `tests/integration/test_sql_readonly_postgres.py` with `@pytest.mark.postgres` (skipped in CI).
- `pytest tests/unit/test_sql_attack_set.py tests/integration/test_sql_readonly_checksum.py tests/integration/test_connection_scoping.py -v`. Cost $0 (no live LLM).

## Dependencies
Blocked by: none.  Blocks: 3518 (T0 wiring), 3006 (isolation matrix, parked).

## Definition of done
- [ ] AC met  - [ ] tests pass in CI  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- Points: the backlog sizes this as tests only (3). Making the "100% rejected" AC pass needs validator and SQLite layer-3 code changes, so 5 is more realistic. Alternative: keep 3, mark failing cases `xfail(strict=True)` and file a fix ticket. That would leave a known `dblink` bypass open while DPs connect databases, so fixing here is recommended.
- Azure SQL has no session-level read-only switch; layer 3 there can only be a read-only DB user (`db_datareader`). Document it in the connection UI and the security page, or block Azure until then.
- Catalog reads are legitimate for schema questions. Decide whether the denylist is unconditional or allowed only on the schema-introspection path (`db_service` step 1), which never goes through the LLM.
