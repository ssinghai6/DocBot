# DOCBOT-2401 Two-org isolation sweep in CI
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-24 Stage A | 1 | S3 | P0 | 5 | senior-dev | - | Not on main (docs/DOCBOT-2400 branch) |

## User story
As a fractional CFO uploading a client's books, I want proof that no other DocBot org can read my sessions, connections, lineage or vectors, so that I can put client data in DocBot without a cross-tenant leak.

## Why now
Design partners start uploading real client ledgers in S1-S3, and Firm workspace (2562+) builds on the org boundary. Today isolation is checked per route family (`tests/integration/test_admin_org_scoping.py`, `tests/unit/test_route_guards.py`), not as one sweep across every route plus the vector store. The strategy doc (docs/DOCBOT-2400 branch, Stage A) makes this sweep a never-cut item and reusable later as PLAT-0212.

## Scope
In:
- A pytest sweep that enumerates every FastAPI route from `app.routes` (the same source `tests/unit/test_route_guards.py` uses) and, for each route taking a resource id, creates the resource as org A and calls it as org B.
- Coverage for sessions, artifacts, db_connections, schema cache, lineage, traces, API keys, connectors, tie-out exports and Chroma retrieval (`api/utils/vector_store.py`).
- An allowlist file for routes with no resource id, each with a written reason, mirroring `PUBLIC_ROUTES`.
- Runs in the existing CI job (`pytest -m "not external and not postgres"`) on SQLite.

Out:
- RLS, Alembic, pgvector, BYOK and KMS (Stage A exclusions: rebuilt in the platform).
- Fixing every leak found beyond small fixes: larger fixes get their own ticket.

## Acceptance criteria
- [ ] Given two orgs A and B with one user each and one of every resource owned by A, when the sweep calls every id-taking route as B, then 100% of calls return 403 or 404 and none return A's data.
- [ ] Given a new route is added without being in the sweep or the allowlist, when CI runs, then the sweep test fails naming the route.
- [ ] Given a session created by org A with an indexed PDF, when org B asks a question with A's `session_id`, then no chunk from A's Chroma collection appears in the response or SSE stream.
- [ ] Given the sweep, when it runs in CI, then it finishes in under 60 seconds and makes no network calls (LLM, E2B and embeddings mocked).
- [ ] Given the sweep lands, when the tracker row is updated, then the count of routes covered and allowlisted is recorded there.

## Deliverable and approach
- `tests/integration/test_two_org_isolation.py` with fixtures from `tests/conftest.py`; build `CallerScope` for org A and B via the same auth helpers the route-guard test uses.
- Parametrize over `app.routes`; map path params (`session_id`, `connection_id`, `run_id`, `key_id`) to A's fixture ids.
- Any leak found: fix in `api/utils/caller_scope.py` `scope_filter`, `api/db_service.py` `connection_access_clause` or `api/resource_scope_service.py` `run_row_accessible`, not per route.
- Keep the harness importable so PLAT-0212 can reuse it.

## Test / verification
- The sweep itself in CI; a deliberately broken route in a local run proves it fails.
- Route count printed in the test summary and compared with `len(app.routes)`.

## Dependencies
Blocked by: none (strategy branch rows only need to land in the tracker)
Blocks: 2402, 2403, 2562, 2563 (Firm tier critical path)

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Should org B get 404 everywhere (hides existence) or is 403 acceptable on admin routes?
- Do `/v1` API-key routes join the same sweep or a second pass with key scope?
