# DOCBOT-2406 Single-replica assertion + scheduler leader lock
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-24 Stage A | 1 | S4 | P1 | 1 | senior-dev | - | Not on main |

## User story
As the founder, I want the API to refuse to run background jobs on a second replica, so that connector syncs and cleanups never run twice and Chroma on one volume is not written by two processes.

## Why now
APScheduler starts inside the FastAPI lifespan on every process (`api/index.py` around line 700, `api/connectors/sync_jobs.py`). `max_instances=1` only guards within one process. On AWS, Chroma on EFS forces `desired_count <= 1`, but nothing in the app enforces it, and the 2519 cutover briefly has Railway and AWS both up.

## Scope
In:
- Postgres advisory lock (`pg_try_advisory_lock`) taken before `scheduler.start()`; only the holder schedules.
- Startup log line and health field showing leader or follower.
- SQLite/dev path: always leader.

Out:
- RLS, Alembic, pgvector, BYOK and KMS (Stage A exclusions: rebuilt in the platform).
- Multi-replica support for Chroma.

## Acceptance criteria
- [ ] Given two API processes on one Postgres, when both start, then exactly one logs scheduler leader and the other logs follower and registers no jobs.
- [ ] Given the leader stops, when the follower restarts, then it becomes leader within one restart.
- [ ] Given SQLite in tests, when the app starts, then the scheduler runs as today and the existing sync job tests pass.

## Deliverable and approach
- Small helper in `api/connectors/sync_jobs.py` called from the lifespan before `register_all_active_connectors()`; hold the lock on a dedicated connection for the process lifetime.

## Test / verification
- `@pytest.mark.postgres` test with two connections; unit test for the SQLite branch.

## Dependencies
Blocked by: none
Blocks: 2519 (both stacks up during cutover)

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Should the follower also refuse to serve writes to Chroma, or is scheduler-only enough while `desired_count <= 1`?
