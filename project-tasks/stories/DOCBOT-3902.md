# DOCBOT-3902 Tie-out run as async API job with polling
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-39 API, SDK and MCP Server | 2 | parked | P1 | 5 | senior-dev | - | Parked (not scheduled) |

## User story
As an API client, I want to start a tie-out or autopilot run as a job and poll it, so that long runs do not hold a connection open.

## Why now
Parked in master backlog section 7.13; scheduled only after the playground slice (S1, S2) ships or a buyer asks.

## Scope
In:
- As written in master backlog section 7.13, reusing `/v1` scopes, test keys and the sandbox.

Out:
- Anything outside the section 7.13 AC.

## Acceptance criteria
- [ ] Given `POST /v1/jobs`, when a tie-out is started, then it returns a job id at once.
- [ ] Given a finished job, when `GET /v1/jobs/{id}` is called, then its result equals the UI result for the same input.
- [ ] Given a job of another key's org, when it is polled, then the API returns 404.

## Deliverable and approach
Build on the `/v1` contract (3901); no parallel API surface. Repo rules from `CLAUDE.md` apply (F-4 routes, models + Zod mirror, auth guard, tests in `tests/`). Plan: `docs/wiki/api/04-implementation-backlog.md` (stories), `docs/wiki/api/02-public-api-plan.md` (design), `docs/wiki/api/03-public-api-reference.md` (contract).

## Test / verification
Unit tests plus an example run.

## Dependencies
Blocked by: 3901, 3305.
Blocks: 3903, 3904.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] OpenProject ticket updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- When to pull this from the parking lot?
