# DOCBOT-3903 Signed webhooks: run.completed, discrepancy.found, workpaper.approved
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-39 API, SDK and MCP Server | 2 | parked | P2 | 3 | senior-dev | - | Parked (not scheduled) |

## User story
As an API client, I want signed webhooks when runs finish or discrepancies are found, so that I do not have to poll.

## Why now
Parked in master backlog section 7.13; scheduled only after the playground slice (S1, S2) ships or a buyer asks.

## Scope
In:
- As written in master backlog section 7.13, reusing `/v1` scopes, test keys and the sandbox.

Out:
- Anything outside the section 7.13 AC.

## Acceptance criteria
- [ ] Given a registered endpoint, when a run completes, then a `run.completed` event is POSTed with an HMAC signature header.
- [ ] Given the endpoint fails, when delivery is retried, then retries back off and stop after the limit.
- [ ] Given a payload with a bad signature, when the documented check runs, then it fails.

## Deliverable and approach
Build on the `/v1` contract (3901); no parallel API surface. Repo rules from `CLAUDE.md` apply (F-4 routes, models + Zod mirror, auth guard, tests in `tests/`). Plan: `docs/wiki/api/04-implementation-backlog.md` (stories), `docs/wiki/api/02-public-api-plan.md` (design), `docs/wiki/api/03-public-api-reference.md` (contract).

## Test / verification
Unit tests plus an example run.

## Dependencies
Blocked by: 3902.
Blocks: none.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] OpenProject ticket updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- When to pull this from the parking lot?
