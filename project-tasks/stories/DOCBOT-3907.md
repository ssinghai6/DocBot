# DOCBOT-3907 TypeScript SDK
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-39 API, SDK and MCP Server | 3 | parked | P3 | 3 | frontend | - | Parked (not scheduled) |

## User story
As a TypeScript developer, I want a published SDK with Zod types, so that I get typed, validated responses.

## Why now
Parked in master backlog section 7.13; scheduled only after the playground slice (S1, S2) ships or a buyer asks.

## Scope
In:
- As written in master backlog section 7.13, reusing `/v1` scopes, test keys and the sandbox.

Out:
- Anything outside the section 7.13 AC.

## Acceptance criteria
- [ ] Given the spec, when the SDK is generated, then its types come from `src/lib/schemas/v1.ts`.
- [ ] Given the package, when it is installed, then the sandbox quickstart example runs.
- [ ] Given a response that does not match the schema, when it is parsed, then Zod raises.

## Deliverable and approach
Build on the `/v1` contract (3901); no parallel API surface. Repo rules from `CLAUDE.md` apply (F-4 routes, models + Zod mirror, auth guard, tests in `tests/`). Plan: `docs/wiki/api/04-implementation-backlog.md` (stories), `docs/wiki/api/02-public-api-plan.md` (design), `docs/wiki/api/03-public-api-reference.md` (contract).

## Test / verification
Unit tests plus an example run.

## Dependencies
Blocked by: 3901.
Blocks: none.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] OpenProject ticket updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- When to pull this from the parking lot?
