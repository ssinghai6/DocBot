# DOCBOT-3904 MCP server: read-only tools, OAuth, workspace scoping, audited
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-39 API, SDK and MCP Server | 2 | parked | P1 | 8 | ai-engineer | - | Parked (not scheduled) |

## User story
As an analyst using an MCP client, I want DocBot as read-only MCP tools, so that I can ask DocBot from my own agent.

## Why now
Parked in master backlog section 7.13; scheduled only after the playground slice (S1, S2) ships or a buyer asks.

## Scope
In:
- As written in master backlog section 7.13, reusing `/v1` scopes, test keys and the sandbox.

Out:
- Anything outside the section 7.13 AC.

## Acceptance criteria
- [ ] Given an MCP client with OAuth, when it lists tools, then 4 read-only tools appear.
- [ ] Given a tool call, when it runs, then it is scoped to the user's workspace and audited.
- [ ] Given a new user, when they onboard through MCP, then they reach the sandbox data first.

## Deliverable and approach
Build on the `/v1` contract (3901); no parallel API surface. Repo rules from `CLAUDE.md` apply (F-4 routes, models + Zod mirror, auth guard, tests in `tests/`). Plan: `docs/wiki/api/04-implementation-backlog.md` (stories), `docs/wiki/api/02-public-api-plan.md` (design), `docs/wiki/api/03-public-api-reference.md` (contract).

## Test / verification
Unit tests plus an example run.

## Dependencies
Blocked by: 3901, 3902.
Blocks: none.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] OpenProject ticket updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- When to pull this from the parking lot?
