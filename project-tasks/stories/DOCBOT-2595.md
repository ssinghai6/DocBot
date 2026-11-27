# DOCBOT-2595 NEW: Number and file the public API playground backlog (EPIC-39)
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | - | P2 | 1 | PM | - | In progress (2026-11-23) |

## User story
As the founder, I want the public API playground plan filed as real numbered tickets, so that the work can be picked into sprints and built next.

## Why now
The plan (DOCBOT-2590) used NEW-xx placeholders. Nothing can be scheduled or synced to the board until each story has an id, a backlog row and a story file.

## Scope
In:
- Real DOCBOT ids for every planned story; reuse 3901-3907 and 2907 where the plan maps them.
- Backlog rows (sections 7.3, 7.13, 8) and a story file per id.
- Placeholders replaced in `docs/wiki/api/01..04`.
- `EPIC-39 API` track mapped to the EPIC-39 epic in the sync scripts.

Out:
- Any app code.
- Writing to the OpenProject board (the sync workflow does it after merge).

## Acceptance criteria
- [ ] Given the API docs in `docs/wiki/api/`, when searched for `NEW-`, then no placeholder id is left.
- [ ] Given each new or reused id, when `python3 scripts/validate_stories.py --only <id> --source-ref HEAD` runs, then it reports 0 missing and 0 failing.
- [ ] Given the backlog at HEAD, when `python3 scripts/openproject_sync.py --dry-run --source-ref HEAD` runs, then the new stories are planned under the EPIC-39 epic.
- [ ] Given bugs DOCBOT-2591 and DOCBOT-2592, when the plan is filed, then they are referenced as dependencies and not re-filed.

## Deliverable and approach
Map each placeholder to an id (2596-2599, then 2611-2618, since 2600-2610 are EPIC-26). Scheduled stories go in section 8 with track `EPIC-39 API`; reused parking-lot rows are edited in place. Repo rules from `CLAUDE.md` apply (F-4 routes, models + Zod mirror, auth guard, tests in `tests/`). Plan: `docs/wiki/api/04-implementation-backlog.md` (stories), `docs/wiki/api/02-public-api-plan.md` (design), `docs/wiki/api/03-public-api-reference.md` (contract).

## Test / verification
Validator per id; sync dry run at HEAD.

## Dependencies
Blocked by: none.
Blocks: 2596-2599, 2611-2618 (filing only).

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] OpenProject ticket updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Sprints S1 and S2 are now over capacity; which existing tickets move out?
