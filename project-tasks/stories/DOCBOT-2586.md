# DOCBOT-2586 NEW: Move ticket tracking from Plane to self-hosted OpenProject (pm.docbot.space)
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | - | P2 | 2 | PM | - | In progress (2026-11-22) |

## User story
As the founder, I want every DocBot ticket (old, current and planned) on a free, self-hosted agile board with epics, features, user stories, sprints and story points, so that I can plan and track velocity without paying for Plane or Jira.

## Why now
Plane's paid tier is no longer an option and Jira would not stay free. AWS credits cover a small Lightsail server, so OpenProject (open source) costs nothing out of pocket.

## Scope
In:
- OpenProject 17 on AWS Lightsail (2 GB, us-east-1) behind Caddy with Let's Encrypt at https://pm.docbot.space.
- `scripts/openproject_sync.py`: full sync (Epic -> Feature -> User story, sprints S0..S7, points, priority, full story markdown) plus every other ticket in the repo docs (EPIC-01..23 history, EPIC-27..40 parking lot); `--set-state` per ticket with comment.
- `.github/workflows/openproject-sync.yml` (renamed from `plane-sync.yml`) using repo secret `OPENPROJECT_API_KEY`.
- CLAUDE.md OpenProject Sync Rule replaces the Plane rule; `scripts/README-openproject.md` covers server ops.

Out:
- Deleting the Plane workspace (founder does it after a week on the new board).
- Triage of the 39 On hold tickets whose status the docs never recorded.

## Acceptance criteria
- [ ] Given the board at https://pm.docbot.space, when the founder logs in, then the DocBot project shows epics, features and user stories with sprints S0..S7 and story points.
- [ ] Given a full sync just ran, when it runs again, then it reports 0 created and 0 updated.
- [ ] Given a story moved forward on the board, when the full sync runs, then its status does not move backwards.
- [ ] Given a ticket not on the board, when `--set-state DOCBOT-XXXX "In Progress" --title "T"` runs, then it is created In progress with a comment if `--note` is given.
- [ ] Given no env var and no Keychain item, when the script runs, then it exits 2 with the Keychain command and makes no request.

## Deliverable and approach
Reuse the Plane backlog parser (`plane_sync.load_stories`) and add an archive loader for the other docs. Talk to the OpenProject API v3 with an API token; sprints are created once on the server with `rails runner` because the sprint API is read-only.

## Test / verification
`python3 scripts/openproject_sync.py --self-test`; real run loaded 450 stories under 41 epics; second run 0 created / 0 updated; `--set-state` on this ticket.

## Dependencies
Blocked by: none.
Blocks: none.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] OpenProject ticket updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Resize the server to 4 GB if pages get slow (memory is near the 2 GB limit)?
