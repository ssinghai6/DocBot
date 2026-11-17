# DOCBOT-2580 NEW: Plane auto-sync rule and per-ticket state command
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | - | P2 | 1 | PM | - | In progress (2026-11-17) |

## User story
As the founder, I want every ticket I ask Claude to build to appear in Plane and move through In Progress, Testing and Done on its own, so that the board is always true without me updating it.

## Why now
Tracking moved from Jira to Plane on 2026-11-17. CLAUDE.md still told agents to update Jira, and `plane_sync.py` could only bulk-sync the backlog, not move one ticket when a branch, PR or merge happens.

## Scope
In:
- `scripts/plane_sync.py --set-state TICKET STATE [--title] [--note]`: move one issue, create it if missing, add a comment.
- A `Testing` state in Plane (created by the script if missing) for PR open, CI or founder test.
- Key lookup: `PLANE_API_KEY` env var, else the macOS Keychain item `PLANE_API_KEY`.
- CLAUDE.md: Plane Sync Rule replaces the Jira rule; tracker pointers updated.
- Commit `plane_sync.py`, `validate_stories.py` and `README-plane.md` (no secrets in them) so every worktree has them.

Out:
- A GitHub Action that moves states on PR events (possible later; needs the key as a repo secret).

## Acceptance criteria
- [ ] Given a ticket not in Plane, when `--set-state DOCBOT-XXXX "In Progress" --title "T"` runs, then the issue is created in In Progress with `external_id=DOCBOT-XXXX`.
- [ ] Given an existing issue, when `--set-state DOCBOT-XXXX Testing --note "PR #N"` runs, then its state is Testing and the comment is on the issue.
- [ ] Given no env var and no Keychain item, when the script runs, then it exits 2 with the one-line Keychain command and makes no request.
- [ ] Given an issue moved to Testing or Done, when the full sync runs, then the state does not move backwards.
- [ ] Given CLAUDE.md on `main`, when it is read, then it has the Plane Sync Rule and no instruction to update Jira.

## Deliverable and approach
Small additions to the existing sync script (reuses its client, throttle and WAF encoding). Rule text in CLAUDE.md lists the five moments: create, start, PR/testing, merge, cancel.

## Test / verification
`python3 scripts/plane_sync.py --self-test`; `--set-state` on this ticket itself (In Progress, Testing on PR, Done on merge); full sync dry run shows 0 unexpected changes.

## Dependencies
Blocked by: founder stores the rotated Plane key in the Keychain.
Blocks: none.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Add the GitHub Action later so Plane also updates when the founder merges a PR outside a Claude session?
