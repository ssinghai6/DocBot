# DOCBOT-2604 Prod golden-path hardening + freeze
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | - | P0 | 2 | senior-dev | 11-27 | In progress (EPIC-26, due by 12-01) |

## User story
As the founder demoing to an investor and to first prospects in the week of 11-30, I want the demo golden path proven 10 out of 10 times on prod with p95 latency recorded and main frozen from 11-27, so that nothing I show breaks because of a last-minute merge.

## Why now
Live demo on 12-01 and proof-runs for booked calls (2550) both run on prod (Vercel + Railway). Cutover is mid-January, so prod for the pitch is Railway.

## Scope
In:
- 10 consecutive golden-path runs on docbot.space (same 5 steps as 2603)
- p50/p95 latency per step recorded
- Fixes for any blocker found, each on its own `feature/DOCBOT-2604-*` branch with a test
- Freeze rule: from 2026-11-27 no merges to main except fixes for this path
Out:
- AWS env (Wave A uses api-dev, not prod)
- New features or copy changes after 11-27

## Acceptance criteria
- [ ] Given prod on or after 11-25, when the golden path runs 10 times on a fresh demo session each time, then 10/10 complete with the expected discrepancy flagged and cited
- [ ] Given the run log, when read, then it records p50 and p95 seconds per step and the commit SHA deployed
- [ ] Given any failed run, when fixed, then the fix has a unit or integration test in `tests/` and the 10-run set is repeated from zero
- [ ] Given main between 2026-11-27 and 2026-12-01, when `git log origin/main --since=2026-11-27` is checked, then every commit references DOCBOT-2604 or a golden-path fix
- [ ] Given Sentry for the run window, when filtered to prod, then no unhandled error is attributed to the golden-path routes

## Deliverable and approach
Run log: `project-tasks/preseed/2604-golden-path-runs.md` (run #, timestamp, SHA, per-step seconds, pass/fail, note). Routes touched by the path: demo init in `api/demo_service.py`, hybrid chat in `api/hybrid_service.py`, discrepancy in `api/utils/discrepancy_detector.py`, export from DOCBOT-2014, autopilot in `api/autopilot_service.py`. Reuse the 2505 smoke script pattern if it is merged; otherwise run by hand.

## Test / verification
Run log reviewed by founder; CI green on every fix; final run on 11-30 per the day-by-day plan.

## Dependencies
Blocked by: none  Blocks: 2603

## Definition of done
- [ ] AC met  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- Freeze conflicts: 2537 (landing copy) merges 11-25 and AWS code PRs (2507/2508/2511/2512) merge on 11-30 per the day-by-day plan. Confirm infra-only PRs under `infra/aws/**` are exempt from the freeze because they do not deploy to prod.
