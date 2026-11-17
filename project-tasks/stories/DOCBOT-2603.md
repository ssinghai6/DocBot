# DOCBOT-2603 Demo script + backup video
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | - | P0 | 2 | founder | 11-26 | In progress (EPIC-26, due by 12-01) |

## User story
As the founder running a live demo on the 2026-12-01 investor call, I want a timed click-by-click script and a 3-minute backup video that plays offline, so that the "flagged discrepancy with both citations" moment lands even if prod, Wi-Fi or Groq fails.

## Why now
The demo moment is slide 2 of the deck. A failed live demo with no fallback costs the call. The prod freeze (2604) is 11-27, so the script must be final before then.

## Scope
In:
- Golden path on docbot.space: Try Demo, tie-out question, flagged discrepancy with page + table citation, XLSX export (DOCBOT-2014), one autopilot question
- Exact prompts to type, expected on-screen result per step, timing per step, recovery line if a step stalls
- 3-minute screen recording of the same path with captions, stored locally
Out:
- Prospect-doc demo for sales calls (2550)
- Private model mode (2609 moved after Dec 1)

## Acceptance criteria
- [ ] Given the script, when a second person follows it on prod, then each of the 5 steps produces the expected result written in the script, in under 6 minutes total
- [ ] Given the tie-out step, when run, then the answer shows at least one flagged figure with both the document page and the database table/query cited (the QuickBite demo data from `api/demo_service.py`)
- [ ] Given the export step, when the XLSX is downloaded, then it opens in Excel with the discrepancy row present
- [ ] Given the backup video, when played on the founder laptop with Wi-Fi off, then it plays end to end, is 3:00 or shorter, and has captions naming each step
- [ ] Given the script, when read, then each step has a one-line recovery ("if this stalls, switch to video at 0:45")

## Deliverable and approach
Script: `project-tasks/preseed/2603-demo-script.md` (table: step, click/prompt, expected result, seconds, recovery). Video: `project-tasks/preseed/docbot-demo-backup-3min.mp4`, plus a copy on the desktop. Not in `public/`, so it does not ship with the site.
Inputs: `api/demo_service.py` (demo init via `POST /api/demo/init`), `tests/eval/gold_tieout_set.py` for a question known to flag, 2014 export.

## Test / verification
Dry run by someone other than the founder on prod after the 11-27 freeze; log time per step in the script file. Video tested offline once.

## Dependencies
Blocked by: 2604 (golden path stable on prod)  Blocks: 2605 (video optional in data room), 2606 (mock calls use it)

## Definition of done
- [ ] AC met  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- Pick the tie-out question now: use one from `gold_tieout_set.py` whose live pass rate in 2543 is 10/10, not the segment Note 12 question unless 2543 shows >= 9/10.
