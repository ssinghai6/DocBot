# DOCBOT-2550 Tie-out demo script on prospect docs
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 Sales ops | 0 | S0 | P0 | 2 | PM | 11-26 | Open; pulled before Dec 1 (2026-11-17) |

## User story
As a fractional CFO on a 15-minute discovery call, I want to watch DocBot tie my own 10-K or board pack to my books with a page and a ledger row for each figure, so that I can judge it on my data instead of a canned demo.

## Why now
Calls start 11-23 and the Dec 1 gate needs >= 6 held calls and 1 LOI or verbal DP yes. The free tie-out (2532) is the offer; this script is how it is shown live without surprises.

## Scope
In:
- A 15-minute call script: 2 min context, 8 min live tie-out on the prospect's docs (pre-run with 2532), 3 min exceptions walkthrough with page and row, 2 min ask.
- A fallback path to the built-in Tidewell demo when the prospect's files are not ready or a run misbehaves.
- Talk track and banned-claims list; pre-call and post-call checklists (including data deletion).
Out:
- Code changes; new demo datasets; recording a public video.

## Acceptance criteria
- [ ] Given the script, when it is rehearsed end to end on one public 10-K plus a QBO-style export, then it fits in 15 minutes (timed) and every figure shown on screen has a PDF page and a ledger row or query.
- [ ] Given a prospect run fails or returns a wrong page, when the operator follows the fallback, then the demo switches to `POST /api/demo/init` with `dataset="tieout"` in under 1 minute and shows the planted gaps (e.g. Lab Services in Note 12).
- [ ] Given the talk track, when it is reviewed against the claims rules, then it contains no "enterprise-grade", no FinanceBench/FinDER/FaithEval numbers, no 20/20 baseline, no "private mode demoed", and quotes only the 2543 number with date and model.
- [ ] Given each call held by 11-26, when it ends, then the 2551 gate tracker has a row with prospect, date, files used, outcome (pain quote, DP yes/no) and deletion confirmation.

## Deliverable and approach
- Script at `project-tasks/playbooks/DOCBOT-2550-tieout-demo-script.md`. Branch `docs/DOCBOT-2550-tieout-demo-script`.
- Before the call: run the 2532 playbook on the prospect's files (books as SQLite via `upload_sqlite`, `api/file_upload_service.py:55`) so the live portion replays known-good questions; keep the exported `.xlsx` from `GET /api/lineage/{run_id}/tieout.xlsx` ready as a backup.
- Live: ask the pre-tested hybrid questions; click the TieOutCard citations; show the export.
- Fallback: Tidewell demo (`api/demo_service.py`, `dataset="tieout"`), whose gold answers are in `tests/eval/gold_tieout_set.py`.
- Never run a confidential prospect CSV through the E2B CSV path before 2812 (sandbox egress) ships.

## Test / verification
- Two timed rehearsals (one on the demo, one on a public 10-K + QBO-style CSV) logged in the script file.
- First real use on a booked call by 11-26, logged in 2551.
- Cost: under $0.05 per call in LLM spend.

## Dependencies
Blocked by: 2532 (proof-run playbook), 2543 (the only number quoted).  Blocks: 2549 (DP agreement follows a DP yes), Dec 1 gate (held calls, LOI).

## Definition of done
- [ ] AC met  - [ ] tests pass in CI  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- Page citations on real uploaded PDFs can be wrong until 2312 ships (S4); the script must only show citations that were verified in the pre-run. Consider pulling 2312 forward.
- Screen-sharing a prospect's board pack on a call is fine with their consent; confirm whether prospects may send files before the call or only share on screen.
