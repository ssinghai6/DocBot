# DOCBOT-2532 Proof-run playbook (QBO/Xero CSV + board-pack PDF to cited tie-out table)
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 GTM | 0 | S0 | P0 | 5 | PM | 11-24 | Open; pulled before Dec 1 (2026-11-17) |

## User story
As a fractional CFO who just closed a client's month, I want to send DocBot my QuickBooks or Xero export and the board pack PDF and get back, within an hour, a table of every board-pack figure marked "ties" or "exception" with the PDF page and the ledger row, so that I can see the review I normally do by hand done and cited before I pay anything.

## Why now
"Free tie-out of your last month" is the outreach hook for the 150-contact push before 12-01. Without a repeatable runbook, each proof run is improvised and slow, and a wrong page or a wrong row in front of a prospect costs the deal.

## Scope
In:
- A step-by-step runbook from intake to a delivered Excel file in under 60 minutes, using only what is on `main` today.
- Intake checklist, data-handling rules, question bank per board-pack line, QA checklist, and delivery email template.
- Dry runs on the Tidewell demo set and on one public 10-K plus a QBO-style CSV.
Out:
- Code changes (QBO/Xero parsing is 2577; page citation fix is 2312; tie lines in the export would be a new ticket).
- Live connectors to QBO/Xero (parked, 3701-3712).
- Any accuracy claim other than the 2543 number.

## Acceptance criteria
- [ ] Given a closed month (one QBO or Xero CSV export plus one board-pack PDF), when an operator follows the playbook, then they produce an `.xlsx` with one row per board-pack figure showing report value, PDF page, books value, ledger row or query, delta and status, in under 60 minutes of wall time (timed and logged).
- [ ] Given the Tidewell demo (`POST /api/demo/init` with `dataset="tieout"`), when the playbook is dry-run, then the output flags exactly the 5 planted gaps and none of the 9 tie figures in `tests/eval/gold_tieout_set.py`.
- [ ] Given one public 10-K PDF and a QBO-style CSV, when the playbook is dry-run, then every line's PDF page and ledger row are verified by hand against the source and any wrong page is logged against 2312.
- [ ] Given the output file, when it is opened in Excel, then it opens without warnings and no cell starts with `=`, `+`, `-` or `@` as a formula (the existing `api/tieout_export.py` export already guards this; the manual sheet must too).
- [ ] Given prospect files, when the run ends, then the playbook's deletion step (session and connection removed, local copies deleted) is done and logged with date and prospect name in the 2551 gate tracker.

## Deliverable and approach
- Playbook at `project-tasks/playbooks/DOCBOT-2532-proof-run.md`. Branch `docs/DOCBOT-2532-proof-run-playbook`.
- Path through today's code:
  1. Board pack PDF via the upload route (pages assigned in `api/index.py:1042-1086`).
  2. Books: convert the QBO/Xero export to a SQLite file locally (strip title rows, drop or flag "Total" rows, turn `(1,234.56)` into `-1234.56`) and upload it through `upload_sqlite` (`api/file_upload_service.py:55`). This puts the books on the SQL pipeline, so answers carry `result_preview` rows and the SQL query as the DB citation.
  3. Ask one hybrid tie-out question per board-pack section (question bank in the playbook); each answer's discrepancies come from `detect_discrepancies` (`api/utils/discrepancy_detector.py:158`).
  4. Download exceptions with `GET /api/lineage/{run_id}/tieout.xlsx` (`api/routes/tieout_routes.py:37`); add tie lines by hand from the answer's cited rows into the same sheet.
  5. QA every row: open the PDF at the cited page, find the ledger row, mark verified.
- Why SQLite and not the CSV upload for S0: raw QBO exports through `upload_csv` go to E2B, where the generic preamble (`csv_preprocessor.generate_section_preamble`, lines 232-260) coerces `1,234.56` and `(1,234.56)` to NaN, and CSV answers carry no row citation. The CSV path also runs in an E2B sandbox that can still get internet (`sandbox_policy.py:74-76`) until 2812 ships. The SQLite route avoids both for prospect data.
- Wording rules: proof-method language only, the 2543 number with date and model if any number is quoted, never "enterprise-grade", never "private mode demoed".

## Test / verification
- Two timed dry runs (demo set, public 10-K + QBO-style CSV) recorded in the playbook with start/end times and per-line verification marks.
- No automated tests (PM deliverable); the AI-engineer pairing checks the demo dry run against `GOLD_TIEOUT` values.
- Cost: a proof run is about 10 to 20 hybrid questions at about 3 calls each, under $0.10 at gpt-oss-20b rates.

## Dependencies
Blocked by: none (uses `main` as is); 2543 for any number quoted.  Blocks: 2550, 2577, 2545, 2549 (DP agreement uses the proof run as the free offer).

## Definition of done
- [ ] AC met  - [ ] tests pass in CI  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- The tie-out export only contains exceptions: `detect_discrepancies` drops tying pairs (`continue` in the matching loop) and the route returns 404 when there are no discrepancies (`tieout_routes.py:50-51`). A clean month gives no file. The playbook works around it by hand; a small code ticket to keep tie lines in lineage would remove about 15 minutes per run. Raise with the senior PM.
- Page citations on uploaded multi-page PDFs are reconstructed by prefix matching and can be wrong (see 2312, scheduled S4). The playbook mandates manual page checks until 2312 ships; consider pulling 2312 into S1.
- Data handling: prospect board packs are confidential. Confirm consent wording in the intake email and whether runs happen on prod (Railway) or the AWS dev env.
