# DOCBOT-2545 Gold set v2: public 10-K + accounting-export shapes
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AI proof | 1 | S3 | P1 | 3 | ai-engineer | - | Open |

## User story
As a fractional CFO evaluating DocBot on my own month-end, I want the tie-out benchmark to include a real public 10-K and a QBO/Xero-style export, not only the two built-in demo datasets, so that the accuracy number reflects the inputs I actually hand over.

## Why now
The 12-question gold set (`tests/eval/gold_tieout_set.py`) covers only `api/demo_service.py` data, where chunks and pages are hand-set. Design partners upload real PDFs and accounting CSVs (2532, 2577), which go through different code paths (PDF upload page mapping in `api/index.py`, CSV to E2B). The benchmark should measure those paths before the number is refreshed.

## Scope
In:
- At least 10 new `TieoutQuestion`s over (a) one public 10-K PDF and (b) a synthetic QBO/Xero-style export built from the 2577 fixtures.
- Each gold value traced to a document page or an export row in a new `source_ref` field.
- Runner support for the two new corpora in `tests/eval/tieout_eval.py`; scorer (`score_run`, `_check_figure`) unchanged.
- README section 5 update stating which corpus a cited number used.
Out:
- XBRL synthetic generator (3502/3503).
- Committing the 10-K file (public filings are not committed, `docbot-robustness-test-data.md` section 7.6).
- FinanceBench or any internal-only dataset.

## Acceptance criteria
- [ ] Given `gold_tieout_set.py`, when `pytest tests/eval/test_tieout_eval.py` runs in CI, then `GOLD_TIEOUT` has >= 22 questions, the new ones span both new corpora with at least 3 must-flag and 3 must-tie figures, and every new `GoldFigure` has a non-empty `source_ref` (e.g. `10-K p.47` or `qbo_pl.csv row 18`).
- [ ] Given the committed QBO/Xero fixture, when the consistency test re-derives each export-side gold value from the CSV, then it matches gold exactly (CI, offline, no keys).
- [ ] Given the 10-K fetched by the 3501 fetcher into the git-ignored cache (pinned URL + sha256), when the external consistency test runs, then every 10-K gold value is found on its stated page via PyMuPDF text extraction.
- [ ] Given `python -m tests.eval.tieout_eval --mode offline --runs 1`, when it runs, then all v2 questions score through the unchanged scorer (no runner crash), and live mode `--only <v2 ids>` produces per-question pass rates.
- [ ] Given README section 5, when a reader checks a cited number, then the README states the corpus (demo-only 12 questions vs v2 22+ questions) and the citation template names it.

## Deliverable and approach
- `tests/eval/gold_tieout_set.py`: add `source_ref: str = ""` to `GoldFigure`; new datasets `"public10k"` and `"acct_export"`; new questions.
- `tests/eval/tieout_eval.py`: extend `_sqlite_for`/`_chunks_for` (lines 131-140) and the live branch (lines 248-266, today hard-wired to `init_demo_session`) with loaders for the new corpora. The 10-K goes through the real PDF upload chunking so pages are measured, not hand-set.
- Fixtures: reuse 2577's synthetic exports under `tests/fixtures/accounting_exports/`; 10-K fetched via `tests/eval/fetch_eval_data.py` (3501) with pinned URL and sha256 in `tests/eval/datasets.json`.
- Branch `feature/DOCBOT-2545-gold-set-v2`.

## Test / verification
- CI: `pytest tests/eval/test_tieout_eval.py -v` (gold consistency for the export corpus, scorer tests unchanged).
- External: new `tests/eval/test_gold_v2_10k.py` marked `@pytest.mark.external` (needs the fetched 10-K).
- Live: `python -m tests.eval.tieout_eval --mode live --runs 10` over 22+ questions, about 660 calls, about $0.45 at gpt-oss-20b rates (about $2.70 at the Llama estimate); cap $3 via the 2544 spend sink.

## Dependencies
Blocked by: 2532 (proof-run inputs define the shapes), 2577 (export fixtures + CSV citations), 3501 (fetcher + manifest), 2544 (spend cap).  Blocks: refreshed public number, 3503.

## Definition of done
- [ ] AC met  - [ ] tests pass in CI  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- "Scorer unchanged" conflicts with the CSV path. CSV answers reach the detector as `csv_answer` text (`api/hybrid_service.py:478-482`, `discrepancy_detector.py:206-208`), with an empty `result_preview` and no `sources`/`sql_query`, so `db_citation` is empty and `_rows_contain` finds nothing. Every export figure would fail checks 2 and 3 unless 2577 adds row citations for CSV. Decide: make 2577's CSV citation a hard dependency (preferred), or relax the scorer for the export corpus (breaks "unchanged").
- Page dependency is out of order: real-PDF page metadata is unreliable until 2312 (scheduled S4), but this story is S3. Either pull 2312 into S3 or before it, or expect 10-K "doc page not cited" failures that measure the 2312 bug rather than the model.
- EDGAR ingest (`api/edgar_service.py:176-186`) produces chunks with no `page` key, so the 10-K must be a PDF (company IR site) rather than EDGAR HTML. Confirm the source filing.
- Points: 3 is tight given the runner loaders for two new corpora; 5 is more realistic.
