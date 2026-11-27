# DOCBOT-2577 QBO/Xero CSV export intake for DP onboarding (CSV only, no connector)
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 Sales ops | 1 | S1 | P0 | 5 | senior-dev | - | New (2026-11-17) |

## User story
As a fractional CFO onboarding as a design partner, I want to upload the P&L, balance sheet, trial balance and GL exactly as QuickBooks Online or Xero exports them, so that I can run the month-end tie-out without cleaning files first and see each books figure cited to its export row.

## Why now
The S1 gate is ">= 1 DP onboarded through 2577 CSV intake". The S0 proof runs (2532) convert exports to SQLite by hand; that does not scale past the founder.

## Scope
In:
- Detection and parsing of QBO and Xero report exports (P&L, BS, TB, GL) in `api/utils/csv_preprocessor.py`: title rows, date-range header, "Total" rows, indented sub-accounts, parenthesized negatives, thousands separators, blank and `-` cells.
- A cleaned E2B preamble and `DataProfile` built from the parsed frame.
- Row citations on the CSV path so a tie-out answer and its discrepancies cite export rows.
- Synthetic fixtures (8 files: 2 vendors x 4 reports) and tests.
Out:
- Live QBO/Xero API connectors (3701-3712, parked).
- Routing CSVs through the SQL pipeline (`CLAUDE.md`: CSV queries never use the SQL pipeline).
- Excel `.xlsx` uploads.

## Acceptance criteria
- [ ] Given each of the 8 fixtures, when `split_csv_sections` and the new parser run, then the vendor, report type and period (e.g. "January 1-31, 2026" or "For the month ended 31 January 2026") are detected, and every amount equals the hand-computed value, including `(1,234.56)` as -1234.56 and `1,234.56` as 1234.56.
- [ ] Given a fixture with "Total ..." and subtotal rows, when the frame is built, then those rows carry `is_total=True` and a sum over leaf accounts equals the export's own total row (no double counting).
- [ ] Given indented sub-accounts, when parsed, then each row has `account_path` (e.g. `Expenses > Payroll > Wages`) and `depth`.
- [ ] Given a fixture uploaded through `upload_csv` and one tie-out question each (8 questions), when run against live E2B + Groq, then each answer states the correct value and cites the export row number, and any discrepancy's `db_citation.table` names the uploaded file.
- [ ] Given a generic non-accounting CSV, when uploaded, then behaviour is unchanged (existing tests in `tests/integration/test_file_upload_service.py` stay green).

## Deliverable and approach
- `api/utils/csv_preprocessor.py`: new `detect_accounting_export(raw) -> Optional[AccountingExport]`, `parse_amount(str) -> Optional[float]`, and a preamble branch in `generate_section_preamble` (lines 232-260) that emits a cleaned `df` with `account`, `account_path`, `depth`, `amount` (or period columns), `is_total`, `_row` (original CSV line). The current generic preamble calls `pd.to_numeric(..., errors='coerce')`, which turns `1,234.56` and `(1,234.56)` into NaN, so numeric columns stay strings today. `_find_header` (lines 83-100) only scans 5 rows; QBO puts 3 to 5 title rows before the header, so detection must run first.
- `build_data_profile` (line 325) runs on the cleaned frame so the LLM sees accounts and amounts, not title text.
- `api/file_upload_service.py:_parse_csv_metadata` (line 215): store the detected export metadata in the creds blob next to sections.
- Citations: CSV results reach the detector only as `csv_answer` text (`api/hybrid_service.py:478-482`, `discrepancy_detector.py:206-208`), and `sources`/`sql_query` are absent, so `db_citation` is empty today. Set `metadata_result["sources"] = [<filename>]` in `_collect_sql_result` (`hybrid_service.py:426`) for CSV, and update the CSV code-gen prompt in `sandbox_service.generate_csv_analysis_code` (line 995) so printed tables include the `_row` column.
- Fixtures: `tests/fixtures/accounting_exports/{qbo,xero}_{pl,bs,tb,gl}.csv`, synthetic, written from our own made-up company (no customer data).
- Branch `feature/DOCBOT-2577-qbo-xero-csv-intake`.

## Test / verification
- `tests/unit/test_accounting_export_csv.py`: detection, amounts, totals, indentation, period, generic-CSV passthrough (pure, no I/O beyond fixtures).
- `tests/integration/test_file_upload_service.py`: extend with `upload_csv` on two fixtures (profile + export metadata stored).
- `tests/external/test_accounting_export_tieout.py` with `@pytest.mark.external` (E2B + Groq): 8 tie-out questions; cost about 8 x 2 calls plus 8 sandbox runs, under $0.10.
- CI: `pytest tests/ -v -m "not external and not postgres"`.

## Dependencies
Blocked by: 2532 (defines the target shapes).  Blocks: S1 DP onboarding gate, 2545 (export corpus), 3703 (normalized accounting schema, parked). Must not go to a DP before 2812 (see open questions).

## Definition of done
- [ ] AC met  - [ ] tests pass in CI  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- Points: 5 covers parsing; the CSV row-citation plumbing (hybrid metadata, code-gen prompt, detector citation) is new work and likely pushes this to 8. Keep 5 and split citations into a follow-up, or re-point.
- Order risk in S1: 2577 is first and 2812 third. CSV queries run as internal codegen with `allow_pip_egress=True` (`sandbox_service._run_csv_in_sandbox`, line 1305), and `sandbox_policy.internet_allowed` grants egress when the code imports statsmodels and similar. A DP ledger should not enter that sandbox until 2812 is live in prod. Gate the first DP upload on 2812, not on 2577's merge.
- Xero exports in some regions use `31 Jan 2026` dates and trailing minus signs; confirm the DP list's regions to size the fixture set.
