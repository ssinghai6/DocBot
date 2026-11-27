# DOCBOT-2639 BUG: Tie-out misses a planted break on a second real filing (Chipotle) and a signed-amount ledger
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | S0 | P0 | 3 | senior-dev | 2026-11-30 | In progress (2026-11-26) |

## User story
As a fractional CFO tying a client's 10-K to their own general ledger, I want the tie-out to work on any filing and any normal ledger layout, so that a break is flagged on my client's books, not only on the filing the product was tuned on.

## Why now
DOCBOT-2637/2638 were verified on Etsy only. Real-data test 2026-11-26 on merged main (ae8dbb0): Chipotle FY2025 10-K (live from EDGAR) against a Postgres ledger built from Chipotle's reported income statement, with signed amounts (credits negative), a voided $8M duplicate rent entry in both DBs, and a planted $12.5M Labor accrual. Planted break caught 0/9; false flags on clean books; the SQL counted the voided entry in every run; revenue came back as -$11.9B and synthesis reported a $23.9B mismatch on clean books. Investor call 2026-12-01.

## Scope
In:
1. Book-guided retrieval searched 8 deep; a number-dense statement table ranks ~17th by embedding but first after rerank. It now searches 40 deep (`BOOK_GUIDED_SEARCH_K`).
2. SQL prompt rule 5 (v6): keep only posted entries when a status column marks entries; with one signed amount column, revenue / income / profit lines are -SUM(amount).
3. Detector: whole-number statement-table cells tie out within half a unit (floor 0.01%, cap 0.5%), so a $12.5M (0.42%) gap is a break. Rounded prose keeps 0.5%.
4. Detector: one verdict per books line across statement tables. A cash-flow add-back or segment-table copy of a caption does not flag when the income-statement copy ties out; when several table copies disagree only the closest is flagged. Prose figures (MD&A) still flag on their own (demo planted break unchanged).
5. SQL sign rule wording: an expense line is SUM(amount), a revenue line -SUM(amount), a profit line -SUM(amount) over all its accounts together (the first wording made the model negate expenses and add other income).
6. Detector: "operations" and "operating" are the same token ("Income from operations" == "operating income").
7. Detector row labels: WHERE literals only (not CASE WHEN in SELECT), never status / state filters; a constant metric tag (`'Net Income' AS metric`) is matched alongside WHERE literals; wide rows report unpaired lines by column so book-guided retrieval fires.

Out:
- SQL that picks the wrong account classes for net income (seen once: other income left out).
- Draft / reversed entries beyond a status column.

## Acceptance criteria
- [x] Given a statement cell 2,991,680 (thousands) and books 3,004,180,000, then Labor is flagged; books 2,991,680,400 tie out.
- [x] Given the income statement and the cash-flow statement both list an impairment caption, when the income statement ties out, then no flag.
- [x] Given `WHERE status = 'posted'` or a CASE WHEN in SELECT, then row labels are unchanged.
- [x] Given the demo planted break (MD&A prose vs statement), then it still flags.
- [x] Live rerun (Bedrock gpt-oss-120b): Chipotle 18/18 (planted Labor break flagged 6/6, operating income and net income flagged 3/3 on the summary question, 0 false flags on clean books 9/9, voided entry excluded 18/18); Etsy regression 12/12.

## Deliverable and approach
See Scope. Tests: `tests/unit/test_generic_ledger_tieout_2639.py` (fail on the old code except the two over-flagging guards).

## Test / verification
CI suite; live reruns against local Postgres 18 with Bedrock gpt-oss-120b (only harness change: the SSRF resolver allows localhost).

## Dependencies
Blocked by: DOCBOT-2638 (merged).
Blocks: investor demo 2026-12-01.

## Definition of done
- [x] AC met
- [x] tests/CI or evidence
- [ ] OpenProject ticket updated
- [x] tracker row updated
- [x] no secrets / no banned claims

## Open questions
- Merge after the 2026-11-25 cut-off needs founder approval.
