# DOCBOT-3512 RB-12 PII eval: Gretel finance subset + must-not-mask set (pins the 10-digit bug class)
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-35 | 1 | S1 | P0 | 3 | ai-engineer | - | New (scheduled from parking lot) |

## User story
As a fractional CFO reading a tie-out on a client's ledger, I want PII masking to hide real phone numbers, SSNs, card numbers and emails but never touch an amount, CIK, accession number or account number, so that a masked answer never turns a revenue figure into `***-***-7890`.

## Why now
Design-partner ledgers flow through masked SSE, sandbox and audit paths from S1. The 2003 fix pinned the bare 10-digit case, but a local check shows the bug class is still open when a finance word that doubles as a phone keyword precedes a 10-digit amount.

## Scope
In:
- A committed must-not-mask set (our own strings, >= 200) and a Gretel finance subset (500 rows, fixed ids, from the 3501 cache).
- An offline eval reporting per-entity recall and false masks for `mask_pii`, `mask_rows` and `mask_pii_dataframe_output`.
- The narrow `pii_masking` fix needed to reach 0 false masks.
Out:
- DLP on ingest (3105); new entity types (IBAN, BBAN masking).

## Acceptance criteria
- [ ] Given the must-not-mask set (CIK zero-padded `0000320193`, accession `0001193125-09-214859`, routing + account, raw XBRL USD values like `5520000000`, invoice ids, `call options worth $4523000000`, and finance-keyword phrases such as `Contact center revenue 1234567890`, `Mobile segment revenue: 5551234567`, `Cell tower lease 2025550143`, `Call report filed 0001193125`, `Contact: CIK 0000320193`), when the eval runs, then 0 strings are changed by any of the three masking functions.
- [ ] Given the Gretel subset, when the eval runs, then masking recall is >= 0.90 for each of phone, SSN, credit card and email (a span counts as masked when its raw text is gone from the output).
- [ ] Given the existing `FINANCIAL_FIXTURES` and `REAL_PII_FIXTURES` in `tests/unit/test_pii_masking.py`, when the fix lands, then all still pass.
- [ ] Given CI, when `pytest tests/eval/test_pii_eval.py` runs with `DOCBOT_EVAL_REQUIRE_DATA=1`, then it runs offline in under 60 s and prints recall per entity and the false-mask count.

## Deliverable and approach
- Finding (local run of `mask_pii` on `main`): `_PHONE_CONTEXT_RE` (`api/utils/pii_masking.py:89-102`) masks a bare 10-digit number after `contact|call|mobile|cell|tel|...` when the gap has no digits or currency, so `Contact center revenue 1234567890` becomes `Contact center revenue ***-***-7890` and `Contact: CIK 0000320193` becomes `Contact: CIK ***-***-0193`. `_SSN_RE` also masks SSN-shaped ids (`PO 123-45-6789`), and `_PHONE_RE` masks `Invoice 555-123-4567`.
- Fix (narrow): require the context word to be followed only by separators or `number|no.|#|:` (no other words), and never match after `CIK|accession|account|acct|invoice|PO|ref`. Keep `_NUM_START`/`_NUM_END`.
- `tests/eval/pii_eval.py` (scorer) and `tests/eval/subsets/pii_must_not_mask.json` (committed, ours); Gretel loaded through 3501 `load_cached("gretel_pii_finance")` with a fixed id list in `tests/eval/subsets/gretel_pii_ids.json`.
- Branch `feature/DOCBOT-3512-pii-eval`.

## Test / verification
- `tests/eval/test_pii_eval.py` (CI, offline) and new regression cases appended to `tests/unit/test_pii_masking.py`.
- `pytest tests/unit/test_pii_masking.py tests/eval/test_pii_eval.py -v`. Cost $0 (no LLM).

## Dependencies
Blocked by: 3501.  Blocks: 3105, 3518.

## Definition of done
- [ ] AC met  - [ ] tests pass in CI  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- Points: 3 assumes eval only, but the 0-false-mask AC fails on `main` today, so the regex fix is in scope; 5 is safer.
- Gretel is multilingual; phone formats from 7 languages are wider than the US/UK/IN/EU patterns, so recall >= 0.90 may need an English-only subset. Decide before the first run.
- Should SSN-shaped invoice or PO ids be must-not-mask? Masking them is a privacy-safe failure but breaks tie-out on those ids.
