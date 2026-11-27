# DOCBOT-2532 Proof-run playbook

The free tie-out we offer a prospect: they send one closed month (a QuickBooks Online or Xero export plus the board pack or lender report PDF). We return one `.xlsx` with every board-pack figure marked "Ties out" or "Exception", with the PDF page on one side and the ledger row or SQL query on the other.

Target: under 60 minutes of operator wall time from files received to email sent. Uses only what is on `main` today. No code changes.

Owner: founder (operator). Reviewer for the demo dry run: AI-engineer lane.

---

## 0. Claim rules (read before every email or call)

- Describe the method, not a score. Say "every flagged figure carries the PDF page and the ledger row so you can check it yourself". Do not quote any accuracy or pass-rate number. The only number we may ever quote is the DOCBOT-2543 result, with its date and model, and only once 2543 has published it.
- The offline eval result in section 7 is an internal regression check. It is not an accuracy number. Never quote it.
- Never say "enterprise-grade". Never say "private mode demoed".
- No em dashes in anything sent to a prospect.
- Say "a second pair of eyes on the tie-out", not "replaces your review". The operator still checks every row (section 5).

---

## 1. Intake checklist

Send with the offer email. Do not start until every box is ticked.

- [ ] Written consent from the prospect (reply to the intake email is enough) to the wording below.
- [ ] One board pack or lender report, PDF, text-based (not a scan). If it is a scan, stop and ask for the native export.
- [ ] One books export for the same period, CSV, from QBO or Xero. Preferred report, in order: Trial Balance, Profit and Loss (by month), General Ledger detail. One file per report.
- [ ] Period stated in writing (for example "January 2026, month ended 2026-01-31").
- [ ] Prospect name, contact, and the date files were received, written in the 2551 gate tracker ("free tie-outs run" row) before the run starts.
- [ ] The prospect has removed, or agreed we remove, bank and card account numbers, SSNs and employee names from both files (see 2.3).
- [ ] Retention period agreed: **PENDING founder decision (DOCBOT-2532 open question). Placeholder: delete within [N] days of delivery.** Do not promise a number until the founder sets N.

Consent wording for the intake email (fill the brackets):

> To run the free tie-out we will upload your [month] board pack and books export to DocBot under a single account used only for this review. Files are used only to produce your tie-out workpaper. We delete the uploaded files, the DocBot session and the database connection, and our local copies, within [N, PENDING] days of sending you the result, and we confirm deletion by email. Reply "agreed" to go ahead.

Open question for the founder: run on prod (Railway) or the AWS dev env. Until decided, use prod with a dedicated operator account, `APP_AUTH_MODE` login, never the anonymous demo.

---

## 2. Data-handling rules (hard rules, no exceptions)

### 2.1 Books go through the SQLite path only

- Convert the QBO/Xero CSV to a SQLite file on the operator laptop (section 3, step 2) and upload with `POST /api/db/upload` (`db_upload_sqlite`, `api/index.py`). This puts the books on the bounded SQL pipeline, so every answer carries the SQL query and result rows as the books citation.
- **Never upload prospect CSV through `/api/db/upload/csv` or the chat CSV upload.** That path runs in an E2B sandbox that can still reach the internet until DOCBOT-2812 (sandbox egress off) ships, and its preamble turns `1,234.56` and `(1,234.56)` into NaN.
- Never run Python analysis or Autopilot against prospect books for the same reason (E2B).

### 2.2 PDFs

- Upload the board pack through the normal PDF upload (`POST /api/upload`).
- Page citations on uploaded multi-page PDFs are reconstructed and can be wrong (DOCBOT-2312, open). **Every cited page must be hand-checked** (section 5). Log each wrong page against 2312 with the run date, the figure label, the cited page and the true page.

### 2.3 PII and account numbers

- The PII masker has a live bug (DOCBOT-3512): a 10-digit number after a finance word can be masked as a phone number, and it can also miss real identifiers. Do not rely on it.
- Before upload, the operator redacts by hand in both files: bank and card account numbers, routing numbers, SSN/EIN/TIN, employee names on payroll lines, customer personal names. Replace with `[REDACTED]` in the CSV before conversion. For the PDF, use a real redaction tool (Acrobat Redact or `qpdf`-based), never a black box drawn over text.
- After every answer, check that no amount shows as `***-***-NNNN`. If one does, read the value from the SQL result rows, not the answer text, and note it in the QA log as a 3512 hit.

### 2.4 General

- Prospect files live only in one local folder: `~/proof-runs/<prospect>-<YYYY-MM-DD>/`. Never in the repo, never in a shared drive, never in Slack.
- Never paste prospect figures or file contents into any LLM chat outside DocBot.
- Never forward the files. Never keep them after the deletion step.

---

## 3. Step-by-step run (timed)

Write the start time in the run log (section 8) before step 1.

**Step 1. Prepare the PDF (about 5 min).** Confirm text is selectable. Redact per 2.3. Note the page numbers of each section you will tie (P&L summary, balance sheet, KPI page, covenant page). Use the printed page number in the PDF viewer's page index (1-based), which is what DocBot cites.

**Step 2. Convert the books to SQLite (about 5 min).** Redact per 2.3, then run the converter below on the operator laptop. It skips the QBO/Xero title rows, makes `(1,234.56)` into `-1234.56`, strips `$` and thousands separators, and keeps subtotal rows with `is_total = 1` so a SUM over detail rows never double counts.

```python
"""qbo_to_sqlite.py  Usage: python qbo_to_sqlite.py export.csv books.sqlite [table_name]"""
import csv, re, sqlite3, sys

AMOUNT = re.compile(r"^\(?-?\$?[\d,]+(\.\d+)?\)?$")

def to_number(cell):
    s = cell.strip()
    if not s or not AMOUNT.match(s):
        return None
    v = float(s.strip("()").replace("$", "").replace(",", ""))
    return -v if s.startswith("(") and s.endswith(")") else v

def convert(src, dst, table="books"):
    with open(src, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    # Header = first row with a non-empty cell after column 1 (skips title / company / date rows).
    start = next(i for i, r in enumerate(rows) if any(c.strip() for c in r[1:]))
    header = [re.sub(r"\W+", "_", h.strip()).strip("_").lower() or f"col{i}" for i, h in enumerate(rows[start])]
    header[0] = "account"
    out = []
    for r in rows[start + 1:]:
        if not any(c.strip() for c in r):
            continue
        label = r[0].strip()
        is_total = int(label.lower().startswith(("total", "net ", "gross profit")))
        r = (r + [""] * len(header))[: len(header)]
        out.append([label] + [to_number(c) if to_number(c) is not None else c.strip() for c in r[1:]] + [is_total])
    cols = header + ["is_total"]
    con = sqlite3.connect(dst)
    con.execute(f'DROP TABLE IF EXISTS "{table}"')
    con.execute(f'CREATE TABLE "{table}" ({", ".join(chr(34) + h + chr(34) for h in cols)})')
    con.executemany(f'INSERT INTO "{table}" VALUES ({",".join("?" * len(cols))})', out)
    con.commit(); con.close()
    return len(out)

if __name__ == "__main__":
    print(convert(sys.argv[1], sys.argv[2], *(sys.argv[3:4])), "rows written")
```

Tested on a QBO-shaped P&L sample (title rows, `"1,234.56"`, `"(200.00)"`, `"$1,034.56"`, Total and Net rows): output `account | total | is_total` with `1234.56`, `-200.0`, and both subtotals flagged. Check after conversion: `sqlite3 books.sqlite "select * from books limit 20"` and confirm one known figure by eye against the CSV. One table per report (`pnl`, `tb`, `gl`) in the same `.sqlite` file is fine.

Spot checks the converter does not do (do them by hand): multi-column period exports (one column per month, pick the right one in every question), Xero "Account Code" columns (keep as text), and amounts in a currency other than USD.

**Step 3. Create the session and connect the books (about 3 min).** Log in with the operator account. Upload the PDF. Upload `books.sqlite` with the same `session_id`. The UI switches to hybrid mode when both are present.

**Step 4. Ask the tie-out questions (about 20 min).** One hybrid question per board-pack section, from the question bank (section 4). Use the Finance Expert persona. Ask in the chat (hybrid route `POST /api/hybrid/chat`). After each answer, open the Inspector, Lineage tab, and note the `run_id`.

**Step 5. Export exceptions (about 5 min).** For each answer that flagged discrepancies, download the workpaper from the Lineage tab (`GET /api/lineage/{run_id}/tieout.xlsx`). Known gap: the export only holds exceptions. Tying figures are dropped by the detector and a clean answer returns 404 "No tie-out lines for this answer". Add tie lines by hand (step 6). A code ticket to keep tie lines in lineage would save about 15 minutes per run; raise with the senior PM.

**Step 6. Build the single delivery sheet (about 10 min).** Start from the first downloaded `.xlsx` and paste the others under it, then add one row per tying figure from the answer's cited SQL rows. Columns, in this order:

| Metric | Period | Report value | Report source (file, page) | Books value | Books source (table / query) | Delta (books - report) | Delta % | Severity | Status |
|---|---|---|---|---|---|---|---|---|---|

Formula safety (Excel must open without warnings): the export already stores text starting with `=` as text. For hand-typed cells, numbers go in as plain numbers; any text cell that would start with `=`, `+`, `-` or `@` (for example a SQL fragment or a negative written as text) is typed with a leading apostrophe `'`. Before saving, use Find (`Ctrl+F`, "Look in: Formulas") for `=` and confirm zero formula cells. Save as `.xlsx`, close, reopen, confirm no warning bar.

**Step 7. QA (about 10 min).** Section 5. Nothing goes out until every row is marked verified.

**Step 8. Send and delete.** Send with the template in section 6. Run the deletion step (section 9) and write the end time in the run log.

---

## 4. Question bank (one per board-pack section)

Swap `[period]` and the line name. Ask about the specific lines the board pack shows; do not ask open questions like "is everything right".

| Board-pack section | Question |
|---|---|
| Revenue summary | Does total revenue for [period] in the board pack agree with the books? |
| Revenue by line / segment | Does revenue by [segment/product line] for [period] in the board pack agree with the books? |
| Gross margin | Does gross profit for [period] in the board pack agree with the books? |
| Opex by category | Does [sales and marketing / R&D / G&A] expense for [period] in the board pack agree with the general ledger? |
| Operating income / EBITDA | Does operating income for [period] in the board pack agree with the books? (EBITDA: ask for the components separately; the add-backs are not in the books.) |
| Net income | Does net income for [period] in the board pack agree with the books? |
| Cash | Does the cash balance at [period end] in the board pack agree with the trial balance? |
| AR / AP | Does accounts receivable at [period end] in the board pack agree with the trial balance? (Repeat for AP.) |
| Debt / covenant page | Does the debt balance at [period end] in the lender report agree with the trial balance? |
| Budget vs actual | Do the actual figures in the budget-vs-actual table for [period] agree with the books? (Ties actuals only; budget is not in the books.) |

Cost: a run is about 10 to 20 hybrid questions at about 3 LLM calls each, well under $0.10 per run at current rates.

---

## 5. QA checklist (every row, every run)

For each row in the delivery sheet:

- [ ] Open the PDF at the cited page. The report value is on that page, with the same label and period. If the page is wrong, fix the cell and log a 2312 hit (run date, figure, cited page, true page).
- [ ] Find the ledger row: run the cited SQL in `sqlite3 books.sqlite`, or find the account in the original CSV. The books value matches.
- [ ] Units match (thousands vs dollars). A board pack in $000 against books in $ is a unit gap, not an exception.
- [ ] Status is right: "Ties out" only when the delta is zero after rounding to the board pack's precision; otherwise "Exception".
- [ ] Sign is right (credits, contra-revenue, parentheses).
- [ ] No masked amount (`***-***-NNNN`) anywhere in the sheet (3512).
- [ ] Mark the row verified in the run log with the operator initials.

Whole-sheet checks:

- [ ] Every figure on the tied board-pack pages has a row (count figures on the page, count rows).
- [ ] No exception without both citations.
- [ ] No prospect PII in the sheet (2.3).
- [ ] Opens in Excel with no warning; zero formula cells (step 6).

---

## 6. Result email template

Subject: Your [month] tie-out: [X] exceptions, [Y] figures tie

> Hi [name],
>
> Attached is the tie-out of your [month] [board pack / lender report] against your [QBO / Xero] export.
>
> - [Y] figures tie to the books.
> - [X] figures do not. Each one shows the report value with its PDF page, the books value with the ledger row or query we used, and the difference.
>
> The biggest one: [metric], page [p], reports [report value] against [books value] in [account / table], a difference of [delta].
>
> Every row was checked by hand against your PDF and your export before sending. Please treat it as a second pair of eyes on your own review, not a replacement for it.
>
> As agreed, we will delete your files, the DocBot session and the connection by [date, PENDING retention decision] and confirm by email.
>
> If this saved you time, I would like 20 minutes to show how this runs every month on your own login. [booking link]
>
> [founder name]

Before sending: no accuracy numbers, no "enterprise-grade", no em dashes.

---

## 7. Dry run 1: Tidewell demo set (`/api/demo/init?dataset=tieout`)

### 7.1 Offline verification (done 2026-11-17, no API keys)

What the demo plants (source of truth `api/demo_service.py` `TIEOUT_EXPECTED_DISCREPANCIES`, gold set `tests/eval/gold_tieout_set.py`):

- Tidewell (`dataset=tieout`) plants **3** gaps: Total revenue (418.2M report vs 412.6M books, p.4), Lab Services (104.9M vs 102.7M, p.9), Sales and marketing expense (54.9M vs 55.2M, p.5).
- QuickBite (`dataset=quickbite`) plants **2**: Q4 Net Income (330M vs 325M, p.7), Restaurant Advertising (890M vs 885M, p.5).
- Together these are the **5** planted gaps in `GOLD_TIEOUT`. The **9** tie figures are all in Tidewell: Commercial Instruments, Software and Analytics, R&D, G&A, Net income, Operating income, Gross profit, Cost of revenue, Income tax expense.

Note for the story: the AC says "the Tidewell demo flags exactly the 5 planted gaps". Tidewell alone has 3; the 5 span both demo datasets. The check below covers both.

Command and result:

```
python -m tests.eval.tieout_eval --mode offline --runs 1
=== Tie-out accuracy eval (mode=offline, tolerance=0.1%) ===
  12/12 questions passed (segment_note12, total_revenue_mdna, sales_marketing_mdna,
  rnd_mdna, gna_mdna, net_income_mdna, operating_income_mdna, gross_profit_mdna,
  cost_of_revenue_is, income_tax_is, q4_net_income, restaurant_advertising)

pytest tests/unit/test_demo_tieout.py tests/unit/test_discrepancy_detector.py -q
53 passed
```

What "passed" means (`score_run` in `tests/eval/tieout_eval.py`): each of the 5 planted figures is flagged exactly once with the right report and books values, the right doc page and a DB citation; none of the 9 tie figures is flagged, and each has its books value in the SQL rows and its page cited; any flag on any other metric fails the run as a false positive.

Limit: offline mode replaces the three LLM stages (intent, SQL generation, synthesis) with the gold route and gold SQL. It proves the demo data, retrieval, detector and scorer, not the live model. It is not an accuracy number.

### 7.2 Live dry run: NEEDS FOUNDER LIVE RUN

Needs `groq_api_key` in the environment. Run from the repo root:

```
python -m tests.eval.tieout_eval --mode live --runs 3
```

Then the timed UI walk-through: `/chat`, Try Demo (tie-out dataset), ask the 10 Tidewell questions in `GOLD_TIEOUT`, export from the Lineage tab, build the sheet per step 6, QA per section 5. Expected: 3 exceptions (Total revenue p.4, Lab Services p.9, Sales and marketing p.5) and 7 tie rows; repeat on `dataset=quickbite` for the other 2. Record in the run log below.

---

## 8. Run log

One row per run. Times in local time. "Verified" = every row passed section 5.

| Date | Prospect / dataset | Operator | Start | End | Wall min | Figures | Exceptions | Wrong pages (2312) | 3512 hits | Verified | Deleted (date) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-17 | Tidewell + QuickBite, offline eval only | agent | n/a | n/a | n/a | 14 | 5 | 0 | 0 | eval only | n/a (demo data) |
| PENDING | Tidewell, live UI timed run | founder | | | | 10 | | | | | n/a |
| PENDING | Public 10-K + QBO-style CSV, timed run | founder | | | | | | | | | n/a |

Dry run 2 (public 10-K + QBO-style CSV) is **NEEDS FOUNDER LIVE RUN**: pick one public 10-K PDF from EDGAR, build a QBO-style P&L CSV from its income statement (with title rows, parentheses negatives, Total rows), convert with step 2, run steps 3 to 7, and hand-verify every page and ledger row. Log any wrong page against 2312.

---

## 9. Deletion step (every prospect run)

Do all of it, then log it.

1. In DocBot: delete the chat session (`DELETE /api/session/{session_id}`).
2. Disconnect the books connection (`DELETE /api/db/disconnect/{connection_id}`). The uploaded SQLite file in `/tmp` also expires after 2 hours.
3. Local: `rm -rf ~/proof-runs/<prospect>-<YYYY-MM-DD>/` (original PDF, CSV, `books.sqlite`, delivery `.xlsx`). Keep only the sent email in the mailbox.
4. Empty the downloads folder of any `tieout*.xlsx` from the run.
5. Log: in the 2551 gate tracker, add the prospect name, run date, and deletion date to the "free tie-outs run" entry. Fill "Deleted (date)" in the run log above.
6. Email the prospect: "Your files, the DocBot session and the connection were deleted on [date]."

Retention window: **PENDING founder decision.** Until set, delete the same day the result is sent.

Known gap for the founder (not verified in this playbook): deleting the session and connection may not remove answer lineage rows, and the audit log is append-only by design. Check what each keeps and decide whether the consent wording must mention this before the first real prospect run.

---

## 10. Blocked on founder

- Retention days N (sections 1, 6, 9).
- Prod (Railway) vs AWS dev env for prospect runs; consent wording sign-off.
- Live demo dry run (7.2) and the timed 10-K + QBO-style dry run (section 8). Both need live LLM keys.
- Whether the consent wording must cover retained lineage and audit rows.
- Any quoted number waits for DOCBOT-2543.
