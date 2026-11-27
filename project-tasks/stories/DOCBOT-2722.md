# DOCBOT-2722 BUG: EDGAR batch ingest: only the last filing is queryable
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | S0 | P0 | 2 | senior-dev | 2026-11-30 | Testing (2026-11-27) |

## User story
As a fractional CFO, I want "Ingest All N 10-Ks" to give me one session with every filing, so that multi-year questions are answered from all the filings the label says are loaded.

## Why now
Investor call 2026-12-01. `/api/edgar/ingest-batch` made one session per filing; the panel loaded only the last one but labelled it "(N filings)". Chat, hybrid and autopilot take exactly one session_id.

## Scope
In:
1. New `merge_stores` (api/utils/vector_store.py) and `combine_filing_sessions` (api/edgar_service.py): copy every successful filing's vectors (metadata intact, so citations name the right filing) into one new caller-owned session; extracted fields merged.
2. `/api/edgar/ingest-batch` returns `combined_session_id`; per-filing `results` unchanged. One success reuses its own session; none gives null.
3. EdgarPanel uses the combined id; label and toast show the count actually loaded.

Out:
- The batch can outlive the proxy (no pending flow like DOCBOT-2706). Not redesigned.

## Acceptance criteria
- [x] 3 filings give one session holding all chunks with per-filing filing_date/source.
- [x] A failed filing is left out; label counts loaded filings.
- [x] Only this caller's own ingest session ids are read; the session row carries the caller's owner columns.

## Test / verification
`tests/unit/test_edgar_batch_combined_2722.py` (4 tests, real Chroma in a tmp dir, fake embedder).

## Dependencies
None.
