# DOCBOT-2312 Correct page citations
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-23 open bugs (referenced) | 1 | S4 | P1 | 3 | ai-engineer | - | Open |

## User story
As a fractional CFO checking a tie-out against a 60-page board pack, I want every cited figure to point to the page it is actually on, so that I can verify an exception in seconds instead of searching the PDF.

## Why now
Tie-out cards, the xlsx export and the 2541 scorer all rely on `page` metadata. The demo uses hand-set pages, so the eval never exercised the real upload path. Prospect PDFs (2532, 2550) and the gold set v2 10-K (2545) do use it.

## Scope
In:
- Per-page chunk metadata computed from character offsets for uploaded PDFs, per file.
- Moving the page-mapping logic out of `api/index.py` into `api/utils/chunker.py`.
- A multi-page, multi-file fixture test that asserts pages.
Out:
- EDGAR HTML filings (no pages); OCR of scanned PDFs; the citation verifier (3401).

## Acceptance criteria
- [ ] Given a generated 3-page PDF with a unique marker per page and a header repeated on every page, when it is uploaded and chunked, then every chunk containing marker Pk has `page == k` (100% of chunks).
- [ ] Given two PDFs uploaded together, when chunked, then no chunk mixes text from both files and each chunk's `source` is its own file.
- [ ] Given a chunk that spans a page break, when chunked, then `page` is the page where the chunk starts and `page_end` records the last page.
- [ ] Given both splitters (SemanticChunker for financial docs and RecursiveCharacterTextSplitter), when the fixture test runs, then both pass the page assertions.
- [ ] Given `python -m tests.eval.tieout_eval --mode offline --runs 1`, when run after the change, then all 12 questions still pass.

## Deliverable and approach
- Root cause, `api/index.py:1042-1086`: per-page `Document`s are built with `page = page_num + 1`, then all pages of all files are joined with spaces into one `full_text` and chunked as one document (`chunk_document`, `api/utils/chunker.py`). Pages are recovered afterwards by `_best_metadata`: an exact 80-char prefix match, else the first page whose text contains the chunk's first 40 chars, else the first page of the first file. Chunks that start mid-page, across a page break, or with repeated header text get the wrong page, and chunks can span two files.
- New `chunk_pages(pages: list[Document], doc_type: str) -> list[Document]` in `api/utils/chunker.py`: per file, join pages while recording page start offsets; recursive splitter with `add_start_index=True`; for SemanticChunker locate each chunk with a forward cursor `text.find`; map offsets to pages with `bisect`. `index.py` calls it (thin route, per `CLAUDE.md`).
- Branch `feature/DOCBOT-2312-page-citations`.

## Test / verification
- `tests/unit/test_chunk_pages.py`: fixture PDFs built in-test with PyMuPDF; stub embeddings for the semantic path (no network).
- `pytest tests/unit/test_chunk_pages.py tests/unit/test_chunker.py tests/eval/test_tieout_eval.py -v`. Cost $0.

## Dependencies
Blocked by: none.  Blocks: 3401, 2545 (10-K page checks), accurate pages in 2532/2550 proof runs.

## Definition of done
- [ ] AC met  - [ ] tests pass in CI  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- Priority and sprint: P1 in S4 (late January), but the S0 proof runs and demos rely on PDF pages, and 2545 (S3) needs them. Recommend pulling this into S1 or before 2545; the playbooks add manual page checks until then.
- Existing Chroma stores keep their old page metadata; decide whether to re-index active design-partner sessions after the fix.
