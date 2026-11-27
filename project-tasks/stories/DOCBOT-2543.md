# DOCBOT-2543 Founder live tie-out eval run (record the public tie-out number)
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AI proof | 0 | - | P0 | 1 | founder | 11-18 | Open (pulled into EPIC-26, due 11-18) |

## User story
As the founder pitching investors on 2026-12-01 and offering prospects a free tie-out, I want one live, dated run of the tie-out eval with real model calls, so that the deck, data room (2605) and proof section carry exactly one honest accuracy number with its date, model, N and commit SHA.

## Why now
The harness (2541, PR #101, `cedb58b`) and the segment fix (2542, PR #102, `46b8efb`) are both on `main`, so nothing blocks the run. This is the only accuracy number allowed in public copy; every other number (FinanceBench, FinDER, FaithEval, the 20/20 extraction baseline) is banned.

## Scope
In:
- One live run: `python -m tests.eval.tieout_eval --mode live --runs 10` (12 questions x 10 runs = 120 runs) from a clean `main` checkout with real keys.
- A committed result file with a metadata header (date, SHA, model, N, gold-set size, fallback count, rough spend).
- Updating the claim wording on the proof section, deck slide and data room using the README section 5 template.
Out:
- Any code change to the harness (automation of the header and spend cap is 2544).
- New gold questions (2545).
- Running against a deployed backend (the README warns it would mostly measure the 6h LLM cache).

## Acceptance criteria
- [ ] Given a clean checkout of `main` at or after `46b8efb` with `groq_api_key` set, when `python -m tests.eval.tieout_eval --mode live --runs 10 2>&1 | tee /tmp/tieout-live.log` runs, then the printed report lists all 12 question ids from `tests/eval/gold_tieout_set.py` with 10 runs each and an `overall X/120` line.
- [ ] Given the run output, when the result file `tests/eval/results/tieout-live-2026-11-18.md` is committed, then it contains: UTC date, `git rev-parse HEAD`, model `openai/gpt-oss-20b` (the value of `GROQ_MODEL` at `api/utils/llm_provider.py:151`), N=10, gold set size 12 (2 demo datasets), per-question pass counts and the overall X/120.
- [ ] Given the run, when `segment_note12` scores below 9/10, then 2542 is reopened in Plane and the number is not published until a rerun passes.
- [ ] Given the captured log, when it is grepped for `"fallback_triggered": true` (JSON llm_call lines from `_log_llm_call`), then the count is 0; if it is above 0 the result file states "Groq with Gemini fallback on K calls" and the founder decides between rerun or disclosure before publishing.
- [ ] Given the measured X/120, when the proof section, deck and 2605 data room are updated, then the wording follows the README template ("On our 12-question tie-out benchmark (2 demo datasets, 10 runs each), DocBot matched both sides to within 0.1% with a citation for each and flagged discrepancies exactly when present in X of 120 runs", plus date and model) and no other accuracy number appears.

## Deliverable and approach
- No code change. Run from the repo root with `.venv` active; `load_dotenv()` in `tieout_eval.py` reads `.env`.
- `report()` in `tests/eval/tieout_eval.py:269-281` prints only pass counts, so the founder writes the metadata header by hand: `date -u`, `git rev-parse HEAD`, model from `llm_provider.GROQ_MODEL`.
- Live mode calls `hybrid_chat` in-process (`live_run`, `tieout_eval.py:190-227`) on a fresh `init_demo_session` per run, so there is no chat history or few-shot carry-over.
- Result file path: `tests/eval/results/tieout-live-2026-11-18.md` (demo data is our own synthetic set, safe to commit). Branch `docs/DOCBOT-2543-live-tieout-result` (docs-only).
- Copy updates go through the PM lane (landing proof section, deck slide, 2605 data room).

## Test / verification
- Pre-check (no keys): `pytest tests/eval/test_tieout_eval.py -v` green, and `python -m tests.eval.tieout_eval --mode offline --runs 1` passes 12/12 so a live failure is a model issue, not a pipeline issue.
- Live: the command above. About 3 LLM calls per run, 360 calls total. Cost at the `_COST_PER_1K_TOKENS` rate for `openai/gpt-oss-20b` ($0.075 / $0.30 per 1M) is about $0.25; at the Llama-3.3 estimate in `docbot-robustness-test-data.md` section 5.3 it is about $1.50. Cap: stop and investigate if the Groq console shows more than $3 for the run.
- Optional cross-check: `pytest tests/eval/test_tieout_eval.py -m external -s` asserts >= 90% per question (same harness).

## Dependencies
Blocked by: 2541 (merged), 2542 (merged).  Blocks: 2544, 2605 (data room eval result), 2550 (demo talk track cites the number), deck v1.

## Definition of done
- [ ] AC met  - [ ] tests pass in CI  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- `CLAUDE.md` and older copy say the main model is Groq Llama 3.3-70b, but `GROQ_MODEL` is `openai/gpt-oss-20b` (`llm_provider.py:151`). The public claim must name the model that actually ran. Confirm the deck wording uses gpt-oss-20b.
- On a free or low-tier Groq key, 360 calls can hit 429s, which silently fall back to Gemini (`chat_completion`, `llm_provider.py:749-816`). If that happens, rerun on a paid key rather than publishing a mixed-model number.
