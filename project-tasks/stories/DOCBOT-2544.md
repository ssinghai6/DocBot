# DOCBOT-2544 Scheduled live tie-out eval with spend cap
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AI proof | 1 | S2 | P1 | 2 | ai-engineer | - | Open |

## User story
As the founder who publishes one live tie-out number, I want the live tie-out eval to run on a schedule with a hard dollar cap, so that a prompt, model or retrieval change that breaks tie-out accuracy fails a job within a week instead of surfacing on a design partner's month-end.

## Why now
Design partners start paying in S1/S2 and prompts keep changing. Today the live test (`tests/eval/test_tieout_eval.py::test_live_tieout_pass_rate`, lines 122-132) only runs when someone runs it by hand, and `nightly-eval.yml` has no Groq key, so the live test is always skipped there.

## Scope
In:
- A separate `tieout-live` job in `.github/workflows/nightly-eval.yml` with its own Groq secret.
- A spend meter and hard cap in `tests/eval/tieout_eval.py` (abort when the cap is reached).
- Run metadata (date, SHA, model, N, spend, fallback count) printed by `report()` and posted to `$GITHUB_STEP_SUMMARY`.
Out:
- New gold questions (2545), XBRL generator cases (3502/3503), the tiered T0/T1/T2 wiring (3518).
- Running against a deployed backend.

## Acceptance criteria
- [ ] Given the workflow, when the `tieout-live` job runs on its schedule or `workflow_dispatch`, then it executes `pytest tests/eval/test_tieout_eval.py -m external -k live -s` with `groq_api_key` from a dedicated repo secret, and the existing retrieval job is unchanged and still has no Groq key.
- [ ] Given `EVAL_TIEOUT_RUNS` (default 10) and `EVAL_SPEND_CAP_USD` (default 3.00), when cumulative estimated spend reaches the cap, then the harness stops issuing runs, prints the spend and fails the job with "spend cap reached" (never silently passes).
- [ ] Given any question whose pass rate is below 90% of its runs, when the job finishes, then the job fails and the summary names the question and its first failure reason.
- [ ] Given a finished run, when the job summary is opened, then it shows UTC date, commit SHA, `GROQ_MODEL`, N per question, overall X/Y, total estimated spend and the number of Gemini fallback calls.
- [ ] Given a unit test with fake llm_call payloads, when spend crosses the cap, then the cap logic aborts, and calls with `estimated_cost_usd = null` are counted at a conservative fixed per-call estimate, not zero.

## Deliverable and approach
- `tests/eval/tieout_eval.py`: register a sink with `llm_provider.set_trace_sink()` (`api/utils/llm_provider.py:138`) that sums `payload["estimated_cost_usd"]` and counts `fallback_triggered`; check the total before each live run in `run_eval()` (lines 230-266); extend `report()` (lines 269-281) with the metadata lines. Restore the previous sink afterwards (index.py sets one in prod).
- `tests/eval/test_tieout_eval.py::test_live_tieout_pass_rate`: append `report()` output to `$GITHUB_STEP_SUMMARY` when set.
- `.github/workflows/nightly-eval.yml`: new job `tieout-live`, cron `0 7 * * 1` (weekly, Monday) plus `workflow_dispatch` with a `runs` input; env `groq_api_key: ${{ secrets.GROQ_API_KEY_EVAL }}`, `EVAL_TIEOUT_RUNS`, `EVAL_SPEND_CAP_USD: "3.00"`, fake `DATABASE_URL` (same as the retrieval job; `_live_env` at `tieout_eval.py:174` needs it set).
- Keep the Groq key out of the retrieval job: that job runs `pytest tests/eval -m external`, which would also pick up the live tie-out test if the key were present.
- New env vars: `EVAL_SPEND_CAP_USD` (eval-only, not app config). Repo secret: `GROQ_API_KEY_EVAL` (a separate Groq key with its own console spend limit as a second cap).
- Branch `feature/DOCBOT-2544-scheduled-tieout-eval`.

## Test / verification
- New `tests/unit/test_tieout_spend_cap.py`: feeds fake sink payloads, asserts abort at cap, null-cost handling, fallback count; no network.
- `pytest tests/eval/test_tieout_eval.py tests/unit/test_tieout_spend_cap.py -v` green in CI (`-m "not external and not postgres"`).
- One `workflow_dispatch` run with `runs=2` to prove the wiring; then the weekly run with 10.
- Cost: 12 questions x 10 runs x about 3 calls = 360 calls, about $0.25 at gpt-oss-20b rates, about $1.50 at the Llama estimate. Hard cap $3 per run; about $1 to $6 per month weekly.

## Dependencies
Blocked by: 2543 (first manual run proves live mode works with real keys).  Blocks: 3518 (tiered CI wiring reuses the spend sink), 2545 (v2 questions join the same scheduled run).

## Definition of done
- [ ] AC met  - [ ] tests pass in CI  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- Nightly vs weekly: with the 90% per-question gate, a nightly run with few runs is flaky (at 3 runs a single failure fails the job). Proposed: weekly with 10 runs (same N as the published number). Confirm, or accept nightly at 10 runs (about $0.25 to $1.50 per night).
- `chat_completion`'s Gemini fallback logs no token counts, so its `estimated_cost_usd` is null; the conservative per-call estimate is a stopgap. Fixing fallback token logging belongs to llm_provider, not this story.
- Scheduled numbers are regression signals only; the public number stays the dated 2543 run unless the founder decides to refresh it.
