# DOCBOT-3501 RB-01 Dataset fetch + cache script, licence manifest (DATA_LICENSES.md), git-ignored cache
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-35 | 1 | S1 | P0 | 3 | ai-engineer | - | New (scheduled from parking lot) |

## User story
As the AI engineer adding robustness evals, I want one fetcher that downloads pinned public datasets into a git-ignored cache and a manifest that records each licence and its allowed uses, so that every eval uses the same data and no non-commercial dataset ever leaks into training, few-shot banks, prompts or public claims.

## Why now
3512 (PII) and 3513 (prompt guard) need Gretel and deepset data in S1. Without a manifest, licence rules (FinanceBench, FinDER, FaithEval internal-only) live only in a planning doc.

## Scope
In:
- Fetcher module with pinned revisions and sha256 checks for the S1 sets: `deepset/prompt-injections`, `gretelai/synthetic_pii_finance_multilingual`.
- Machine-readable manifest for every dataset in `docbot-robustness-test-data.md` section 1, with licence and use flags; human-readable `DATA_LICENSES.md` with required attributions.
- Git-ignored cache directory and a CI mode that fails (not skips) when required data is missing.
- Guard tests for the licence rules.
Out:
- Fetch implementations for sets not needed in S1 (FinanceBench, BIRD, TAT-QA and others get manifest rows only).
- The evals themselves (3512, 3513 and later).

## Acceptance criteria
- [ ] Given a clean checkout, when `python -m tests.eval.fetch_eval_data --set deepset_pi --set gretel_pii_finance` runs, then files land in the cache dir at the pinned revision, every sha256 matches the manifest, and a rerun downloads nothing.
- [ ] Given a manifest entry without a pinned revision or sha256, when the fetcher runs, then it refuses with a clear error.
- [ ] Given `tests/eval/datasets.json`, when `pytest tests/unit/test_data_licenses.py` runs, then every entry has licence, URL, revision and the flags `commercial_ok`, `internal_only`, `never_train`, `never_fewshot`, `never_public_claim`, `attribution`; FinanceBench, FinDER and FaithEval have `internal_only`, `never_train`, `never_fewshot` and `never_public_claim` all true.
- [ ] Given the repo, when the guard test scans `api/` and `finetune/`, then no file references the cache dir or an internal-only dataset id, and `git check-ignore` confirms the cache dir is ignored.
- [ ] Given `git status` after a full fetch, when inspected, then only code, `datasets.json` and `DATA_LICENSES.md` are tracked; no dataset file is committed.

## Deliverable and approach
- `tests/eval/fetch_eval_data.py` (run as a module). Not `scripts/fetch_eval_data.py` as the robustness plan says: `scripts/` is git-ignored (`.gitignore:64-65`, "scripts (contain credentials)"), so a script there would never be committed.
- Downloads via `huggingface_hub.hf_hub_download(repo_type="dataset", revision=<commit sha>)` (present transitively through `langchain-huggingface`; verify and pin in `requirements-dev.txt` if not) and `httpx` for non-HF URLs. No `datasets` package.
- Cache dir `tests/eval/.data/` (override `DOCBOT_EVAL_DATA_DIR`), added to `.gitignore`.
- `tests/eval/datasets.json` + `tests/eval/DATA_LICENSES.md` (attribution for TAT-QA, TAT-DQA, TrustSQL; share-alike note for BIRD and BIPIA; internal-only list).
- Helper `load_cached(set_name)` used by 3512/3513; when `DOCBOT_EVAL_REQUIRE_DATA=1` (set in `ci.yml`) a missing file fails the test; otherwise it skips locally.
- `ci.yml`: a step before pytest that runs the fetcher for the S1 sets, with `actions/cache` keyed on `datasets.json`.
- Branch `feature/DOCBOT-3501-eval-data-fetch`.

## Test / verification
- `tests/unit/test_data_licenses.py` (manifest schema, internal-only flags, no references from `api/` or `finetune/`, cache ignored) and `tests/unit/test_fetch_eval_data.py` (pin and sha256 checks with a mocked downloader; no network).
- `pytest tests/unit/test_data_licenses.py tests/unit/test_fetch_eval_data.py -v`.
- Cost: $0 LLM; about 10 to 50 MB download for the two S1 sets (parquet).

## Dependencies
Blocked by: none.  Blocks: 3512, 3513, 2545 (10-K fetch), 3502, 3505, 3506, 3516, 3521.

## Definition of done
- [ ] AC met  - [ ] tests pass in CI  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- AC conflict: "nothing committed but code + manifest" here vs "runs offline in CI" in 3513. The proposal above (CI fetch step + `actions/cache`) keeps unit tests offline but adds a network step to the CI job. The alternative is to commit frozen subsets of the two Apache-2.0 sets (deepset 662 rows, a 500-row Gretel subset) with a NOTICE, which the licences allow. Senior PM to pick one.
- Both S1 sets ship as parquet; reading them needs `pyarrow`, which is not pinned in `requirements.txt`. Confirm it is present via pandas or add it to `requirements-dev.txt`.
- Training-leak risk outside this ticket: `finetune/export_dataset.py` (around line 303) exports every `agent_traces` row for a pipeline with no org, consent or eval filter. If any NC dataset (or design-partner data) is ever run through a deployed backend, it becomes fine-tune data. Add a rule that NC evals run in-process only, and file a follow-up to tag and exclude eval traces.
