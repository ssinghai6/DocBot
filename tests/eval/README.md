# DocBot Evaluation Harness

Real, reproducible metrics for DocBot's core capabilities. Three evals, split by
what they need to run.

| Eval | Measures | Needs API keys? | Runs in CI? |
|------|----------|-----------------|-------------|
| `test_discrepancy_eval.py` | Discrepancy precision / recall / F1 | No (pure code) | ✅ Yes — every push/PR (`.github/workflows/ci.yml`) |
| `test_retrieval_eval.py` | Retrieval Recall@k on the demo 10-K | HuggingFace embeddings | ✅ Nightly (`.github/workflows/nightly-eval.yml`, DOCBOT-1503) — needs `HUGGINGFACE_API_KEY` repo secret |
| `eval_latency.py` | TTFT + p50/p95 latency | Running backend | ❌ manual |
| `tieout_eval.py` / `test_tieout_eval.py` | Tie-out accuracy: pass rate per gold question over N runs (DOCBOT-2541) | Live mode: Groq/Gemini. Offline mode + scorer tests: none | Scorer + gold-set checks: yes. Live: manual (`-m external`) |
| `run_eval_batch.py` | LLM-judge scoring of real `agent_traces` rows (tool/intent plausibility + evidence support) | Live `DATABASE_URL` + Groq/Gemini | ❌ manual (DOCBOT-1519) |

**Nightly gate (DOCBOT-1503)**: `test_retrieval_recall` hard-asserts `recall[5] >= 0.7` — a nightly run below that baseline fails the job. Runs `pytest tests/eval -m external` on a schedule (not per-push) since it needs a live embeddings call. `eval_latency.py` stays manual — it's a script against a *running* backend, not a pytest test, and nightly CI doesn't spin the backend up.

## 1. Discrepancy detection (the differentiator)

The headline feature — flagging conflicts between a document and a database — is
pure code, so it scores deterministically with no API keys.

```bash
pytest tests/eval/test_discrepancy_eval.py -s
```

Reports precision, recall, F1, and per-case TP/FP/FN over `gold_discrepancy_set.py`.
**Precision and false-positive count are guarded hard** — a single spurious
discrepancy (the "-100%" / "+19566%" class of bug) destroys demo trust.

Add cases by appending to `GOLD_CASES` in `gold_discrepancy_set.py`.

## 2. Retrieval quality (Recall@k)

Builds a real vector store from the demo 10-K chunks and checks how often the
correct source page appears in the top-k.

```bash
# needs huggingface_api_key
pytest tests/eval/test_retrieval_eval.py -s -m external
#   or standalone:
python -m tests.eval.test_retrieval_eval
```

Reports Recall@1 / @3 / @5. Extend `GOLD_QA` with (question, expected-pages).

## 3. Latency (perceived speed)

Times the streaming chat pipeline against a running backend.

```bash
# local backend on :8000
python -m tests.eval.eval_latency
# against prod
DOCBOT_BASE_URL=https://<backend>.railway.app python -m tests.eval.eval_latency
```

Reports **TTFT** (time-to-first-token — the key streaming UX metric) and total
latency p50/p95.

## 4. LLM-judge trace eval (DOCBOT-1519)

Samples recent rows from the live `agent_traces` table (DOCBOT-1512) and asks
an LLM judge whether each trace's tool/intent selection was plausible and
whether its final answer looks supported by the retrieved evidence. Persists
every judgment to `eval_judgments` for later reference (feeds Tier 3
fine-tuning data labeling eventually).

```bash
# needs a live DATABASE_URL (real agent_traces data) + Groq/Gemini key
python -m tests.eval.run_eval_batch --pipeline autopilot --limit 50
python -m tests.eval.run_eval_batch --limit 20 --since-hours 24
```

Manual script, same category as `eval_latency.py` — requires a live external
dependency (a populated production/staging DB) that CI can't provide, so it
is not wired into `ci.yml` or `nightly-eval.yml`. The underlying functions
(`api/eval_service.py`'s `sample_recent_traces` / `judge_trace` /
`run_eval_batch`) are unit-tested with mocks in
`tests/unit/test_eval_service.py`, which does run in CI.

## 5. Tie-out accuracy (DOCBOT-2541) -- the method a public accuracy claim cites

Measures the end-to-end product claim: asked whether a report figure agrees
with the books, does DocBot return the right numbers for both sides, cite both
sides, and flag a discrepancy exactly when there is one?

**Gold set** (`gold_tieout_set.py`): 12 tie-out questions over the two demo
datasets in `api/demo_service.py` -- 10 on Tidewell (draft 10-K vs trial
balance, incl. "Does segment revenue in Note 12 agree with the books?") and 2
on QuickBite. Each question lists its figures: report value, books value
(absolute USD), the document page, and whether a discrepancy must be flagged
(5 flagged, 9 must tie). `test_gold_values_match_demo_data` re-derives every
gold number from the demo chunks and SQLite, so the gold set cannot drift.

**Scorer** (`tieout_eval.score_run`, pure code, no LLM judge). A run passes
only if every check holds:
1. Every figure that must be flagged has exactly one discrepancy with that
   metric whose report and books values each equal gold within tolerance.
2. That discrepancy cites both sides: document file + the gold page, and a DB
   table or query.
3. Every figure that must tie has no discrepancy, its books value appears in
   the SQL result rows, and its document page is among the cited sources.
4. No discrepancy is flagged on any metric outside the dataset's planted set
   (false positive = fail).
5. The run did not error.

**Tolerance**: 0.1% relative, after snapping a power-of-1000 unit gap (books in
$M vs report in $). The smallest planted gap is 0.55%, so a gap can never pass
as a tie.

**Runs**: each question is run N times (`--runs 10`), each live run on a fresh
demo session (no history, no few-shot carry-over). In-process the exact-match
LLM cache is not wired, so every run is an independent model sample; running
against a deployed backend would mostly measure the 6h cache. Report the pass
rate per question and overall.

```bash
# CI (no keys): scorer unit tests + gold-set consistency
pytest tests/eval/test_tieout_eval.py
# offline: real retrieval + gold SQL + real detector + scorer. LLM stages are
# replaced by gold, so this is NOT an accuracy number.
python -m tests.eval.tieout_eval --mode offline --runs 10
# live: the number a public claim may cite (needs groq_api_key, ~3 LLM calls/run)
python -m tests.eval.tieout_eval --mode live --runs 10
pytest tests/eval/test_tieout_eval.py -m external -s   # same, asserts >= 90% per question
```

Cite only a **live** result, with its date, model, N and the gold-set size,
e.g. "On our 12-question tie-out benchmark (2 demo datasets, 10 runs each),
DocBot matched both sides to within 0.1% with a citation for each and flagged
discrepancies exactly when present in X of 120 runs." Offline numbers are
pipeline checks, not accuracy.

## What to cite (honestly)

- **Discrepancy precision/recall/F1** — solid, deterministic, reproducible.
- **Recall@k** — real retrieval quality on the demo corpus (state the corpus).
- **TTFT / p95** — measured, environment-dependent (state local vs prod).
- **Tie-out pass rate** — live mode only (section 5), with date, model and N.

Do **not** cite the older `tests/external/test_llm_extraction_baseline.py` as a
RAG number — it feeds ground-truth context to the LLM and bypasses retrieval.
These evals are the retrieval-inclusive replacements.
```
