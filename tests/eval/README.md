# DocBot Evaluation Harness

Real, reproducible metrics for DocBot's core capabilities. Three evals, split by
what they need to run.

| Eval | Measures | Needs API keys? | Runs in CI? |
|------|----------|-----------------|-------------|
| `test_discrepancy_eval.py` | Discrepancy precision / recall / F1 | No (pure code) | ✅ Yes — every push/PR (`.github/workflows/ci.yml`) |
| `test_retrieval_eval.py` | Retrieval Recall@k on the demo 10-K | HuggingFace embeddings | ✅ Nightly (`.github/workflows/nightly-eval.yml`, DOCBOT-1503) — needs `HUGGINGFACE_API_KEY` repo secret |
| `eval_latency.py` | TTFT + p50/p95 latency | Running backend | ❌ manual |
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

## What to cite (honestly)

- **Discrepancy precision/recall/F1** — solid, deterministic, reproducible.
- **Recall@k** — real retrieval quality on the demo corpus (state the corpus).
- **TTFT / p95** — measured, environment-dependent (state local vs prod).

Do **not** cite the older `tests/external/test_llm_extraction_baseline.py` as a
RAG number — it feeds ground-truth context to the LLM and bypasses retrieval.
These evals are the retrieval-inclusive replacements.
```
