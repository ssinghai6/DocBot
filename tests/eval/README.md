# DocBot Evaluation Harness

Real, reproducible metrics for DocBot's core capabilities. Three evals, split by
what they need to run.

| Eval | Measures | Needs API keys? | Runs in CI? |
|------|----------|-----------------|-------------|
| `test_discrepancy_eval.py` | Discrepancy precision / recall / F1 | No (pure code) | ✅ Yes |
| `test_retrieval_eval.py` | Retrieval Recall@k on the demo 10-K | HuggingFace embeddings | ❌ external |
| `eval_latency.py` | TTFT + p50/p95 latency | Running backend | ❌ manual |

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

## What to cite (honestly)

- **Discrepancy precision/recall/F1** — solid, deterministic, reproducible.
- **Recall@k** — real retrieval quality on the demo corpus (state the corpus).
- **TTFT / p95** — measured, environment-dependent (state local vs prod).

Do **not** cite the older `tests/external/test_llm_extraction_baseline.py` as a
RAG number — it feeds ground-truth context to the LLM and bypasses retrieval.
These evals are the retrieval-inclusive replacements.
```
