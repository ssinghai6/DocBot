# DocBot Fine-Tuning Pipeline (DOCBOT-1523)

## What this is, and why

This is Tier 3 of a 3-tier AI-maturity plan for DocBot's Analytical
Autopilot. Tiers 1 and 2 already shipped:

- `api/trace_service.py` (DOCBOT-1512) logs every autopilot/hybrid/chat run
  to an `agent_traces` table.
- `api/autopilot_service.py`'s `_select_tool_llm` (DOCBOT-1513) replaced a
  pure heuristic router with a model-driven one (one Groq call per
  investigation step, falling back to the heuristic on any failure).
- `api/eval_service.py` (DOCBOT-1519) judges sampled traces with an LLM and
  persists verdicts to `eval_judgments`, scoring (among other things)
  whether the tool choice was plausible (`tool_selection_verdict`).

The original plan's gate for Tier 3 was: **don't build fine-tuning until
there's real evidence the heuristic/router needs it.** A seed batch of real
traces, judged via `python -m tests.eval.run_eval_batch`, came back with
genuine signal — several traces were flagged `tool_selection_verdict`
incorrect/uncertain, not clean. That's the evidence. This directory is the
response: a pipeline that distills `_select_tool_llm`'s own decisions into a
small open-weight model via QLoRA, using the router's own execution traces
as training data, **hindsight-relabeled** using the eval harness's
verdicts — a trace judged "incorrect" is never used as a positive example
of the tool it picked.

This is standalone, offline ML tooling — not FastAPI runtime code — so it
lives in this top-level `finetune/` directory (not `api/`, per
`CLAUDE.md`'s module rules; not `scripts/`, which is gitignored in this
repo) with its own `requirements.txt` kept separate from the production
app's dependencies.

## Why the training data is an approximation (read before trusting it)

`agent_traces` logs one row per *entire autopilot run*, not one row per
*step* — but `_select_tool_llm` makes a decision per step, and the schema
doesn't carry the per-step text, nor the `has_db`/`has_docs`/`has_csv`
flags or persona hint that were live at call time. `finetune/export_dataset.py`
reconstructs an approximation of the router's real input (see that file's
module docstring for the exact heuristics: using the run's whole question
as a step-text stand-in, inferring source flags from which tools appear in
the aggregate `tool_chosen` column, and emitting one example per distinct
tool for a multi-tool run rather than guessing step correspondence).

**This is a known, documented limitation, not a bug to silently work
around.** The correct long-term fix is to log the per-step router decision
directly in `trace_service` (a future ticket) so this export stops needing
to approximate. Until then, this pipeline is built to be correct and
reusable as more real traces accumulate — not hand-tuned to extract the
most signal from today's small seed batch.

## Regenerating the dataset

```bash
python -m finetune.export_dataset --min-examples 1 --output finetune/data/routing_sft.jsonl
```

Requires `DATABASE_URL` pointing at the Postgres instance holding
`agent_traces`/`eval_judgments` (same env var the FastAPI app uses). Re-run
this any time more traces have been logged and judged
(`python -m tests.eval.run_eval_batch`) to grow the dataset — it's additive
in effect (each run re-derives the full JSONL from the current DB state,
it doesn't append).

The script prints a warning if the resulting dataset has fewer than 50
examples — too small for any production fine-tune, but it still writes
whatever is available, since the whole point is to be re-run as the corpus
grows. It never hard-fails on a tiny dataset.

`finetune/data/` is gitignored — it holds real (if synthetic-seeded) usage
data derived from production traces and should never be committed.

## Running the actual training

**This requires a rented GPU.** This is a solo-dev project with no standing
GPU infrastructure — provisioning and paying for compute (RunPod, Modal, or
similar) is a human action this ticket does not automate. A 7B model under
4-bit QLoRA needs roughly 10-16GB of VRAM; a single mid-tier cloud GPU
(e.g. an RTX 4090 or A10) is sufficient.

Once you have a GPU box:

```bash
pip install -r finetune/requirements.txt
python finetune/train_qlora.py --data finetune/data/routing_sft.jsonl
```

Useful flags:
- `--base-model` (default `Qwen/Qwen2.5-7B-Instruct`)
- `--lora-rank` (default `16`)
- `--run-name` (default `routing-v1`) — the adapter is saved to
  `finetune/output/<run-name>/`
- `--epochs`, `--batch-size`

### Validating the script without a GPU

```bash
python finetune/train_qlora.py --data finetune/data/routing_sft.jsonl --dry-run
```

This validates dataset loading + schema, a tokenizer smoke test, and LoRA
config construction wherever the relevant optional dependency is available
in the current environment, and explicitly reports what it skipped (e.g.
`peft`/`trl`/`bitsandbytes` not installed, no GPU). It never attempts to
load the actual 7B model or run a training step — that step is always
reported as `SKIP` in dry-run mode, by design. The run exits non-zero only
on a genuine structural failure (e.g. malformed JSONL), not on a missing
optional dependency.

## Evaluating/shadow-testing the adapter before any cutover

This ticket does **not** build adapter serving or shadow-test integration —
that's the explicit next step, flagged here rather than built, because it
needs its own design (a cheap hosted LoRA endpoint, e.g. Modal, Fireworks,
or Together, serving the adapter alongside the base model).

The mechanism for comparing the fine-tuned router against the live
`_select_tool_llm` already exists: `api/eval_service.py`'s LLM-judge
harness. The intended workflow once serving exists:

1. Stand up the LoRA adapter behind a hosted endpoint.
2. Run both routers (the live Groq call and the adapter) against the same
   sample of real investigation steps, in shadow mode — i.e. the adapter's
   output is logged but never actually used to pick the tool.
3. Judge both sets of decisions with the same `judge_trace` /
   `run_eval_batch` harness and compare `tool_selection_verdict`
   distributions.
4. Only promote the adapter to replace (or gate) the live call if it
   measurably beats the current router's verdict distribution on a
   dataset large enough to trust.

Until that shadow-serving integration is built, this pipeline's output
(the trained LoRA adapter under `finetune/output/`) is an artifact to
evaluate offline, not something wired into the live `api/autopilot_service.py`
code path. `_select_tool_llm` is read-only reference material for this
ticket and was not modified.

## Files

| File | Purpose |
|------|---------|
| `finetune/export_dataset.py` | Queries `agent_traces` LEFT JOIN `eval_judgments`, hindsight-relabels, writes `finetune/data/routing_sft.jsonl` |
| `finetune/train_qlora.py` | QLoRA training script (transformers + peft + trl + bitsandbytes); `--dry-run` validates without a GPU |
| `finetune/requirements.txt` | Training-only deps (not in the main `requirements.txt`) |
| `finetune/data/` | Exported JSONL datasets (gitignored) |
| `finetune/output/` | Trained LoRA adapters (gitignored) |
| `tests/unit/test_export_dataset.py` | Pure-logic tests for the hindsight-relabeling rules (mocked data, no DB) |
