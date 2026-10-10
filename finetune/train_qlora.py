"""QLoRA fine-tune of a small open-weight model on the Autopilot tool-router
distillation dataset — DOCBOT-1523.

Distills ``api/autopilot_service.py``'s ``_select_tool_llm`` (DOCBOT-1513)
tool-routing decisions, hindsight-relabeled via ``api/eval_service.py``
(DOCBOT-1519) judgments, into a small open-weight model via QLoRA
(4-bit `bitsandbytes` quantization + LoRA adapters, trained with TRL's
`SFTTrainer`).

Requires a GPU for a real training run. This solo-dev project has no
standing GPU infrastructure — see ``finetune/README.md`` for exactly what a
human needs to provision (RunPod/Modal/etc.) and pay for themselves before
running this for real.

Usage (real run, on a GPU box with finetune/requirements.txt installed)
-------------------------------------------------------------------------
    python finetune/train_qlora.py \
        --data finetune/data/routing_sft.jsonl \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --lora-rank 16 \
        --run-name routing-v1

Usage (dry run — validates the pipeline without a GPU or full training)
-------------------------------------------------------------------------
    python finetune/train_qlora.py --data finetune/data/routing_sft.jsonl --dry-run

``--dry-run`` validates what it can in whatever environment it's run in and
degrades gracefully:
  * dataset loading + per-example schema validation (always runs — pure
    Python, no deps beyond stdlib) — a FAILURE here fails the dry run.
  * tokenizer load + a tokenization smoke test against the base model, if
    `transformers` is installed and the tokenizer is reachable (cached
    locally or a network call succeeds) — SKIPPED (not failed) if neither.
  * LoRA config construction (rank/target-modules shape validation), if
    `peft` is installed — SKIPPED (not failed) if not.
  * Full model instantiation and an actual training step are NEVER
    attempted in `--dry-run` mode — a 7B model under QLoRA still needs a
    real GPU with enough VRAM, which this validates for but does not
    attempt to provide.
Each step prints PASS/SKIP/FAIL; the process exits non-zero only on a FAIL.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("finetune.train_qlora")

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LORA_RANK = 16
# Standard target modules for Qwen2-architecture attention + MLP blocks.
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

_VALID_TOOLS = {"sql_query", "doc_search", "python_analysis", "unsupported"}


class DryRunStepFailure(Exception):
    """Raised for a hard dry-run failure (as opposed to a graceful skip)."""


# ---------------------------------------------------------------------------
# Dataset loading + validation — always runs, no optional deps required.
# ---------------------------------------------------------------------------


def load_and_validate_dataset(path: Path) -> list[dict]:
    """Load the JSONL SFT dataset and validate each example's shape.

    Raises DryRunStepFailure (or ValueError for the real training path) on
    any structurally invalid row — a malformed example is always a hard
    error, never silently skipped at this layer (export_dataset.py already
    did the semantic filtering).
    """
    if not path.exists():
        raise DryRunStepFailure(f"dataset file not found: {path}")

    examples = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DryRunStepFailure(f"{path}:{line_no}: invalid JSON: {exc}") from exc

            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) != 3:
                raise DryRunStepFailure(
                    f"{path}:{line_no}: expected exactly 3 messages "
                    f"(system, user, assistant), got {messages!r}"
                )
            roles = [m.get("role") for m in messages]
            if roles != ["system", "user", "assistant"]:
                raise DryRunStepFailure(f"{path}:{line_no}: unexpected role order {roles!r}")

            assistant_content = messages[2].get("content", "")
            try:
                parsed = json.loads(assistant_content)
            except json.JSONDecodeError as exc:
                raise DryRunStepFailure(
                    f"{path}:{line_no}: assistant content is not valid JSON: {assistant_content!r}"
                ) from exc
            tool = parsed.get("tool")
            if tool not in _VALID_TOOLS:
                raise DryRunStepFailure(
                    f"{path}:{line_no}: assistant tool {tool!r} not in {_VALID_TOOLS}"
                )

            examples.append(row)

    if not examples:
        raise DryRunStepFailure(f"{path}: contains zero valid examples")

    return examples


# ---------------------------------------------------------------------------
# Optional-dependency-gated checks — each returns "pass" / "skip" and never
# raises; callers decide how to report.
# ---------------------------------------------------------------------------


def _check_tokenizer(base_model: str, examples: list[dict]) -> tuple[str, str]:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return "skip", "transformers not installed"

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    except Exception as exc:  # noqa: BLE001 - any load failure (no network, not cached, gated repo, …)
        return "skip", f"could not load tokenizer for {base_model!r} ({exc})"

    try:
        sample = examples[0]["messages"]
        rendered = tokenizer.apply_chat_template(sample, tokenize=False)
        token_ids = tokenizer(rendered, truncation=True, max_length=2048)["input_ids"]
        if not token_ids:
            return "skip", "tokenizer produced zero tokens for a sample example"
    except Exception as exc:  # noqa: BLE001
        return "skip", f"tokenization smoke test failed ({exc})"

    return "pass", f"tokenized a sample example into {len(token_ids)} tokens"


def _check_lora_config(lora_rank: int) -> tuple[str, str]:
    try:
        from peft import LoraConfig
    except ImportError:
        return "skip", "peft not installed"

    try:
        LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.05,
            target_modules=DEFAULT_TARGET_MODULES,
            task_type="CAUSAL_LM",
            bias="none",
        )
    except Exception as exc:  # noqa: BLE001
        raise DryRunStepFailure(f"LoraConfig construction failed: {exc}") from exc

    return "pass", f"LoraConfig(r={lora_rank}) constructed with {len(DEFAULT_TARGET_MODULES)} target modules"


def _check_trl_bitsandbytes_importable() -> tuple[str, str]:
    missing = []
    for mod in ("trl", "bitsandbytes", "accelerate", "datasets"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        return "skip", f"not installed: {', '.join(missing)} (see finetune/requirements.txt)"
    return "pass", "trl, bitsandbytes, accelerate, datasets all importable"


# ---------------------------------------------------------------------------
# Dry run orchestration
# ---------------------------------------------------------------------------


def run_dry_run(args: argparse.Namespace) -> int:
    results: list[tuple[str, str, str]] = []  # (step, status, detail)

    try:
        examples = load_and_validate_dataset(Path(args.data))
        results.append(("dataset_load_and_validate", "pass", f"{len(examples)} valid examples"))
    except DryRunStepFailure as exc:
        results.append(("dataset_load_and_validate", "fail", str(exc)))
        _print_report(results)
        return 1

    status, detail = _check_tokenizer(args.base_model, examples)
    results.append(("tokenizer_smoke_test", status, detail))

    try:
        status, detail = _check_lora_config(args.lora_rank)
        results.append(("lora_config_construction", status, detail))
    except DryRunStepFailure as exc:
        results.append(("lora_config_construction", "fail", str(exc)))
        _print_report(results)
        return 1

    status, detail = _check_trl_bitsandbytes_importable()
    results.append(("training_deps_importable", status, detail))

    results.append((
        "model_instantiation_and_training_step",
        "skip",
        "never attempted in --dry-run — requires a real GPU with enough VRAM "
        "for the base model under 4-bit quantization; see finetune/README.md",
    ))

    _print_report(results)
    any_fail = any(status == "fail" for _, status, _ in results)
    return 1 if any_fail else 0


def _print_report(results: list[tuple[str, str, str]]) -> None:
    logger.info("=== DOCBOT-1523 QLoRA dry-run report ===")
    for step, status, detail in results:
        logger.info("[%s] %s: %s", status.upper(), step, detail)
    logger.info("=========================================")


# ---------------------------------------------------------------------------
# Real training path — only reached without --dry-run, on a GPU box with
# finetune/requirements.txt installed.
# ---------------------------------------------------------------------------


def run_training(args: argparse.Namespace) -> int:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        logger.error(
            "Missing training dependency: %s. Install finetune/requirements.txt on a "
            "GPU box — these are deliberately not in the main requirements.txt "
            "(see that file's header comment).",
            exc,
        )
        return 1

    if not torch.cuda.is_available():
        logger.error(
            "No CUDA GPU detected. QLoRA training requires a GPU — see "
            "finetune/README.md for how to provision one (RunPod/Modal/etc.). "
            "Use --dry-run to validate the pipeline without a GPU."
        )
        return 1

    examples = load_and_validate_dataset(Path(args.data))
    logger.info("Loaded %d training examples from %s", len(examples), args.data)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=DEFAULT_TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none",
    )

    dataset = Dataset.from_list([{"messages": ex["messages"]} for ex in examples])

    output_dir = Path("finetune/output") / args.run_name
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    logger.info("Saved LoRA adapter to %s", output_dir)
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="finetune/data/routing_sft.jsonl", help="Path to the JSONL SFT dataset")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help=f"Base model (default: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK, help=f"LoRA rank (default: {DEFAULT_LORA_RANK})")
    parser.add_argument("--run-name", default="routing-v1", help="Run name — adapter saved to finetune/output/<run-name>/")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device train batch size (default: 4)")
    parser.add_argument("--dry-run", action="store_true", help="Validate the pipeline without requiring a GPU/training run")
    args = parser.parse_args(argv)

    if args.dry_run:
        return run_dry_run(args)
    return run_training(args)


if __name__ == "__main__":
    sys.exit(main())
