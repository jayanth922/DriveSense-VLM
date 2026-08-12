#!/usr/bin/env python3
"""Phase 2a: Run LoRA SFT training for DriveSense-VLM.

Usage:
    python scripts/run_training.py
    python scripts/run_training.py --config configs/training.yaml
    python scripts/run_training.py --resume
    python scripts/run_training.py --dry-run
    python scripts/run_training.py --dry-run --mock
    python scripts/run_training.py --debug
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drivesense.utils.config import load_config, merge_configs  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_training")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="DriveSense-VLM Phase 2a: LoRA SFT training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default="configs/training.yaml", help="Path to training.yaml")
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model, process 3 examples, run 1 forward pass, then exit",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use a tiny mock model for dry-run (no model download required)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Override config: 1 epoch, 10 steps — for quick HPC validation",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_full_config(config_path: str) -> dict:
    """Load and merge model.yaml + data.yaml + training.yaml.

    Args:
        config_path: Path to training.yaml.

    Returns:
        Merged config dict.
    """
    cfg_dir = Path(config_path).parent
    model_cfg = load_config(cfg_dir / "model.yaml")
    data_cfg = load_config(cfg_dir / "data.yaml")
    training_cfg = load_config(config_path)
    return merge_configs(model_cfg, data_cfg, training_cfg)


def apply_debug_overrides(config: dict) -> dict:
    """Patch config in-place for --debug mode.

    Args:
        config: Merged config dict.

    Returns:
        Same dict with training overrides applied.
    """
    debug_cfg = config.get("training", {}).get("debug", {})
    training = config.setdefault("training", {})
    training["max_steps"] = int(debug_cfg.get("max_steps", 10))
    training["num_epochs"] = int(debug_cfg.get("num_epochs", 1))
    training["logging_steps"] = int(debug_cfg.get("logging_steps", 1))
    training["eval_strategy"] = "steps"
    training["save_strategy"] = str(debug_cfg.get("save_strategy", "no"))
    training["report_to"] = "none"
    logger.info(
        "DEBUG mode: max_steps=%d, num_epochs=%d",
        training["max_steps"],
        training["num_epochs"],
    )
    return config


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


def run_dry_run(config: dict, mock: bool = False) -> None:
    """Validate training setup without committing to a full run.

    Steps: load model + processor, load 3 examples, run 1 forward pass,
    print stats, then exit.

    Args:
        config: Merged config dict.
        mock: Use a tiny CPU mock model instead of downloading the real one.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        logger.error("torch not installed. Install with: pip install -e '.[training]'")
        sys.exit(1)

    if mock:
        model, processor = _create_mock_model_processor()
    else:
        from drivesense.training.sft_trainer import setup_model_and_processor
        model, processor, _ = setup_model_and_processor(config)

    from drivesense.training.sft_trainer import DriveSenseDataCollator, DriveSenseSFTDataset

    ann_cfg = config.get("annotation", {})
    sft_dir = Path(ann_cfg.get("sft_output_dir", "outputs/data/sft_ready"))
    train_file = sft_dir / "sft_train.jsonl"

    if not train_file.exists():
        logger.error("SFT data not found: %s", train_file)
        logger.info("Run: python scripts/run_annotation_pipeline.py --mock-llm")
        sys.exit(1)

    max_seq = config.get("training", {}).get("max_seq_length", 2048)
    dataset = DriveSenseSFTDataset(train_file, processor, max_seq_length=max_seq)
    logger.info("Dataset size: %d examples", len(dataset))

    # Time a real per-device micro-batch (bs = per_device_train_batch_size) so the
    # estimate reflects the actual model + config, not a guessed constant.
    pdb = int(config.get("training", {}).get("per_device_train_batch_size", 4))
    n_samples = min(max(1, pdb), len(dataset))
    examples = [dataset[i] for i in range(n_samples)]
    collator = DriveSenseDataCollator(processor, max_seq_length=max_seq)
    batch = collator(examples)

    import torch
    device = next(model.parameters()).device
    batch_gpu = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

    logger.info("Running 1 forward pass ...")
    with torch.no_grad():
        outputs = model(**batch_gpu)

    micro_s = None if mock else _measure_step_seconds(model, batch_gpu)
    _print_dry_run_stats(model, dataset, config, outputs.loss.item(), micro_s, n_samples)


def _measure_step_seconds(model: object, batch_gpu: dict) -> float | None:
    """Median seconds for one real forward+backward micro-step (``None`` off CUDA).

    Replaces the old hardcoded 75 s/step guess: the estimate is now grounded in the
    actual attention impl, image-token count, batch size and gradient-checkpointing
    setting on THIS GPU. 2 warmup + 5 measured iters, CUDA-synchronised.
    """
    import time

    import torch
    if not torch.cuda.is_available():
        return None
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    times: list[float] = []
    for i in range(7):
        for p in trainable:
            p.grad = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(**batch_gpu)
        out.loss.backward()
        torch.cuda.synchronize()
        if i >= 2:
            times.append(time.perf_counter() - t0)
    model.zero_grad(set_to_none=True)
    return sorted(times)[len(times) // 2] if times else None


def _print_dry_run_stats(
    model: object, dataset: object, config: dict, loss: float,
    micro_s: float | None = None, microbatch_size: int | None = None,
) -> None:
    """Print a summary of model parameters and estimated training time.

    ``micro_s`` is the MEASURED seconds per forward+backward micro-batch (from
    :func:`_measure_step_seconds`); when present the time estimate is real, not the
    old 75 s/step guess. ``None`` (e.g. ``--mock`` or CPU) falls back to a clearly
    labelled rough estimate.
    """
    import torch

    try:
        params = list(model.parameters())  # type: ignore[union-attr]
        trainable = sum(p.numel() for p in params if p.requires_grad)
        total = sum(p.numel() for p in params)
    except Exception:  # noqa: BLE001
        trainable, total = 0, 0

    t = config.get("training", {})
    grad_accum = t.get("gradient_accumulation_steps", 4)
    bs_eff = t.get("per_device_train_batch_size", 4) * grad_accum
    n = len(dataset)  # type: ignore[arg-type]
    steps_per_epoch = max(1, n // bs_eff)
    epochs = t.get("num_epochs", 5)
    est_steps = steps_per_epoch * epochs

    if micro_s is not None:
        s_per_step = micro_s * grad_accum
        est_time_h = est_steps * s_per_step / 3600
        time_note = (f"{est_time_h:.2f} h  [MEASURED {s_per_step:.2f}s/optim-step = "
                     f"{micro_s:.3f}s/microbatch(bs={microbatch_size}) x {grad_accum} accum]")
    else:
        est_time_h = est_steps * 75 / 3600
        time_note = (f"{est_time_h:.1f} h  [ROUGH 75s/step guess — NOT measured; "
                     "run on GPU without --mock]")

    print("\n" + "=" * 60)
    print("  DriveSense-VLM — Dry-run Summary")
    print("=" * 60)
    print(f"  Trainable params : {trainable:,} / {total:,} ({100*trainable/max(total,1):.2f}%)")
    print(f"  Train examples   : {n}")
    print(f"  Effective batch  : {bs_eff}")
    print(f"  Steps / epoch    : {steps_per_epoch}")
    print(f"  Total steps      : {est_steps}  ({epochs} epochs)")
    print(f"  Est. time        : {time_note}")
    print(f"  Forward pass loss: {loss:.4f}")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM used        : {alloc:.1f} / {total_vram:.1f} GB")
    print("=" * 60)
    print("  Dry run passed — remove --dry-run to start training.")
    print("=" * 60 + "\n")


def _create_mock_model_processor() -> tuple:
    """Return a tiny CPU model + mock processor for dry-run testing.

    Returns:
        ``(model, processor)`` tuple backed by a trivial linear network.
    """
    import torch
    import torch.nn as nn

    class _MockModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(10, 10)

        def forward(
            self,
            input_ids: object = None,
            labels: object = None,
            **kwargs: object,
        ) -> object:
            import types
            logits = torch.zeros(1, 5, 10)
            loss = torch.tensor(1.0, requires_grad=True)
            out = types.SimpleNamespace(loss=loss, logits=logits)
            return out

        def parameters(self, recurse: bool = True):  # noqa: ANN201, ANN202
            return super().parameters(recurse)

        def save_pretrained(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)

    class _MockProcessor:
        class _Tok:
            pad_token_id = 0

        tokenizer = _Tok()

        def apply_chat_template(
            self,
            messages: list,
            tokenize: bool = False,
            add_generation_prompt: bool = False,
        ) -> str:
            return " ".join(str(m) for m in messages)

        def __call__(self, text: object = None, images: object = None,
                     return_tensors: str = "pt", **kw: object) -> dict:
            return {
                "input_ids": torch.randint(1, 100, (1, 8)),
                "attention_mask": torch.ones(1, 8, dtype=torch.long),
            }

        def save_pretrained(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)

    return _MockModel(), _MockProcessor()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = load_full_config(args.config)

    if args.debug:
        config = apply_debug_overrides(config)

    if args.dry_run:
        run_dry_run(config, mock=args.mock)
        return

    try:
        from drivesense.training.sft_trainer import train
    except ImportError as exc:
        logger.error(
            "Training deps not installed: %s\nInstall: pip install -e '.[training]'", exc
        )
        sys.exit(1)

    if args.debug:
        # train() applies the debug overrides when this is set (no temp config file).
        os.environ["DRIVESENSE_DEBUG"] = "1"

    # train() ALWAYS reloads config from args.config — mutating a locally-loaded
    # config dict here would have no effect (that was the --resume bug). Pass the
    # resume choice as an explicit override instead.
    metrics = train(args.config, resume_override="latest" if args.resume else None)
    logger.info("Training complete.")
    print("\n--- Training Metrics ---")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
