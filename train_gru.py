"""Train and fine-tune the compact GRU model on Candlezip distillation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from gru_trainer.data import DistillationDataset
from gru_trainer.model import SmallGRU
from gru_trainer.train import evaluate_model, reinforcement_finetune, train_distillation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/distillation_dataset.pt"),
        help="Path produced by prepare_distillation.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/gru"),
        help="Directory to store checkpoints and metrics.",
    )
    parser.add_argument("--seq-len", type=int, default=256, help="Training sequence length.")
    parser.add_argument("--distill-epochs", type=int, default=2, help="Number of distillation epochs.")
    parser.add_argument("--distill-batch-size", type=int, default=64, help="Batch size for supervised training.")
    parser.add_argument("--distill-lr", type=float, default=3e-3, help="Learning rate for distillation.")
    parser.add_argument("--rl-steps", type=int, default=200, help="Number of RL fine-tuning steps.")
    parser.add_argument("--rl-batch-size", type=int, default=64, help="Batch size during RL fine-tuning.")
    parser.add_argument("--rl-lr", type=float, default=5e-4, help="Learning rate for RL fine-tuning.")
    parser.add_argument("--rl-entropy", type=float, default=0.001, help="Entropy bonus coefficient.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=20, help="Steps between training logs.")
    parser.add_argument(
        "--max-train-tokens",
        type=int,
        default=0,
        help="Optionally cap the number of training tokens for quick experiments.",
    )
    parser.add_argument(
        "--max-val-tokens",
        type=int,
        default=0,
        help="Optionally cap the number of validation tokens.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = torch.load(args.dataset)
    vocab = artifact["vocab"]
    train_tokens = artifact["train_tokens"]
    val_tokens = artifact["val_tokens"]

    if args.max_train_tokens:
        train_tokens = train_tokens[: args.max_train_tokens]
    if args.max_val_tokens:
        val_tokens = val_tokens[: args.max_val_tokens]

    if train_tokens.numel() <= args.seq_len or val_tokens.numel() <= args.seq_len:
        raise ValueError("Dataset too small for the requested sequence length")

    device = torch.device(args.device)

    train_dataset = DistillationDataset(train_tokens, args.seq_len)
    val_dataset = DistillationDataset(val_tokens, args.seq_len)

    model = SmallGRU(len(vocab)).to(device)

    print("Starting distillation phase...")
    distill_result = train_distillation(
        model,
        train_dataset,
        val_dataset,
        device=device,
        epochs=args.distill_epochs,
        batch_size=args.distill_batch_size,
        lr=args.distill_lr,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        log_interval=args.log_interval,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    distill_ckpt = args.output_dir / "distilled.pt"
    torch.save({"model": model.state_dict(), "vocab": vocab}, distill_ckpt)
    print(f"Saved distilled model to {distill_ckpt}")

    baseline_bpc = distill_result.val_bits_per_char
    print(f"Baseline bits/char for RL: {baseline_bpc:.4f}")

    print("Starting reinforcement fine-tuning phase...")
    rl_result = reinforcement_finetune(
        model,
        train_dataset,
        device=device,
        steps=args.rl_steps,
        batch_size=args.rl_batch_size,
        lr=args.rl_lr,
        entropy_bonus=args.rl_entropy,
        grad_clip=args.grad_clip,
        baseline_bpc=baseline_bpc,
        num_workers=args.num_workers,
    )

    final_ckpt = args.output_dir / "gru_final.pt"
    torch.save({"model": model.state_dict(), "vocab": vocab}, final_ckpt)
    print(f"Saved RL-finetuned model to {final_ckpt}")

    # Evaluate on validation set after RL.
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.rl_batch_size, shuffle=False, drop_last=False
    )
    _, final_bpc = evaluate_model(model, val_loader, device=device)
    print(f"Final validation bits/char after RL: {final_bpc:.4f}")

    metrics: Dict[str, Any] = {
        "distillation": [entry for entry in distill_result.history],
        "rl": rl_result.history,
        "final_bits_per_char": final_bpc,
        "baseline_bits_per_char": baseline_bpc,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    metrics_path = args.output_dir / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()

