"""Training utilities for the GRU agent."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class DistillationResult:
    train_bits_per_char: float
    val_bits_per_char: float
    history: List[Dict[str, float]]


@dataclass
class RLResult:
    final_bits_per_char: float
    history: List[Dict[str, float]]


def _bits_from_loss(loss: torch.Tensor) -> torch.Tensor:
    return loss / math.log(2)


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    device: torch.device,
) -> Tuple[float, float]:
    """Return (loss, bits_per_char) for the given dataset."""

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits, _ = model(inputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += targets.numel()

    if total_tokens == 0:
        raise ValueError("Evaluation dataset has no tokens")

    avg_loss = total_loss / total_tokens
    bits_per_char = avg_loss / math.log(2)
    return avg_loss, bits_per_char


def train_distillation(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    *,
    device: torch.device,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    num_workers: int = 0,
    log_interval: int = 100,
) -> DistillationResult:
    """Run supervised distillation to mimic Candlezip agent transcripts."""

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(1, epochs * max(1, len(train_loader)))
    )

    best_val_loss = float("inf")
    history: List[Dict[str, float]] = []
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_tokens = 0

        for step, (inputs, targets) in enumerate(train_loader, start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimiser.zero_grad(set_to_none=True)
            logits, _ = model(inputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()
            scheduler.step()

            running_loss += loss.item() * targets.numel()
            running_tokens += targets.numel()

            if step % log_interval == 0:
                avg_loss = running_loss / running_tokens
                print(
                    f"Epoch {epoch} | Step {step}/{len(train_loader)} | "
                    f"train bits/char={avg_loss / math.log(2):.4f}",
                    flush=True,
                )
                running_loss = 0.0
                running_tokens = 0

        train_loss, train_bpc = evaluate_model(model, train_loader, device=device)
        val_loss, val_bpc = evaluate_model(model, val_loader, device=device)
        history.append({
            "epoch": epoch,
            "train_bits_per_char": train_bpc,
            "val_bits_per_char": val_bpc,
        })
        print(
            f"Epoch {epoch} complete | train bits/char={train_bpc:.4f} | val bits/char={val_bpc:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return DistillationResult(train_bits_per_char=train_bpc, val_bits_per_char=val_bpc, history=history)


def reinforcement_finetune(
    model: nn.Module,
    dataset: Dataset,
    *,
    device: torch.device,
    steps: int = 200,
    batch_size: int = 64,
    lr: float = 5e-4,
    entropy_bonus: float = 0.001,
    grad_clip: float = 1.0,
    baseline_bpc: float | None = None,
    num_workers: int = 0,
) -> RLResult:
    """Fine-tune the model with a lightweight REINFORCE objective.

    The reward signal is derived from *bits saved*: we maintain a moving baseline of
    bits per character and encourage the model to achieve lower values. This mirrors
    the Agentic Compression principle by optimising the model directly for entropy
    reduction rather than proxy metrics.
    """

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    running_bpc = baseline_bpc if baseline_bpc is not None else 8.0
    history: List[Dict[str, float]] = []
    total_steps = 0

    while total_steps < steps:
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimiser.zero_grad(set_to_none=True)
            logits, _ = model(inputs)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
            mean_log_prob = selected.mean()
            nll_bits = (-selected).mean() / math.log(2)
            reward = running_bpc - nll_bits.detach()
            running_bpc = 0.9 * running_bpc + 0.1 * nll_bits.detach().item()
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean() / math.log(2)

            loss = -(reward * mean_log_prob) - entropy_bonus * entropy
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()

            history.append({
                "step": total_steps,
                "reward": reward.item(),
                "bits_per_char": nll_bits.item(),
                "entropy_bits": entropy.item(),
            })

            total_steps += 1
            if total_steps >= steps:
                break

    # Evaluate final performance using the same dataset for simplicity. In practice,
    # a held-out set or Candlezip integration would be preferable.
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    _, final_bpc = evaluate_model(model, eval_loader, device=device)
    return RLResult(final_bits_per_char=final_bpc, history=history)

