"""Utilities for GRU distillation and RL training."""

from .data import (
    collect_agent_corpora,
    build_vocabulary,
    encode_corpora,
    DistillationDataset,
)
from .model import SmallGRU
from .train import (
    train_distillation,
    reinforcement_finetune,
    evaluate_model,
)

__all__ = [
    "collect_agent_corpora",
    "build_vocabulary",
    "encode_corpora",
    "DistillationDataset",
    "SmallGRU",
    "train_distillation",
    "reinforcement_finetune",
    "evaluate_model",
]
