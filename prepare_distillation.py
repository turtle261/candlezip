"""Prepare distillation datasets from Candlezip agent outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from gru_trainer.data import build_vocabulary, collect_agent_corpora, encode_corpora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_300s_nomem"),
        help="Directory containing Candlezip run outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/distillation_dataset.pt"),
        help="Where to store the processed dataset (torch.save).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of tokens reserved for validation.",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=8,
        help="Discard agent responses shorter than this many characters.",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default="\n\n",
        help="Delimiter inserted between segments while concatenating.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpora, stats = collect_agent_corpora(args.results_dir, min_text_length=args.min_text_length)

    print(f"Collected {len(corpora)} corpora from {args.results_dir}")
    total_chars = 0
    for entry in stats:
        total_chars += entry.num_characters
        print(
            f"- {entry.dataset}: segments={entry.num_segments} chars={entry.num_characters} "
            f"source={entry.source_path}."
        )

    vocab, stoi = build_vocabulary(corpora)
    print(f"Vocabulary size (including <pad>): {len(vocab)}")

    encoded = encode_corpora(corpora, stoi, delimiter=args.delimiter)

    train_tensors = []
    val_tensors = []
    for name, tokens in encoded.items():
        split = max(1, int(tokens.numel() * (1.0 - args.val_fraction)))
        split = min(split, tokens.numel() - 1)
        train_tensors.append(tokens[:split])
        val_tensors.append(tokens[split:])
        print(
            f"Dataset {name}: train_tokens={tokens[:split].numel()} val_tokens={tokens[split:].numel()}"
        )

    train_tokens = torch.cat(train_tensors)
    val_tokens = torch.cat(val_tensors)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact: Dict[str, object] = {
        "vocab": vocab,
        "stoi": stoi,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "stats": [
            {
                "dataset": entry.dataset,
                "num_segments": entry.num_segments,
                "num_characters": entry.num_characters,
                "source_path": str(entry.source_path),
            }
            for entry in stats
        ],
        "config": {
            "results_dir": str(args.results_dir),
            "val_fraction": args.val_fraction,
            "min_text_length": args.min_text_length,
            "delimiter": args.delimiter,
        },
    }

    torch.save(artifact, args.output)
    print(f"Saved dataset to {args.output}")
    print(
        json.dumps(
            {
                "total_characters": total_chars,
                "train_tokens": train_tokens.numel(),
                "val_tokens": val_tokens.numel(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

