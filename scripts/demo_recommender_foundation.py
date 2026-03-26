#!/usr/bin/env python3
"""Run a minimal end-to-end demo of the MASI recommender foundation.

The script exercises three components:

1. semantic-ID vocabulary and history serialization,
2. cross-modal MLM pretraining objective,
3. autoregressive token generation objective.

It is intentionally small and deterministic so it can serve as a regression
check while the larger dataset and feature-extraction pipeline are still under
construction.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from masi.common.config import load_json_config
from masi.common.io import ensure_directory, write_json
from masi.recommender.generative import GenerativeSIDRecommender
from masi.recommender.mlm import CrossModalMLMPretrainer
from masi.recommender.sasrec import SASRecConfig, SASRecModel
from masi.recommender.sequence_data import (
    CrossModalMLMDataset,
    GenerativeSequenceDataset,
    build_demo_histories,
)
from masi.recommender.training import training_step
from masi.recommender.vocabulary import TokenVocabulary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/recommender_demo.json",
        help="Path to the recommender demo config.",
    )
    return parser.parse_args()


def _load_config(config_path: str) -> tuple[dict[str, object], Path]:
    """Load a JSON config and resolve the repository root."""

    loaded = load_json_config(config_path)
    repo_root = loaded.path.parent.parent
    return loaded.data, repo_root


def _prepare_tensors(config: dict[str, object]) -> tuple[TokenVocabulary, DataLoader, DataLoader, dict[str, list[str]]]:
    """Build the synthetic datasets and loaders used by the demo."""

    fused_ids, user_histories = build_demo_histories()
    vocabulary = TokenVocabulary.build(fused_ids)
    item_tokens = {fused_id.item_id: vocabulary.encode(fused_id.to_tokens()) for fused_id in fused_ids}

    generative_dataset = GenerativeSequenceDataset(
        user_histories=user_histories,
        item_tokens=item_tokens,
        vocabulary=vocabulary,
        history_max_tokens=int(config["history_max_tokens"]),
        target_max_tokens=int(config["target_max_tokens"]),
    )
    mlm_dataset = CrossModalMLMDataset(
        fused_ids=fused_ids,
        vocabulary=vocabulary,
        max_length=int(config["mlm_max_tokens"]),
    )

    batch_size = int(config["batch_size"])
    generative_loader = DataLoader(generative_dataset, batch_size=batch_size, shuffle=False, collate_fn=generative_dataset.collate)
    mlm_loader = DataLoader(mlm_dataset, batch_size=batch_size, shuffle=False, collate_fn=mlm_dataset.collate)
    return vocabulary, generative_loader, mlm_loader, user_histories


def main() -> None:
    """Execute the demo training passes and write a JSON summary."""

    args = parse_args()
    config, repo_root = _load_config(args.config)

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)

    vocabulary, generative_loader, mlm_loader, user_histories = _prepare_tensors(config)

    common_model_args = {
        "vocab_size": len(vocabulary.token_to_id),
        "hidden_dim": int(config["hidden_dim"]),
        "num_heads": int(config["num_heads"]),
        "num_layers": int(config["num_layers"]),
        "dropout": float(config["dropout"]),
        "pad_token_id": vocabulary.pad_id,
    }

    generative_model = GenerativeSIDRecommender(
        **common_model_args,
        max_sequence_length=int(config["history_max_tokens"]) + int(config["target_max_tokens"]),
    )
    mlm_model = CrossModalMLMPretrainer(
        **common_model_args,
        max_sequence_length=int(config["mlm_max_tokens"]),
    )

    sasrec_model = SASRecModel(
        SASRecConfig(
            num_items=1 + len({item_id for history in user_histories.values() for item_id in history}),
            max_sequence_length=4,
            hidden_dim=int(config["hidden_dim"]),
            num_heads=int(config["num_heads"]),
            num_layers=int(config["num_layers"]),
            dropout=float(config["dropout"]),
            pad_token_id=0,
        )
    )

    generative_optimizer = torch.optim.AdamW(generative_model.parameters(), lr=float(config["learning_rate"]))
    mlm_optimizer = torch.optim.AdamW(mlm_model.parameters(), lr=float(config["learning_rate"]))

    generative_batch = next(iter(generative_loader))
    mlm_batch = next(iter(mlm_loader))

    # For the autoregressive objective we concatenate the history prefix and the
    # target sequence so the model learns next-token prediction over one stream.
    generative_inputs = torch.cat(
        [generative_batch["history_token_ids"], generative_batch["target_token_ids"]],
        dim=1,
    )
    generative_loss = training_step(
        model=generative_model,
        optimizer=generative_optimizer,
        batch_inputs=generative_inputs,
        batch_labels=generative_inputs,
        objective="autoregressive",
        pad_token_id=vocabulary.pad_id,
    )
    mlm_loss = training_step(
        model=mlm_model,
        optimizer=mlm_optimizer,
        batch_inputs=mlm_batch["input_token_ids"],
        batch_labels=mlm_batch["label_token_ids"],
        objective="mlm",
        pad_token_id=vocabulary.pad_id,
    )

    item_id_lookup = {item_id: index + 1 for index, item_id in enumerate(sorted({item for history in user_histories.values() for item in history}))}
    sasrec_sequence_batch = []
    for history in user_histories.values():
        encoded = [item_id_lookup[item_id] for item_id in history]
        padded = encoded + [0] * (4 - len(encoded))
        sasrec_sequence_batch.append(padded[:4])
    sasrec_logits = sasrec_model(torch.tensor(sasrec_sequence_batch, dtype=torch.long))

    outputs_root = ensure_directory(repo_root / str(config["outputs_root"]))
    summary = {
        "seed": seed,
        "vocab_size": len(vocabulary.token_to_id),
        "num_generative_examples": len(generative_loader.dataset),
        "num_mlm_examples": len(mlm_loader.dataset),
        "autoregressive_loss": generative_loss,
        "mlm_loss": mlm_loss,
        "sasrec_output_shape": list(sasrec_logits.shape),
        "special_tokens": vocabulary.decode(list(range(len(vocabulary.token_to_id)))[:7]),
    }
    summary_path = write_json(summary, outputs_root / "recommender_demo_summary.json")
    print(json.dumps(summary, indent=2))
    print(f"Wrote recommender demo summary to {summary_path}")


if __name__ == "__main__":
    main()
