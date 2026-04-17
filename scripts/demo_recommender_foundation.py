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

from masi.common.config import find_repo_root, load_json_config
from masi.common.io import ensure_directory, write_json
from masi.common.toggles import MethodToggleConfig
from masi.recommender.amazon_data import build_real_amazon_histories
from masi.recommender.evaluation import build_leave_one_out_split
from masi.recommender.generative import GenerativeSIDRecommender
from masi.recommender.mlm import CrossModalMLMPretrainer
from masi.recommender.sasrec import SASRecConfig, SASRecModel
from masi.recommender.sequence_data import (
    CrossModalMLMDataset,
    GenerativeSequenceDataset,
    resolve_token_budgets,
)
from masi.recommender.training import training_step
from masi.recommender.vocabulary import TokenVocabulary
from masi.tokenization.masi_tokens import load_fused_ids


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/recommender_amazon_csj_demo.json",
        help="Path to the recommender demo config.",
    )
    return parser.parse_args()


def _load_config(config_path: str) -> tuple[dict[str, object], Path]:
    """Load a JSON config and resolve the repository root."""

    loaded = load_json_config(config_path)
    repo_root = find_repo_root(loaded.path)
    return loaded.data, repo_root


def _select_device() -> torch.device:
    """Select the best available PyTorch device for Phase 3 demos."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _prepare_tensors(
    config: dict[str, object]
) -> tuple[TokenVocabulary, DataLoader, DataLoader, dict[str, list[str]], dict[str, object], dict[str, int]]:
    """Build recommender datasets and loaders from imported Amazon data.

    If a real MASI fused-token artifact exists, it takes precedence over the
    older review-field proxy token path.
    """

    dataset_config = dict(config["dataset"])
    toggles = MethodToggleConfig.from_mapping(config.get("method_toggles"))
    max_review_records = dataset_config.get("max_review_records")
    imported = build_real_amazon_histories(
        reviews_path=str(dataset_config["reviews_path"]),
        min_user_interactions=int(dataset_config["min_user_interactions"]),
        min_item_interactions=int(dataset_config.get("min_item_interactions", dataset_config["min_user_interactions"])),
        max_users=int(dataset_config.get("max_users", 0) or 0),
        max_items=int(dataset_config.get("max_items", 0) or 0),
        max_review_records=None if max_review_records is None else int(max_review_records),
        collapse_consecutive_duplicates=bool(dataset_config.get("collapse_consecutive_duplicates", False)),
    )

    token_artifact_raw = str(config.get("token_artifact_path", "")).strip()
    token_artifact_path = Path(token_artifact_raw).expanduser() if token_artifact_raw else None
    if token_artifact_path is not None and token_artifact_path.is_file():
        fused_ids = load_fused_ids(token_artifact_path)
        imported.summary["token_source"] = "phase1_phase2_masi_artifact"
        imported.summary["token_artifact_path"] = str(token_artifact_path.resolve())
    else:
        fused_ids = imported.fused_ids
        imported.summary["token_source"] = "review_field_proxy_fallback"

    available_item_ids = {fused_id.item_id for fused_id in fused_ids}
    user_histories = {}
    for user_id, history in imported.user_histories.items():
        filtered_history = [item_id for item_id in history if item_id in available_item_ids]
        if len(filtered_history) >= int(dataset_config["min_user_interactions"]):
            user_histories[user_id] = filtered_history
    imported.summary["users_after_token_filter"] = len(user_histories)
    imported.summary["items_available_from_tokens"] = len(available_item_ids)

    vocabulary = TokenVocabulary.build(
        fused_ids,
        use_text_modality=toggles.use_text_modality,
        use_visual_modality=toggles.use_visual_modality,
        use_late_fusion=toggles.use_late_fusion,
    )
    item_tokens = {
        fused_id.item_id: vocabulary.encode(
            fused_id.to_tokens(
                use_text_modality=toggles.use_text_modality,
                use_visual_modality=toggles.use_visual_modality,
                use_late_fusion=toggles.use_late_fusion,
            )
        ) for fused_id in fused_ids
    }
    target_max_tokens, mlm_max_tokens = resolve_token_budgets(
        item_tokens=item_tokens,
        configured_target_max_tokens=int(config["target_max_tokens"]) if config.get("target_max_tokens") is not None else None,
        configured_mlm_max_tokens=int(config["mlm_max_tokens"]) if config.get("mlm_max_tokens") is not None else None,
    )

    generative_dataset = GenerativeSequenceDataset(
        user_histories=user_histories,
        item_tokens=item_tokens,
        vocabulary=vocabulary,
        history_max_tokens=int(config["history_max_tokens"]),
        target_max_tokens=target_max_tokens,
    )
    mlm_dataset = CrossModalMLMDataset(
        fused_ids=fused_ids,
        vocabulary=vocabulary,
        max_length=mlm_max_tokens,
        use_text_modality=toggles.use_text_modality,
        use_visual_modality=toggles.use_visual_modality,
        use_late_fusion=toggles.use_late_fusion,
    )

    batch_size = int(config["batch_size"])
    generative_loader = DataLoader(
        generative_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=generative_dataset.collate,
    )
    mlm_loader = DataLoader(
        mlm_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=mlm_dataset.collate,
    )
    split = build_leave_one_out_split(
        user_histories=user_histories,
        cold_start_ratio=float(config.get("cold_start_ratio", 0.25)),
        min_train_history=1,
        seed=int(config["seed"]),
        use_cold_start_evaluation=toggles.use_cold_start_evaluation,
    )
    imported.summary["split_summary"] = split.summary
    return vocabulary, generative_loader, mlm_loader, user_histories, imported.summary, {
        "target_max_tokens": target_max_tokens,
        "mlm_max_tokens": mlm_max_tokens,
    }


def main() -> None:
    """Execute the demo training passes and write a JSON summary."""

    args = parse_args()
    config, repo_root = _load_config(args.config)

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _select_device()

    vocabulary, generative_loader, mlm_loader, user_histories, import_summary, resolved_budgets = _prepare_tensors(config)

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
        max_sequence_length=int(config["history_max_tokens"]) + resolved_budgets["target_max_tokens"],
    ).to(device)
    mlm_model = CrossModalMLMPretrainer(
        **common_model_args,
        max_sequence_length=resolved_budgets["mlm_max_tokens"],
    ).to(device)

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
    ).to(device)
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
        batch_inputs=mlm_batch["input_token_ids"].to(device),
        batch_labels=mlm_batch["label_token_ids"].to(device),
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
        "device": str(device),
        "vocab_size": len(vocabulary.token_to_id),
        "resolved_target_max_tokens": resolved_budgets["target_max_tokens"],
        "resolved_mlm_max_tokens": resolved_budgets["mlm_max_tokens"],
        "num_generative_examples": len(generative_loader.dataset),
        "num_mlm_examples": len(mlm_loader.dataset),
        "autoregressive_loss": generative_loss,
        "mlm_loss": mlm_loss,
        "sasrec_output_shape": list(sasrec_logits.shape),
        "special_tokens": vocabulary.decode(list(range(len(vocabulary.token_to_id)))[:7]),
        "import_summary": import_summary,
    }
    summary_path = write_json(summary, outputs_root / "recommender_demo_summary.json")
    print(json.dumps(summary, indent=2))
    print(f"Wrote recommender demo summary to {summary_path}")


if __name__ == "__main__":
    main()
