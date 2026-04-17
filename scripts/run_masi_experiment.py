#!/usr/bin/env python3
"""Run the bounded MASI experiment pipeline with ablation toggles.

This script covers the proposal's later-stage coding path:

1. load fused semantic IDs and chronological Amazon histories,
2. build deterministic warm-start and zero-shot leave-one-out splits,
3. optionally pretrain with cross-modal MLM,
4. fine-tune the generative recommender on chronological histories,
5. evaluate `HR@10`, `NDCG@10`, `Coverage@10`, and retrieval latency.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from masi.common.config import find_repo_root, load_json_config
from masi.common.io import ensure_directory, write_json
from masi.common.toggles import MethodToggleConfig
from masi.recommender.amazon_data import build_real_amazon_histories
from masi.recommender.evaluation import build_leave_one_out_split, evaluate_generative_ranking
from masi.recommender.generative import GenerativeSIDRecommender
from masi.recommender.mlm import CrossModalMLMPretrainer
from masi.recommender.retrieval import build_item_token_lookup, match_generated_sequence_to_items
from masi.recommender.sequence_data import (
    CrossModalMLMDataset,
    GenerativeSequenceDataset,
    resolve_token_budgets,
    serialize_history_tokens,
)
from masi.recommender.training import initialize_generative_from_mlm, run_training_epochs
from masi.recommender.vocabulary import TokenVocabulary
from masi.tokenization.masi_tokens import load_fused_ids


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/masi_experiment_amazon_csj_demo.json",
        help="Path to the experiment config.",
    )
    return parser.parse_args()


def _load_config(config_path: str) -> tuple[dict[str, object], Path]:
    """Load config JSON and resolve the repository root."""

    loaded = load_json_config(config_path)
    repo_root = find_repo_root(loaded.path)
    return loaded.data, repo_root


def _select_device() -> torch.device:
    """Select the best available PyTorch device."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _optional_positive_int(value: object) -> int | None:
    """Convert JSON values into optional positive integer limits."""

    if value is None:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _prepare_inputs(
    *,
    config: dict[str, object],
    toggles: MethodToggleConfig,
) -> tuple[list, dict[str, list[str]], dict[str, object]]:
    """Load histories and fused IDs from the bounded Amazon subset."""

    dataset_config = dict(config["dataset"])
    imported = build_real_amazon_histories(
        reviews_path=str(dataset_config["reviews_path"]),
        min_user_interactions=int(dataset_config["min_user_interactions"]),
        min_item_interactions=_optional_positive_int(dataset_config.get("min_item_interactions")),
        max_users=int(dataset_config.get("max_users", 0) or 0),
        max_items=int(dataset_config.get("max_items", 0) or 0),
        max_review_records=_optional_positive_int(dataset_config.get("max_review_records")),
        collapse_consecutive_duplicates=bool(dataset_config.get("collapse_consecutive_duplicates", False)),
    )

    token_artifact_raw = str(config.get("token_artifact_path", "")).strip()
    token_artifact_path = Path(token_artifact_raw).expanduser() if token_artifact_raw else None
    if token_artifact_path is not None and token_artifact_path.is_file():
        fused_ids = load_fused_ids(token_artifact_path)
        imported.summary["token_source"] = "phase1_phase2_masi_artifact"
        imported.summary["token_artifact_path"] = str(token_artifact_path.resolve())
    elif bool(config.get("require_token_artifact", False)):
        raise FileNotFoundError(
            "The experiment config requires a Phase 1 + Phase 2 token artifact, "
            f"but no file was found at {token_artifact_path}."
        )
    else:
        fused_ids = imported.fused_ids
        imported.summary["token_source"] = "review_field_proxy_fallback"

    available_item_ids = {fused_id.item_id for fused_id in fused_ids}
    filtered_histories: dict[str, list[str]] = {}
    for user_id, history in imported.user_histories.items():
        kept = [
            item_id for item_id in history
            if item_id in available_item_ids
        ]
        if len(kept) >= int(config["min_sequence_items"]):
            filtered_histories[user_id] = kept

    active_fused_ids = []
    used_item_ids = {item_id for history in filtered_histories.values() for item_id in history}
    for fused_id in fused_ids:
        if fused_id.item_id not in used_item_ids:
            continue
        if not toggles.use_text_modality:
            fused_id = type(fused_id)(item_id=fused_id.item_id, text_codes=[], visual_codes=fused_id.visual_codes)
        if not toggles.use_visual_modality:
            fused_id = type(fused_id)(item_id=fused_id.item_id, text_codes=fused_id.text_codes, visual_codes=[])
        active_fused_ids.append(fused_id)

    imported.summary["users_after_token_filter"] = len(filtered_histories)
    imported.summary["items_after_token_filter"] = len(active_fused_ids)
    return active_fused_ids, filtered_histories, imported.summary


def main() -> None:
    """Run the bounded experiment and write metrics to disk."""

    args = parse_args()
    config, repo_root = _load_config(args.config)
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    toggles = MethodToggleConfig.from_mapping(config.get("method_toggles"))
    device = _select_device()

    fused_ids, user_histories, import_summary = _prepare_inputs(config=config, toggles=toggles)
    vocabulary = TokenVocabulary.build(
        fused_ids,
        use_text_modality=toggles.use_text_modality,
        use_visual_modality=toggles.use_visual_modality,
        use_late_fusion=toggles.use_late_fusion,
    )
    item_tokens = build_item_token_lookup(
        fused_ids=fused_ids,
        vocabulary=vocabulary,
        use_text_modality=toggles.use_text_modality,
        use_visual_modality=toggles.use_visual_modality,
        use_late_fusion=toggles.use_late_fusion,
    )
    target_max_tokens, mlm_max_tokens = resolve_token_budgets(
        item_tokens=item_tokens,
        configured_target_max_tokens=_optional_positive_int(config.get("target_max_tokens")),
        configured_mlm_max_tokens=_optional_positive_int(config.get("mlm_max_tokens")),
    )

    split = build_leave_one_out_split(
        user_histories=user_histories,
        cold_start_ratio=float(config["cold_start_ratio"]),
        min_train_history=int(config["min_train_history"]),
        seed=seed,
        use_cold_start_evaluation=toggles.use_cold_start_evaluation,
    )

    train_dataset = GenerativeSequenceDataset(
        user_histories=split.train_histories,
        item_tokens=item_tokens,
        vocabulary=vocabulary,
        history_max_tokens=int(config["history_max_tokens"]),
        target_max_tokens=target_max_tokens,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=len(train_dataset) > 0,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=train_dataset.collate,
    )

    mlm_dataset = CrossModalMLMDataset(
        fused_ids=fused_ids,
        vocabulary=vocabulary,
        max_length=mlm_max_tokens,
        use_text_modality=toggles.use_text_modality,
        use_visual_modality=toggles.use_visual_modality,
        use_late_fusion=toggles.use_late_fusion,
    )
    mlm_loader = DataLoader(
        mlm_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=len(mlm_dataset) > 0,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=mlm_dataset.collate,
    )

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
        max_sequence_length=int(config["history_max_tokens"]) + target_max_tokens,
    ).to(device)
    mlm_model = CrossModalMLMPretrainer(
        **common_model_args,
        max_sequence_length=mlm_max_tokens,
    ).to(device)

    mlm_history: list[float] = []
    if toggles.use_cross_modal_mlm and len(mlm_dataset) > 0:
        mlm_optimizer = torch.optim.AdamW(mlm_model.parameters(), lr=float(config["learning_rate"]))
        mlm_history = run_training_epochs(
            model=mlm_model,
            optimizer=mlm_optimizer,
            data_loader=mlm_loader,
            objective="mlm",
            pad_token_id=vocabulary.pad_id,
            input_key="input_token_ids",
            label_key="label_token_ids",
            epochs=int(config["mlm_epochs"]),
            device=device,
        )
        initialize_generative_from_mlm(
            generative_model=generative_model,
            mlm_model=mlm_model,
            use_cross_modal_mlm=True,
        )

    generative_history: list[float] = []
    if toggles.use_generative_finetuning and len(train_dataset) > 0:
        generative_optimizer = torch.optim.AdamW(generative_model.parameters(), lr=float(config["learning_rate"]))
        class _AutoregressiveLoader:
            def __iter__(self_nonlocal):
                for batch in train_loader:
                    yield {
                        "input_token_ids": torch.cat(
                            [batch["history_token_ids"], batch["target_token_ids"]],
                            dim=1,
                        ),
                        "label_token_ids": torch.cat(
                            [batch["history_token_ids"], batch["target_token_ids"]],
                            dim=1,
                        ),
                    }

        generative_history = run_training_epochs(
            model=generative_model,
            optimizer=generative_optimizer,
            data_loader=_AutoregressiveLoader(),
            objective="autoregressive",
            pad_token_id=vocabulary.pad_id,
            input_key="input_token_ids",
            label_key="label_token_ids",
            epochs=int(config["autoregressive_epochs"]),
            device=device,
        )

    candidate_item_ids = sorted(item_tokens)
    top_k = int(config["top_k"])
    warm_metrics = evaluate_generative_ranking(
        model=generative_model,
        examples=split.warm_examples,
        candidate_item_ids=candidate_item_ids,
        item_tokens=item_tokens,
        vocabulary=vocabulary,
        max_sequence_length=generative_model.max_sequence_length,
        device=device,
        top_k=top_k,
    )
    cold_metrics = evaluate_generative_ranking(
        model=generative_model,
        examples=split.cold_examples,
        candidate_item_ids=candidate_item_ids,
        item_tokens=item_tokens,
        vocabulary=vocabulary,
        max_sequence_length=generative_model.max_sequence_length,
        device=device,
        top_k=top_k,
    )

    sample_generation = {}
    if split.warm_examples:
        example = split.warm_examples[0]
        prefix = serialize_history_tokens(
            history_item_ids=example.history_item_ids,
            item_tokens=item_tokens,
            vocabulary=vocabulary,
        )
        generated = generative_model.greedy_decode(
            prefix_token_ids=torch.tensor([prefix], dtype=torch.long, device=device),
            max_new_tokens=target_max_tokens,
            stop_token_id=vocabulary.eos_id,
        )[0].detach().cpu().tolist()[len(prefix):]
        sample_generation = {
            "user_id": example.user_id,
            "history_item_ids": example.history_item_ids,
            "target_item_id": example.target_item_id,
            "generated_token_ids": generated,
            "matched_items": [
                candidate.item_id
                for candidate in match_generated_sequence_to_items(
                    generated_token_ids=generated,
                    item_tokens=item_tokens,
                )[:5]
            ],
        }

    outputs_root = ensure_directory(repo_root / str(config["outputs_root"]))
    checkpoint_root = config.get("checkpoint_root")
    checkpoint_paths: dict[str, str] = {}
    if checkpoint_root:
        resolved_checkpoint_root = ensure_directory(repo_root / str(checkpoint_root))
        generative_checkpoint_path = resolved_checkpoint_root / "generative_recommender.pt"
        torch.save(
            {
                "config": config,
                "method_toggles": asdict(toggles),
                "model_state_dict": generative_model.state_dict(),
                "vocabulary": vocabulary.token_to_id,
                "item_tokens": item_tokens,
            },
            generative_checkpoint_path,
        )
        checkpoint_paths["generative_recommender"] = str(generative_checkpoint_path)

        if mlm_history:
            mlm_checkpoint_path = resolved_checkpoint_root / "cross_modal_mlm.pt"
            torch.save(
                {
                    "config": config,
                    "method_toggles": asdict(toggles),
                    "model_state_dict": mlm_model.state_dict(),
                    "vocabulary": vocabulary.token_to_id,
                },
                mlm_checkpoint_path,
            )
            checkpoint_paths["cross_modal_mlm"] = str(mlm_checkpoint_path)

    summary = {
        "seed": seed,
        "device": str(device),
        "method_toggles": asdict(toggles),
        "resolved_target_max_tokens": target_max_tokens,
        "resolved_mlm_max_tokens": mlm_max_tokens,
        "mlm_status": "trained" if mlm_history else ("disabled" if not toggles.use_cross_modal_mlm else "skipped_no_examples"),
        "generative_finetuning_status": "trained" if generative_history else ("disabled" if not toggles.use_generative_finetuning else "skipped_no_examples"),
        "vocab_size": len(vocabulary.token_to_id),
        "num_items": len(candidate_item_ids),
        "num_train_examples": len(train_dataset),
        "num_mlm_examples": len(mlm_dataset),
        "mlm_loss_history": mlm_history,
        "autoregressive_loss_history": generative_history,
        "split_summary": split.summary,
        "warm_metrics": warm_metrics,
        "cold_metrics": cold_metrics,
        "sample_generation": sample_generation,
        "import_summary": import_summary,
        "checkpoint_paths": checkpoint_paths,
    }
    summary_path = write_json(summary, outputs_root / "experiment_summary.json")
    print(json.dumps(summary, indent=2))
    print(f"Wrote experiment summary to {summary_path}")


if __name__ == "__main__":
    main()
