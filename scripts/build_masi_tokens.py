#!/usr/bin/env python3
"""Build actual Phase 1 + Phase 2 MASI tokens for a bounded Amazon subset.

The pipeline follows the proposal order:

1. freeze CLIP and extract text/image features,
2. train behavior-aware projection heads with collaborative positives,
3. quantize text and image embeddings independently with separate RQ-VAE-style codebooks,
4. late-fuse the resulting codes into `[TXT] ... [VIS] ...` token sequences.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import random
from pathlib import Path

import torch

from masi.alignment.behavior_alignment import train_behavior_aware_alignment
from masi.common.config import find_repo_root, load_json_config
from masi.common.io import ensure_directory, write_json
from masi.common.toggles import MethodToggleConfig
from masi.data.amazon_csj_assets import (
    download_item_images,
    fetch_metadata_slice_for_items,
    load_metadata_slice_from_file,
)
from masi.recommender.amazon_data import select_real_amazon_subset
from masi.tokenization.masi_tokens import (
    build_fused_ids_from_quantized_codes,
    encode_clip_embeddings,
    select_device,
    write_fused_ids,
)
from masi.tokenization.rqvae import train_rqvae_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/masi_tokens_amazon_csj_demo.json",
        help="Path to the MASI token-generation config.",
    )
    return parser.parse_args()


def _load_config(config_path: str) -> tuple[dict[str, object], Path]:
    """Load the JSON config and infer the repository root."""

    loaded = load_json_config(config_path)
    repo_root = find_repo_root(loaded.path)
    return loaded.data, repo_root


def _filter_histories_for_available_items(
    *,
    user_histories: dict[str, list[str]],
    available_item_ids: set[str],
    min_user_interactions: int,
) -> dict[str, list[str]]:
    """Trim user histories to items with full modality coverage."""

    filtered: dict[str, list[str]] = {}
    for user_id, history in user_histories.items():
        trimmed = [item_id for item_id in history if item_id in available_item_ids]
        if len(trimmed) >= min_user_interactions:
            filtered[user_id] = trimmed
    return filtered


def _optional_positive_int(value: object) -> int | None:
    """Convert JSON values into optional positive integer limits."""

    if value is None:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def main() -> None:
    """Execute the bounded MASI token-building pipeline."""

    args = parse_args()
    config, repo_root = _load_config(args.config)
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset_config = dict(config["dataset"])
    assets_config = dict(config["assets"])
    clip_config = dict(config["clip"])
    alignment_config = dict(config["alignment"])
    tokenization_config = dict(config["tokenization"])
    toggles = MethodToggleConfig.from_mapping(config.get("method_toggles"))

    subset = select_real_amazon_subset(
        reviews_path=str(dataset_config["reviews_path"]),
        min_user_interactions=int(dataset_config["min_user_interactions"]),
        min_item_interactions=_optional_positive_int(dataset_config.get("min_item_interactions")),
        max_users=int(dataset_config.get("max_users", 0) or 0),
        max_items=int(dataset_config.get("max_items", 0) or 0),
        max_review_records=_optional_positive_int(dataset_config.get("max_review_records")),
        collapse_consecutive_duplicates=bool(dataset_config.get("collapse_consecutive_duplicates", False)),
    )

    modality_records = dict(subset.item_records)
    metadata_by_item: dict[str, dict[str, object]] = {}
    metadata_source = "review_side_images_and_text"
    metadata_local_path = assets_config.get("metadata_local_path")
    if metadata_local_path:
        resolved_metadata_path = (repo_root / str(metadata_local_path)).expanduser()
        if resolved_metadata_path.exists():
            metadata_by_item = load_metadata_slice_from_file(
                item_ids=set(subset.item_records),
                metadata_path=resolved_metadata_path,
            )
            metadata_source = "local_metadata_file"
    if not metadata_by_item and bool(assets_config.get("use_remote_metadata", False)):
        # Remote metadata remains a fallback path for environments where the
        # full metadata file has not been downloaded locally.
        metadata_by_item = fetch_metadata_slice_for_items(
            item_ids=set(subset.item_records),
            output_path=repo_root / str(assets_config["metadata_cache_path"]),
        )
        if metadata_by_item:
            metadata_source = "remote_metadata_stream"
    if metadata_by_item:
        modality_records.update(metadata_by_item)

    image_paths = download_item_images(
        metadata_by_item=modality_records,
        image_cache_dir=repo_root / str(assets_config["image_cache_dir"]),
    )

    device = select_device()
    text_embeddings, image_embeddings = encode_clip_embeddings(
        metadata_by_item=modality_records,
        image_paths_by_item=image_paths,
        model_name=str(clip_config["model_name"]),
        batch_size=int(clip_config["batch_size"]),
        device=device,
        use_text_modality=toggles.use_text_modality,
        use_visual_modality=toggles.use_visual_modality,
    )

    available_item_ids = set(subset.item_records)
    if toggles.use_text_modality:
        available_item_ids = available_item_ids.intersection(text_embeddings)
    if toggles.use_visual_modality:
        available_item_ids = available_item_ids.intersection(image_embeddings)
    # We only keep items with full text+image coverage because MASI's late
    # fusion assumes both modality-specific code sequences exist.
    filtered_histories = _filter_histories_for_available_items(
        user_histories=subset.user_histories,
        available_item_ids=available_item_ids,
        min_user_interactions=int(dataset_config["min_user_interactions"]),
    )

    # Re-trim embeddings to the items that still appear after coverage filtering.
    used_item_ids = set(item_id for history in filtered_histories.values() for item_id in history)
    if toggles.use_text_modality:
        text_embeddings = {item_id: text_embeddings[item_id] for item_id in used_item_ids}
    else:
        text_embeddings = {}
    if toggles.use_visual_modality:
        image_embeddings = {item_id: image_embeddings[item_id] for item_id in used_item_ids}
    else:
        image_embeddings = {}

    alignment_result = train_behavior_aware_alignment(
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        user_histories=filtered_histories,
        projection_dim=int(alignment_config["projection_dim"]),
        epochs=int(alignment_config["epochs"]),
        batch_size=int(alignment_config["batch_size"]),
        learning_rate=float(alignment_config["learning_rate"]),
        temperature=float(alignment_config["temperature"]),
        hard_negative_count=int(alignment_config["hard_negative_count"]),
        window_size=int(alignment_config["window_size"]),
        dropout=float(alignment_config["dropout"]),
        device=device,
        seed=seed,
        use_behavior_alignment=toggles.use_behavior_alignment,
    )

    text_quantizer_model = None
    image_quantizer_model = None
    text_quantization = None
    image_quantization = None
    if toggles.use_text_modality:
        text_quantizer_model, text_quantization = train_rqvae_model(
            # Text and image quantizers are trained separately on purpose. This is
            # the core late-fusion design decision in the proposal.
            embeddings_by_item=alignment_result.aligned_text_embeddings,
            latent_dim=int(tokenization_config["latent_dim"]),
            depth=int(tokenization_config["depth"]),
            codebook_size=int(tokenization_config["codebook_size"]),
            epochs=int(tokenization_config["epochs"]),
            batch_size=int(tokenization_config["batch_size"]),
            learning_rate=float(tokenization_config["learning_rate"]),
            commitment_weight=float(tokenization_config["commitment_weight"]),
            device=device,
            seed=seed,
            refit_codebooks_with_residual_kmeans=bool(
                tokenization_config.get("refit_codebooks_with_residual_kmeans", False)
            ),
        )
    if toggles.use_visual_modality:
        image_quantizer_model, image_quantization = train_rqvae_model(
            embeddings_by_item=alignment_result.aligned_image_embeddings,
            latent_dim=int(tokenization_config["latent_dim"]),
            depth=int(tokenization_config["depth"]),
            codebook_size=int(tokenization_config["codebook_size"]),
            epochs=int(tokenization_config["epochs"]),
            batch_size=int(tokenization_config["batch_size"]),
            learning_rate=float(tokenization_config["learning_rate"]),
            commitment_weight=float(tokenization_config["commitment_weight"]),
            device=device,
            seed=seed,
            refit_codebooks_with_residual_kmeans=bool(
                tokenization_config.get("refit_codebooks_with_residual_kmeans", False)
            ),
        )

    fused_ids = build_fused_ids_from_quantized_codes(
        # The fused artifact is the handoff boundary into the recommender
        # modules: after this point the sequence model works on discrete tokens.
        item_ids=sorted(used_item_ids),
        text_codes_by_item=text_quantization.code_indices_by_item if text_quantization else {},
        image_codes_by_item=image_quantization.code_indices_by_item if image_quantization else {},
        use_text_modality=toggles.use_text_modality,
        use_visual_modality=toggles.use_visual_modality,
    )

    outputs_root = ensure_directory(repo_root / str(config["outputs_root"]))
    fused_ids_path = write_fused_ids(fused_ids, repo_root / str(config["fused_ids_path"]))
    if not toggles.use_behavior_alignment:
        alignment_status = "disabled"
    elif alignment_result.positive_pairs and alignment_result.loss_history:
        alignment_status = "trained"
    elif not alignment_result.positive_pairs:
        alignment_status = "skipped_no_positive_pairs"
    else:
        alignment_status = "skipped"
    checkpoint_root = config.get("checkpoint_root")
    checkpoint_paths: dict[str, str] = {}
    if checkpoint_root:
        resolved_checkpoint_root = ensure_directory(repo_root / str(checkpoint_root))
        if alignment_result.model_state_dict is not None:
            alignment_checkpoint_path = resolved_checkpoint_root / "behavior_alignment.pt"
            torch.save(
                {
                    "config": config,
                    "method_toggles": asdict(toggles),
                    "model_state_dict": alignment_result.model_state_dict,
                },
                alignment_checkpoint_path,
            )
            checkpoint_paths["behavior_alignment"] = str(alignment_checkpoint_path)
        if text_quantization is not None and text_quantizer_model is not None:
            text_quantizer_checkpoint_path = resolved_checkpoint_root / "text_rqvae.pt"
            torch.save(
                {
                    "config": config,
                    "modality": "text",
                    "model_state_dict": text_quantizer_model.state_dict(),
                    "code_indices_by_item": text_quantization.code_indices_by_item,
                },
                text_quantizer_checkpoint_path,
            )
            checkpoint_paths["text_rqvae"] = str(text_quantizer_checkpoint_path)
        if image_quantization is not None and image_quantizer_model is not None:
            image_quantizer_checkpoint_path = resolved_checkpoint_root / "vision_rqvae.pt"
            torch.save(
                {
                    "config": config,
                    "modality": "vision",
                    "model_state_dict": image_quantizer_model.state_dict(),
                    "code_indices_by_item": image_quantization.code_indices_by_item,
                },
                image_quantizer_checkpoint_path,
            )
            checkpoint_paths["vision_rqvae"] = str(image_quantizer_checkpoint_path)
    summary = {
        "seed": seed,
        "device": str(device),
        "subset_summary": subset.summary,
        "metadata_records_found": len(metadata_by_item),
        "image_files_downloaded": len(image_paths),
        "items_with_full_modalities": len(used_item_ids),
        "users_after_modality_filter": len(filtered_histories),
        "method_toggles": asdict(toggles),
        "multimodal_source": metadata_source,
        "alignment_status": alignment_status,
        "positive_pairs": len(alignment_result.positive_pairs),
        "alignment_steps": len(alignment_result.loss_history),
        "alignment_last_loss": alignment_result.loss_history[-1] if alignment_result.loss_history else None,
        "text_quantization_last_loss": text_quantization.reconstruction_loss_history[-1]
        if text_quantization and text_quantization.reconstruction_loss_history else None,
        "image_quantization_last_loss": image_quantization.reconstruction_loss_history[-1]
        if image_quantization and image_quantization.reconstruction_loss_history else None,
        "unique_text_code_sequences": len({tuple(fused_id.text_codes) for fused_id in fused_ids}),
        "unique_visual_code_sequences": len({tuple(fused_id.visual_codes) for fused_id in fused_ids}),
        "fused_ids_path": str(fused_ids_path),
        "checkpoint_paths": checkpoint_paths,
    }
    summary_path = write_json(summary, outputs_root / "masi_token_summary.json")
    print(json.dumps(summary, indent=2))
    print(f"Wrote MASI token summary to {summary_path}")


if __name__ == "__main__":
    main()
