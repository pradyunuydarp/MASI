#!/usr/bin/env python3
"""Run the SASRec baseline on the same CSJ split contract used by MASI."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from masi.baselines import run_sasrec_baseline
from masi.common.config import find_repo_root, load_json_config
from masi.common.io import ensure_directory
from masi.common.runtime import (
    detect_runtime_environment,
    find_kaggle_dataset_root,
    resolve_input_path,
    resolve_storage_root,
    resolve_torch_device,
)
from masi.recommender.amazon_data import build_real_amazon_histories


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/baseline_sasrec_full_dataset.json",
        help="Path to the SASRec baseline config.",
    )
    parser.add_argument(
        "--storage-root",
        default=None,
        help="Optional storage root for data, checkpoints, and outputs.",
    )
    return parser.parse_args()


def _optional_positive_int(value: object) -> int | None:
    """Convert loose config values into optional positive integers."""

    if value is None:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def main() -> None:
    """Load MASI-format data, train SASRec, and write baseline metrics."""

    args = parse_args()
    loaded = load_json_config(args.config)
    config = loaded.data
    repo_root = find_repo_root(Path(__file__))
    runtime_config = dict(config.get("runtime", {}))
    storage_root = resolve_storage_root(
        repo_root=repo_root,
        runtime_config=runtime_config,
        cli_storage_root=args.storage_root,
    )

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = resolve_torch_device(runtime_config)

    dataset_config = dict(config["dataset"])
    dataset_root = find_kaggle_dataset_root(
        dataset_slugs=dataset_config.get("kaggle_input_slugs"),
        required_relative_paths=[
            dataset_config.get("reviews_relpath") or dataset_config.get("reviews_path"),
        ],
    )
    reviews_path = resolve_input_path(
        repo_root=repo_root,
        storage_root=storage_root,
        configured_path=str(dataset_config.get("reviews_path", "")),
        kaggle_dataset_root=dataset_root,
        relative_path=str(dataset_config.get("reviews_relpath", "")).strip() or None,
    )
    if reviews_path is None or not reviews_path.exists():
        raise FileNotFoundError(
            "Missing reviews input for SASRec baseline. "
            f"Configured path: {dataset_config.get('reviews_path')}"
        )

    imported = build_real_amazon_histories(
        reviews_path=str(reviews_path),
        min_user_interactions=int(dataset_config["min_user_interactions"]),
        min_item_interactions=_optional_positive_int(dataset_config.get("min_item_interactions")),
        max_users=int(dataset_config.get("max_users", 0) or 0),
        max_items=int(dataset_config.get("max_items", 0) or 0),
        max_review_records=_optional_positive_int(dataset_config.get("max_review_records")),
        review_record_offset=int(dataset_config.get("review_record_offset", 0) or 0),
        user_rank_offset=int(dataset_config.get("user_rank_offset", 0) or 0),
        item_rank_offset=int(dataset_config.get("item_rank_offset", 0) or 0),
        collapse_consecutive_duplicates=bool(dataset_config.get("collapse_consecutive_duplicates", False)),
    )
    imported.summary["reviews_path"] = str(reviews_path.resolve())
    imported.summary["resolved_dataset_root"] = str(dataset_root.resolve()) if dataset_root is not None else None
    imported.summary["environment"] = detect_runtime_environment()

    run_name = str(runtime_config.get("run_name", "baseline_sasrec_full_dataset"))
    outputs_root = config.get("outputs_root")
    if outputs_root:
        outputs_path = Path(str(outputs_root)).expanduser()
        outputs_path = outputs_path if outputs_path.is_absolute() else storage_root / outputs_path
    else:
        outputs_path = storage_root / "outputs" / run_name
    checkpoint_root = config.get("checkpoint_root")
    checkpoint_path = None
    if checkpoint_root:
        raw_checkpoint_path = Path(str(checkpoint_root)).expanduser()
        checkpoint_path = raw_checkpoint_path if raw_checkpoint_path.is_absolute() else storage_root / raw_checkpoint_path
    else:
        checkpoint_path = outputs_path / "checkpoints"

    summary = run_sasrec_baseline(
        user_histories=imported.user_histories,
        import_summary=imported.summary,
        config=config,
        device=device,
        outputs_root=ensure_directory(outputs_path),
        checkpoint_root=checkpoint_path,
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote SASRec baseline summary to {outputs_path / 'baseline_summary.json'}")


if __name__ == "__main__":
    main()
