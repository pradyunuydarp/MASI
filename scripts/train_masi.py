#!/usr/bin/env python3
"""Run the bounded MASI CSJ pipeline from one entrypoint.

The canonical workflow now assumes a prepared sliced CSJ dataset with:

1. bounded review records,
2. bounded metadata records,
3. optional preloaded item images.

The script still keeps the raw CSJ download path available as an explicit
fallback for local preparation workflows, but prepared subsets are now the
default training contract for Kaggle and other bounded runs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from masi.common.config import find_repo_root, load_json_config
from masi.common.io import ensure_directory, write_json
from masi.common.runtime import (
    detect_runtime_environment,
    find_kaggle_dataset_root,
    resolve_input_path,
    resolve_path,
    resolve_storage_root,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/masi_train_csj_subset_large.json",
        help="Path to the one-click training config.",
    )
    parser.add_argument(
        "--storage-root",
        default=None,
        help="Optional storage root for raw data, checkpoints, and outputs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run stage scripts even if their output summaries already exist.",
    )
    return parser.parse_args()


def _run_python_script(
    *,
    repo_root: Path,
    script_path: Path,
    arguments: list[str],
) -> None:
    """Execute a repository Python script with the correct `PYTHONPATH`."""

    env = dict(os.environ)
    src_path = str((repo_root / "src").resolve())
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"
    command = [sys.executable, str(script_path), *arguments]
    subprocess.run(command, check=True, cwd=repo_root, env=env)


def _ensure_dataset_inputs(
    *,
    repo_root: Path,
    config: dict[str, object],
    reviews_path: Path,
    metadata_path: Path,
    dataset_root: Path | None,
) -> dict[str, object]:
    """Ensure the configured dataset inputs exist, optionally downloading raw files."""

    runtime_config = dict(config.get("runtime", {}))
    raw_dir = ensure_directory(reviews_path.parent)
    downloaded_reviews = False
    downloaded_metadata = False

    if not reviews_path.exists() and bool(runtime_config.get("download_reviews_if_missing", True)):
        download_args = ["--output-dir", str(raw_dir)]
        if bool(runtime_config.get("full_reviews", True)):
            download_args.append("--full-reviews")
        else:
            review_range_end = int(runtime_config.get("review_range_end", (2 * 1024 * 1024 * 1024) - 1))
            download_args.extend(["--review-range-end", str(review_range_end)])
        if bool(runtime_config.get("download_metadata_if_missing", True)) and not metadata_path.exists():
            download_args.append("--download-metadata")
        _run_python_script(
            repo_root=repo_root,
            script_path=repo_root / "scripts" / "download_amazon_csj_dataset.py",
            arguments=download_args,
        )
        downloaded_reviews = True
        downloaded_metadata = metadata_path.exists()

    if not metadata_path.exists() and bool(runtime_config.get("download_metadata_if_missing", True)):
        _run_python_script(
            repo_root=repo_root,
            script_path=repo_root / "scripts" / "download_amazon_csj_dataset.py",
            arguments=[
                "--output-dir", str(raw_dir),
                "--download-metadata",
                "--skip-reviews",
            ],
        )
        downloaded_metadata = True

    if not reviews_path.exists():
        raise FileNotFoundError(
            f"Missing reviews file at {reviews_path}. "
            "Attach the configured prepared subset dataset or update `dataset.reviews_path`."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata file at {metadata_path}. "
            "Attach the configured prepared subset dataset or update `dataset.metadata_path`."
        )

    return {
        "downloaded_reviews": downloaded_reviews,
        "downloaded_metadata": downloaded_metadata,
        "reviews_path": str(reviews_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "dataset_root": str(dataset_root.resolve()) if dataset_root is not None else None,
    }


def main() -> None:
    """Run the full MASI training pipeline."""

    args = parse_args()
    loaded = load_json_config(args.config)
    config = loaded.data
    repo_root = find_repo_root(Path(__file__))
    runtime_config = dict(config.get("runtime", {}))
    environment = detect_runtime_environment()
    storage_root = resolve_storage_root(
        repo_root=repo_root,
        runtime_config=runtime_config,
        cli_storage_root=args.storage_root,
    )

    dataset_config = dict(config["dataset"])
    assets_config = dict(config.get("assets", {}))

    run_name = str(runtime_config.get("run_name", "masi_train_csj"))
    run_root = ensure_directory(storage_root / "outputs" / run_name)
    resolved_config_root = ensure_directory(run_root / "resolved_configs")
    checkpoint_root = ensure_directory(run_root / "checkpoints")

    dataset_root = find_kaggle_dataset_root(
        dataset_slugs=dataset_config.get("kaggle_input_slugs"),
        required_relative_paths=[
            dataset_config.get("reviews_relpath") or dataset_config.get("reviews_path"),
            dataset_config.get("metadata_relpath") or dataset_config.get("metadata_path"),
        ],
    )
    reviews_path = resolve_input_path(
        repo_root=repo_root,
        storage_root=storage_root,
        configured_path=str(dataset_config.get("reviews_path", "")),
        kaggle_dataset_root=dataset_root,
        relative_path=str(dataset_config.get("reviews_relpath", "")).strip() or None,
    )
    metadata_path = resolve_input_path(
        repo_root=repo_root,
        storage_root=storage_root,
        configured_path=str(dataset_config.get("metadata_path", "")),
        kaggle_dataset_root=dataset_root,
        relative_path=str(dataset_config.get("metadata_relpath", "")).strip() or None,
    )
    assert reviews_path is not None
    assert metadata_path is not None

    resolved_dataset_root = dataset_root
    if resolved_dataset_root is None and reviews_path.exists() and metadata_path.exists() and reviews_path.parent == metadata_path.parent:
        resolved_dataset_root = reviews_path.parent

    data_setup = _ensure_dataset_inputs(
        repo_root=repo_root,
        config=config,
        reviews_path=reviews_path,
        metadata_path=metadata_path,
        dataset_root=resolved_dataset_root,
    )

    token_outputs_root = ensure_directory(run_root / "phase12_tokens")
    experiment_outputs_root = ensure_directory(run_root / "phase3_experiment")
    image_cache_dir = resolve_path(
        storage_root,
        str(assets_config.get("image_cache_dir", f"data/processed/{run_name}/images")),
    )
    metadata_cache_path = resolve_path(
        storage_root,
        str(assets_config.get("metadata_cache_path", f"data/processed/{run_name}/metadata.slice.jsonl")),
    )
    assert image_cache_dir is not None
    assert metadata_cache_path is not None

    preloaded_image_dir = resolve_input_path(
        repo_root=repo_root,
        storage_root=storage_root,
        configured_path=str(assets_config.get("preloaded_images_path", "")).strip() or None,
        kaggle_dataset_root=resolved_dataset_root,
        relative_path=str(assets_config.get("preloaded_images_relpath", "")).strip() or None,
    )
    preloaded_image_dirs = [
        str(preloaded_image_dir.resolve())
    ] if preloaded_image_dir is not None and preloaded_image_dir.exists() else []

    token_config = {
        "seed": int(config["seed"]),
        "dataset": {
            "reviews_path": str(reviews_path.resolve()),
            "min_user_interactions": int(dataset_config["min_user_interactions"]),
            "min_item_interactions": int(dataset_config.get("min_item_interactions", dataset_config["min_user_interactions"])),
            "max_users": dataset_config.get("max_users"),
            "max_items": dataset_config.get("max_items"),
            "max_review_records": dataset_config.get("max_review_records"),
            "collapse_consecutive_duplicates": bool(dataset_config.get("collapse_consecutive_duplicates", False)),
        },
        "assets": {
            "metadata_local_path": str(metadata_path.resolve()),
            "use_remote_metadata": bool(assets_config.get("use_remote_metadata", False)),
            "metadata_cache_path": str(metadata_cache_path.resolve()),
            "image_cache_dir": str(image_cache_dir.resolve()),
            "preloaded_image_dirs": preloaded_image_dirs,
            "download_missing_images": bool(assets_config.get("download_missing_images", True)),
            "image_download_workers": int(assets_config.get("image_download_workers", 1)),
            "image_download_retries": int(assets_config.get("image_download_retries", 0)),
            "image_download_timeout_seconds": int(assets_config.get("image_download_timeout_seconds", 30)),
            "image_download_resume": bool(assets_config.get("image_download_resume", True)),
        },
        "clip": dict(config["clip"]),
        "alignment": dict(config["alignment"]),
        "tokenization": dict(config["tokenization"]),
        "checkpointing": dict(config.get("checkpointing", {})),
        "method_toggles": dict(config.get("method_toggles", {})),
        "outputs_root": str(token_outputs_root.resolve()),
        "fused_ids_path": str((token_outputs_root / "fused_semantic_ids.jsonl").resolve()),
        "checkpoint_root": str((checkpoint_root / "phase12_tokens").resolve()),
    }
    token_config_path = write_json(token_config, resolved_config_root / "token_build.json")

    experiment_config = {
        "seed": int(config["seed"]),
        "history_max_tokens": int(config["experiment"]["history_max_tokens"]),
        "target_max_tokens": config["experiment"].get("target_max_tokens"),
        "mlm_max_tokens": config["experiment"].get("mlm_max_tokens"),
        "batch_size": int(config["experiment"]["batch_size"]),
        "learning_rate": float(config["experiment"]["learning_rate"]),
        "hidden_dim": int(config["experiment"]["hidden_dim"]),
        "num_heads": int(config["experiment"]["num_heads"]),
        "num_layers": int(config["experiment"]["num_layers"]),
        "dropout": float(config["experiment"]["dropout"]),
        "mlm_epochs": int(config["experiment"]["mlm_epochs"]),
        "autoregressive_epochs": int(config["experiment"]["autoregressive_epochs"]),
        "top_k": int(config["experiment"]["top_k"]),
        "cold_start_ratio": float(config["experiment"]["cold_start_ratio"]),
        "min_train_history": int(config["experiment"]["min_train_history"]),
        "min_sequence_items": int(config["experiment"]["min_sequence_items"]),
        "outputs_root": str(experiment_outputs_root.resolve()),
        "checkpoint_root": str((checkpoint_root / "phase3_experiment").resolve()),
        "token_artifact_path": token_config["fused_ids_path"],
        "require_token_artifact": True,
        "checkpointing": dict(config.get("checkpointing", {})),
        "method_toggles": dict(config.get("method_toggles", {})),
        "dataset": token_config["dataset"],
    }
    experiment_config_path = write_json(experiment_config, resolved_config_root / "experiment.json")

    token_summary_path = token_outputs_root / "masi_token_summary.json"
    experiment_summary_path = experiment_outputs_root / "experiment_summary.json"
    resume_if_artifacts_exist = bool(runtime_config.get("resume_if_artifacts_exist", True))

    if args.force or not (resume_if_artifacts_exist and token_summary_path.exists()):
        _run_python_script(
            repo_root=repo_root,
            script_path=repo_root / "scripts" / "build_masi_tokens.py",
            arguments=["--config", str(token_config_path)],
        )

    if args.force or not (resume_if_artifacts_exist and experiment_summary_path.exists()):
        _run_python_script(
            repo_root=repo_root,
            script_path=repo_root / "scripts" / "run_masi_experiment.py",
            arguments=["--config", str(experiment_config_path)],
        )

    with token_summary_path.open("r", encoding="utf-8") as handle:
        token_summary = json.load(handle)
    with experiment_summary_path.open("r", encoding="utf-8") as handle:
        experiment_summary = json.load(handle)

    manifest = {
        "environment": environment,
        "storage_root": str(storage_root.resolve()),
        "run_root": str(run_root.resolve()),
        "resolved_config_paths": {
            "token_build": str(token_config_path.resolve()),
            "experiment": str(experiment_config_path.resolve()),
        },
        "data_setup": data_setup,
        "resolved_dataset_root": str(resolved_dataset_root.resolve()) if resolved_dataset_root is not None else None,
        "preloaded_image_dirs": preloaded_image_dirs,
        "token_summary_path": str(token_summary_path.resolve()),
        "experiment_summary_path": str(experiment_summary_path.resolve()),
        "token_summary": token_summary,
        "experiment_summary": experiment_summary,
    }
    manifest_path = write_json(manifest, run_root / "run_manifest.json")
    print(json.dumps({"run_manifest_path": str(manifest_path), **manifest}, indent=2))


if __name__ == "__main__":
    main()
