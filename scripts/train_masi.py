#!/usr/bin/env python3
"""Run the full MASI CSJ pipeline from one entrypoint.

This launcher is intended for the proposal's primary experiment setting:

1. Amazon Reviews 2023 Clothing/Shoes/Jewelry with iterative 5-core filtering,
2. frozen CLIP feature extraction,
3. Phase 1 behavior-aware alignment,
4. Phase 2 independent text/vision quantization with late fusion,
5. Phase 3 cross-modal MLM pretraining, sequential fine-tuning, and evaluation.

The script writes resolved stage configs, executes the phase scripts in order,
and stores a top-level run manifest so the same config can be used on Kaggle,
Colab, or a local lab machine.
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
from masi.common.runtime import detect_runtime_environment, resolve_path, resolve_storage_root


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/masi_train_csj_full.json",
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


def _ensure_raw_dataset(
    *,
    repo_root: Path,
    config: dict[str, object],
    reviews_path: Path,
    metadata_path: Path,
) -> dict[str, object]:
    """Download the raw CSJ files when the config requests it."""

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
            "Attach the configured raw dataset input or update `dataset.reviews_path`."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata file at {metadata_path}. "
            "Attach the configured raw dataset input or update `dataset.metadata_path`."
        )

    return {
        "downloaded_reviews": downloaded_reviews,
        "downloaded_metadata": downloaded_metadata,
        "reviews_path": str(reviews_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
    }


def main() -> None:
    """Run the full MASI training pipeline."""

    args = parse_args()
    loaded = load_json_config(args.config)
    config = loaded.data
    repo_root = find_repo_root(loaded.path)
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

    reviews_path = resolve_path(storage_root, str(dataset_config["reviews_path"]))
    metadata_path = resolve_path(storage_root, str(dataset_config["metadata_path"]))
    assert reviews_path is not None
    assert metadata_path is not None

    data_setup = _ensure_raw_dataset(
        repo_root=repo_root,
        config=config,
        reviews_path=reviews_path,
        metadata_path=metadata_path,
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
        "token_summary_path": str(token_summary_path.resolve()),
        "experiment_summary_path": str(experiment_summary_path.resolve()),
        "token_summary": token_summary,
        "experiment_summary": experiment_summary,
    }
    manifest_path = write_json(manifest, run_root / "run_manifest.json")
    print(json.dumps({"run_manifest_path": str(manifest_path), **manifest}, indent=2))


if __name__ == "__main__":
    main()
