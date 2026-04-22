#!/usr/bin/env python3
"""Pre-download and validate Amazon CSJ item images for a bounded MASI run."""

from __future__ import annotations

import argparse
import json

from masi.common.config import find_repo_root, load_json_config
from masi.common.io import ensure_directory, write_json
from masi.common.runtime import detect_runtime_environment, resolve_path, resolve_storage_root
from masi.data.amazon_csj_assets import (
    download_item_images_with_options,
    resolve_metadata_records_for_items,
)
from masi.recommender.amazon_data import select_real_amazon_subset


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for bounded image downloading."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/masi_train_csj_medium_kaggle.json",
        help="Path to the bounded training config used to choose the item subset.",
    )
    parser.add_argument(
        "--storage-root",
        default=None,
        help="Optional storage root for raw data, image caches, and outputs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override the configured image download worker count.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=None,
        help="Override the configured retry count for each image.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="Override the configured curl timeout for each image.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse valid cached images and resume partial `.part` downloads.",
    )
    return parser.parse_args()


def _optional_positive_int(value: object) -> int | None:
    """Convert JSON values into optional positive integer limits."""

    if value is None:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def main() -> None:
    """Materialize a metadata slice and image cache for one bounded MASI config."""

    args = parse_args()
    loaded = load_json_config(args.config)
    config = loaded.data
    repo_root = find_repo_root(loaded.path)
    runtime_config = dict(config.get("runtime", {}))
    dataset_config = dict(config["dataset"])
    assets_config = dict(config.get("assets", {}))

    environment = detect_runtime_environment()
    storage_root = resolve_storage_root(
        repo_root=repo_root,
        runtime_config=runtime_config,
        cli_storage_root=args.storage_root,
    )

    reviews_path = resolve_path(storage_root, str(dataset_config["reviews_path"]))
    metadata_path = resolve_path(storage_root, str(dataset_config["metadata_path"]))
    image_cache_dir = resolve_path(
        storage_root,
        str(assets_config.get("image_cache_dir", "data/processed/amazon_csj_images/images")),
    )
    metadata_cache_path = resolve_path(
        storage_root,
        str(assets_config.get("metadata_cache_path", "data/processed/amazon_csj_images/metadata.slice.jsonl")),
    )
    assert reviews_path is not None
    assert metadata_path is not None
    assert image_cache_dir is not None
    assert metadata_cache_path is not None

    if not reviews_path.exists():
        raise FileNotFoundError(
            f"Missing reviews file at {reviews_path}. "
            "Attach the configured raw dataset input or run scripts/download_amazon_csj_dataset.py first."
        )

    subset = select_real_amazon_subset(
        reviews_path=str(reviews_path),
        min_user_interactions=int(dataset_config["min_user_interactions"]),
        min_item_interactions=_optional_positive_int(dataset_config.get("min_item_interactions")),
        max_users=int(dataset_config.get("max_users", 0) or 0),
        max_items=int(dataset_config.get("max_items", 0) or 0),
        max_review_records=_optional_positive_int(dataset_config.get("max_review_records")),
        collapse_consecutive_duplicates=bool(dataset_config.get("collapse_consecutive_duplicates", False)),
    )

    metadata_result = resolve_metadata_records_for_items(
        item_ids=set(subset.item_records),
        metadata_local_path=metadata_path if metadata_path.exists() else None,
        metadata_cache_path=metadata_cache_path,
        use_remote_metadata=bool(assets_config.get("use_remote_metadata", False)),
    )

    modality_records = dict(subset.item_records)
    if metadata_result.metadata_by_item:
        modality_records.update(metadata_result.metadata_by_item)

    workers = int(args.workers if args.workers is not None else assets_config.get("image_download_workers", 1))
    retries = int(args.retries if args.retries is not None else assets_config.get("image_download_retries", 0))
    timeout_seconds = int(
        args.timeout_seconds
        if args.timeout_seconds is not None
        else assets_config.get("image_download_timeout_seconds", 30)
    )
    resume = bool(args.resume or assets_config.get("image_download_resume", True))

    download_result = download_item_images_with_options(
        metadata_by_item=modality_records,
        image_cache_dir=image_cache_dir,
        workers=workers,
        retries=retries,
        timeout_seconds=timeout_seconds,
        resume=resume,
    )

    run_name = str(runtime_config.get("run_name", "masi_train_csj"))
    run_root = ensure_directory(storage_root / "outputs" / run_name)
    manifest = {
        "environment": environment,
        "storage_root": str(storage_root.resolve()),
        "run_root": str(run_root.resolve()),
        "run_name": run_name,
        "subset_summary": subset.summary,
        "selected_item_count": len(subset.item_records),
        "selected_user_count": len(subset.user_histories),
        "metadata_records_found": len(metadata_result.metadata_by_item),
        "metadata_source": metadata_result.metadata_source,
        "metadata_cache_path": str(metadata_cache_path.resolve()),
        "image_cache_dir": str(image_cache_dir.resolve()),
        "workers": workers,
        "retries": retries,
        "timeout_seconds": timeout_seconds,
        "resume": resume,
        "successful_item_count": len(download_result.image_paths_by_item),
        "downloaded_item_count": len(download_result.downloaded_item_ids),
        "reused_item_count": len(download_result.skipped_existing_item_ids),
        "failed_item_count": len(download_result.failed_item_ids),
        "missing_url_item_count": len(download_result.missing_url_item_ids),
        "failed_item_ids": download_result.failed_item_ids,
        "missing_url_item_ids": download_result.missing_url_item_ids,
    }
    manifest_path = write_json(manifest, run_root / "image_download_manifest.json")
    print(json.dumps({"manifest_path": str(manifest_path.resolve()), **manifest}, indent=2))


if __name__ == "__main__":
    main()
