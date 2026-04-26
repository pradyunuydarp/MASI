#!/usr/bin/env python3
"""Download and validate item images for a prepared bounded Amazon CSJ subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from masi.common.io import ensure_directory, write_json
from masi.data.amazon_csj_assets import download_item_images_with_options
from masi.data.amazon_csj_subset import iter_jsonl


def parse_args() -> argparse.Namespace:
    """Parse local subset-image download arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-path", required=True, help="Path to the prepared subset metadata JSONL file.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Prepared subset root where `images/` and `image_download_manifest.json` will be written.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel download workers.")
    parser.add_argument("--retries", type=int, default=2, help="Number of retries for each item image.")
    parser.add_argument("--timeout-seconds", type=int, default=30, help="Curl timeout for each image download.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse valid existing images and resume partial `.part` downloads.",
    )
    return parser.parse_args()


def main() -> None:
    """Download one validated image per subset item."""

    args = parse_args()
    metadata_path = Path(args.metadata_path).expanduser().resolve()
    output_dir = ensure_directory(args.output_dir)
    image_output_dir = ensure_directory(output_dir / "images")

    metadata_by_item: dict[str, dict[str, object]] = {}
    for record in iter_jsonl(metadata_path):
        item_id = str(record.get("parent_asin", "")).strip()
        if item_id and item_id not in metadata_by_item:
            metadata_by_item[item_id] = record

    download_result = download_item_images_with_options(
        metadata_by_item=metadata_by_item,
        image_cache_dir=image_output_dir,
        workers=int(args.workers),
        retries=int(args.retries),
        timeout_seconds=int(args.timeout_seconds),
        resume=bool(args.resume),
    )

    manifest = {
        "source_metadata_path": str(metadata_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "image_cache_dir": str(image_output_dir.resolve()),
        "workers": int(args.workers),
        "retries": int(args.retries),
        "timeout_seconds": int(args.timeout_seconds),
        "resume": bool(args.resume),
        "selected_item_count": len(metadata_by_item),
        "successful_item_count": len(download_result.image_paths_by_item),
        "downloaded_item_count": len(download_result.downloaded_item_ids),
        "reused_item_count": len(download_result.skipped_existing_item_ids),
        "failed_item_count": len(download_result.failed_item_ids),
        "missing_url_item_count": len(download_result.missing_url_item_ids),
        "failed_item_ids": download_result.failed_item_ids,
        "missing_url_item_ids": download_result.missing_url_item_ids,
    }
    manifest_path = write_json(manifest, output_dir / "image_download_manifest.json")
    print(json.dumps({"image_download_manifest_path": str(manifest_path.resolve()), **manifest}, indent=2))


if __name__ == "__main__":
    main()
