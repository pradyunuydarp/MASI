#!/usr/bin/env python3
"""Download the Amazon Reviews 2023 Clothing/Shoes/Jewelry raw files.

The proposal targets the Amazon Reviews 2023 Clothing/Shoes/Jewelry subset.
The raw review file is very large, so this script defaults to downloading a
bounded prefix suitable for local development on machines with limited free
disk. Pass `--full-reviews` only if you have enough space for the complete raw
review file.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from masi.common.io import ensure_directory


REVIEW_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "raw/review_categories/Clothing_Shoes_and_Jewelry.jsonl?download=true"
)
METADATA_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl?download=true"
)


def parse_args() -> argparse.Namespace:
    """Parse downloader arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data/raw/amazon_reviews_2023",
        help="Directory where the raw JSONL files should be stored.",
    )
    parser.add_argument(
        "--full-reviews",
        action="store_true",
        help="Download the full raw reviews file instead of a bounded prefix.",
    )
    parser.add_argument(
        "--review-range-end",
        type=int,
        default=(2 * 1024 * 1024 * 1024) - 1,
        help="Inclusive byte offset used when downloading a bounded review prefix.",
    )
    parser.add_argument(
        "--download-metadata",
        action="store_true",
        help="Download the raw metadata file as well.",
    )
    parser.add_argument(
        "--skip-reviews",
        action="store_true",
        help="Skip the reviews download and only fetch metadata when requested.",
    )
    return parser.parse_args()


def download(url: str, destination: Path, *, range_end: int | None = None) -> None:
    """Download one file with optional byte-range bounding via `curl`."""

    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if range_end is None:
        command = ["curl", "-L", "--continue-at", "-", "--output", str(destination), url]
        subprocess.run(command, check=True)
        return

    complete_size = int(range_end) + 1
    existing_size = destination.stat().st_size if destination.exists() else 0
    if existing_size >= complete_size:
        return

    if existing_size == 0:
        command = ["curl", "-L", "--output", str(destination), "--range", f"0-{range_end}", url]
        subprocess.run(command, check=True)
        return

    partial_destination = destination.with_suffix(destination.suffix + ".part")
    if partial_destination.exists():
        partial_destination.unlink()

    command = [
        "curl",
        "-L",
        "--output",
        str(partial_destination),
        "--range",
        f"{existing_size}-{range_end}",
        url,
    ]
    subprocess.run(command, check=True)
    with partial_destination.open("rb") as source_handle, destination.open("ab") as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle)
    partial_destination.unlink()


def main() -> None:
    """Download the proposal-aligned Amazon CSJ raw files."""

    args = parse_args()
    output_dir = ensure_directory(args.output_dir)

    review_path = output_dir / "Clothing_Shoes_and_Jewelry.jsonl"
    metadata_path = output_dir / "meta_Clothing_Shoes_and_Jewelry.jsonl"

    if not args.skip_reviews:
        download(
            REVIEW_URL,
            review_path,
            range_end=None if args.full_reviews else int(args.review_range_end),
        )
    if args.download_metadata:
        download(METADATA_URL, metadata_path)
    if not args.skip_reviews:
        print(f"Downloaded reviews to {review_path}")
    if args.download_metadata:
        print(f"Downloaded metadata to {metadata_path}")


if __name__ == "__main__":
    main()
