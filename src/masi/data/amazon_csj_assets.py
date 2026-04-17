"""Metadata and image helpers for the Amazon CSJ subset.

These utilities intentionally work on a *selected item subset* rather than the
entire raw category. The raw metadata file is too large to mirror casually on
disk in this workspace, so we stream only the records needed for the currently
selected MASI demo subset.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from masi.common.io import ensure_directory


METADATA_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl?download=true"
)


JsonRecord = dict[str, object]


def fetch_metadata_slice_for_items(
    *,
    item_ids: set[str],
    output_path: str | Path,
    remote_url: str = METADATA_URL,
) -> dict[str, JsonRecord]:
    """Stream the remote metadata file and cache records for selected items.

    The function stops as soon as it has found every requested `parent_asin`,
    which keeps network and disk usage bounded for local development.
    """

    cache_path = Path(output_path).expanduser().resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        cached: dict[str, JsonRecord] = {}
        with cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    cached[str(record.get("parent_asin", "")).strip()] = record
        if item_ids.issubset(set(cached)):
            return {item_id: cached[item_id] for item_id in item_ids}

    found: dict[str, JsonRecord] = {}
    process = subprocess.Popen(
        ["curl", "-L", "-s", remote_url],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            item_id = str(record.get("parent_asin", "")).strip()
            if item_id in item_ids and item_id not in found:
                found[item_id] = record
                if len(found) == len(item_ids):
                    break
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    with cache_path.open("w", encoding="utf-8") as handle:
        for item_id in sorted(found):
            handle.write(json.dumps(found[item_id], ensure_ascii=False) + "\n")

    return found


def load_metadata_slice_from_file(
    *,
    item_ids: set[str],
    metadata_path: str | Path,
) -> dict[str, JsonRecord]:
    """Load only the requested item records from a local metadata JSONL file."""

    source_path = Path(metadata_path).expanduser().resolve()
    found: dict[str, JsonRecord] = {}
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            item_id = str(record.get("parent_asin", "")).strip()
            if item_id in item_ids and item_id not in found:
                found[item_id] = record
                if len(found) == len(item_ids):
                    break
    return found


def select_primary_image_url(metadata_record: JsonRecord) -> str:
    """Choose the best available image URL from a metadata or review record."""

    images = metadata_record.get("images", [])
    if not isinstance(images, list):
        return ""

    for image in images:
        if not isinstance(image, dict):
            continue
        for key in ("hi_res", "large", "thumb", "large_image_url", "medium_image_url", "small_image_url"):
            value = str(image.get(key, "")).strip()
            if value:
                return value
    return ""


def download_item_images(
    *,
    metadata_by_item: dict[str, JsonRecord],
    image_cache_dir: str | Path,
) -> dict[str, Path]:
    """Download one product image per item and return the successful paths."""

    cache_dir = ensure_directory(image_cache_dir)
    image_paths: dict[str, Path] = {}

    for item_id, metadata in metadata_by_item.items():
        image_url = select_primary_image_url(metadata)
        if not image_url:
            continue

        destination = cache_dir / f"{item_id}.jpg"
        if not destination.exists():
            result = subprocess.run(
                ["curl", "-L", "--fail", "-s", "--max-time", "30", "-o", str(destination), image_url],
                check=False,
            )
            if result.returncode != 0:
                if destination.exists():
                    destination.unlink()
                continue

        try:
            with Image.open(destination) as image:
                image.verify()
        except (OSError, UnidentifiedImageError):
            if destination.exists():
                destination.unlink()
            continue

        image_paths[item_id] = destination

    return image_paths
