"""Metadata and image helpers for the Amazon CSJ subset.

These utilities intentionally work on a *selected item subset* rather than the
entire raw category. The raw metadata file is too large to mirror casually on
disk in this workspace, so we stream only the records needed for the currently
selected MASI demo subset.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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


@dataclass(slots=True)
class MetadataResolutionResult:
    """Metadata records resolved for a bounded item subset."""

    metadata_by_item: dict[str, JsonRecord]
    metadata_source: str
    metadata_cache_path: Path | None


@dataclass(slots=True)
class ImageDownloadResult:
    """Outcome summary for one bounded image-download batch."""

    image_paths_by_item: dict[str, Path]
    downloaded_item_ids: list[str]
    skipped_existing_item_ids: list[str]
    failed_item_ids: list[str]
    missing_url_item_ids: list[str]


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


def write_metadata_slice(
    *,
    metadata_by_item: dict[str, JsonRecord],
    output_path: str | Path,
) -> Path:
    """Persist a deterministic metadata slice for the selected item subset."""

    cache_path = Path(output_path).expanduser().resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        for item_id in sorted(metadata_by_item):
            handle.write(json.dumps(metadata_by_item[item_id], ensure_ascii=False) + "\n")
    return cache_path


def resolve_metadata_records_for_items(
    *,
    item_ids: set[str],
    metadata_local_path: str | Path | None,
    metadata_cache_path: str | Path | None,
    use_remote_metadata: bool,
    remote_url: str = METADATA_URL,
) -> MetadataResolutionResult:
    """Resolve metadata while preferring a cached local slice before the raw file."""

    resolved_cache_path = Path(metadata_cache_path).expanduser().resolve() if metadata_cache_path else None
    resolved_local_path = Path(metadata_local_path).expanduser().resolve() if metadata_local_path else None

    metadata_by_item: dict[str, JsonRecord] = {}
    source_steps: list[str] = []

    if resolved_cache_path and resolved_cache_path.exists():
        cached_records = load_metadata_slice_from_file(
            item_ids=item_ids,
            metadata_path=resolved_cache_path,
        )
        if cached_records:
            metadata_by_item.update(cached_records)
            source_steps.append("cached_metadata_slice")

    if resolved_local_path and resolved_local_path.exists():
        missing_item_ids = item_ids.difference(metadata_by_item)
        if missing_item_ids:
            local_records = load_metadata_slice_from_file(
                item_ids=missing_item_ids,
                metadata_path=resolved_local_path,
            )
            if local_records:
                metadata_by_item.update(local_records)
                source_steps.append("local_metadata_file")

    if use_remote_metadata:
        missing_item_ids = item_ids.difference(metadata_by_item)
        if missing_item_ids:
            cache_target = resolved_cache_path or Path("metadata.slice.jsonl").expanduser().resolve()
            remote_records = fetch_metadata_slice_for_items(
                item_ids=missing_item_ids,
                output_path=cache_target,
                remote_url=remote_url,
            )
            if remote_records:
                metadata_by_item.update(remote_records)
                source_steps.append("remote_metadata_stream")
                resolved_cache_path = cache_target

    if resolved_cache_path and metadata_by_item:
        write_metadata_slice(
            metadata_by_item=metadata_by_item,
            output_path=resolved_cache_path,
        )

    metadata_source = "+".join(source_steps) if source_steps else "review_side_images_and_text"
    return MetadataResolutionResult(
        metadata_by_item=metadata_by_item,
        metadata_source=metadata_source,
        metadata_cache_path=resolved_cache_path,
    )


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


def _validate_image(path: Path) -> bool:
    """Return whether a cached file is a readable image."""

    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError):
        return False
    return True


def _download_single_item_image(
    *,
    item_id: str,
    metadata: JsonRecord,
    cache_dir: Path,
    retries: int,
    timeout_seconds: int,
    resume: bool,
) -> tuple[str, str, Path | None]:
    """Download one image file and return its terminal status."""

    image_url = select_primary_image_url(metadata)
    if not image_url:
        return "missing_url", item_id, None

    destination = cache_dir / f"{item_id}.jpg"
    partial_destination = cache_dir / f"{item_id}.jpg.part"

    if resume and destination.exists():
        if _validate_image(destination):
            return "skipped_existing", item_id, destination
        destination.unlink()

    if destination.exists():
        destination.unlink()
    if not resume and partial_destination.exists():
        partial_destination.unlink()

    attempts = max(1, int(retries) + 1)
    for _ in range(attempts):
        command = ["curl", "-L", "--fail", "-s", "--max-time", str(int(timeout_seconds))]
        if resume and partial_destination.exists():
            command.extend(["--continue-at", "-"])
        command.extend(["-o", str(partial_destination), image_url])
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            continue
        if _validate_image(partial_destination):
            if destination.exists():
                destination.unlink()
            partial_destination.replace(destination)
            return "downloaded", item_id, destination
        if partial_destination.exists():
            partial_destination.unlink()

    if destination.exists() and not _validate_image(destination):
        destination.unlink()
    return "failed", item_id, None


def download_item_images(
    *,
    metadata_by_item: dict[str, JsonRecord],
    image_cache_dir: str | Path,
) -> ImageDownloadResult:
    """Download one product image per item and return a resumable batch summary."""

    return download_item_images_with_options(
        metadata_by_item=metadata_by_item,
        image_cache_dir=image_cache_dir,
        workers=1,
        retries=0,
        timeout_seconds=30,
        resume=True,
    )


def download_item_images_with_options(
    *,
    metadata_by_item: dict[str, JsonRecord],
    image_cache_dir: str | Path,
    workers: int,
    retries: int,
    timeout_seconds: int,
    resume: bool,
) -> ImageDownloadResult:
    """Download one verified product image per item with threaded workers."""

    cache_dir = ensure_directory(image_cache_dir)
    image_paths: dict[str, Path] = {}
    downloaded_item_ids: list[str] = []
    skipped_existing_item_ids: list[str] = []
    failed_item_ids: list[str] = []
    missing_url_item_ids: list[str] = []

    items = sorted(metadata_by_item.items())
    max_workers = max(1, int(workers))

    def consume(result: tuple[str, str, Path | None]) -> None:
        status, item_id, path = result
        if status == "downloaded":
            assert path is not None
            image_paths[item_id] = path
            downloaded_item_ids.append(item_id)
            return
        if status == "skipped_existing":
            assert path is not None
            image_paths[item_id] = path
            skipped_existing_item_ids.append(item_id)
            return
        if status == "missing_url":
            missing_url_item_ids.append(item_id)
            return
        failed_item_ids.append(item_id)

    if max_workers == 1:
        for item_id, metadata in items:
            consume(
                _download_single_item_image(
                    item_id=item_id,
                    metadata=metadata,
                    cache_dir=cache_dir,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                    resume=resume,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _download_single_item_image,
                    item_id=item_id,
                    metadata=metadata,
                    cache_dir=cache_dir,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                    resume=resume,
                )
                for item_id, metadata in items
            ]
            for future in as_completed(futures):
                consume(future.result())

    return ImageDownloadResult(
        image_paths_by_item=image_paths,
        downloaded_item_ids=sorted(downloaded_item_ids),
        skipped_existing_item_ids=sorted(skipped_existing_item_ids),
        failed_item_ids=sorted(failed_item_ids),
        missing_url_item_ids=sorted(missing_url_item_ids),
    )
