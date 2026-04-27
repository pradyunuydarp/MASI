#!/usr/bin/env python3
"""Prepare a deterministic bounded Amazon CSJ subset for local or Kaggle use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from masi.common.io import ensure_directory, write_json
from masi.data.amazon_csj_assets import resolve_metadata_records_for_items, write_metadata_slice
from masi.data.amazon_csj_subset import select_real_amazon_subset


PRESET_LIMITS = {
    "smoke": {
        "max_review_records": 50000,
        "max_users": 64,
        "max_items": 128,
    },
    "medium": {
        "max_review_records": 150000,
        "max_users": 256,
        "max_items": 512,
    },
    "full_dataset": {
        "max_review_records": 60000000,
        "max_users": 102400,
        "max_items": 204800,
    },
    "large": {
        "max_review_records": 400000,
        "max_users": 512,
        "max_items": 1024,
    },
}


def parse_args() -> argparse.Namespace:
    """Parse local subset-preparation arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews-path", required=True, help="Path to the local raw CSJ reviews JSONL file.")
    parser.add_argument("--metadata-path", required=True, help="Path to the local raw CSJ metadata JSONL file.")
    parser.add_argument("--output-dir", required=True, help="Directory where the prepared subset dataset will be written.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_LIMITS),
        default="large",
        help="Bounded subset preset to use before applying explicit overrides.",
    )
    parser.add_argument("--max-review-records", type=int, default=None, help="Override the preset review scan limit.")
    parser.add_argument("--max-users", type=int, default=None, help="Override the preset maximum user count.")
    parser.add_argument("--max-items", type=int, default=None, help="Override the preset maximum item count.")
    parser.add_argument(
        "--min-user-interactions",
        type=int,
        default=5,
        help="Minimum number of interactions per user during iterative k-core filtering.",
    )
    parser.add_argument(
        "--min-item-interactions",
        type=int,
        default=5,
        help="Minimum number of interactions per item during iterative k-core filtering.",
    )
    parser.add_argument(
        "--collapse-consecutive-duplicates",
        action="store_true",
        help="Collapse consecutive repeated item IDs when building user histories.",
    )
    return parser.parse_args()


def _resolve_limit(args: argparse.Namespace, name: str) -> int:
    """Resolve one bounded limit from the chosen preset and CLI overrides."""

    preset_value = int(PRESET_LIMITS[str(args.preset)][name])
    override_value = getattr(args, name)
    return preset_value if override_value is None else int(override_value)


def main() -> None:
    """Materialize a prepared bounded CSJ subset dataset."""

    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    reviews_output_path = output_dir / "Clothing_Shoes_and_Jewelry.jsonl"
    metadata_output_path = output_dir / "meta_Clothing_Shoes_and_Jewelry.jsonl"

    max_review_records = _resolve_limit(args, "max_review_records")
    max_users = _resolve_limit(args, "max_users")
    max_items = _resolve_limit(args, "max_items")

    subset = select_real_amazon_subset(
        reviews_path=args.reviews_path,
        min_user_interactions=int(args.min_user_interactions),
        min_item_interactions=int(args.min_item_interactions),
        max_users=max_users,
        max_items=max_items,
        max_review_records=max_review_records,
        collapse_consecutive_duplicates=bool(args.collapse_consecutive_duplicates),
    )

    metadata_result = resolve_metadata_records_for_items(
        item_ids=set(subset.item_records),
        metadata_local_path=args.metadata_path,
        metadata_cache_path=metadata_output_path,
        use_remote_metadata=False,
    )
    metadata_records = dict(subset.item_records)
    if metadata_result.metadata_by_item:
        metadata_records.update(metadata_result.metadata_by_item)

    with reviews_output_path.open("w", encoding="utf-8") as handle:
        for record in subset.interaction_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_metadata_slice(
        metadata_by_item=metadata_records,
        output_path=metadata_output_path,
    )

    selected_item_ids = sorted(metadata_records)
    manifest = {
        "preset": str(args.preset),
        "source_reviews_path": str(Path(args.reviews_path).expanduser().resolve()),
        "source_metadata_path": str(Path(args.metadata_path).expanduser().resolve()),
        "output_dir": str(output_dir.resolve()),
        "output_paths": {
            "reviews": str(reviews_output_path.resolve()),
            "metadata": str(metadata_output_path.resolve()),
        },
        "limits": {
            "max_review_records": max_review_records,
            "max_users": max_users,
            "max_items": max_items,
            "min_user_interactions": int(args.min_user_interactions),
            "min_item_interactions": int(args.min_item_interactions),
            "collapse_consecutive_duplicates": bool(args.collapse_consecutive_duplicates),
        },
        "subset_summary": subset.summary,
        "metadata_source": metadata_result.metadata_source,
        "selected_user_count": len(subset.user_histories),
        "selected_item_count": len(selected_item_ids),
        "selected_interaction_count": len(subset.interaction_records),
        "metadata_records_loaded_from_local_file": len(metadata_result.metadata_by_item),
        "metadata_records_filled_from_review_side": len(set(subset.item_records).difference(metadata_result.metadata_by_item)),
        "selected_item_ids": selected_item_ids,
    }
    manifest_path = write_json(manifest, output_dir / "subset_manifest.json")
    print(json.dumps({"subset_manifest_path": str(manifest_path.resolve()), **manifest}, indent=2))


if __name__ == "__main__":
    main()
