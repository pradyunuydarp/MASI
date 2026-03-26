#!/usr/bin/env python3
"""Create a small synthetic dataset and run the MASI preprocessing pipeline.

The real Amazon Reviews 2023 assets are too large to commit to the repository.
This script gives the project an immediately demo-able path that exercises the
same code paths the real dataset will use.
"""

from __future__ import annotations

import argparse
import json

from masi.common.config import load_json_config
from masi.common.io import write_json
from masi.data.amazon2023 import prepare_dataset
from masi.data.contracts import PipelineConfig


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the synthetic demo workflow."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/data_prep_demo.json",
        help="Path to the preprocessing JSON config.",
    )
    return parser.parse_args()


def build_demo_tables() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Construct a compact synthetic dataset with warm and cold-style items.

    The examples are intentionally small but structured enough to exercise:

    - text field concatenation,
    - image availability validation,
    - iterative k-core filtering,
    - manifest generation for downstream stages.
    """

    metadata = [
            {
                "item_id": "dress_001",
                "title": "Floral summer dress",
                "brand": "Aster",
                "category": "Dresses",
                "image_url": "https://example.com/dress_001.jpg",
            },
            {
                "item_id": "dress_002",
                "title": "Minimal black dress",
                "brand": "Aster",
                "category": "Dresses",
                "image_url": "https://example.com/dress_002.jpg",
            },
            {
                "item_id": "shoe_001",
                "title": "White leather sneakers",
                "brand": "Northwind",
                "category": "Shoes",
                "image_url": "https://example.com/shoe_001.jpg",
            },
            {
                "item_id": "bag_001",
                "title": "Structured tote bag",
                "brand": "Northwind",
                "category": "Accessories",
                "image_url": "",
            },
        ]

    interactions = [
            {"user_id": "u_001", "item_id": "dress_001", "rating": 5.0, "timestamp": 1704067200},
            {"user_id": "u_001", "item_id": "dress_002", "rating": 4.0, "timestamp": 1704153600},
            {"user_id": "u_002", "item_id": "dress_001", "rating": 4.0, "timestamp": 1704240000},
            {"user_id": "u_002", "item_id": "shoe_001", "rating": 5.0, "timestamp": 1704326400},
            {"user_id": "u_003", "item_id": "dress_002", "rating": 5.0, "timestamp": 1704412800},
            {"user_id": "u_003", "item_id": "shoe_001", "rating": 4.0, "timestamp": 1704499200},
            {"user_id": "u_004", "item_id": "bag_001", "rating": 3.0, "timestamp": 1704585600},
        ]
    return metadata, interactions


def main() -> None:
    """Persist demo data, run preprocessing, and write a summary artifact."""

    args = parse_args()
    loaded_config = load_json_config(args.config)
    repo_root = loaded_config.path.parent.parent
    pipeline_config = PipelineConfig.from_dict(loaded_config.data, base_dir=repo_root)

    metadata, interactions = build_demo_tables()
    pipeline_config.paths.raw_root.mkdir(parents=True, exist_ok=True)

    with pipeline_config.paths.metadata_file.open("w", encoding="utf-8") as handle:
        for record in metadata:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with pipeline_config.paths.reviews_file.open("w", encoding="utf-8") as handle:
        for record in interactions:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    _, _, summary = prepare_dataset(pipeline_config)
    summary_path = write_json(summary.to_dict(), pipeline_config.paths.outputs_root / "demo_summary.json")
    print(f"Wrote demo summary to {summary_path}")


if __name__ == "__main__":
    main()
