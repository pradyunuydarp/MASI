#!/usr/bin/env python3
"""Build a deterministic dataset manifest for the MASI preprocessing stage.

This script is intentionally narrow in scope: it validates raw inputs and
produces summary artifacts that document exactly what was available before
feature extraction or graph construction begins.
"""

from __future__ import annotations

import argparse
import json

from masi.common.config import load_json_config
from masi.common.io import ensure_directory, write_json
from masi.data.amazon2023 import prepare_dataset
from masi.data.contracts import PipelineConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for manifest generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/data_prep_demo.json",
        help="Path to the preprocessing JSON config.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the manifest generation workflow."""

    args = parse_args()
    loaded_config = load_json_config(args.config)
    repo_root = loaded_config.path.parent.parent
    pipeline_config = PipelineConfig.from_dict(loaded_config.data, base_dir=repo_root)

    filtered_metadata, filtered_interactions, summary = prepare_dataset(pipeline_config)

    processed_root = ensure_directory(pipeline_config.paths.processed_root)
    outputs_root = ensure_directory(pipeline_config.paths.outputs_root)

    metadata_path = processed_root / "metadata.filtered.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in filtered_metadata:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    interactions_path = processed_root / "interactions.filtered.jsonl"
    with interactions_path.open("w", encoding="utf-8") as handle:
        for record in filtered_interactions:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest_payload = {
        "config_path": str(loaded_config.path),
        "processed_root": str(processed_root),
        "outputs_root": str(outputs_root),
        "summary": summary.to_dict(),
    }
    manifest_path = write_json(manifest_payload, outputs_root / "dataset_manifest.json")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
