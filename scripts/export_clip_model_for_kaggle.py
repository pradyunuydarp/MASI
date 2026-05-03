#!/usr/bin/env python3
"""Download CLIP into a standalone directory for Kaggle Dataset/Model reuse."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

from transformers import CLIPModel, CLIPProcessor


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model ID to export.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/kaggle_models/openai_clip-vit-base-patch32",
        help="Directory that will receive the standalone model files.",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Also create a .zip archive next to the output directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    """Download and save CLIP model/processor files."""

    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    started_at = time.time()
    processor = CLIPProcessor.from_pretrained(args.model_name, token=hf_token)
    model = CLIPModel.from_pretrained(
        args.model_name,
        token=hf_token,
        low_cpu_mem_usage=False,
        use_safetensors=True,
    )
    processor.save_pretrained(output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)

    manifest = {
        "model_name": args.model_name,
        "output_dir": str(output_dir),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "usage": {
            "environment_variable": f"MASI_CLIP_MODEL_DIR={output_dir}",
            "config_key": {"clip": {"local_model_path": str(output_dir)}},
        },
    }
    manifest_path = output_dir / "masi_clip_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    archive_path = None
    if args.archive:
        archive_path = shutil.make_archive(str(output_dir), "zip", root_dir=output_dir)
        manifest["archive_path"] = archive_path
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"manifest_path": str(manifest_path), **manifest}, indent=2))


if __name__ == "__main__":
    main()
