"""Token-building helpers for the MASI Phase 1 -> Phase 2 pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from masi.common.progress import make_progress_bar
from masi.common.runtime import select_torch_device
from masi.recommender.vocabulary import FusedSemanticId


def select_device(runtime_config: dict[str, object] | None = None) -> torch.device:
    """Backward-compatible wrapper around the central MASI device selector."""

    return select_torch_device(runtime_config)


def build_text_input(metadata_record: dict[str, object]) -> str:
    """Create a proposal-aligned text description for CLIP encoding."""

    title = str(metadata_record.get("title", "")).strip()
    review_text = str(metadata_record.get("text", "")).strip()
    description = metadata_record.get("description", [])
    features = metadata_record.get("features", [])
    categories = metadata_record.get("categories", [])
    store = str(metadata_record.get("store", "")).strip()
    brand = str(metadata_record.get("brand", "")).strip()
    details = metadata_record.get("details", {})

    parts = [title]
    if review_text:
        parts.append(review_text)
    if isinstance(description, list):
        parts.extend(str(fragment).strip() for fragment in description[:3] if str(fragment).strip())
    if isinstance(features, list):
        parts.extend(str(feature).strip() for feature in features[:4] if str(feature).strip())
    if isinstance(categories, list):
        parts.extend(str(category).strip() for category in categories[:3] if str(category).strip())
    if isinstance(details, dict):
        for key in ("Brand", "Department", "Material", "Color"):
            value = str(details.get(key, "")).strip()
            if value:
                parts.append(f"{key}: {value}")
    if brand:
        parts.append(f"Brand: {brand}")
    if store:
        parts.append(f"Store: {store}")
    # We intentionally keep the text recipe deterministic so that Phase 1
    # embeddings can be reproduced exactly from the same bounded subset.
    return " | ".join(part for part in parts if part)


def _unwrap_clip_output(output: object) -> torch.Tensor:
    """Extract the embedding tensor from CLIP helper outputs across versions."""

    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "text_embeds") and isinstance(output.text_embeds, torch.Tensor):
        return output.text_embeds
    if hasattr(output, "image_embeds") and isinstance(output.image_embeds, torch.Tensor):
        return output.image_embeds
    if hasattr(output, "pooler_output") and isinstance(output.pooler_output, torch.Tensor):
        return output.pooler_output
    raise TypeError(f"Unsupported CLIP output type: {type(output)}")


def _resolve_clip_model_source(model_name: str) -> str:
    """Prefer a locally attached CLIP directory when one is configured."""

    configured = (
        os.environ.get("MASI_CLIP_MODEL_DIR")
        or os.environ.get("MASI_CLIP_MODEL_PATH")
        or model_name
    )
    candidate = Path(configured).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return configured


def _move_processor_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move a processor output batch to the selected device efficiently."""

    non_blocking = device.type == "cuda"
    return {
        key: value.to(device=device, non_blocking=non_blocking)
        for key, value in batch.items()
    }


def encode_clip_embeddings(
    *,
    metadata_by_item: dict[str, dict[str, object]],
    image_paths_by_item: dict[str, Path],
    model_name: str,
    batch_size: int,
    device: torch.device,
    use_text_modality: bool = True,
    use_visual_modality: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Encode text and image embeddings for items with full modality coverage."""

    if not use_text_modality and not use_visual_modality:
        raise ValueError("At least one modality must be enabled for CLIP encoding.")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    model_source = _resolve_clip_model_source(model_name)
    model = CLIPModel.from_pretrained(
        model_source,
        token=hf_token,
        low_cpu_mem_usage=False,
        use_safetensors=True,
    ).to(device)
    processor = CLIPProcessor.from_pretrained(model_source, token=hf_token)
    model.eval()

    valid_item_ids = [
        item_id for item_id in sorted(metadata_by_item)
        if (
            (not use_text_modality or build_text_input(metadata_by_item[item_id]))
            and (not use_visual_modality or item_id in image_paths_by_item)
        )
    ]
    text_embeddings: dict[str, torch.Tensor] = {}
    image_embeddings: dict[str, torch.Tensor] = {}

    total_batches = (len(valid_item_ids) + batch_size - 1) // batch_size
    with torch.no_grad(), make_progress_bar(
        total=total_batches,
        desc="CLIP encode",
        unit="batch",
    ) as progress:
        for start in range(0, len(valid_item_ids), batch_size):
            batch_item_ids = valid_item_ids[start : start + batch_size]
            # Text and image batches are built in the same item order so their
            # outputs remain perfectly aligned when the projection heads are
            # trained in Phase 1.
            images = []
            batch_text_embeddings = None
            batch_image_embeddings = None
            if use_text_modality:
                text_inputs = [build_text_input(metadata_by_item[item_id]) for item_id in batch_item_ids]
                text_batch = processor(text=text_inputs, return_tensors="pt", padding=True, truncation=True)
                text_batch = _move_processor_batch_to_device(text_batch, device)
                batch_text_embeddings = _unwrap_clip_output(model.get_text_features(**text_batch))
                batch_text_embeddings = torch.nn.functional.normalize(batch_text_embeddings, dim=-1).cpu()
                del text_batch
            if use_visual_modality:
                images = [Image.open(image_paths_by_item[item_id]).convert("RGB") for item_id in batch_item_ids]
                image_batch = processor(images=images, return_tensors="pt")
                image_batch = _move_processor_batch_to_device(image_batch, device)
                batch_image_embeddings = _unwrap_clip_output(model.get_image_features(**image_batch))
                batch_image_embeddings = torch.nn.functional.normalize(batch_image_embeddings, dim=-1).cpu()
                del image_batch

            for index, item_id in enumerate(batch_item_ids):
                # Embeddings are cached in dictionaries keyed by item ID because
                # later stages work in item space, not fixed row-index space.
                if batch_text_embeddings is not None:
                    text_embeddings[item_id] = batch_text_embeddings[index]
                if batch_image_embeddings is not None:
                    image_embeddings[item_id] = batch_image_embeddings[index]
                if images:
                    images[index].close()
            images.clear()
            progress.set_postfix(
                {
                    "items": min(start + len(batch_item_ids), len(valid_item_ids)),
                    "total": len(valid_item_ids),
                }
            )
            progress.update(1)

    return text_embeddings, image_embeddings


def build_fused_ids_from_quantized_codes(
    *,
    item_ids: list[str],
    text_codes_by_item: dict[str, list[int]],
    image_codes_by_item: dict[str, list[int]],
    use_text_modality: bool = True,
    use_visual_modality: bool = True,
) -> list[FusedSemanticId]:
    """Late-fuse separate text and vision codebooks into MASI token sequences."""

    fused_ids: list[FusedSemanticId] = []
    for item_id in item_ids:
        if use_text_modality and item_id not in text_codes_by_item:
            continue
        if use_visual_modality and item_id not in image_codes_by_item:
            continue
        if not use_text_modality and not use_visual_modality:
            continue
        fused_ids.append(
            FusedSemanticId(
                item_id=item_id,
                text_codes=[
                    f"txt_c{level}_{code}" for level, code in enumerate(text_codes_by_item.get(item_id, []))
                ],
                visual_codes=[
                    f"vis_c{level}_{code}" for level, code in enumerate(image_codes_by_item.get(item_id, []))
                ],
            )
        )
    return fused_ids


def write_fused_ids(fused_ids: list[FusedSemanticId], output_path: str | Path) -> Path:
    """Persist fused semantic IDs as JSONL for the recommender stage."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for fused_id in fused_ids:
            handle.write(
                json.dumps(
                    {
                        "item_id": fused_id.item_id,
                        "text_codes": fused_id.text_codes,
                        "visual_codes": fused_id.visual_codes,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def load_fused_ids(path: str | Path) -> list[FusedSemanticId]:
    """Load fused semantic IDs from a JSONL artifact."""

    artifact_path = Path(path).expanduser().resolve()
    fused_ids: list[FusedSemanticId] = []
    with artifact_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            fused_ids.append(
                FusedSemanticId(
                    item_id=str(payload["item_id"]),
                    text_codes=list(payload["text_codes"]),
                    visual_codes=list(payload["visual_codes"]),
                )
            )
    return fused_ids
