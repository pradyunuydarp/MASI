"""Real Amazon Reviews 2023 sequence builders for the recommender stack."""

from __future__ import annotations

from dataclasses import dataclass

from masi.data.amazon_csj_subset import AmazonSubset, JsonRecord, select_real_amazon_subset
from masi.recommender.vocabulary import FusedSemanticId


@dataclass(slots=True)
class AmazonSequenceBuildResult:
    """Container for imported recommender inputs built from raw Amazon data."""

    fused_ids: list[FusedSemanticId]
    user_histories: dict[str, list[str]]
    summary: dict[str, object]


def _normalize_token(value: str) -> str:
    """Normalize free text into a compact token fragment."""

    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in value.strip())
    collapsed = "_".join(part for part in cleaned.split("_") if part)
    return collapsed or "unknown"


def _extract_text_codes(record: JsonRecord) -> list[str]:
    """Create deterministic text-side proxy tokens from imported fields.

    These are *not* the final MASI semantic IDs. They are placeholders that
    allow the recommender stack to run on real imported items before Phase 1 and
    Phase 2 are implemented.
    """

    title = str(record.get("title", "")).strip()
    text = str(record.get("text", "")).strip()
    categories = record.get("categories", [])
    store = str(record.get("store", "")).strip()

    title_tokens = [_normalize_token(token) for token in title.split()[:2] if token.strip()]
    text_tokens = [_normalize_token(token) for token in text.split()[:2] if token.strip()]
    category_tokens = []
    if isinstance(categories, list):
        category_tokens = [_normalize_token(str(token)) for token in categories[:2]]

    codes = [f"txt_{token}" for token in [*title_tokens, *text_tokens, *category_tokens] if token]
    if store:
        codes.append(f"txt_store_{_normalize_token(store)}")
    return codes[:4] or ["txt_unknown"]


def _extract_visual_proxy_codes(record: JsonRecord) -> list[str]:
    """Create deterministic visual-side proxy tokens from imported image fields.

    We do not have CLIP vision embeddings or vision RQ-VAE outputs yet. These
    proxy codes preserve a separate visual-token channel using image availability
    and asset structure from the real imported file.
    """

    images = record.get("images", [])
    details = record.get("details", {})

    image_count = len(images) if isinstance(images, list) else 0
    hi_res_count = 0
    variants: list[str] = []
    if isinstance(images, list):
        for image in images[:3]:
            if isinstance(image, dict):
                if str(image.get("hi_res", "")).strip() or str(image.get("large_image_url", "")).strip():
                    hi_res_count += 1
                variant = str(image.get("variant", "") or image.get("attachment_type", "")).strip()
                if variant:
                    variants.append(_normalize_token(variant))

    department = ""
    if isinstance(details, dict):
        department = str(details.get("Department", "")).strip()

    codes = [
        f"vis_imgcount_{image_count}",
        f"vis_hires_{hi_res_count}",
        *(f"vis_variant_{variant}" for variant in variants[:2]),
    ]
    if department:
        codes.append(f"vis_dept_{_normalize_token(department)}")
    return codes[:4] or ["vis_unknown"]


def build_fused_ids_from_records(records_by_item: dict[str, JsonRecord]) -> list[FusedSemanticId]:
    """Convert imported item records into placeholder fused semantic IDs."""

    fused_ids: list[FusedSemanticId] = []
    for item_id, record in records_by_item.items():
        fused_ids.append(
            FusedSemanticId(
                item_id=item_id,
                text_codes=_extract_text_codes(record),
                visual_codes=_extract_visual_proxy_codes(record),
            )
        )
    return fused_ids


def build_real_amazon_histories(
    *,
    reviews_path: str | Path,
    min_user_interactions: int,
    min_item_interactions: int | None = None,
    max_users: int,
    max_items: int,
    max_review_records: int | None,
    collapse_consecutive_duplicates: bool = False,
) -> AmazonSequenceBuildResult:
    """Build recommender-ready histories from the real Amazon CSJ subset.

    This is a thin wrapper over :func:`select_real_amazon_subset` that keeps the
    older recommender API intact while the repository transitions from
    placeholder review proxies to Phase 1/2-generated semantic IDs.
    """

    subset = select_real_amazon_subset(
        reviews_path=reviews_path,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
        max_users=max_users,
        max_items=max_items,
        max_review_records=max_review_records,
        collapse_consecutive_duplicates=collapse_consecutive_duplicates,
    )
    fused_ids = build_fused_ids_from_records(subset.item_records)
    return AmazonSequenceBuildResult(
        fused_ids=fused_ids,
        user_histories=subset.user_histories,
        summary=subset.summary,
    )
