"""Real Amazon Reviews 2023 sequence builders for the recommender stack.

This module replaces the earlier hardcoded recommender demo histories with a
streaming importer over the proposal's target domain:

- reviews from `Clothing_Shoes_and_Jewelry.jsonl`
The implementation deliberately avoids loading the full raw category into
memory. Instead, it scans bounded numbers of review lines, constructs user
histories from `parent_asin` interactions, and derives placeholder item tokens
from real review records. If a metadata file is available later, the same
module can consume it as a richer source of item attributes.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from masi.data.amazon2023 import run_iterative_k_core
from masi.recommender.vocabulary import FusedSemanticId


JsonRecord = dict[str, object]


@dataclass(slots=True)
class AmazonSubset:
    """Selected review-derived subset used by downstream MASI stages."""

    item_records: dict[str, JsonRecord]
    user_histories: dict[str, list[str]]
    summary: dict[str, object]


@dataclass(slots=True)
class AmazonSequenceBuildResult:
    """Container for imported recommender inputs built from raw Amazon data."""

    fused_ids: list[FusedSemanticId]
    user_histories: dict[str, list[str]]
    summary: dict[str, object]


def iter_jsonl(path: str | Path, *, limit: int | None = None) -> Iterator[JsonRecord]:
    """Yield JSON objects from a JSONL file with an optional line limit.

    The locally cached review file may be intentionally partial when disk space
    is constrained. In that case the final line can be truncated; we skip that
    last malformed record rather than failing the whole import.
    """

    file_path = Path(path).expanduser().resolve()
    with file_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


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


def select_real_amazon_subset(
    *,
    reviews_path: str | Path,
    min_user_interactions: int,
    min_item_interactions: int | None = None,
    max_users: int,
    max_items: int,
    max_review_records: int | None,
    collapse_consecutive_duplicates: bool = False,
) -> AmazonSubset:
    """Select a bounded real-data subset from Amazon CSJ reviews.

    The procedure is:

    1. stream review records and collect chronological `parent_asin` histories,
    2. keep users with at least `min_user_interactions`,
    3. keep the most active users up to `max_users`,
    4. collect representative review-side records for the referenced items.
    """

    min_item_interactions = (
        min_user_interactions if min_item_interactions is None
        else int(min_item_interactions)
    )
    review_record_limit = (
        None if max_review_records is None or int(max_review_records) <= 0
        else int(max_review_records)
    )
    user_limit = None if int(max_users) <= 0 else int(max_users)
    item_limit = None if int(max_items) <= 0 else int(max_items)

    raw_interactions: list[JsonRecord] = []
    review_summary_by_item: dict[str, JsonRecord] = {}
    review_records_scanned = 0

    for review in iter_jsonl(reviews_path, limit=review_record_limit):
        review_records_scanned += 1
        user_id = str(review.get("user_id", "")).strip()
        parent_asin = str(review.get("parent_asin", "")).strip()
        timestamp = int(review.get("timestamp", 0) or 0)
        if not user_id or not parent_asin or timestamp <= 0:
            continue
        raw_interactions.append(
            {
                "user_id": user_id,
                "parent_asin": parent_asin,
                "timestamp": timestamp,
            }
        )
        if parent_asin not in review_summary_by_item:
            review_summary_by_item[parent_asin] = {
                "title": review.get("title", ""),
                "text": review.get("text", ""),
                "images": review.get("images", []),
                "details": {},
            }

    filtered_interactions = run_iterative_k_core(
        raw_interactions,
        user_col="user_id",
        item_col="parent_asin",
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )

    if user_limit is not None:
        user_counts = Counter(str(record["user_id"]) for record in filtered_interactions)
        keep_users = {
            user_id
            for user_id, _ in sorted(
                user_counts.items(),
                key=lambda pair: (-pair[1], pair[0]),
            )[:user_limit]
        }
        filtered_interactions = [
            record for record in filtered_interactions
            if str(record["user_id"]) in keep_users
        ]

    if item_limit is not None:
        item_counts = Counter(str(record["parent_asin"]) for record in filtered_interactions)
        keep_items = {
            item_id
            for item_id, _ in sorted(
                item_counts.items(),
                key=lambda pair: (-pair[1], pair[0]),
            )[:item_limit]
        }
        filtered_interactions = [
            record for record in filtered_interactions
            if str(record["parent_asin"]) in keep_items
        ]

    if user_limit is not None or item_limit is not None:
        filtered_interactions = run_iterative_k_core(
            filtered_interactions,
            user_col="user_id",
            item_col="parent_asin",
            min_user_interactions=min_user_interactions,
            min_item_interactions=min_item_interactions,
        )

    interactions_by_user: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for record in filtered_interactions:
        interactions_by_user[str(record["user_id"])].append(
            (int(record["timestamp"]), str(record["parent_asin"]))
        )

    filtered_histories: dict[str, list[str]] = {}
    for user_id, interactions in interactions_by_user.items():
        ordered_items = [item_id for _, item_id in sorted(interactions, key=lambda pair: pair[0])]

        if collapse_consecutive_duplicates:
            collapsed: list[str] = []
            for item_id in ordered_items:
                if not collapsed or collapsed[-1] != item_id:
                    collapsed.append(item_id)
            ordered_items = collapsed

        if len(ordered_items) >= min_user_interactions:
            filtered_histories[user_id] = ordered_items

    available_item_ids = set(review_summary_by_item)
    final_histories: dict[str, list[str]] = {}
    for user_id, history in filtered_histories.items():
        available_history = [item_id for item_id in history if item_id in available_item_ids]
        if len(available_history) >= min_user_interactions:
            final_histories[user_id] = available_history

    used_item_ids = set(item_id for history in final_histories.values() for item_id in history)
    final_records = {item_id: review_summary_by_item[item_id] for item_id in used_item_ids}
    summary = {
        "review_records_scanned": review_records_scanned,
        "interaction_rows_after_5core": len(filtered_interactions),
        "num_users": len(final_histories),
        "num_items": len(final_records),
        "min_user_interactions": min_user_interactions,
        "min_item_interactions": min_item_interactions,
        "collapse_consecutive_duplicates": collapse_consecutive_duplicates,
        "max_users": user_limit,
        "max_items": item_limit,
        "max_review_records": review_record_limit,
        "reviews_path": str(Path(reviews_path).expanduser().resolve()),
        "source_mode": "real_amazon_reviews_only",
        "five_core_applied": True,
    }
    return AmazonSubset(
        item_records=final_records,
        user_histories=final_histories,
        summary=summary,
    )
