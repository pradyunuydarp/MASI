"""Shared bounded-subset selection utilities for Amazon CSJ.

These helpers keep the repository's local subset-preparation scripts and the
training pipeline on the same deterministic selection contract.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from masi.data.amazon2023 import run_iterative_k_core


JsonRecord = dict[str, object]


@dataclass(slots=True)
class AmazonSubset:
    """Selected review-derived subset used by downstream MASI stages."""

    item_records: dict[str, JsonRecord]
    user_histories: dict[str, list[str]]
    interaction_records: list[JsonRecord]
    summary: dict[str, object]


def iter_jsonl(path: str | Path, *, limit: int | None = None) -> Iterator[JsonRecord]:
    """Yield JSON objects from a JSONL file with an optional line limit."""

    file_path = Path(path).expanduser().resolve()
    with file_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


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
    """Select a bounded real-data subset from Amazon CSJ reviews."""

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
    review_records: list[JsonRecord] = []
    review_summary_by_item: dict[str, JsonRecord] = {}
    review_records_scanned = 0

    for review in iter_jsonl(reviews_path, limit=review_record_limit):
        review_records_scanned += 1
        user_id = str(review.get("user_id", "")).strip()
        parent_asin = str(review.get("parent_asin", "")).strip()
        timestamp = int(review.get("timestamp", 0) or 0)
        if not user_id or not parent_asin or timestamp <= 0:
            continue

        record_index = len(review_records)
        review_records.append(review)
        raw_interactions.append(
            {
                "user_id": user_id,
                "parent_asin": parent_asin,
                "timestamp": timestamp,
                "__record_index": record_index,
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

    used_item_ids = {item_id for history in final_histories.values() for item_id in history}
    final_records = {item_id: review_summary_by_item[item_id] for item_id in used_item_ids}
    selected_review_records = [
        review_records[int(record["__record_index"])]
        for record in filtered_interactions
        if str(record["user_id"]) in final_histories and str(record["parent_asin"]) in used_item_ids
    ]
    summary = {
        "review_records_scanned": review_records_scanned,
        "interaction_rows_after_5core": len(filtered_interactions),
        "selected_interaction_rows": len(selected_review_records),
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
        interaction_records=selected_review_records,
        summary=summary,
    )
