#!/usr/bin/env python3
"""Prepare a deterministic bounded Amazon CSJ subset for local or Kaggle use."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from masi.common.io import ensure_directory, write_json
from masi.data.amazon_csj_assets import resolve_metadata_records_for_items, write_metadata_slice
from masi.data.amazon_csj_subset import iter_jsonl, select_real_amazon_subset


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
        "max_review_records": 20000000,
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


def _execute_keep_prune(connection: sqlite3.Connection, query: str) -> int:
    """Run one SQL prune query and return how many rows were removed."""

    before = int(connection.total_changes)
    connection.execute(query)
    connection.commit()
    return int(connection.total_changes) - before


def _run_sqlite_k_core(
    *,
    connection: sqlite3.Connection,
    min_user_interactions: int,
    min_item_interactions: int,
) -> int:
    """Apply iterative k-core pruning to the disk-backed interaction table."""

    total_removed = 0
    while True:
        removed_users = _execute_keep_prune(
            connection,
            f"""
            WITH bad_users AS (
              SELECT user_id
              FROM interactions
              WHERE keep = 1
              GROUP BY user_id
              HAVING COUNT(*) < {int(min_user_interactions)}
            )
            UPDATE interactions
            SET keep = 0
            WHERE keep = 1 AND user_id IN (SELECT user_id FROM bad_users)
            """,
        )
        removed_items = _execute_keep_prune(
            connection,
            f"""
            WITH bad_items AS (
              SELECT parent_asin
              FROM interactions
              WHERE keep = 1
              GROUP BY parent_asin
              HAVING COUNT(*) < {int(min_item_interactions)}
            )
            UPDATE interactions
            SET keep = 0
            WHERE keep = 1 AND parent_asin IN (SELECT parent_asin FROM bad_items)
            """,
        )
        removed = removed_users + removed_items
        total_removed += removed
        if removed == 0:
            return total_removed


def _prepare_disk_backed_subset(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    reviews_output_path: Path,
    metadata_output_path: Path,
    max_review_records: int,
    max_users: int,
    max_items: int,
) -> dict[str, object]:
    """Prepare a large subset without keeping all raw reviews in memory."""

    sqlite_path = output_dir / "subset_selection.sqlite3"
    if sqlite_path.exists():
        sqlite_path.unlink()

    connection = sqlite3.connect(str(sqlite_path))
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA temp_store = FILE")
    connection.execute(
        """
        CREATE TABLE interactions (
          row_id INTEGER PRIMARY KEY,
          user_id TEXT NOT NULL,
          parent_asin TEXT NOT NULL,
          timestamp INTEGER NOT NULL,
          review_json TEXT NOT NULL,
          keep INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE item_summary (
          parent_asin TEXT PRIMARY KEY,
          summary_json TEXT NOT NULL
        )
        """
    )

    review_record_limit = None if int(max_review_records) <= 0 else int(max_review_records)
    review_records_scanned = 0
    valid_interaction_rows = 0
    insert_batch: list[tuple[str, str, int, str]] = []
    summary_batch: list[tuple[str, str]] = []

    print(
        "Building disk-backed interaction index: "
        f"limit={review_record_limit}, sqlite={sqlite_path}",
        flush=True,
    )
    for review in iter_jsonl(args.reviews_path, limit=review_record_limit):
        review_records_scanned += 1
        user_id = str(review.get("user_id", "")).strip()
        parent_asin = str(review.get("parent_asin", "")).strip()
        timestamp = int(review.get("timestamp", 0) or 0)
        if not user_id or not parent_asin or timestamp <= 0:
            continue

        valid_interaction_rows += 1
        review_json = json.dumps(review, ensure_ascii=False)
        insert_batch.append((user_id, parent_asin, timestamp, review_json))
        summary = {
            "title": review.get("title", ""),
            "text": review.get("text", ""),
            "images": review.get("images", []),
            "details": {},
        }
        summary_batch.append((parent_asin, json.dumps(summary, ensure_ascii=False)))

        if len(insert_batch) >= 50000:
            connection.executemany(
                "INSERT INTO interactions (user_id, parent_asin, timestamp, review_json) VALUES (?, ?, ?, ?)",
                insert_batch,
            )
            connection.executemany(
                "INSERT OR IGNORE INTO item_summary (parent_asin, summary_json) VALUES (?, ?)",
                summary_batch,
            )
            connection.commit()
            insert_batch.clear()
            summary_batch.clear()
            print(
                f"indexed scanned={review_records_scanned} valid={valid_interaction_rows}",
                flush=True,
            )

    if insert_batch:
        connection.executemany(
            "INSERT INTO interactions (user_id, parent_asin, timestamp, review_json) VALUES (?, ?, ?, ?)",
            insert_batch,
        )
        connection.executemany(
            "INSERT OR IGNORE INTO item_summary (parent_asin, summary_json) VALUES (?, ?)",
            summary_batch,
        )
        connection.commit()

    print("Creating selection indexes", flush=True)
    connection.execute("CREATE INDEX idx_interactions_keep_user ON interactions (keep, user_id)")
    connection.execute("CREATE INDEX idx_interactions_keep_item ON interactions (keep, parent_asin)")
    connection.execute("CREATE INDEX idx_interactions_keep_user_time ON interactions (keep, user_id, timestamp)")
    connection.commit()

    min_user_interactions = int(args.min_user_interactions)
    min_item_interactions = int(args.min_item_interactions)
    print("Applying disk-backed 5-core filter", flush=True)
    _run_sqlite_k_core(
        connection=connection,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )
    interaction_rows_after_5core = int(
        connection.execute("SELECT COUNT(*) FROM interactions WHERE keep = 1").fetchone()[0]
    )

    if int(max_users) > 0:
        print(f"Applying max user limit: {int(max_users)}", flush=True)
        connection.execute("DROP TABLE IF EXISTS keep_users")
        connection.execute(
            f"""
            CREATE TEMP TABLE keep_users AS
            SELECT user_id
            FROM interactions
            WHERE keep = 1
            GROUP BY user_id
            ORDER BY COUNT(*) DESC, user_id ASC
            LIMIT {int(max_users)}
            """
        )
        _execute_keep_prune(
            connection,
            "UPDATE interactions SET keep = 0 WHERE keep = 1 AND user_id NOT IN (SELECT user_id FROM keep_users)",
        )

    if int(max_items) > 0:
        print(f"Applying max item limit: {int(max_items)}", flush=True)
        connection.execute("DROP TABLE IF EXISTS keep_items")
        connection.execute(
            f"""
            CREATE TEMP TABLE keep_items AS
            SELECT parent_asin
            FROM interactions
            WHERE keep = 1
            GROUP BY parent_asin
            ORDER BY COUNT(*) DESC, parent_asin ASC
            LIMIT {int(max_items)}
            """
        )
        _execute_keep_prune(
            connection,
            "UPDATE interactions SET keep = 0 WHERE keep = 1 AND parent_asin NOT IN (SELECT parent_asin FROM keep_items)",
        )

    if int(max_users) > 0 or int(max_items) > 0:
        print("Re-applying 5-core after user/item caps", flush=True)
        _run_sqlite_k_core(
            connection=connection,
            min_user_interactions=min_user_interactions,
            min_item_interactions=min_item_interactions,
        )

    selected_user_count = int(connection.execute("SELECT COUNT(DISTINCT user_id) FROM interactions WHERE keep = 1").fetchone()[0])
    selected_item_count = int(connection.execute("SELECT COUNT(DISTINCT parent_asin) FROM interactions WHERE keep = 1").fetchone()[0])
    selected_interaction_count = int(connection.execute("SELECT COUNT(*) FROM interactions WHERE keep = 1").fetchone()[0])

    print(f"Writing selected reviews to {reviews_output_path}", flush=True)
    with reviews_output_path.open("w", encoding="utf-8") as handle:
        for (review_json,) in connection.execute(
            "SELECT review_json FROM interactions WHERE keep = 1 ORDER BY user_id, timestamp, row_id"
        ):
            handle.write(str(review_json) + "\n")

    selected_item_ids = [
        str(row[0])
        for row in connection.execute(
            "SELECT DISTINCT parent_asin FROM interactions WHERE keep = 1 ORDER BY parent_asin"
        )
    ]
    item_records = {
        str(parent_asin): json.loads(str(summary_json))
        for parent_asin, summary_json in connection.execute(
            """
            SELECT item_summary.parent_asin, item_summary.summary_json
            FROM item_summary
            INNER JOIN (
              SELECT DISTINCT parent_asin
              FROM interactions
              WHERE keep = 1
            ) selected_items
            ON item_summary.parent_asin = selected_items.parent_asin
            """
        )
    }

    print("Resolving selected metadata records", flush=True)
    metadata_result = resolve_metadata_records_for_items(
        item_ids=set(selected_item_ids),
        metadata_local_path=args.metadata_path,
        metadata_cache_path=metadata_output_path,
        use_remote_metadata=False,
    )
    metadata_records = dict(item_records)
    if metadata_result.metadata_by_item:
        metadata_records.update(metadata_result.metadata_by_item)
    write_metadata_slice(
        metadata_by_item=metadata_records,
        output_path=metadata_output_path,
    )

    summary = {
        "review_records_scanned": review_records_scanned,
        "valid_interaction_rows": valid_interaction_rows,
        "interaction_rows_after_5core": interaction_rows_after_5core,
        "selected_interaction_rows": selected_interaction_count,
        "num_users": selected_user_count,
        "num_items": selected_item_count,
        "min_user_interactions": min_user_interactions,
        "min_item_interactions": min_item_interactions,
        "collapse_consecutive_duplicates": bool(args.collapse_consecutive_duplicates),
        "max_users": int(max_users),
        "max_items": int(max_items),
        "max_review_records": review_record_limit,
        "reviews_path": str(Path(args.reviews_path).expanduser().resolve()),
        "source_mode": "real_amazon_reviews_only_disk_backed",
        "five_core_applied": True,
        "selection_sqlite_path": str(sqlite_path.resolve()),
    }
    connection.close()

    return {
        "subset_summary": summary,
        "metadata_source": metadata_result.metadata_source,
        "selected_user_count": selected_user_count,
        "selected_item_count": len(metadata_records),
        "selected_interaction_count": selected_interaction_count,
        "metadata_records_loaded_from_local_file": len(metadata_result.metadata_by_item),
        "metadata_records_filled_from_review_side": len(set(item_records).difference(metadata_result.metadata_by_item)),
        "selected_item_ids": sorted(metadata_records),
    }


def main() -> None:
    """Materialize a prepared bounded CSJ subset dataset."""

    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    reviews_output_path = output_dir / "Clothing_Shoes_and_Jewelry.jsonl"
    metadata_output_path = output_dir / "meta_Clothing_Shoes_and_Jewelry.jsonl"

    max_review_records = _resolve_limit(args, "max_review_records")
    max_users = _resolve_limit(args, "max_users")
    max_items = _resolve_limit(args, "max_items")

    if str(args.preset) == "full_dataset":
        preparation_result = _prepare_disk_backed_subset(
            args=args,
            output_dir=output_dir,
            reviews_output_path=reviews_output_path,
            metadata_output_path=metadata_output_path,
            max_review_records=max_review_records,
            max_users=max_users,
            max_items=max_items,
        )
    else:
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

        preparation_result = {
            "subset_summary": subset.summary,
            "metadata_source": metadata_result.metadata_source,
            "selected_user_count": len(subset.user_histories),
            "selected_item_count": len(metadata_records),
            "selected_interaction_count": len(subset.interaction_records),
            "metadata_records_loaded_from_local_file": len(metadata_result.metadata_by_item),
            "metadata_records_filled_from_review_side": len(set(subset.item_records).difference(metadata_result.metadata_by_item)),
            "selected_item_ids": sorted(metadata_records),
        }

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
        **preparation_result,
    }
    manifest_path = write_json(manifest, output_dir / "subset_manifest.json")
    print(json.dumps({"subset_manifest_path": str(manifest_path.resolve()), **manifest}, indent=2))


if __name__ == "__main__":
    main()
