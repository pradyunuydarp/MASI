"""Amazon Reviews 2023 preprocessing utilities for MASI.

The initial implementation avoids third-party dependencies so the repository
remains runnable in a fresh Python environment. Records are represented as
lists of dictionaries, which is sufficient for the first-stage preprocessing
tasks defined in the proposal.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from masi.data.contracts import DatasetConfig, PipelineConfig


SUPPORTED_SUFFIXES = {".jsonl", ".json", ".csv"}
Record = dict[str, object]


@dataclass(slots=True)
class DatasetSummary:
    """Compact summary artifact emitted by preprocessing scripts."""

    dataset_name: str
    subset: str
    metadata_rows: int
    interaction_rows: int
    filtered_users: int
    filtered_items: int
    missing_text_rows: int
    missing_image_rows: int
    columns: dict[str, list[str]]

    def to_dict(self) -> dict[str, object]:
        """Convert the summary to a plain dictionary for JSON serialization."""

        return asdict(self)


def load_table(path: str | Path) -> list[Record]:
    """Load a supported tabular file into a list of dictionaries.

    Parameters
    ----------
    path:
        Path to a JSON Lines, JSON array, or CSV file.

    Raises
    ------
    ValueError
        If the suffix is unsupported.
    """

    file_path = Path(path).expanduser().resolve()
    if file_path.suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file format for {file_path}. "
            f"Expected one of: {sorted(SUPPORTED_SUFFIXES)}"
        )

    if file_path.suffix == ".jsonl":
        with file_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    if file_path.suffix == ".json":
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        raise ValueError(f"Expected JSON array in {file_path}")

    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def validate_required_columns(records: list[Record], required: Iterable[str], *, table_name: str) -> None:
    """Ensure a record collection contains the minimum required keys.

    Fast failure is better than allowing missing-column bugs to surface much
    later during training when the real source of the issue is harder to see.
    """

    if not records:
        raise ValueError(f"{table_name} is empty.")

    available = set().union(*(record.keys() for record in records))
    missing = [column for column in required if column not in available]
    if missing:
        raise ValueError(f"Missing required columns in {table_name}: {missing}")


def run_iterative_k_core(
    interactions: list[Record],
    *,
    user_col: str,
    item_col: str,
    min_user_interactions: int,
    min_item_interactions: int,
) -> list[Record]:
    """Apply deterministic iterative k-core filtering.

    The proposal calls for a 5-core dataset. In bipartite interaction data this
    must be iterative because removing low-degree users changes item degrees,
    which may in turn invalidate other users.
    """

    filtered = list(interactions)
    while True:
        prior_size = len(filtered)

        user_counts = Counter(str(record[user_col]) for record in filtered)
        keep_users = {user for user, count in user_counts.items() if count >= min_user_interactions}
        filtered = [record for record in filtered if str(record[user_col]) in keep_users]

        item_counts = Counter(str(record[item_col]) for record in filtered)
        keep_items = {item for item, count in item_counts.items() if count >= min_item_interactions}
        filtered = [record for record in filtered if str(record[item_col]) in keep_items]

        if len(filtered) == prior_size:
            break

    return filtered


def build_text_feature(records: list[Record], text_fields: list[str]) -> list[str]:
    """Concatenate text fields into one normalized model input string.

    Later CLIP text encoding needs a deterministic text view of each item.
    Joining here, rather than ad hoc in notebooks, ensures the exact same text
    recipe can be reused by scripts and training jobs.
    """

    joined_text = []
    for record in records:
        tokens = [str(record.get(field, "")).strip() for field in text_fields]
        joined_text.append(" | ".join(token for token in tokens if token))
    return joined_text


def summarize_dataset(
    config: PipelineConfig,
    metadata: list[Record],
    interactions: list[Record],
) -> DatasetSummary:
    """Create a compact dataset summary aligned with proposal Phase 1 needs."""

    text_view = build_text_feature(metadata, config.dataset.text_fields)
    missing_text_rows = sum(1 for text in text_view if len(text) == 0)

    if config.dataset.image_fields:
        missing_image_rows = 0
        for record in metadata:
            total_length = sum(len(str(record.get(field, "")).strip()) for field in config.dataset.image_fields)
            if total_length == 0:
                missing_image_rows += 1
    else:
        missing_image_rows = len(metadata)

    return DatasetSummary(
        dataset_name=config.dataset.name,
        subset=config.dataset.subset,
        metadata_rows=int(len(metadata)),
        interaction_rows=int(len(interactions)),
        filtered_users=len({str(record[config.dataset.user_col]) for record in interactions}),
        filtered_items=len({str(record[config.dataset.item_col]) for record in interactions}),
        missing_text_rows=missing_text_rows,
        missing_image_rows=missing_image_rows,
        columns={
            "metadata": sorted(set().union(*(record.keys() for record in metadata))) if metadata else [],
            "interactions": sorted(set().union(*(record.keys() for record in interactions))) if interactions else [],
        },
    )


def prepare_dataset(config: PipelineConfig) -> tuple[list[Record], list[Record], DatasetSummary]:
    """Load, validate, filter, and summarize the configured dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, DatasetSummary]
        Filtered metadata, filtered interactions, and a JSON-friendly summary.
    """

    metadata = load_table(config.paths.metadata_file)
    interactions = load_table(config.paths.reviews_file)

    validate_required_columns(
        metadata,
        [config.dataset.item_col, *config.dataset.text_fields, *config.dataset.image_fields],
        table_name="metadata",
    )
    validate_required_columns(
        interactions,
        [
            config.dataset.user_col,
            config.dataset.item_col,
            config.dataset.rating_col,
            config.dataset.timestamp_col,
        ],
        table_name="interactions",
    )

    filtered_interactions = run_iterative_k_core(
        interactions,
        user_col=config.dataset.user_col,
        item_col=config.dataset.item_col,
        min_user_interactions=config.dataset.min_user_interactions,
        min_item_interactions=config.dataset.min_item_interactions,
    )

    keep_items = {str(record[config.dataset.item_col]) for record in filtered_interactions}
    filtered_metadata = [
        record for record in metadata if str(record[config.dataset.item_col]) in keep_items
    ]
    summary = summarize_dataset(config, filtered_metadata, filtered_interactions)
    return filtered_metadata, filtered_interactions, summary
