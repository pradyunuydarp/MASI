"""Data contracts for the MASI preprocessing pipeline.

The proposal specifies the Amazon Reviews 2023 Clothing/Shoes/Jewelry subset as
the initial benchmark domain. Before model code exists, we need explicit and
validated assumptions about:

1. where raw files live,
2. which columns are required,
3. which derived artifacts later phases can rely on.

This module centralizes those assumptions so later agents do not have to infer
them from scripts or notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PathConfig:
    """Filesystem locations used by the preprocessing pipeline."""

    raw_root: Path
    processed_root: Path
    outputs_root: Path
    metadata_file: Path
    reviews_file: Path

    @classmethod
    def from_dict(cls, payload: dict[str, Any], base_dir: Path) -> "PathConfig":
        """Build a path config while resolving relative paths from one base.

        The config files in this repository use repo-relative paths so the same
        config can be reused from scripts, notebooks, and CI without requiring
        the caller to pre-normalize every field.
        """

        def resolve(value: str) -> Path:
            return (base_dir / value).resolve()

        return cls(
            raw_root=resolve(payload["raw_root"]),
            processed_root=resolve(payload["processed_root"]),
            outputs_root=resolve(payload["outputs_root"]),
            metadata_file=resolve(payload["metadata_file"]),
            reviews_file=resolve(payload["reviews_file"]),
        )


@dataclass(slots=True)
class DatasetConfig:
    """Dataset-specific schema and filtering parameters."""

    name: str
    subset: str
    user_col: str
    item_col: str
    rating_col: str
    timestamp_col: str
    text_fields: list[str] = field(default_factory=list)
    image_fields: list[str] = field(default_factory=list)
    min_user_interactions: int = 5
    min_item_interactions: int = 5

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetConfig":
        """Build a strongly typed dataset config from raw config content."""

        return cls(
            name=payload["name"],
            subset=payload["subset"],
            user_col=payload["user_col"],
            item_col=payload["item_col"],
            rating_col=payload["rating_col"],
            timestamp_col=payload["timestamp_col"],
            text_fields=list(payload.get("text_fields", [])),
            image_fields=list(payload.get("image_fields", [])),
            min_user_interactions=int(payload.get("min_user_interactions", 5)),
            min_item_interactions=int(payload.get("min_item_interactions", 5)),
        )


@dataclass(slots=True)
class PipelineConfig:
    """Single object passed around by preprocessing scripts."""

    seed: int
    paths: PathConfig
    dataset: DatasetConfig

    @classmethod
    def from_dict(cls, payload: dict[str, Any], base_dir: Path) -> "PipelineConfig":
        """Construct a full preprocessing config from a JSON dictionary."""

        return cls(
            seed=int(payload.get("seed", 42)),
            paths=PathConfig.from_dict(payload["paths"], base_dir=base_dir),
            dataset=DatasetConfig.from_dict(payload["dataset"]),
        )
