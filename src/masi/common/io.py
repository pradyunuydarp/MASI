"""I/O utilities shared across scripts.

The helpers here prefer explicit JSON artifacts because later training stages
will need deterministic, machine-readable manifests for dataset lineage,
feature extraction runs, and experiment metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist and return its path."""

    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Serialize a dictionary to JSON with stable formatting.

    Stable indentation and key ordering make git diffs and handoffs easier to
    inspect, which matters for a research repository where artifacts are often
    reviewed by humans before they are consumed by automation.
    """

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
