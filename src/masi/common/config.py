"""Configuration loading helpers for MASI scripts.

The initial repository slice intentionally relies on the Python standard
library only. JSON is used instead of YAML so that the preprocessing scripts
run in a bare Python environment without requiring package installation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


@dataclass(slots=True)
class LoadedConfig:
    """Container returned by :func:`load_json_config`.

    Attributes
    ----------
    path:
        Absolute path to the JSON file that was loaded.
    data:
        Parsed JSON content as a nested dictionary structure.
    """

    path: Path
    data: dict[str, Any]


def load_json_config(config_path: str | Path) -> LoadedConfig:
    """Load a JSON config file and return an absolute-path aware wrapper.

    Parameters
    ----------
    config_path:
        Path to the JSON file. Relative paths are resolved against the current
        working directory of the calling script.

    Returns
    -------
    LoadedConfig
        A simple wrapper that preserves both the parsed data and the canonical
        on-disk location for reproducibility logs.
    """

    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle) or {}
    return LoadedConfig(path=path, data=data)
