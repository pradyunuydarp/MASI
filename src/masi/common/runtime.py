"""Runtime helpers for MASI training and notebook workflows."""

from __future__ import annotations

import os
from pathlib import Path

from masi.common.io import ensure_directory


def detect_runtime_environment() -> str:
    """Infer the current execution environment."""

    if os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if os.getenv("COLAB_RELEASE_TAG") or os.getenv("COLAB_GPU"):
        return "colab"
    return "local"


def resolve_storage_root(
    *,
    repo_root: Path,
    runtime_config: dict[str, object],
    cli_storage_root: str | None,
) -> Path:
    """Resolve the storage root used for datasets and run artifacts."""

    environment = detect_runtime_environment()

    if cli_storage_root:
        return ensure_directory(cli_storage_root)

    env_override = os.getenv("MASI_STORAGE_ROOT")
    if env_override:
        return ensure_directory(env_override)

    configured_storage_root = runtime_config.get("storage_root")
    if configured_storage_root:
        return ensure_directory(str(configured_storage_root))

    if environment == "kaggle":
        return ensure_directory("/kaggle/working/masi_artifacts")
    if environment == "colab":
        return ensure_directory("/content/masi_artifacts")
    return repo_root


def resolve_path(storage_root: Path, path_value: str | None) -> Path | None:
    """Resolve a configured path against the storage root unless absolute."""

    if not path_value:
        return None
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate
    return storage_root / candidate
