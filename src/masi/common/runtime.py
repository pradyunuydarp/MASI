"""Runtime helpers for MASI training and notebook workflows."""

from __future__ import annotations

from collections.abc import Sequence
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


def _normalize_string_list(values: Sequence[object] | object | None) -> list[str]:
    """Convert loose config values into a clean list of non-empty strings."""

    if values is None:
        return []
    if isinstance(values, (str, Path)):
        items = [values]
    else:
        items = list(values)
    normalized: list[str] = []
    for item in items:
        value = str(item).strip()
        if value:
            normalized.append(value)
    return normalized


def find_kaggle_dataset_root(
    *,
    dataset_slugs: Sequence[object] | object | None,
    required_relative_paths: Sequence[object] | object | None = None,
    input_root: str | Path = "/kaggle/input",
) -> Path | None:
    """Find a Kaggle input dataset by slug across direct and nested mounts."""

    slugs = _normalize_string_list(dataset_slugs)
    required_paths = _normalize_string_list(required_relative_paths)
    if not slugs:
        return None

    root = Path(input_root).expanduser()
    if not root.exists():
        return None

    seen: set[Path] = set()
    search_roots = [root, root / "datasets"]
    for slug in slugs:
        candidates: list[Path] = []
        direct_candidate = root / slug
        if direct_candidate.is_dir():
            candidates.append(direct_candidate)
        nested_glob_root = root / "datasets"
        if nested_glob_root.is_dir():
            candidates.extend(
                candidate for candidate in sorted(nested_glob_root.glob(f"*/{slug}"))
                if candidate.is_dir()
            )
        for search_root in search_roots:
            if not search_root.is_dir():
                continue
            for candidate in sorted(search_root.rglob(slug)):
                if candidate.is_dir():
                    candidates.append(candidate)

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if required_paths and not all((resolved / relative_path).exists() for relative_path in required_paths):
                continue
            return resolved
    return None


def resolve_input_path(
    *,
    repo_root: Path,
    storage_root: Path,
    configured_path: str | None,
    kaggle_dataset_root: Path | None = None,
    relative_path: str | None = None,
) -> Path | None:
    """Resolve an existing input path across explicit, local, and Kaggle roots."""

    candidates: list[Path] = []
    normalized_relative = str(relative_path).strip() if relative_path else ""

    if configured_path:
        candidate = Path(configured_path).expanduser()
        if candidate.is_absolute():
            candidates.append(candidate)
        else:
            candidates.extend([storage_root / candidate, repo_root / candidate])

    if normalized_relative:
        relative_candidate = Path(normalized_relative).expanduser()
        candidates.extend([storage_root / relative_candidate, repo_root / relative_candidate])
        if kaggle_dataset_root is not None:
            candidates.append(kaggle_dataset_root / relative_candidate)

    seen: set[Path] = set()
    ordered_candidates: list[Path] = []
    for candidate in candidates:
        resolved_candidate = candidate.resolve() if candidate.exists() else candidate
        if resolved_candidate in seen:
            continue
        seen.add(resolved_candidate)
        ordered_candidates.append(candidate)

    for candidate in ordered_candidates:
        if candidate.exists():
            return candidate.resolve()

    if ordered_candidates:
        return ordered_candidates[0]
    return None
