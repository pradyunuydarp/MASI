"""Runtime helpers for MASI training and notebook workflows."""

from __future__ import annotations

from collections.abc import Sequence
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from masi.common.io import ensure_directory

if TYPE_CHECKING:
    import torch


LOGGER = logging.getLogger(__name__)


def detect_runtime_environment() -> str:
    """Infer the current execution environment."""

    if os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if os.getenv("COLAB_RELEASE_TAG") or os.getenv("COLAB_GPU"):
        return "colab"
    return "local"


def _ensure_runtime_logging() -> None:
    """Make INFO runtime messages visible in scripts that do not configure logging."""

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def _mps_is_available() -> bool:
    """Return whether the current PyTorch build can use Apple MPS."""

    torch = _torch()
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def _torch():
    """Import torch lazily so data-only runtime helpers keep working without it."""

    import torch

    return torch


def select_torch_device(runtime_config: dict[str, object] | None = None) -> torch.device:
    """Select a PyTorch device from MASI runtime config.

    `runtime.device = "auto"` prefers CUDA, then MPS, then CPU. Explicit CUDA
    or MPS requests fall back to the next available supported backend with a
    warning instead of failing long notebook workflows after setup.
    """

    torch = _torch()
    config = runtime_config or {}
    requested = str(config.get("device", "auto")).strip().lower() or "auto"
    valid = {"auto", "cuda", "mps", "cpu"}
    if requested not in valid:
        raise ValueError(f"Unsupported runtime.device={requested!r}; expected one of {sorted(valid)}.")

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        LOGGER.warning("runtime.device='cuda' requested but CUDA is unavailable; falling back to auto selection.")
    if requested == "mps":
        if _mps_is_available():
            return torch.device("mps")
        LOGGER.warning("runtime.device='mps' requested but MPS is unavailable; falling back to auto selection.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log_torch_device_summary(
    device: torch.device,
    *,
    enabled: bool = True,
    logger: logging.Logger | None = None,
) -> None:
    """Log the selected torch device and CUDA details when available."""

    if not enabled:
        return
    torch = _torch()
    _ensure_runtime_logging()
    active_logger = logger or LOGGER
    active_logger.info("MASI selected torch device: %s", device)

    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        active_logger.info("CUDA device %s: %s", index, torch.cuda.get_device_name(index))
        try:
            allocated_gib = torch.cuda.memory_allocated(index) / (1024 ** 3)
            reserved_gib = torch.cuda.memory_reserved(index) / (1024 ** 3)
            active_logger.info(
                "CUDA memory before stage: allocated=%.3f GiB reserved=%.3f GiB",
                allocated_gib,
                reserved_gib,
            )
            summary = torch.cuda.memory_summary(index, abbreviated=True)
            active_logger.info("CUDA memory summary:\n%s", summary)
        except RuntimeError as exc:
            active_logger.warning("CUDA memory summary unavailable: %s", exc)


def resolve_torch_device(runtime_config: dict[str, object] | None = None) -> torch.device:
    """Select and log the torch device according to MASI runtime config."""

    config = runtime_config or {}
    device = select_torch_device(config)
    log_torch_device_summary(
        device,
        enabled=bool(config.get("log_device_summary", True)),
    )
    return device


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer state tensors after loading a checkpoint payload."""

    torch = _torch()
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def object_to_cpu(value: Any) -> Any:
    """Recursively detach tensors to CPU for checkpoint and artifact writes."""

    torch = _torch()
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: object_to_cpu(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [object_to_cpu(nested) for nested in value]
    if isinstance(value, tuple):
        return tuple(object_to_cpu(nested) for nested in value)
    return value


def module_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a CPU copy of a module state dict for portable checkpointing."""

    return {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}


def optimizer_state_dict_to_cpu(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Return a CPU copy of an optimizer state dict for portable checkpointing."""

    return object_to_cpu(optimizer.state_dict())


def clear_device_cache(device: torch.device) -> None:
    """Release backend cache memory between heavyweight MASI stages."""

    torch = _torch()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


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
