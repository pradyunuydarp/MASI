"""Progress-bar helpers for notebook and terminal training runs."""

from __future__ import annotations

import os
from typing import Any


class _NoOpProgress:
    """Minimal fallback used when tqdm is unavailable or progress is disabled."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.total = kwargs.get("total")

    def __enter__(self) -> "_NoOpProgress":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def update(self, n: int = 1) -> None:
        return None

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


def progress_enabled() -> bool:
    """Return whether long-running scripts should emit progress bars."""

    value = os.environ.get("MASI_DISABLE_PROGRESS", "").strip().lower()
    return value not in {"1", "true", "yes", "on"}


def make_progress_bar(
    *,
    total: int,
    desc: str,
    unit: str = "step",
    leave: bool = True,
):
    """Create a compact tqdm progress bar with a no-op fallback."""

    if total <= 0 or not progress_enabled():
        return _NoOpProgress(total=total)

    try:
        from tqdm.auto import tqdm
    except Exception:
        return _NoOpProgress(total=total)

    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=leave,
        mininterval=1.0,
        smoothing=0.1,
    )
