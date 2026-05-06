"""Checkpoint helpers for long-running MASI training stages."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from masi.common.io import ensure_directory, write_json
from masi.common.runtime import object_to_cpu


@dataclass(slots=True)
class StepCheckpointManager:
    """Persist periodic step-based checkpoints for one training stage."""

    checkpoint_root: Path
    stage_name: str
    save_steps: int | None
    keep_last: int | None = 2

    def __post_init__(self) -> None:
        self.checkpoint_root = ensure_directory(self.checkpoint_root)

    @property
    def stage_directory(self) -> Path:
        """Return the directory used for this stage's periodic checkpoints."""

        return ensure_directory(self.checkpoint_root / self.stage_name)

    @property
    def enabled(self) -> bool:
        """Return whether periodic checkpointing is active."""

        return self.save_steps is not None and self.save_steps > 0

    def maybe_save(
        self,
        *,
        global_step: int,
        payload: dict[str, Any],
    ) -> Path | None:
        """Persist a checkpoint when the configured step interval is reached."""

        if not self.enabled or global_step <= 0 or global_step % int(self.save_steps) != 0:
            return None
        return self.save(global_step=global_step, payload=payload)

    def save(
        self,
        *,
        global_step: int,
        payload: dict[str, Any],
    ) -> Path:
        """Persist a checkpoint immediately and update the stage manifest."""

        checkpoint_path = self.stage_directory / f"step_{global_step:07d}.pt"
        torch.save(object_to_cpu(payload), checkpoint_path)
        write_json(
            {
                "global_step": global_step,
                "checkpoint_path": str(checkpoint_path),
            },
            self.stage_directory / "latest.json",
        )
        self._prune_old_checkpoints()
        return checkpoint_path

    def list_checkpoints(self) -> list[str]:
        """Return the currently retained periodic checkpoints."""

        return [str(path) for path in sorted(self.stage_directory.glob("step_*.pt"))]

    def latest_checkpoint(self) -> str | None:
        """Return the latest retained periodic checkpoint path, if any."""

        checkpoints = sorted(self.stage_directory.glob("step_*.pt"))
        if not checkpoints:
            return None
        return str(checkpoints[-1])

    def _prune_old_checkpoints(self) -> None:
        """Keep only the newest retained step checkpoints when configured."""

        if self.keep_last is None or self.keep_last <= 0:
            return
        checkpoints = sorted(self.stage_directory.glob("step_*.pt"))
        if len(checkpoints) <= self.keep_last:
            return
        for path in checkpoints[: -self.keep_last]:
            path.unlink(missing_ok=True)


def find_stage_resume_checkpoint(
    *,
    checkpoint_root: Path,
    final_checkpoint_name: str,
    step_stage_name: str,
) -> Path | None:
    """Find the best checkpoint to restore for a training stage.

    Completed stages prefer their final checkpoint. Interrupted stages fall back
    to the latest retained periodic checkpoint, whose manifest may contain an
    absolute path from the previous Kaggle session.
    """

    final_checkpoint = checkpoint_root / final_checkpoint_name
    if final_checkpoint.exists():
        return final_checkpoint

    step_directory = checkpoint_root / step_stage_name
    latest_manifest = step_directory / "latest.json"
    if latest_manifest.exists():
        with latest_manifest.open("r", encoding="utf-8") as handle:
            latest_payload = json.load(handle)
        raw_checkpoint_path = latest_payload.get("checkpoint_path")
        if raw_checkpoint_path:
            manifest_checkpoint = Path(str(raw_checkpoint_path))
            if manifest_checkpoint.exists():
                return manifest_checkpoint
            sibling_checkpoint = step_directory / manifest_checkpoint.name
            if sibling_checkpoint.exists():
                return sibling_checkpoint

    periodic_checkpoints = sorted(step_directory.glob("step_*.pt"))
    if periodic_checkpoints:
        return periodic_checkpoints[-1]
    return None


def load_checkpoint_payload(checkpoint_path: Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint payload and validate the expected dictionary shape."""

    payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected checkpoint payload dict at {checkpoint_path}, got {type(payload)!r}.")
    return payload
