"""Shared method toggles for MASI ablations and staged experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(slots=True)
class MethodToggleConfig:
    """Boolean switches that gate optional MASI stages.

    The proposal's main ablations focus on whether Phase 1 behavior alignment
    and Phase 3 cross-modal MLM are enabled. We also expose text/visual/late
    fusion switches so downstream experiments can disable individual modalities
    without rewriting the pipeline.
    """

    use_behavior_alignment: bool = True
    use_text_modality: bool = True
    use_visual_modality: bool = True
    use_late_fusion: bool = True
    use_cross_modal_mlm: bool = True
    use_generative_finetuning: bool = True
    use_cold_start_evaluation: bool = True

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | None) -> "MethodToggleConfig":
        """Build toggle config from a JSON-style mapping."""

        if mapping is None:
            return cls()
        values: dict[str, bool] = {}
        for field_name in cls.__dataclass_fields__:
            raw_value = mapping.get(field_name)
            if raw_value is None:
                continue
            values[field_name] = bool(raw_value)
        return cls(**values)


__all__ = ["MethodToggleConfig"]
