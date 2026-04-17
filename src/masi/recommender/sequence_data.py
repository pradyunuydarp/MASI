"""Dataset builders for sequential and cross-modal recommendation tasks.

The proposal needs two distinct recommender-facing data views:

1. user interaction histories converted into token sequences for generative
   training and sequential fine-tuning,
2. item-level fused semantic IDs converted into masked-token examples for
   cross-modal MLM pretraining.

This module implements both views with explicit, inspectable examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

import torch
from torch.utils.data import Dataset

from masi.recommender.vocabulary import FusedSemanticId, TokenVocabulary


@dataclass(slots=True)
class GenerativeTrainingExample:
    """One autoregressive next-item training example."""

    user_id: str
    history_item_ids: list[str]
    target_item_id: str
    input_token_ids: list[int]
    label_token_ids: list[int]


@dataclass(slots=True)
class CrossModalMLMExample:
    """One cross-modal masked-language-modeling example."""

    item_id: str
    direction: str
    input_token_ids: list[int]
    label_token_ids: list[int]


def _pad_sequence(sequence: list[int], *, max_length: int, pad_id: int) -> list[int]:
    """Pad or truncate a token sequence from the left.

    Left truncation mirrors common recommender practice: the most recent
    interactions are usually the most predictive, so older context is the part
    we drop first when sequence budgets are exceeded.
    """

    if len(sequence) >= max_length:
        return sequence[-max_length:]
    return sequence + [pad_id] * (max_length - len(sequence))


def serialize_history_tokens(
    *,
    history_item_ids: list[str],
    item_tokens: dict[str, list[int]],
    vocabulary: TokenVocabulary,
) -> list[int]:
    """Serialize a chronological item history into one flat token stream."""

    flattened_history: list[int] = [vocabulary.bos_id]
    for item_id in history_item_ids:
        if item_id not in item_tokens:
            continue
        flattened_history.extend(item_tokens[item_id])
        flattened_history.append(vocabulary.sep_id)
    return flattened_history


def serialize_target_tokens(
    *,
    target_item_id: str,
    item_tokens: dict[str, list[int]],
    vocabulary: TokenVocabulary,
) -> list[int]:
    """Serialize the next-item target sequence."""

    return [
        *item_tokens[target_item_id],
        vocabulary.eos_id,
    ]


def resolve_token_budgets(
    *,
    item_tokens: dict[str, list[int]],
    configured_target_max_tokens: int | None,
    configured_mlm_max_tokens: int | None,
) -> tuple[int, int]:
    """Resolve safe per-item token budgets for autoregressive and MLM stages."""

    max_item_token_length = max((len(tokens) for tokens in item_tokens.values()), default=0)
    required_target_max_tokens = max_item_token_length + 1  # item tokens + <EOS>
    required_mlm_max_tokens = max_item_token_length + 2  # <BOS> + item tokens + <EOS>

    target_max_tokens = max(
        required_target_max_tokens,
        0 if configured_target_max_tokens is None else int(configured_target_max_tokens),
    )
    mlm_max_tokens = max(
        required_mlm_max_tokens,
        0 if configured_mlm_max_tokens is None else int(configured_mlm_max_tokens),
    )
    return target_max_tokens, mlm_max_tokens


class GenerativeSequenceDataset(Dataset[GenerativeTrainingExample]):
    """Flatten user histories into token-level autoregressive examples.

    This dataset follows the TIGER-style setup conceptually: the model receives
    a serialized history of semantic IDs and is trained to generate the token
    sequence corresponding to the next item.
    """

    def __init__(
        self,
        *,
        user_histories: dict[str, list[str]],
        item_tokens: dict[str, list[int]],
        vocabulary: TokenVocabulary,
        history_max_tokens: int = 128,
        target_max_tokens: int = 16,
    ) -> None:
        self.examples: list[GenerativeTrainingExample] = []
        self.vocabulary = vocabulary
        self.history_max_tokens = history_max_tokens
        self.target_max_tokens = target_max_tokens

        for user_id, item_sequence in user_histories.items():
            # We need at least one history item and one prediction target.
            if len(item_sequence) < 2:
                continue

            for prediction_index in range(1, len(item_sequence)):
                history_items = item_sequence[:prediction_index]
                target_item = item_sequence[prediction_index]

                if target_item not in item_tokens:
                    continue
                flattened_history = serialize_history_tokens(
                    history_item_ids=history_items,
                    item_tokens=item_tokens,
                    vocabulary=vocabulary,
                )
                target_tokens = serialize_target_tokens(
                    target_item_id=target_item,
                    item_tokens=item_tokens,
                    vocabulary=vocabulary,
                )

                self.examples.append(
                    GenerativeTrainingExample(
                        user_id=user_id,
                        history_item_ids=list(history_items),
                        target_item_id=target_item,
                        input_token_ids=_pad_sequence(
                            flattened_history,
                            max_length=history_max_tokens,
                            pad_id=vocabulary.pad_id,
                        ),
                        label_token_ids=_pad_sequence(
                            target_tokens,
                            max_length=target_max_tokens,
                            pad_id=vocabulary.pad_id,
                        ),
                    )
                )

    def __len__(self) -> int:
        """Return the number of token-level training examples."""

        return len(self.examples)

    def __getitem__(self, index: int) -> GenerativeTrainingExample:
        """Return one serialized user-history example."""

        return self.examples[index]

    @staticmethod
    def collate(batch: list[GenerativeTrainingExample]) -> dict[str, torch.Tensor]:
        """Convert a batch of examples into tensors for PyTorch models."""

        return {
            "history_token_ids": torch.tensor(
                [example.input_token_ids for example in batch], dtype=torch.long
            ),
            "target_token_ids": torch.tensor(
                [example.label_token_ids for example in batch], dtype=torch.long
            ),
        }


class CrossModalMLMDataset(Dataset[CrossModalMLMExample]):
    """Create masked token reconstruction examples from fused semantic IDs.

    The proposal calls for predicting visual tokens from text tokens and the
    reverse direction. We instantiate that as two deterministic masked examples
    per item:

    - `text_to_visual`: text tokens stay visible, visual tokens are masked
    - `visual_to_text`: visual tokens stay visible, text tokens are masked
    """

    def __init__(
        self,
        *,
        fused_ids: list[FusedSemanticId],
        vocabulary: TokenVocabulary,
        max_length: int = 16,
        use_text_modality: bool = True,
        use_visual_modality: bool = True,
        use_late_fusion: bool = True,
    ) -> None:
        self.examples: list[CrossModalMLMExample] = []
        self.vocabulary = vocabulary
        self.max_length = max_length

        for fused_id in fused_ids:
            tokens = fused_id.to_tokens(
                use_text_modality=use_text_modality,
                use_visual_modality=use_visual_modality,
                use_late_fusion=use_late_fusion,
            )
            if not use_text_modality or not use_visual_modality:
                # Cross-modal MLM is only meaningful when both modalities are
                # present in the active token stream.
                continue
            token_ids = vocabulary.encode(tokens)

            text_start = 1 if use_late_fusion else 0
            text_end = text_start + len(fused_id.text_codes)
            visual_start = text_end + (1 if use_late_fusion else 0)

            text_to_visual = list(token_ids)
            visual_to_text = list(token_ids)
            text_to_visual_labels = [-100] * len(token_ids)
            visual_to_text_labels = [-100] * len(token_ids)

            # Keep the modality markers intact so the model always knows the
            # structural role of the masked span.
            for position in range(visual_start, len(token_ids)):
                text_to_visual[position] = vocabulary.mask_id
                text_to_visual_labels[position] = token_ids[position]

            for position in range(text_start, text_end):
                visual_to_text[position] = vocabulary.mask_id
                visual_to_text_labels[position] = token_ids[position]

            self.examples.append(
                CrossModalMLMExample(
                    item_id=fused_id.item_id,
                    direction="text_to_visual",
                    input_token_ids=_pad_sequence(
                        [vocabulary.bos_id, *text_to_visual, vocabulary.eos_id],
                        max_length=max_length,
                        pad_id=vocabulary.pad_id,
                    ),
                    label_token_ids=_pad_sequence(
                        [-100, *text_to_visual_labels, -100],
                        max_length=max_length,
                        pad_id=-100,
                    ),
                )
            )
            self.examples.append(
                CrossModalMLMExample(
                    item_id=fused_id.item_id,
                    direction="visual_to_text",
                    input_token_ids=_pad_sequence(
                        [vocabulary.bos_id, *visual_to_text, vocabulary.eos_id],
                        max_length=max_length,
                        pad_id=vocabulary.pad_id,
                    ),
                    label_token_ids=_pad_sequence(
                        [-100, *visual_to_text_labels, -100],
                        max_length=max_length,
                        pad_id=-100,
                    ),
                )
            )

    def __len__(self) -> int:
        """Return the number of MLM examples."""

        return len(self.examples)

    def __getitem__(self, index: int) -> CrossModalMLMExample:
        """Return one masked cross-modal item example."""

        return self.examples[index]

    @staticmethod
    def collate(batch: list[CrossModalMLMExample]) -> dict[str, torch.Tensor]:
        """Convert a batch of MLM examples into tensors."""

        return {
            "input_token_ids": torch.tensor(
                [example.input_token_ids for example in batch], dtype=torch.long
            ),
            "label_token_ids": torch.tensor(
                [example.label_token_ids for example in batch], dtype=torch.long
            ),
        }


def build_demo_histories() -> tuple[list[FusedSemanticId], dict[str, list[str]]]:
    """Construct a compact synthetic setup for local recommender demos.

    The dataset is intentionally small. Its role is not to prove recommendation
    quality, but to guarantee that the recommender serialization and model code
    can be executed end to end inside the repository.
    """

    fused_ids = [
        FusedSemanticId(
            "dress_001", ["txt_dress", "txt_floral"], ["vis_red", "vis_flowy"]
        ),
        FusedSemanticId(
            "dress_002", ["txt_dress", "txt_black"], ["vis_black", "vis_minimal"]
        ),
        FusedSemanticId(
            "shoe_001", ["txt_shoe", "txt_sneaker"], ["vis_white", "vis_leather"]
        ),
        FusedSemanticId(
            "bag_001", ["txt_bag", "txt_tote"], ["vis_structured", "vis_large"]
        ),
    ]
    user_histories = {
        "u_001": ["dress_001", "dress_002", "shoe_001"],
        "u_002": ["dress_001", "shoe_001", "bag_001"],
        "u_003": ["dress_002", "shoe_001", "dress_001"],
    }
    return fused_ids, user_histories


def build_negative_item_candidates(
    *,
    item_ids: list[str],
    positive_item_id: str,
    sample_size: int,
    rng: Random,
) -> list[str]:
    """Sample negative items for future retrieval-style evaluations.

    This helper is not yet used in training, but it provides a reproducible
    bridge into leave-one-out evaluation and sampled ranking metrics.
    """

    pool = [item_id for item_id in item_ids if item_id != positive_item_id]
    rng.shuffle(pool)
    return pool[:sample_size]
