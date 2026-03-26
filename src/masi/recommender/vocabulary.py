"""Vocabulary and semantic-ID serialization utilities.

The proposal defines the final item representation as a late-fused token
sequence:

`[TXT] <text codes> [VIS] <vision codes>`

This module turns that conceptual definition into a concrete and reproducible
tokenization contract that the recommender models can consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field


SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<SEP>", "[TXT]", "[VIS]"]


@dataclass(slots=True)
class FusedSemanticId:
    """Structured representation of one item's late-fused semantic identifier.

    Attributes
    ----------
    item_id:
        External item identifier from the dataset.
    text_codes:
        Ordered residual-quantization codes for the text modality.
    visual_codes:
        Ordered residual-quantization codes for the image modality.
    """

    item_id: str
    text_codes: list[str]
    visual_codes: list[str]

    def to_tokens(self) -> list[str]:
        """Render the fused semantic ID into the proposal's token order."""

        return ["[TXT]", *self.text_codes, "[VIS]", *self.visual_codes]


@dataclass(slots=True)
class TokenVocabulary:
    """Bidirectional token vocabulary used by MASI recommender models.

    The vocabulary is intentionally small and explicit. Rather than hiding token
    creation inside the training script, we make the mapping a first-class
    artifact so generated sequences, checkpoints, and error analysis can all
    refer to the same token IDs.
    """

    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: dict[int, str] = field(default_factory=dict)

    @classmethod
    def build(cls, fused_ids: list[FusedSemanticId]) -> "TokenVocabulary":
        """Construct a vocabulary from known fused semantic IDs."""

        token_to_id: dict[str, int] = {}
        id_to_token: dict[int, str] = {}

        for token in SPECIAL_TOKENS:
            index = len(token_to_id)
            token_to_id[token] = index
            id_to_token[index] = token

        for fused_id in fused_ids:
            for token in fused_id.to_tokens():
                if token not in token_to_id:
                    index = len(token_to_id)
                    token_to_id[token] = index
                    id_to_token[index] = token

        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    def encode(self, tokens: list[str]) -> list[int]:
        """Convert a token sequence into integer IDs."""

        return [self.token_to_id[token] for token in tokens]

    def decode(self, token_ids: list[int]) -> list[str]:
        """Convert integer IDs back into tokens."""

        return [self.id_to_token[token_id] for token_id in token_ids]

    @property
    def pad_id(self) -> int:
        """Return the padding token ID."""

        return self.token_to_id["<PAD>"]

    @property
    def bos_id(self) -> int:
        """Return the beginning-of-sequence token ID."""

        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        """Return the end-of-sequence token ID."""

        return self.token_to_id["<EOS>"]

    @property
    def mask_id(self) -> int:
        """Return the MLM mask token ID."""

        return self.token_to_id["<MASK>"]

    @property
    def sep_id(self) -> int:
        """Return the token used to separate consecutive items in a history."""

        return self.token_to_id["<SEP>"]
