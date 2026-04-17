"""Cross-modal MLM pretraining model for MASI Phase 4.

This component operationalizes the proposal's discrete alignment step. The
model receives fused item token sequences with one modality masked and learns to
reconstruct the missing span from the visible modality context.
"""

from __future__ import annotations

import torch
from torch import nn


class CrossModalMLMPretrainer(nn.Module):
    """Masked-token reconstruction model over fused semantic IDs."""

    def __init__(
        self,
        *,
        vocab_size: int,
        max_sequence_length: int = 32,
        hidden_dim: int = 96,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.max_sequence_length = max_sequence_length

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_sequence_length, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        # Keep the MLM pretrainer on the same MPS-compatible execution path as
        # the downstream generative model used in Phase 3.
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token_ids: torch.Tensor) -> torch.Tensor:
        """Predict vocabulary logits for every masked-LM position."""

        batch_size, sequence_length = input_token_ids.shape
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"Input sequence length {sequence_length} exceeds configured maximum "
                f"{self.max_sequence_length}."
            )

        positions = torch.arange(sequence_length, device=input_token_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.token_embedding(input_token_ids) + self.position_embedding(positions)
        hidden = self.dropout(hidden)

        key_padding_mask = input_token_ids.eq(self.pad_token_id)
        encoded = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        encoded = self.output_norm(encoded)
        return self.output_projection(encoded)


def masked_language_modeling_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute MLM loss with `-100` labels ignored, matching PyTorch convention."""

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
