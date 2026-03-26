"""SASRec-style sequential recommender baseline.

This module reproduces the core mechanics of SASRec:

- learned item embeddings,
- learned positional embeddings,
- causal self-attention over interaction histories,
- a last-timestep user state used for next-item scoring.

The implementation is intentionally compact and heavily documented so later
phases can adapt it for stronger baselines or swap in semantic-ID inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class SASRecConfig:
    """Configuration for a SASRec-style encoder."""

    num_items: int
    max_sequence_length: int = 50
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    pad_token_id: int = 0


class SASRecModel(nn.Module):
    """Minimal SASRec implementation for baseline reproduction."""

    def __init__(self, config: SASRecConfig) -> None:
        super().__init__()
        self.config = config

        self.item_embedding = nn.Embedding(
            num_embeddings=config.num_items,
            embedding_dim=config.hidden_dim,
            padding_idx=config.pad_token_id,
        )
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.output_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, item_sequences: torch.Tensor) -> torch.Tensor:
        """Encode a batch of item-ID histories.

        Parameters
        ----------
        item_sequences:
            Tensor of shape `(batch_size, sequence_length)` containing integer
            item IDs with left-padded zeros for missing positions.

        Returns
        -------
        torch.Tensor
            Contextualized hidden states for every timestep.
        """

        batch_size, sequence_length = item_sequences.shape
        positions = torch.arange(sequence_length, device=item_sequences.device).unsqueeze(0).expand(batch_size, -1)

        hidden = self.item_embedding(item_sequences) + self.position_embedding(positions)
        hidden = self.dropout(hidden)

        # Causal attention prevents each position from seeing future items,
        # which preserves the autoregressive semantics of next-item prediction.
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=item_sequences.device, dtype=torch.bool),
            diagonal=1,
        )
        key_padding_mask = item_sequences.eq(self.config.pad_token_id)
        encoded = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        return self.output_norm(encoded)

    def score_all_items(self, item_sequences: torch.Tensor) -> torch.Tensor:
        """Produce next-item logits over the entire item vocabulary."""

        encoded = self.forward(item_sequences)
        last_hidden = encoded[:, -1, :]
        return last_hidden @ self.item_embedding.weight.transpose(0, 1)
