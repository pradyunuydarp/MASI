"""Decoder-only semantic-ID recommender inspired by TIGER-style generation.

The model consumes a flattened history of late-fused semantic-ID tokens and
predicts the token sequence of the next item. For the initial implementation we
use a Transformer encoder with a causal mask, which yields decoder-like
behavior while keeping the code compact and easy to inspect.
"""

from __future__ import annotations

import torch
from torch import nn


class GenerativeSIDRecommender(nn.Module):
    """Autoregressive token generator for semantic-ID recommendation."""

    def __init__(
        self,
        *,
        vocab_size: int,
        max_sequence_length: int = 128,
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
        # MPS does not currently support the nested-tensor fast path used by
        # the default TransformerEncoder configuration during masked ranking.
        self.decoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token_ids: torch.Tensor) -> torch.Tensor:
        """Compute next-token logits for every position in the input sequence."""

        batch_size, sequence_length = input_token_ids.shape
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"Input sequence length {sequence_length} exceeds configured maximum "
                f"{self.max_sequence_length}."
            )

        positions = torch.arange(sequence_length, device=input_token_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.token_embedding(input_token_ids) + self.position_embedding(positions)
        hidden = self.dropout(hidden)

        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=input_token_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        key_padding_mask = input_token_ids.eq(self.pad_token_id)
        decoded = self.decoder(hidden, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        decoded = self.output_norm(decoded)
        return self.output_projection(decoded)

    @torch.no_grad()
    def greedy_decode(
        self,
        *,
        prefix_token_ids: torch.Tensor,
        max_new_tokens: int,
        stop_token_id: int,
    ) -> torch.Tensor:
        """Generate tokens greedily from a history prefix.

        This is sufficient for initial repository demos and deterministic sanity
        checks. Beam search can be added later once retrieval and ranking
        evaluation are wired in.
        """

        generated = prefix_token_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if bool((next_token == stop_token_id).all()):
                break
        return generated
