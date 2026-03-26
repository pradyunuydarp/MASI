"""Small training helpers for the recommender foundation.

These functions are intentionally modest. Their job is to make the new
recommender modules easy to demo and regression-test before the repository
graduates to a fuller experiment runner.
"""

from __future__ import annotations

import torch
from torch import nn

from masi.recommender.mlm import masked_language_modeling_loss


def autoregressive_token_loss(logits: torch.Tensor, labels: torch.Tensor, *, pad_token_id: int) -> torch.Tensor:
    """Compute token-level next-step loss for generative recommendation.

    We align logits and labels in teacher-forcing style:

    - the model sees the input sequence up to timestep `t`,
    - it predicts token `t + 1`,
    - padding labels are ignored.
    """

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    return loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))


def training_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_inputs: torch.Tensor,
    batch_labels: torch.Tensor,
    objective: str,
    pad_token_id: int,
) -> float:
    """Execute one optimization step for either autoregressive or MLM training."""

    model.train()
    optimizer.zero_grad()
    logits = model(batch_inputs)

    if objective == "autoregressive":
        loss = autoregressive_token_loss(logits, batch_labels, pad_token_id=pad_token_id)
    elif objective == "mlm":
        loss = masked_language_modeling_loss(logits, batch_labels)
    else:
        raise ValueError(f"Unsupported objective: {objective}")

    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())
