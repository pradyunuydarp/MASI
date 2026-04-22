"""Small training helpers for the recommender foundation.

These functions are intentionally modest. Their job is to make the new
recommender modules easy to demo and regression-test before the repository
graduates to a fuller experiment runner.
"""

from __future__ import annotations

from statistics import mean
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

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


def run_training_epochs(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    objective: str,
    pad_token_id: int,
    input_key: str,
    label_key: str,
    epochs: int,
    device: torch.device,
    checkpoint_callback: Callable[..., None] | None = None,
) -> list[float]:
    """Train a model for multiple epochs and return mean epoch losses."""

    epoch_losses: list[float] = []
    model.to(device)
    global_step = 0
    for epoch_index in range(epochs):
        batch_losses: list[float] = []
        for batch_index, batch in enumerate(data_loader, start=1):
            batch_inputs = batch[input_key].to(device)
            batch_labels = batch[label_key].to(device)
            loss_value = training_step(
                model=model,
                optimizer=optimizer,
                batch_inputs=batch_inputs,
                batch_labels=batch_labels,
                objective=objective,
                pad_token_id=pad_token_id,
            )
            batch_losses.append(loss_value)
            global_step += 1
            if checkpoint_callback is not None:
                checkpoint_callback(
                    model=model,
                    optimizer=optimizer,
                    objective=objective,
                    global_step=global_step,
                    epoch_index=epoch_index + 1,
                    step_in_epoch=batch_index,
                    loss=loss_value,
                )
        epoch_losses.append(mean(batch_losses) if batch_losses else 0.0)
    return epoch_losses


def initialize_generative_from_mlm(
    *,
    generative_model: nn.Module,
    mlm_model: nn.Module,
    use_cross_modal_mlm: bool = True,
) -> None:
    """Copy shared weights from MLM pretraining into the generative model."""

    if not use_cross_modal_mlm:
        return

    generative_model.token_embedding.load_state_dict(mlm_model.token_embedding.state_dict())
    generative_model.output_norm.load_state_dict(mlm_model.output_norm.state_dict())
    generative_model.output_projection.load_state_dict(mlm_model.output_projection.state_dict())

    with torch.no_grad():
        source_positions = mlm_model.position_embedding.weight.data
        target_positions = generative_model.position_embedding.weight.data
        copy_length = min(source_positions.size(0), target_positions.size(0))
        target_positions[:copy_length].copy_(source_positions[:copy_length])

    mlm_state = mlm_model.encoder.state_dict()
    decoder_state = generative_model.decoder.state_dict()
    transferable = {
        key: value
        for key, value in mlm_state.items()
        if key in decoder_state and decoder_state[key].shape == value.shape
    }
    decoder_state.update(transferable)
    generative_model.decoder.load_state_dict(decoder_state)
