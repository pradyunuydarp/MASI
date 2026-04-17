"""Independent text and vision residual quantization for MASI Phase 2.

This is an RQ-VAE-style implementation tailored for the repository's current
state:

- the inputs are dense item embeddings produced by Phase 1,
- each modality is quantized by its own codebook stack,
- the output is a code sequence per item that can be late-fused for the
  recommender stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class QuantizationResult:
    """Output of one modality's quantization stage."""

    code_indices_by_item: dict[str, list[int]]
    reconstruction_loss_history: list[float]


class ResidualVectorQuantizer(nn.Module):
    """Residual vector quantizer with multiple independent codebooks."""

    def __init__(self, *, depth: int, codebook_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.depth = depth
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.codebooks = nn.Parameter(torch.randn(depth, codebook_size, embedding_dim) * 0.02)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Quantize a latent tensor and return the quantized output plus codes."""

        residual = latent
        quantized_sum = torch.zeros_like(latent)
        code_indices: list[torch.Tensor] = []

        for level in range(self.depth):
            # Each level explains whatever variance the previous levels left in
            # the residual. This mirrors the residual quantization logic used by
            # the proposal's SID generation stage.
            codebook = self.codebooks[level]
            distances = (
                residual.pow(2).sum(dim=1, keepdim=True)
                - 2 * residual @ codebook.transpose(0, 1)
                + codebook.pow(2).sum(dim=1).unsqueeze(0)
            )
            indices = distances.argmin(dim=1)
            quantized = codebook[indices]
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized
            code_indices.append(indices)

        straight_through = latent + (quantized_sum - latent).detach()
        return straight_through, code_indices


class RQVAEModel(nn.Module):
    """Minimal RQ-VAE-style autoencoder around the residual quantizer."""

    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int,
        depth: int,
        codebook_size: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.quantizer = ResidualVectorQuantizer(
            depth=depth,
            codebook_size=codebook_size,
            embedding_dim=latent_dim,
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Encode, quantize, and decode one embedding batch."""

        latent = self.encoder(inputs)
        quantized, code_indices = self.quantizer(latent)
        reconstructed = self.decoder(quantized)
        return reconstructed, code_indices, latent, quantized


def _fit_kmeans(
    *,
    data: torch.Tensor,
    num_centroids: int,
    iterations: int,
    seed: int,
) -> torch.Tensor:
    """Fit a small k-means codebook using PyTorch only."""

    num_points = data.size(0)
    effective_k = min(num_centroids, num_points)
    generator = torch.Generator(device=data.device).manual_seed(seed)
    initial_indices = torch.randperm(num_points, generator=generator, device=data.device)[:effective_k]
    centroids = data[initial_indices].clone()

    for _ in range(iterations):
        # This is deliberately simple k-means because the development subset is
        # small; the goal is stable code assignments rather than large-scale
        # clustering efficiency.
        distances = torch.cdist(data, centroids)
        assignments = distances.argmin(dim=1)

        updated = []
        for centroid_index in range(effective_k):
            members = data[assignments == centroid_index]
            if members.numel() == 0:
                updated.append(centroids[centroid_index])
            else:
                updated.append(members.mean(dim=0))
        centroids = torch.stack(updated, dim=0)

    if effective_k < num_centroids:
        padding = centroids[-1:].repeat(num_centroids - effective_k, 1)
        centroids = torch.cat([centroids, padding], dim=0)
    return centroids


def _fit_residual_codebooks(
    *,
    latent_data: torch.Tensor,
    depth: int,
    codebook_size: int,
    seed: int,
) -> tuple[torch.Tensor, dict[int, list[int]], float]:
    """Fit residual codebooks greedily, mirroring the RQ quantization logic."""

    residual = latent_data.clone()
    quantized_sum = torch.zeros_like(latent_data)
    codebooks = []
    assignments_by_level: dict[int, list[int]] = {}

    for level in range(depth):
        # We fit one codebook at a time on the current residual, then subtract
        # its contribution before fitting the next level. This keeps the text
        # and vision codebooks structurally independent while still producing a
        # multi-level code sequence per item.
        centroids = _fit_kmeans(
            data=residual,
            num_centroids=codebook_size,
            iterations=20,
            seed=seed + level,
        )
        distances = torch.cdist(residual, centroids)
        assignments = distances.argmin(dim=1)
        quantized = centroids[assignments]
        quantized_sum = quantized_sum + quantized
        residual = residual - quantized
        codebooks.append(centroids)
        assignments_by_level[level] = assignments.cpu().tolist()

    reconstruction_error = float(torch.mean((latent_data - quantized_sum) ** 2).cpu().item())
    return torch.stack(codebooks, dim=0), assignments_by_level, reconstruction_error


def train_rqvae_model(
    *,
    embeddings_by_item: dict[str, torch.Tensor],
    latent_dim: int,
    depth: int,
    codebook_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    commitment_weight: float,
    device: torch.device,
    seed: int,
    refit_codebooks_with_residual_kmeans: bool = False,
) -> tuple[RQVAEModel, QuantizationResult]:
    """Train an RQ-VAE-style model on one modality's aligned embeddings."""

    item_ids = sorted(embeddings_by_item)
    data = torch.stack([embeddings_by_item[item_id] for item_id in item_ids])
    input_dim = data.shape[1]
    model = RQVAEModel(
        input_dim=input_dim,
        latent_dim=latent_dim,
        depth=depth,
        codebook_size=codebook_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    loss_history: list[float] = []

    for _ in range(epochs):
        permutation = torch.randperm(data.size(0), generator=generator)
        shuffled = data[permutation]

        for start in range(0, shuffled.size(0), batch_size):
            # The learned encoder/decoder pair gives the quantizer a smoother
            # latent space before we freeze the final code assignments with the
            # residual k-means fit below.
            batch = shuffled[start : start + batch_size].to(device)
            optimizer.zero_grad()
            reconstructed, _, latent, quantized = model(batch)
            reconstruction_loss = F.mse_loss(reconstructed, batch)
            commitment_loss = F.mse_loss(latent, quantized.detach()) + F.mse_loss(quantized, latent.detach())
            loss = reconstruction_loss + commitment_weight * commitment_loss
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.detach().cpu().item()))

    with torch.no_grad():
        latent_data = model.encoder(data.to(device))
        if refit_codebooks_with_residual_kmeans:
            # This fallback is intentionally opt-in. It can help tiny
            # development subsets avoid degenerate assignments, but it is not
            # part of the proposal's default trainable-codebook path.
            fitted_codebooks, assignments_by_level, residual_error = _fit_residual_codebooks(
                latent_data=latent_data,
                depth=depth,
                codebook_size=codebook_size,
                seed=seed,
            )
            model.quantizer.codebooks.data.copy_(fitted_codebooks)
        else:
            quantized_latent, code_indices = model.quantizer(latent_data)
            assignments_by_level = {
                level: code_indices[level].cpu().tolist()
                for level in range(depth)
            }
            residual_error = float(torch.mean((latent_data - quantized_latent) ** 2).cpu().item())

    code_indices_by_item: dict[str, list[int]] = {}
    for item_index, item_id in enumerate(item_ids):
        code_indices_by_item[item_id] = [assignments_by_level[level][item_index] for level in range(depth)]
    loss_history.append(residual_error)

    return model, QuantizationResult(
        code_indices_by_item=code_indices_by_item,
        reconstruction_loss_history=loss_history,
    )
