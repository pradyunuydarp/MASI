"""Behavior-aware contrastive alignment for MASI Phase 1.

The proposal freezes the base CLIP encoders and trains separate projection
heads for text and vision using collaborative positive pairs from the user-item
graph. This module implements that training stage over a bounded item subset.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from random import Random

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class AlignmentResult:
    """Output of the behavior-aware alignment stage."""

    aligned_text_embeddings: dict[str, torch.Tensor]
    aligned_image_embeddings: dict[str, torch.Tensor]
    positive_pairs: list[tuple[str, str]]
    loss_history: list[float]
    model_state_dict: dict[str, object] | None = None


def build_positive_item_pairs(
    user_histories: dict[str, list[str]],
    *,
    window_size: int = 2,
) -> list[tuple[str, str]]:
    """Build collaborative positive item pairs from chronological histories."""

    pairs: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for history in user_histories.values():
        # A short forward window approximates "behavioral co-interest" without
        # leaking the whole future sequence into every positive pair.
        for index, anchor_item in enumerate(history):
            for offset in range(1, window_size + 1):
                target_index = index + offset
                if target_index >= len(history):
                    break
                target_item = history[target_index]
                if anchor_item == target_item:
                    continue
                pair = (anchor_item, target_item)
                reverse = (target_item, anchor_item)
                if pair not in seen_pairs and reverse not in seen_pairs:
                    seen_pairs.add(pair)
                    pairs.append(pair)
    return pairs


def build_graph_negative_pool(
    user_histories: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Build a simple hard-negative pool from high-frequency non-neighbor items."""

    neighbors: dict[str, set[str]] = defaultdict(set)
    item_counter = Counter(item_id for history in user_histories.values() for item_id in history)

    for history in user_histories.values():
        # We only need graph structure here, not repeated occurrences of the
        # same item in one history, so we collapse duplicates before building
        # the per-item neighbor sets.
        unique_history = list(dict.fromkeys(history))
        for item_id in unique_history:
            neighbors[item_id].update(other for other in unique_history if other != item_id)

    sorted_items = [item_id for item_id, _ in item_counter.most_common()]
    negative_pool: dict[str, list[str]] = {}
    for item_id in sorted_items:
        negative_pool[item_id] = [candidate for candidate in sorted_items if candidate != item_id and candidate not in neighbors[item_id]]
    return negative_pool


class ProjectionHead(nn.Module):
    """Two-layer projection head used on top of frozen CLIP embeddings."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project frozen CLIP embeddings into the behavior-aware space."""

        return F.normalize(self.network(embeddings), dim=-1)


class BehaviorAwareAlignmentModel(nn.Module):
    """Separate text and vision projection heads with shared training logic."""

    def __init__(self, input_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()
        self.text_head = ProjectionHead(input_dim, projection_dim, dropout)
        self.image_head = ProjectionHead(input_dim, projection_dim, dropout)


def _sample_hard_negatives(
    *,
    item_id: str,
    negative_pool: dict[str, list[str]],
    sample_size: int,
    rng: Random,
    fallback_items: list[str],
) -> list[str]:
    """Sample graph-based negatives for one anchor item."""

    pool = list(negative_pool.get(item_id, []))
    rng.shuffle(pool)
    sampled = pool[:sample_size]
    if len(sampled) >= sample_size:
        return sampled

    # When the graph-derived pool is too small, we fall back to any other item
    # so that every anchor contributes the same tensor shape to the batch.
    fallback = [candidate for candidate in fallback_items if candidate != item_id and candidate not in sampled]
    rng.shuffle(fallback)
    sampled.extend(fallback[: sample_size - len(sampled)])

    if not sampled:
        sampled = [candidate for candidate in fallback_items if candidate != item_id][:1]

    while len(sampled) < sample_size and sampled:
        sampled.append(sampled[-1])
    return sampled


def _info_nce_loss(
    *,
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    hard_negative_embeddings: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute InfoNCE with in-batch and graph-hard negatives.

    The proposal specifies mixed negative sampling composed of:

    - other positives already present in the batch, and
    - graph-derived hard negatives sampled from non-neighbor items.
    """

    batch_size = anchor_embeddings.size(0)
    positive_logits = (anchor_embeddings * positive_embeddings).sum(dim=-1, keepdim=True) / temperature

    negative_logits_parts: list[torch.Tensor] = []
    if batch_size > 1:
        batch_logits = (anchor_embeddings @ positive_embeddings.transpose(0, 1)) / temperature
        in_batch_mask = ~torch.eye(batch_size, dtype=torch.bool, device=anchor_embeddings.device)
        in_batch_negative_logits = batch_logits.masked_select(in_batch_mask).reshape(batch_size, batch_size - 1)
        negative_logits_parts.append(in_batch_negative_logits)

    if hard_negative_embeddings.numel() > 0:
        graph_negative_logits = torch.einsum("bd,bnd->bn", anchor_embeddings, hard_negative_embeddings) / temperature
        negative_logits_parts.append(graph_negative_logits)

    if negative_logits_parts:
        logits = torch.cat([positive_logits, *negative_logits_parts], dim=1)
    else:
        logits = positive_logits
    labels = torch.zeros(anchor_embeddings.size(0), dtype=torch.long, device=anchor_embeddings.device)
    return F.cross_entropy(logits, labels)


def train_behavior_aware_alignment(
    *,
    text_embeddings: dict[str, torch.Tensor],
    image_embeddings: dict[str, torch.Tensor],
    user_histories: dict[str, list[str]],
    projection_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    hard_negative_count: int,
    window_size: int,
    dropout: float,
    device: torch.device,
    seed: int,
    use_behavior_alignment: bool = True,
) -> AlignmentResult:
    """Train the Phase 1 behavior-aware projection heads."""

    if text_embeddings and image_embeddings:
        item_ids = sorted(set(text_embeddings).intersection(image_embeddings))
    elif text_embeddings:
        item_ids = sorted(text_embeddings)
    else:
        item_ids = sorted(image_embeddings)
    if not item_ids:
        raise ValueError("No shared item IDs between text and image embeddings.")

    if not use_behavior_alignment:
        # The ablation path bypasses projection-head learning but still returns
        # normalized embeddings with the same dictionary contract expected by
        # the quantization and recommender stages.
        return AlignmentResult(
            aligned_text_embeddings={
                item_id: F.normalize(text_embeddings[item_id], dim=0).cpu() for item_id in item_ids
            } if text_embeddings else {},
            aligned_image_embeddings={
                item_id: F.normalize(image_embeddings[item_id], dim=0).cpu() for item_id in item_ids
            } if image_embeddings else {},
            positive_pairs=[],
            loss_history=[],
            model_state_dict=None,
        )

    if not text_embeddings or not image_embeddings:
        # Single-modality ablations keep the same output contract but skip the
        # cross-item projection-head training that assumes both branches exist.
        return AlignmentResult(
            aligned_text_embeddings={
                item_id: F.normalize(text_embeddings[item_id], dim=0).cpu() for item_id in item_ids
            } if text_embeddings else {},
            aligned_image_embeddings={
                item_id: F.normalize(image_embeddings[item_id], dim=0).cpu() for item_id in item_ids
            } if image_embeddings else {},
            positive_pairs=[],
            loss_history=[],
            model_state_dict=None,
        )

    input_dim = next(iter(text_embeddings.values())).shape[-1]
    model = BehaviorAwareAlignmentModel(input_dim=input_dim, projection_dim=projection_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    positive_pairs = [
        pair for pair in build_positive_item_pairs(user_histories, window_size=window_size)
        if pair[0] in text_embeddings and pair[1] in text_embeddings and pair[0] in image_embeddings and pair[1] in image_embeddings
    ]
    negative_pool = build_graph_negative_pool(user_histories)
    rng = Random(seed)
    loss_history: list[float] = []

    if not positive_pairs:
        # Small smoke subsets can lose all collaborative pairs after modality
        # filtering. In that case we keep the proposal's stage boundary but
        # skip optimization rather than failing the whole pipeline.
        return AlignmentResult(
            aligned_text_embeddings={
                item_id: F.normalize(text_embeddings[item_id], dim=0).cpu() for item_id in item_ids
            } if text_embeddings else {},
            aligned_image_embeddings={
                item_id: F.normalize(image_embeddings[item_id], dim=0).cpu() for item_id in item_ids
            } if image_embeddings else {},
            positive_pairs=[],
            loss_history=[],
            model_state_dict=None,
        )

    for _ in range(epochs):
        shuffled_pairs = list(positive_pairs)
        rng.shuffle(shuffled_pairs)

        for start in range(0, len(shuffled_pairs), batch_size):
            # Each batch is constructed in item-pair space rather than user
            # space because the proposal's alignment objective operates over
            # collaborative item-item positives extracted from the graph.
            batch_pairs = shuffled_pairs[start : start + batch_size]
            anchors = [pair[0] for pair in batch_pairs]
            positives = [pair[1] for pair in batch_pairs]

            batch_text_anchor = torch.stack([text_embeddings[item_id] for item_id in anchors]).to(device)
            batch_text_positive = torch.stack([text_embeddings[item_id] for item_id in positives]).to(device)
            batch_image_anchor = torch.stack([image_embeddings[item_id] for item_id in anchors]).to(device)
            batch_image_positive = torch.stack([image_embeddings[item_id] for item_id in positives]).to(device)

            text_negative_sets = []
            image_negative_sets = []
            for item_id in anchors:
                negative_ids = _sample_hard_negatives(
                    item_id=item_id,
                    negative_pool=negative_pool,
                    sample_size=hard_negative_count,
                    rng=rng,
                    fallback_items=item_ids,
                )
                text_negative_sets.append(torch.stack([text_embeddings[negative_id] for negative_id in negative_ids], dim=0))
                image_negative_sets.append(torch.stack([image_embeddings[negative_id] for negative_id in negative_ids], dim=0))

            text_negative_batch = torch.stack(text_negative_sets).to(device)
            image_negative_batch = torch.stack(image_negative_sets).to(device)

            optimizer.zero_grad()

            # Text and image heads are trained independently on the same
            # behavioral supervision so that one modality cannot dominate the
            # representation of the other before quantization.
            projected_text_anchor = model.text_head(batch_text_anchor)
            projected_text_positive = model.text_head(batch_text_positive)
            projected_text_negatives = model.text_head(text_negative_batch.reshape(-1, input_dim)).reshape(
                text_negative_batch.size(0), text_negative_batch.size(1), -1
            )

            projected_image_anchor = model.image_head(batch_image_anchor)
            projected_image_positive = model.image_head(batch_image_positive)
            projected_image_negatives = model.image_head(image_negative_batch.reshape(-1, input_dim)).reshape(
                image_negative_batch.size(0), image_negative_batch.size(1), -1
            )

            text_loss = _info_nce_loss(
                anchor_embeddings=projected_text_anchor,
                positive_embeddings=projected_text_positive,
                hard_negative_embeddings=projected_text_negatives,
                temperature=temperature,
            )
            image_loss = _info_nce_loss(
                anchor_embeddings=projected_image_anchor,
                positive_embeddings=projected_image_positive,
                hard_negative_embeddings=projected_image_negatives,
                temperature=temperature,
            )

            loss = text_loss + image_loss
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.detach().cpu().item()))

    with torch.no_grad():
        aligned_text_embeddings = {
            item_id: model.text_head(text_embeddings[item_id].unsqueeze(0).to(device)).squeeze(0).cpu()
            for item_id in item_ids
        }
        aligned_image_embeddings = {
            item_id: model.image_head(image_embeddings[item_id].unsqueeze(0).to(device)).squeeze(0).cpu()
            for item_id in item_ids
        }

    return AlignmentResult(
        aligned_text_embeddings=aligned_text_embeddings,
        aligned_image_embeddings=aligned_image_embeddings,
        positive_pairs=positive_pairs,
        loss_history=loss_history,
        model_state_dict=model.state_dict(),
    )
