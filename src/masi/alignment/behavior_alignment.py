"""Behavior-aware contrastive alignment for MASI Phase 1.

The proposal freezes the base CLIP encoders and trains separate projection
heads for text and vision using collaborative positive pairs from the user-item
graph. This module implements that training stage over a bounded item subset.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from masi.common.progress import make_progress_bar
from masi.common.runtime import module_state_dict_to_cpu, move_optimizer_state_to_device


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
    """Build a simple hard-negative pool from high-frequency non-neighbor items.

    This helper is kept for small debugging and compatibility paths. The main
    training loop uses row-index neighbor sets instead so larger Kaggle profiles
    do not materialize an O(num_items^2) negative-pool dictionary.
    """

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


def _build_graph_neighbor_indices(
    *,
    user_histories: dict[str, list[str]],
    item_id_to_row: dict[str, int],
) -> tuple[dict[int, set[int]], list[int]]:
    """Build compact graph neighbor sets and popularity-ordered candidates."""

    neighbors_by_row: dict[int, set[int]] = defaultdict(set)
    item_counter = Counter(
        item_id
        for history in user_histories.values()
        for item_id in history
        if item_id in item_id_to_row
    )

    for history in user_histories.values():
        unique_rows = [
            item_id_to_row[item_id]
            for item_id in dict.fromkeys(history)
            if item_id in item_id_to_row
        ]
        for row_index in unique_rows:
            neighbors_by_row[row_index].update(
                other_index for other_index in unique_rows if other_index != row_index
            )

    popularity_indices = [
        item_id_to_row[item_id]
        for item_id, _ in item_counter.most_common()
    ]
    return neighbors_by_row, popularity_indices


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


def _pack_embeddings_by_item(
    *,
    embeddings_by_item: dict[str, torch.Tensor],
    item_ids: list[str],
    device: torch.device,
    keep_on_device: bool,
) -> torch.Tensor:
    """Pack item-keyed embeddings into one dense row-indexed tensor."""

    packed = torch.stack(
        [
            embeddings_by_item[item_id].detach().to(dtype=torch.float32, device="cpu")
            for item_id in item_ids
        ],
        dim=0,
    ).contiguous()
    if keep_on_device:
        packed = packed.to(device=device, non_blocking=device.type == "cuda")
    return packed


def _normalize_packed_embeddings(
    *,
    embeddings_by_item: dict[str, torch.Tensor],
    item_ids: list[str],
) -> dict[str, torch.Tensor]:
    """Normalize item embeddings for ablation paths without changing contracts."""

    return {
        item_id: F.normalize(embeddings_by_item[item_id].detach().to(dtype=torch.float32), dim=0).cpu()
        for item_id in item_ids
    }


def _build_negative_pool_indices(
    *,
    negative_pool: dict[str, list[str]],
    item_id_to_row: dict[str, int],
) -> dict[int, list[int]]:
    """Convert hard-negative item IDs to packed embedding row indices once."""

    negative_pool_indices: dict[int, list[int]] = {}
    for item_id, candidates in negative_pool.items():
        if item_id not in item_id_to_row:
            continue
        row_index = item_id_to_row[item_id]
        negative_pool_indices[row_index] = [
            item_id_to_row[candidate]
            for candidate in candidates
            if candidate in item_id_to_row and candidate != item_id
        ]
    return negative_pool_indices


def _sample_hard_negative_indices(
    *,
    item_index: int,
    neighbor_indices: dict[int, set[int]],
    sample_size: int,
    rng: Random,
    popularity_indices: list[int],
    fallback_indices: list[int],
) -> list[int]:
    """Sample graph-based negatives in packed row-index space."""

    blocked = neighbor_indices.get(item_index, set())
    sampled: list[int] = []
    sampled_set: set[int] = set()

    def add_candidate(candidate: int) -> None:
        if candidate == item_index or candidate in blocked or candidate in sampled_set:
            return
        sampled.append(candidate)
        sampled_set.add(candidate)

    candidate_sources = [popularity_indices, fallback_indices]
    for source in candidate_sources:
        if len(sampled) >= sample_size or not source:
            continue
        # Try random probes first so anchors do not all receive the exact same
        # head of the popularity list, then scan only as much as needed.
        max_random_attempts = min(len(source), max(sample_size * 16, 64))
        for _ in range(max_random_attempts):
            add_candidate(source[rng.randrange(len(source))])
            if len(sampled) >= sample_size:
                break
        if len(sampled) >= sample_size:
            break
        scan_start = rng.randrange(len(source))
        for offset in range(len(source)):
            add_candidate(source[(scan_start + offset) % len(source)])
            if len(sampled) >= sample_size:
                break

    if not sampled:
        sampled = [candidate for candidate in fallback_indices if candidate != item_index][:1]

    while len(sampled) < sample_size and sampled:
        sampled.append(sampled[-1])
    return sampled


def _sample_hard_negatives(
    *,
    item_id: str,
    negative_pool: dict[str, list[str]],
    sample_size: int,
    rng: Random,
    fallback_items: list[str],
) -> list[str]:
    """Sample graph-based negatives for one anchor item."""

    pool = negative_pool.get(item_id, [])
    if len(pool) >= sample_size:
        return rng.sample(pool, sample_size)

    sampled = list(pool)
    rng.shuffle(sampled)
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
    checkpoint_callback: Callable[..., object] | None = None,
    initial_model_state_dict: dict[str, object] | None = None,
    initial_optimizer_state_dict: dict[str, object] | None = None,
    initial_global_step: int = 0,
    keep_embeddings_on_device: bool = True,
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
            aligned_text_embeddings=_normalize_packed_embeddings(
                embeddings_by_item=text_embeddings,
                item_ids=item_ids,
            ) if text_embeddings else {},
            aligned_image_embeddings=_normalize_packed_embeddings(
                embeddings_by_item=image_embeddings,
                item_ids=item_ids,
            ) if image_embeddings else {},
            positive_pairs=[],
            loss_history=[],
            model_state_dict=None,
        )

    if not text_embeddings or not image_embeddings:
        # Single-modality ablations keep the same output contract but skip the
        # cross-item projection-head training that assumes both branches exist.
        return AlignmentResult(
            aligned_text_embeddings=_normalize_packed_embeddings(
                embeddings_by_item=text_embeddings,
                item_ids=item_ids,
            ) if text_embeddings else {},
            aligned_image_embeddings=_normalize_packed_embeddings(
                embeddings_by_item=image_embeddings,
                item_ids=item_ids,
            ) if image_embeddings else {},
            positive_pairs=[],
            loss_history=[],
            model_state_dict=None,
        )

    item_id_to_row = {item_id: row_index for row_index, item_id in enumerate(item_ids)}
    packed_text_embeddings = _pack_embeddings_by_item(
        embeddings_by_item=text_embeddings,
        item_ids=item_ids,
        device=device,
        keep_on_device=keep_embeddings_on_device,
    )
    packed_image_embeddings = _pack_embeddings_by_item(
        embeddings_by_item=image_embeddings,
        item_ids=item_ids,
        device=device,
        keep_on_device=keep_embeddings_on_device,
    )
    input_dim = packed_text_embeddings.shape[-1]
    model = BehaviorAwareAlignmentModel(input_dim=input_dim, projection_dim=projection_dim, dropout=dropout).to(device)
    if initial_model_state_dict:
        model.load_state_dict(initial_model_state_dict)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if initial_optimizer_state_dict:
        optimizer.load_state_dict(initial_optimizer_state_dict)
        move_optimizer_state_to_device(optimizer, device)

    positive_pairs = [
        pair for pair in build_positive_item_pairs(user_histories, window_size=window_size)
        if pair[0] in text_embeddings and pair[1] in text_embeddings and pair[0] in image_embeddings and pair[1] in image_embeddings
    ]
    neighbor_indices, popularity_indices = _build_graph_neighbor_indices(
        user_histories=user_histories,
        item_id_to_row=item_id_to_row,
    )
    if not popularity_indices:
        popularity_indices = list(range(len(item_ids)))
    pair_anchor_indices = torch.tensor(
        [item_id_to_row[anchor_item] for anchor_item, _ in positive_pairs],
        dtype=torch.long,
        device=device if keep_embeddings_on_device else "cpu",
    )
    pair_positive_indices = torch.tensor(
        [item_id_to_row[positive_item] for _, positive_item in positive_pairs],
        dtype=torch.long,
        device=device if keep_embeddings_on_device else "cpu",
    )
    fallback_indices = list(range(len(item_ids)))
    rng = Random(seed)
    loss_history: list[float] = []
    global_step = max(0, int(initial_global_step))

    if not positive_pairs:
        # Small smoke subsets can lose all collaborative pairs after modality
        # filtering. In that case we keep the proposal's stage boundary but
        # skip optimization rather than failing the whole pipeline.
        return AlignmentResult(
            aligned_text_embeddings=_normalize_packed_embeddings(
                embeddings_by_item=text_embeddings,
                item_ids=item_ids,
            ) if text_embeddings else {},
            aligned_image_embeddings=_normalize_packed_embeddings(
                embeddings_by_item=image_embeddings,
                item_ids=item_ids,
            ) if image_embeddings else {},
            positive_pairs=[],
            loss_history=[],
            model_state_dict=None,
        )

    batches_per_epoch = (len(positive_pairs) + batch_size - 1) // batch_size
    total_steps = epochs * batches_per_epoch
    with make_progress_bar(total=total_steps, desc="Phase 1 alignment") as progress:
        for epoch_index in range(epochs):
            shuffled_pair_indices = list(range(len(positive_pairs)))
            rng.shuffle(shuffled_pair_indices)

            for batch_index, start in enumerate(range(0, len(shuffled_pair_indices), batch_size), start=1):
                # Each batch is constructed in item-pair space rather than user
                # space because the proposal's alignment objective operates over
                # collaborative item-item positives extracted from the graph.
                batch_pair_rows = torch.tensor(
                    shuffled_pair_indices[start : start + batch_size],
                    dtype=torch.long,
                    device=pair_anchor_indices.device,
                )
                anchor_indices = pair_anchor_indices.index_select(0, batch_pair_rows)
                positive_indices = pair_positive_indices.index_select(0, batch_pair_rows)

                batch_text_anchor = packed_text_embeddings.index_select(0, anchor_indices)
                batch_text_positive = packed_text_embeddings.index_select(0, positive_indices)
                batch_image_anchor = packed_image_embeddings.index_select(0, anchor_indices)
                batch_image_positive = packed_image_embeddings.index_select(0, positive_indices)

                if hard_negative_count > 0:
                    anchor_indices_cpu = anchor_indices.detach().cpu().tolist()
                    negative_rows = [
                        _sample_hard_negative_indices(
                            item_index=int(item_index),
                            neighbor_indices=neighbor_indices,
                            sample_size=hard_negative_count,
                            rng=rng,
                            popularity_indices=popularity_indices,
                            fallback_indices=fallback_indices,
                        )
                        for item_index in anchor_indices_cpu
                    ]
                    negative_indices = torch.tensor(
                        negative_rows,
                        dtype=torch.long,
                        device=pair_anchor_indices.device,
                    )
                    text_negative_batch = packed_text_embeddings.index_select(
                        0,
                        negative_indices.reshape(-1),
                    ).reshape(anchor_indices.size(0), hard_negative_count, input_dim)
                    image_negative_batch = packed_image_embeddings.index_select(
                        0,
                        negative_indices.reshape(-1),
                    ).reshape(anchor_indices.size(0), hard_negative_count, input_dim)
                else:
                    text_negative_batch = packed_text_embeddings.new_empty(
                        (anchor_indices.size(0), 0, input_dim)
                    )
                    image_negative_batch = packed_image_embeddings.new_empty(
                        (anchor_indices.size(0), 0, input_dim)
                    )

                if not keep_embeddings_on_device:
                    batch_text_anchor = batch_text_anchor.to(device=device, non_blocking=device.type == "cuda")
                    batch_text_positive = batch_text_positive.to(device=device, non_blocking=device.type == "cuda")
                    batch_image_anchor = batch_image_anchor.to(device=device, non_blocking=device.type == "cuda")
                    batch_image_positive = batch_image_positive.to(device=device, non_blocking=device.type == "cuda")
                    text_negative_batch = text_negative_batch.to(device=device, non_blocking=device.type == "cuda")
                    image_negative_batch = image_negative_batch.to(device=device, non_blocking=device.type == "cuda")

                optimizer.zero_grad()

                # Text and image heads are trained independently on the same
                # behavioral supervision so that one modality cannot dominate the
                # representation of the other before quantization.
                projected_text_anchor = model.text_head(batch_text_anchor)
                projected_text_positive = model.text_head(batch_text_positive)
                if text_negative_batch.numel() > 0:
                    projected_text_negatives = model.text_head(text_negative_batch.reshape(-1, input_dim)).reshape(
                        text_negative_batch.size(0), text_negative_batch.size(1), -1
                    )
                else:
                    projected_text_negatives = projected_text_anchor.new_empty(
                        (text_negative_batch.size(0), 0, projected_text_anchor.size(-1))
                    )

                projected_image_anchor = model.image_head(batch_image_anchor)
                projected_image_positive = model.image_head(batch_image_positive)
                if image_negative_batch.numel() > 0:
                    projected_image_negatives = model.image_head(image_negative_batch.reshape(-1, input_dim)).reshape(
                        image_negative_batch.size(0), image_negative_batch.size(1), -1
                    )
                else:
                    projected_image_negatives = projected_image_anchor.new_empty(
                        (image_negative_batch.size(0), 0, projected_image_anchor.size(-1))
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
                loss_value = float(loss.detach().cpu().item())
                loss_history.append(loss_value)
                global_step += 1

                checkpoint_path = None
                if checkpoint_callback is not None:
                    checkpoint_path = checkpoint_callback(
                        model=model,
                        optimizer=optimizer,
                        global_step=global_step,
                        epoch_index=epoch_index + 1,
                        step_in_epoch=batch_index,
                        loss=loss_value,
                    )
                postfix = {"loss": f"{loss_value:.4f}"}
                if checkpoint_path is not None:
                    postfix["ckpt"] = Path(str(checkpoint_path)).name
                progress.set_postfix(postfix)
                progress.update(1)

    aligned_text_embeddings: dict[str, torch.Tensor] = {}
    aligned_image_embeddings: dict[str, torch.Tensor] = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(item_ids), batch_size):
            row_slice = slice(start, start + batch_size)
            text_batch = packed_text_embeddings[row_slice]
            image_batch = packed_image_embeddings[row_slice]
            if not keep_embeddings_on_device:
                text_batch = text_batch.to(device=device, non_blocking=device.type == "cuda")
                image_batch = image_batch.to(device=device, non_blocking=device.type == "cuda")
            projected_text = model.text_head(text_batch).cpu()
            projected_image = model.image_head(image_batch).cpu()
            for offset, item_id in enumerate(item_ids[start : start + batch_size]):
                aligned_text_embeddings[item_id] = projected_text[offset]
                aligned_image_embeddings[item_id] = projected_image[offset]

    return AlignmentResult(
        aligned_text_embeddings=aligned_text_embeddings,
        aligned_image_embeddings=aligned_image_embeddings,
        positive_pairs=positive_pairs,
        loss_history=loss_history,
        model_state_dict=module_state_dict_to_cpu(model),
    )
