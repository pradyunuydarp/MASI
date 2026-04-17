"""Decoding and retrieval helpers for semantic-ID recommendation."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
import time

import torch
from torch.nn import functional as F

from masi.recommender.generative import GenerativeSIDRecommender
from masi.recommender.sequence_data import serialize_history_tokens, serialize_target_tokens
from masi.recommender.vocabulary import FusedSemanticId, TokenVocabulary


@dataclass(slots=True)
class RetrievalCandidate:
    """One ranked catalog candidate."""

    item_id: str
    score: float


@dataclass(slots=True)
class RetrievalResult:
    """Ranked retrieval output for one query."""

    ranked_candidates: list[RetrievalCandidate]
    latency_ms: float


def build_item_token_lookup(
    *,
    fused_ids: list[FusedSemanticId],
    vocabulary: TokenVocabulary,
    use_text_modality: bool = True,
    use_visual_modality: bool = True,
    use_late_fusion: bool = True,
) -> dict[str, list[int]]:
    """Encode fused semantic IDs into reusable token sequences."""

    return {
        fused_id.item_id: vocabulary.encode(
            fused_id.to_tokens(
                use_text_modality=use_text_modality,
                use_visual_modality=use_visual_modality,
                use_late_fusion=use_late_fusion,
            )
        )
        for fused_id in fused_ids
    }


def score_candidate_item(
    *,
    model: GenerativeSIDRecommender,
    history_item_ids: list[str],
    candidate_item_id: str,
    item_tokens: dict[str, list[int]],
    vocabulary: TokenVocabulary,
    max_sequence_length: int,
    device: torch.device,
) -> float:
    """Score one candidate by normalized autoregressive log-probability."""

    history_tokens = serialize_history_tokens(
        history_item_ids=history_item_ids,
        item_tokens=item_tokens,
        vocabulary=vocabulary,
    )
    candidate_tokens = serialize_target_tokens(
        target_item_id=candidate_item_id,
        item_tokens=item_tokens,
        vocabulary=vocabulary,
    )
    combined = history_tokens + candidate_tokens
    if len(combined) > max_sequence_length:
        combined = combined[-max_sequence_length:]
    model_input = torch.tensor([combined], dtype=torch.long, device=device)
    logits = model(model_input)
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    labels = model_input[:, 1:]

    candidate_length = len(candidate_tokens)
    if candidate_length == 0:
        return -inf
    start_index = max(0, labels.size(1) - candidate_length)
    candidate_log_probs = log_probs[:, start_index:, :]
    candidate_labels = labels[:, start_index:]
    gathered = candidate_log_probs.gather(dim=-1, index=candidate_labels.unsqueeze(-1)).squeeze(-1)
    return float(gathered.mean().detach().cpu().item())


def rank_catalog_candidates(
    *,
    model: GenerativeSIDRecommender,
    history_item_ids: list[str],
    candidate_item_ids: list[str],
    item_tokens: dict[str, list[int]],
    vocabulary: TokenVocabulary,
    max_sequence_length: int,
    device: torch.device,
    top_k: int,
) -> RetrievalResult:
    """Rank candidate items by token-sequence likelihood."""

    start_time = time.perf_counter()
    scored = [
        RetrievalCandidate(
            item_id=item_id,
            score=score_candidate_item(
                model=model,
                history_item_ids=history_item_ids,
                candidate_item_id=item_id,
                item_tokens=item_tokens,
                vocabulary=vocabulary,
                max_sequence_length=max_sequence_length,
                device=device,
            ),
        )
        for item_id in candidate_item_ids
        if item_id in item_tokens
    ]
    ranked = sorted(scored, key=lambda candidate: (-candidate.score, candidate.item_id))[:top_k]
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return RetrievalResult(ranked_candidates=ranked, latency_ms=latency_ms)


def match_generated_sequence_to_items(
    *,
    generated_token_ids: list[int],
    item_tokens: dict[str, list[int]],
) -> list[RetrievalCandidate]:
    """Map a generated semantic-ID sequence back to the closest catalog items.

    Exact token matches rank above partial prefix matches. This utility is
    mainly for qualitative inspection, while ranking evaluation uses likelihood
    scoring via :func:`rank_catalog_candidates`.
    """

    comparisons: list[RetrievalCandidate] = []
    for item_id, candidate_tokens in item_tokens.items():
        prefix_overlap = 0
        for generated, candidate in zip(generated_token_ids, candidate_tokens):
            if generated != candidate:
                break
            prefix_overlap += 1
        exact_bonus = 1_000 if generated_token_ids == candidate_tokens else 0
        comparisons.append(
            RetrievalCandidate(
                item_id=item_id,
                score=float(exact_bonus + prefix_overlap),
            )
        )
    return sorted(comparisons, key=lambda candidate: (-candidate.score, candidate.item_id))


__all__ = [
    "RetrievalCandidate",
    "RetrievalResult",
    "build_item_token_lookup",
    "match_generated_sequence_to_items",
    "rank_catalog_candidates",
    "score_candidate_item",
]
