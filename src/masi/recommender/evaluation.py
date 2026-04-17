"""Leave-one-out splits and ranking metrics for MASI experiments."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from statistics import mean

import torch

from masi.recommender.generative import GenerativeSIDRecommender
from masi.recommender.retrieval import rank_catalog_candidates
from masi.recommender.vocabulary import TokenVocabulary


@dataclass(slots=True)
class LeaveOneOutExample:
    """One evaluation query from a chronological user history."""

    user_id: str
    history_item_ids: list[str]
    target_item_id: str
    split: str


@dataclass(slots=True)
class LeaveOneOutSplit:
    """Train histories plus warm/cold leave-one-out evaluation examples."""

    train_histories: dict[str, list[str]]
    warm_examples: list[LeaveOneOutExample]
    cold_examples: list[LeaveOneOutExample]
    cold_item_ids: set[str]
    summary: dict[str, object]


def build_leave_one_out_split(
    *,
    user_histories: dict[str, list[str]],
    cold_start_ratio: float,
    min_train_history: int,
    seed: int,
    use_cold_start_evaluation: bool = True,
) -> LeaveOneOutSplit:
    """Create deterministic warm-start and zero-shot cold-start splits."""

    eligible_histories = {user_id: history for user_id, history in user_histories.items() if len(history) >= 2}
    last_items = sorted({history[-1] for history in eligible_histories.values()})
    rng = Random(seed)
    shuffled_last_items = list(last_items)
    rng.shuffle(shuffled_last_items)

    cold_item_ids: set[str] = set()
    if use_cold_start_evaluation and shuffled_last_items:
        cold_item_count = max(1, int(round(len(shuffled_last_items) * cold_start_ratio)))
        cold_item_ids = set(shuffled_last_items[:cold_item_count])

    train_histories: dict[str, list[str]] = {}
    warm_examples: list[LeaveOneOutExample] = []
    cold_examples: list[LeaveOneOutExample] = []

    for user_id, history in eligible_histories.items():
        target_item_id = history[-1]
        prefix_history = history[:-1]
        train_prefix = [item_id for item_id in prefix_history if item_id not in cold_item_ids]

        if len(train_prefix) >= min_train_history:
            train_histories[user_id] = train_prefix

        example = LeaveOneOutExample(
            user_id=user_id,
            history_item_ids=list(train_prefix),
            target_item_id=target_item_id,
            split="cold" if target_item_id in cold_item_ids else "warm",
        )
        if example.split == "cold":
            if train_prefix:
                cold_examples.append(example)
        else:
            if train_prefix:
                warm_examples.append(example)

    summary = {
        "num_train_users": len(train_histories),
        "num_warm_examples": len(warm_examples),
        "num_cold_examples": len(cold_examples),
        "num_cold_items": len(cold_item_ids),
        "cold_start_ratio": cold_start_ratio if use_cold_start_evaluation else 0.0,
    }
    return LeaveOneOutSplit(
        train_histories=train_histories,
        warm_examples=warm_examples,
        cold_examples=cold_examples,
        cold_item_ids=cold_item_ids,
        summary=summary,
    )


def hit_rate_at_k(*, ranked_item_ids: list[str], target_item_id: str, k: int) -> float:
    """Compute hit rate at K for one ranked list."""

    return 1.0 if target_item_id in ranked_item_ids[:k] else 0.0


def ndcg_at_k(*, ranked_item_ids: list[str], target_item_id: str, k: int) -> float:
    """Compute NDCG at K for one ranked list with one relevant target."""

    try:
        position = ranked_item_ids[:k].index(target_item_id)
    except ValueError:
        return 0.0
    return 1.0 / torch.log2(torch.tensor(float(position + 2))).item()


def coverage_at_k(*, ranked_lists: list[list[str]], catalog_item_count: int, k: int) -> float:
    """Compute recommendation coverage over the catalog."""

    if catalog_item_count <= 0:
        return 0.0
    recommended_items = {item_id for ranked in ranked_lists for item_id in ranked[:k]}
    return len(recommended_items) / float(catalog_item_count)


def evaluate_generative_ranking(
    *,
    model: GenerativeSIDRecommender,
    examples: list[LeaveOneOutExample],
    candidate_item_ids: list[str],
    item_tokens: dict[str, list[int]],
    vocabulary: TokenVocabulary,
    max_sequence_length: int,
    device: torch.device,
    top_k: int,
) -> dict[str, object]:
    """Evaluate a generative recommender on ranking metrics."""

    if not examples:
        return {
            f"hr@{top_k}": 0.0,
            f"ndcg@{top_k}": 0.0,
            f"coverage@{top_k}": 0.0,
            "avg_latency_ms": 0.0,
            "num_examples": 0,
        }

    model.eval()
    ranked_lists: list[list[str]] = []
    hit_scores: list[float] = []
    ndcg_scores: list[float] = []
    latencies: list[float] = []

    with torch.no_grad():
        for example in examples:
            retrieval = rank_catalog_candidates(
                model=model,
                history_item_ids=example.history_item_ids,
                candidate_item_ids=candidate_item_ids,
                item_tokens=item_tokens,
                vocabulary=vocabulary,
                max_sequence_length=max_sequence_length,
                device=device,
                top_k=top_k,
            )
            ranked_item_ids = [candidate.item_id for candidate in retrieval.ranked_candidates]
            ranked_lists.append(ranked_item_ids)
            hit_scores.append(hit_rate_at_k(ranked_item_ids=ranked_item_ids, target_item_id=example.target_item_id, k=top_k))
            ndcg_scores.append(ndcg_at_k(ranked_item_ids=ranked_item_ids, target_item_id=example.target_item_id, k=top_k))
            latencies.append(retrieval.latency_ms)

    return {
        f"hr@{top_k}": mean(hit_scores),
        f"ndcg@{top_k}": mean(ndcg_scores),
        f"coverage@{top_k}": coverage_at_k(
            ranked_lists=ranked_lists,
            catalog_item_count=len(candidate_item_ids),
            k=top_k,
        ),
        "avg_latency_ms": mean(latencies),
        "num_examples": len(examples),
    }


__all__ = [
    "LeaveOneOutExample",
    "LeaveOneOutSplit",
    "build_leave_one_out_split",
    "coverage_at_k",
    "evaluate_generative_ranking",
    "hit_rate_at_k",
    "ndcg_at_k",
]
