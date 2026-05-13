"""SASRec baseline training and evaluation on MASI data splits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from random import Random
from statistics import mean
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from masi.common.io import ensure_directory, write_json
from masi.common.progress import make_progress_bar
from masi.common.runtime import module_state_dict_to_cpu
from masi.recommender.evaluation import (
    LeaveOneOutExample,
    build_leave_one_out_split,
    coverage_at_k,
    hit_rate_at_k,
    ndcg_at_k,
)
from masi.recommender.sasrec import SASRecConfig, SASRecModel


@dataclass(slots=True)
class SASRecTrainingExample:
    """One SASRec next-item training example."""

    user_id: str
    history_item_ids: list[str]
    target_item_id: str
    input_item_ids: list[int]
    label_item_id: int


class SASRecTrainingDataset(Dataset[SASRecTrainingExample]):
    """Convert chronological item histories into SASRec training examples."""

    def __init__(
        self,
        *,
        user_histories: dict[str, list[str]],
        item_to_index: dict[str, int],
        max_sequence_length: int,
        pad_token_id: int = 0,
    ) -> None:
        self.examples: list[SASRecTrainingExample] = []
        self.max_sequence_length = max_sequence_length
        self.pad_token_id = pad_token_id

        for user_id, history in sorted(user_histories.items()):
            indexed_history = [item_to_index[item_id] for item_id in history if item_id in item_to_index]
            if len(indexed_history) < 2:
                continue

            for prediction_index in range(1, len(indexed_history)):
                raw_history = history[:prediction_index]
                raw_target = history[prediction_index]
                input_ids = _left_pad(
                    indexed_history[:prediction_index],
                    max_length=max_sequence_length,
                    pad_token_id=pad_token_id,
                )
                self.examples.append(
                    SASRecTrainingExample(
                        user_id=user_id,
                        history_item_ids=list(raw_history),
                        target_item_id=raw_target,
                        input_item_ids=input_ids,
                        label_item_id=indexed_history[prediction_index],
                    )
                )

    def __len__(self) -> int:
        """Return the number of next-item training examples."""

        return len(self.examples)

    def __getitem__(self, index: int) -> SASRecTrainingExample:
        """Return one SASRec training example."""

        return self.examples[index]

    @staticmethod
    def collate(batch: list[SASRecTrainingExample]) -> dict[str, torch.Tensor]:
        """Convert examples into tensors."""

        return {
            "item_sequences": torch.tensor([example.input_item_ids for example in batch], dtype=torch.long),
            "labels": torch.tensor([example.label_item_id for example in batch], dtype=torch.long),
        }


def _left_pad(sequence: Iterable[int], *, max_length: int, pad_token_id: int) -> list[int]:
    """Left-pad or left-truncate a sequence so the last position is recent."""

    values = list(sequence)[-max_length:]
    return [pad_token_id] * (max_length - len(values)) + values


def _build_item_index(
    *,
    user_histories: dict[str, list[str]],
    warm_examples: list[LeaveOneOutExample],
    cold_examples: list[LeaveOneOutExample],
) -> tuple[dict[str, int], dict[int, str]]:
    """Build a stable item index with zero reserved for padding."""

    item_ids = {
        item_id
        for history in user_histories.values()
        for item_id in history
    }
    item_ids.update(example.target_item_id for example in warm_examples)
    item_ids.update(example.target_item_id for example in cold_examples)

    item_to_index = {item_id: index for index, item_id in enumerate(sorted(item_ids), start=1)}
    index_to_item = {index: item_id for item_id, index in item_to_index.items()}
    return item_to_index, index_to_item


def _candidate_pool_for_example(
    *,
    candidate_item_ids: list[str],
    target_item_id: str,
    max_eval_candidates: int | None,
    seed: int,
    user_id: str,
) -> list[str]:
    """Return a deterministic candidate subset that always includes the target."""

    if max_eval_candidates is None or max_eval_candidates <= 0 or len(candidate_item_ids) <= max_eval_candidates:
        return candidate_item_ids

    cap = max(1, int(max_eval_candidates))
    negatives = [item_id for item_id in candidate_item_ids if item_id != target_item_id]
    digest = hashlib.sha256(f"{seed}:{user_id}:{target_item_id}".encode("utf-8")).hexdigest()
    rng = Random(int(digest[:16], 16))
    sampled_negative_count = max(0, cap - 1)
    if len(negatives) > sampled_negative_count:
        negatives = rng.sample(negatives, sampled_negative_count)
    return sorted([target_item_id, *negatives])


def _train_sasrec(
    *,
    model: SASRecModel,
    train_loader: DataLoader,
    learning_rate: float,
    epochs: int,
    device: torch.device,
    pad_token_id: int,
) -> list[float]:
    """Train SASRec with full-catalog next-item cross entropy."""

    if len(train_loader) == 0 or epochs <= 0:
        return []

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    loss_history: list[float] = []
    total_steps = len(train_loader) * epochs

    with make_progress_bar(total=total_steps, desc="SASRec train", unit="batch") as progress:
        for _epoch_index in range(epochs):
            batch_losses: list[float] = []
            for batch in train_loader:
                sequences = batch["item_sequences"].to(device=device, non_blocking=device.type == "cuda")
                labels = batch["labels"].to(device=device, non_blocking=device.type == "cuda")

                model.train()
                optimizer.zero_grad()
                logits = model.score_all_items(sequences)
                logits[:, pad_token_id] = torch.finfo(logits.dtype).min
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

                loss_value = float(loss.detach().cpu().item())
                batch_losses.append(loss_value)
                progress.set_postfix({"loss": f"{loss_value:.4f}"})
                progress.update(1)
            loss_history.append(mean(batch_losses) if batch_losses else 0.0)

    return loss_history


def _rank_sasrec_candidates(
    *,
    model: SASRecModel,
    example: LeaveOneOutExample,
    candidate_item_ids: list[str],
    item_to_index: dict[str, int],
    index_to_item: dict[int, str],
    max_sequence_length: int,
    device: torch.device,
    top_k: int,
    pad_token_id: int,
) -> tuple[list[str], float]:
    """Rank candidate items for one leave-one-out query."""

    history_indices = [item_to_index[item_id] for item_id in example.history_item_ids if item_id in item_to_index]
    input_ids = _left_pad(history_indices, max_length=max_sequence_length, pad_token_id=pad_token_id)
    candidate_indices = [item_to_index[item_id] for item_id in candidate_item_ids if item_id in item_to_index]
    if not candidate_indices:
        return [], 0.0

    started_at = time.perf_counter()
    sequence_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        scores = model.score_all_items(sequence_tensor)[0]
        scores[pad_token_id] = torch.finfo(scores.dtype).min
        candidate_tensor = torch.tensor(candidate_indices, dtype=torch.long, device=device)
        candidate_scores = scores.index_select(0, candidate_tensor)
        sorted_positions = torch.argsort(candidate_scores, descending=True)
        top_positions = sorted_positions[: min(top_k, sorted_positions.numel())].detach().cpu().tolist()
    latency_ms = (time.perf_counter() - started_at) * 1000.0

    ranked_indices = [candidate_indices[position] for position in top_positions]
    ranked_item_ids = [index_to_item[index] for index in ranked_indices]
    return ranked_item_ids, latency_ms


def _evaluate_sasrec(
    *,
    model: SASRecModel,
    examples: list[LeaveOneOutExample],
    candidate_item_ids: list[str],
    item_to_index: dict[str, int],
    index_to_item: dict[int, str],
    max_sequence_length: int,
    device: torch.device,
    top_k: int,
    max_eval_candidates: int | None,
    seed: int,
    split_name: str,
    pad_token_id: int = 0,
) -> dict[str, object]:
    """Evaluate SASRec on the same ranking metrics as MASI."""

    if not examples:
        return {
            f"hr@{top_k}": 0.0,
            f"ndcg@{top_k}": 0.0,
            f"coverage@{top_k}": 0.0,
            "avg_latency_ms": 0.0,
            "num_examples": 0,
            "candidate_item_count": len(candidate_item_ids),
            "max_eval_candidates": max_eval_candidates,
        }

    model.eval()
    ranked_lists: list[list[str]] = []
    hit_scores: list[float] = []
    ndcg_scores: list[float] = []
    latencies: list[float] = []

    with make_progress_bar(total=len(examples), desc=f"Evaluate SASRec {split_name}", unit="user") as progress:
        for example in examples:
            active_candidate_item_ids = _candidate_pool_for_example(
                candidate_item_ids=candidate_item_ids,
                target_item_id=example.target_item_id,
                max_eval_candidates=max_eval_candidates,
                seed=seed,
                user_id=example.user_id,
            )
            ranked_item_ids, latency_ms = _rank_sasrec_candidates(
                model=model,
                example=example,
                candidate_item_ids=active_candidate_item_ids,
                item_to_index=item_to_index,
                index_to_item=index_to_item,
                max_sequence_length=max_sequence_length,
                device=device,
                top_k=top_k,
                pad_token_id=pad_token_id,
            )
            ranked_lists.append(ranked_item_ids)
            hit_scores.append(hit_rate_at_k(ranked_item_ids=ranked_item_ids, target_item_id=example.target_item_id, k=top_k))
            ndcg_scores.append(ndcg_at_k(ranked_item_ids=ranked_item_ids, target_item_id=example.target_item_id, k=top_k))
            latencies.append(latency_ms)
            progress.set_postfix({"lat_ms": f"{latency_ms:.1f}", "candidates": len(active_candidate_item_ids)})
            progress.update(1)

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
        "candidate_item_count": len(candidate_item_ids),
        "max_eval_candidates": max_eval_candidates,
    }


def run_sasrec_baseline(
    *,
    user_histories: dict[str, list[str]],
    import_summary: dict[str, object],
    config: dict[str, object],
    device: torch.device,
    outputs_root: Path,
    checkpoint_root: Path | None = None,
) -> dict[str, object]:
    """Train and evaluate the SASRec baseline on MASI leave-one-out splits."""

    seed = int(config["seed"])
    baseline_config = dict(config.get("baseline", {}))
    pad_token_id = 0
    top_k = int(baseline_config.get("top_k", 10))
    max_sequence_length = int(baseline_config.get("max_sequence_length", 50))
    max_eval_candidates_raw = baseline_config.get("max_eval_candidates")
    max_eval_candidates = None if max_eval_candidates_raw in (None, 0, "0") else int(max_eval_candidates_raw)

    split = build_leave_one_out_split(
        user_histories=user_histories,
        cold_start_ratio=float(baseline_config.get("cold_start_ratio", 0.2)),
        min_train_history=int(baseline_config.get("min_train_history", 1)),
        seed=seed,
        use_cold_start_evaluation=bool(baseline_config.get("use_cold_start_evaluation", True)),
    )
    item_to_index, index_to_item = _build_item_index(
        user_histories=user_histories,
        warm_examples=split.warm_examples,
        cold_examples=split.cold_examples,
    )
    candidate_item_ids = sorted(item_to_index)

    train_dataset = SASRecTrainingDataset(
        user_histories=split.train_histories,
        item_to_index=item_to_index,
        max_sequence_length=max_sequence_length,
        pad_token_id=pad_token_id,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(baseline_config.get("batch_size", 256)),
        shuffle=len(train_dataset) > 0,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=train_dataset.collate,
    )

    model = SASRecModel(
        SASRecConfig(
            num_items=len(item_to_index) + 1,
            max_sequence_length=max_sequence_length,
            hidden_dim=int(baseline_config.get("hidden_dim", 128)),
            num_heads=int(baseline_config.get("num_heads", 4)),
            num_layers=int(baseline_config.get("num_layers", 3)),
            dropout=float(baseline_config.get("dropout", 0.1)),
            pad_token_id=pad_token_id,
        )
    )
    loss_history = _train_sasrec(
        model=model,
        train_loader=train_loader,
        learning_rate=float(baseline_config.get("learning_rate", 0.001)),
        epochs=int(baseline_config.get("epochs", 30)),
        device=device,
        pad_token_id=pad_token_id,
    )

    warm_metrics = _evaluate_sasrec(
        model=model,
        examples=split.warm_examples,
        candidate_item_ids=candidate_item_ids,
        item_to_index=item_to_index,
        index_to_item=index_to_item,
        max_sequence_length=max_sequence_length,
        device=device,
        top_k=top_k,
        max_eval_candidates=max_eval_candidates,
        seed=seed,
        split_name="warm",
        pad_token_id=pad_token_id,
    )
    cold_metrics = _evaluate_sasrec(
        model=model,
        examples=split.cold_examples,
        candidate_item_ids=candidate_item_ids,
        item_to_index=item_to_index,
        index_to_item=index_to_item,
        max_sequence_length=max_sequence_length,
        device=device,
        top_k=top_k,
        max_eval_candidates=max_eval_candidates,
        seed=seed,
        split_name="cold",
        pad_token_id=pad_token_id,
    )

    outputs_root = ensure_directory(outputs_root)
    item_mapping_path = write_json(
        {"item_to_index": item_to_index},
        outputs_root / "sasrec_item_mapping.json",
    )
    checkpoint_paths: dict[str, str] = {}
    if checkpoint_root is not None:
        checkpoint_root = ensure_directory(checkpoint_root)
        model_path = checkpoint_root / "sasrec_model.pt"
        torch.save(
            {
                "config": config,
                "model_state_dict": module_state_dict_to_cpu(model),
                "item_to_index": item_to_index,
                "loss_history": loss_history,
            },
            model_path,
        )
        checkpoint_paths["sasrec_model"] = str(model_path)

    summary = {
        "baseline": "sasrec",
        "baseline_type": "exact_sequential_id_baseline",
        "seed": seed,
        "device": str(device),
        "num_items": len(item_to_index),
        "num_train_examples": len(train_dataset),
        "training_status": "trained" if loss_history else "skipped_no_examples_or_epochs",
        "loss_history": loss_history,
        "split_summary": split.summary,
        "warm_metrics": warm_metrics,
        "cold_metrics": cold_metrics,
        "import_summary": import_summary,
        "item_mapping_path": str(item_mapping_path),
        "checkpoint_paths": checkpoint_paths,
        "baseline_config": baseline_config,
    }
    write_json(summary, outputs_root / "baseline_summary.json")
    return summary
