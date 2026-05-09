from random import Random

from masi.alignment.behavior_alignment import (
    _build_graph_neighbor_indices,
    _sample_hard_negative_indices,
)


def test_lazy_graph_negative_sampler_excludes_neighbors_without_materialized_pool() -> None:
    item_id_to_row = {f"item_{index}": index for index in range(8)}
    histories = {
        "user_a": ["item_0", "item_1", "item_2"],
        "user_b": ["item_0", "item_3"],
        "user_c": ["item_4", "item_5", "item_6"],
    }

    neighbor_indices, popularity_indices = _build_graph_neighbor_indices(
        user_histories=histories,
        item_id_to_row=item_id_to_row,
    )

    sampled = _sample_hard_negative_indices(
        item_index=item_id_to_row["item_0"],
        neighbor_indices=neighbor_indices,
        sample_size=3,
        rng=Random(7),
        popularity_indices=popularity_indices,
        fallback_indices=list(item_id_to_row.values()),
    )

    assert len(sampled) == 3
    assert item_id_to_row["item_0"] not in sampled
    assert not set(sampled).intersection(neighbor_indices[item_id_to_row["item_0"]])

