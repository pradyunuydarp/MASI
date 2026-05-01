import torch

from masi.recommender.generative import GenerativeSIDRecommender


def test_greedy_decode_uses_sliding_context_for_long_prefixes() -> None:
    model = GenerativeSIDRecommender(
        vocab_size=16,
        max_sequence_length=8,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        pad_token_id=0,
    )
    prefix = torch.arange(20, dtype=torch.long).remainder(16).unsqueeze(0)

    generated = model.greedy_decode(
        prefix_token_ids=prefix,
        max_new_tokens=3,
        stop_token_id=15,
    )

    assert generated.shape[0] == 1
    assert generated.shape[1] >= prefix.shape[1] + 1
    assert generated.shape[1] <= prefix.shape[1] + 3
    torch.testing.assert_close(generated[:, : prefix.shape[1]], prefix)
