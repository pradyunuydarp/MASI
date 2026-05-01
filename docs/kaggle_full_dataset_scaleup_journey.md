# Kaggle Full-Dataset Scale-Up Journey

This note records the interpretation of the saved Kaggle output notebook
`Kaggle_interactions/masi-full-dataset.ipynb` and the next bounded scale-up
plan for `notebooks/08_full_dataset_pipeline.ipynb`.

## What The Saved Kaggle Run Actually Was

The run completed the full MASI pipeline, but it was intentionally capped by
Kaggle-safe limits. It should be treated as a smoke-quality result, not as the
final full-dataset result.

Observed runtime config:

```text
run_name = amazon_csj_full_dataset_kaggle_safe_train
max_users = 512
max_items = 1024
max_review_records = 2,000,000
clip.batch_size = 8
tokenization.epochs = 4
experiment.batch_size = 16
experiment.history_max_tokens = 96
experiment.mlm_epochs = 1
experiment.autoregressive_epochs = 2
```

Observed data after filtering:

```text
users = 373
items after token filter = 1008
train examples = 9585
MLM examples = 2016
warm examples = 306
cold examples = 67
cold items = 49
```

Observed metrics:

```text
warm HR@10 = 0.042483660130718956
warm NDCG@10 = 0.033215539108583256
warm Coverage@10 = 0.24702380952380953

cold HR@10 = 0.0
cold NDCG@10 = 0.0
cold Coverage@10 = 0.1626984126984127
```

Interpretation:

- Warm metrics are low but nonzero because the autoregressive recommender only
  trained for two epochs on a small user/item slice.
- Cold HR and NDCG stayed at zero because cold items are withheld from
  sequential fine-tuning, so they depend heavily on Phase 1 behavioral
  alignment, Phase 2 code quality, and Phase 3 cross-modal MLM. The run used
  only one MLM epoch and four tokenization epochs.
- Coverage is already nontrivial, which means retrieval is not collapsed to a
  tiny item set. The next priority is ranking quality.

## Where To Change Training Scale

For Kaggle, change only the first code cell in
`notebooks/08_full_dataset_pipeline.ipynb`.

The source config `configs/Full_dataset.json` remains the proposal-oriented
base. The notebook derives a runtime config and writes it to:

```text
/kaggle/working/masi_artifacts/configs/Full_dataset.kaggle_safe_runtime.json
```

The knobs are:

```python
KAGGLE_SAFE_PROFILE = "scaled_safe"
KAGGLE_SAFE_PROFILES = {
    "smoke_safe": {...},
    "scaled_safe": {...},
}
```

Use `smoke_safe` only to verify that the notebook still runs end to end. Use
`scaled_safe` for the next metric-improvement run.

## Next Safe Scaling Profile

The current recommended profile is:

```text
max_users = 4096
max_items = 8192
max_review_records = 20,000,000
clip.batch_size = 16
alignment.batch_size = 256
alignment.epochs = 5
alignment.learning_rate = 0.0005
alignment.hard_negative_count = 16
alignment.window_size = 3
tokenization.batch_size = 256
tokenization.epochs = 15
tokenization.learning_rate = 0.0005
experiment.batch_size = 32
experiment.history_max_tokens = 128
experiment.mlm_epochs = 5
experiment.autoregressive_epochs = 15
experiment.learning_rate = 0.0003
experiment.hidden_dim = 256
experiment.num_heads = 8
experiment.num_layers = 4
```

Why this profile:

- It increases users and items enough to make warm/cold splits more stable.
- It gives Phase 1 more behavioral signal before quantization.
- It gives RQ-VAE codebooks enough training time to form useful text/visual
  semantic IDs.
- It gives cross-modal MLM enough steps to help cold items, which are not
  present as next-item targets during sequential fine-tuning.
- It improves model capacity for warm ranking without trying the full
  proposal-scale `102400` users and `204800` items in one Kaggle session.

If Kaggle kills this profile with `SIGKILL: 9`, lower these first:

```text
max_items: 8192 -> 4096
max_users: 4096 -> 2048
clip.batch_size: 16 -> 8
tokenization.batch_size: 256 -> 128
```

Do not disable `USE_KAGGLE_SAFE_LIMITS` until sharded CLIP embedding
persistence exists.

## Expected Important Cell Outputs

After the first setup cell, the output should include the selected profile and
the resolved runtime config:

```text
Repository: /kaggle/working/MASI
Repo source: https://github.com/pradyunuydarp/MASI.git
Source cfg: /kaggle/working/MASI/configs/Full_dataset.json
Run config: /kaggle/working/masi_artifacts/configs/Full_dataset.kaggle_safe_runtime.json
Safe caps:  True
Safe profile: scaled_safe
{
  "max_users": 4096,
  "max_items": 8192,
  "max_review_records": 20000000,
  ...
}
Run root: /kaggle/working/masi_artifacts/outputs/amazon_csj_full_dataset_kaggle_scaled_safe_train
```

After the dataset validation cell:

```text
Prepared full_dataset input is complete.
- /kaggle/input/datasets/dheerajrajanala/masi-amazon-csj-full-dataset/Clothing_Shoes_and_Jewelry.jsonl
- /kaggle/input/datasets/dheerajrajanala/masi-amazon-csj-full-dataset/meta_Clothing_Shoes_and_Jewelry.jsonl
- /kaggle/input/datasets/dheerajrajanala/masi-amazon-csj-full-dataset/images
- /kaggle/input/datasets/dheerajrajanala/masi-amazon-csj-full-dataset/image_download_manifest.json
- /kaggle/input/datasets/dheerajrajanala/masi-amazon-csj-full-dataset/subset_manifest.json
```

After the training cell, the exact metric values will depend on runtime and
surviving filtered items, but the summary should show larger counts than the
smoke run:

```json
{
  "mlm_status": "trained",
  "generative_finetuning_status": "trained",
  "num_items": "... should be up to about 8192 after modality filtering ...",
  "num_train_examples": "... should be much larger than 9585 ...",
  "mlm_loss_history": ["five epoch values"],
  "autoregressive_loss_history": ["fifteen epoch values"],
  "warm_metrics": {
    "hr@10": "...",
    "ndcg@10": "...",
    "coverage@10": "...",
    "num_examples": "..."
  },
  "cold_metrics": {
    "hr@10": "...",
    "ndcg@10": "...",
    "coverage@10": "...",
    "num_examples": "..."
  }
}
```

After the metric inspection cell, copy the printed `warm_metrics`,
`cold_metrics`, `items_with_full_modalities`, and `run_manifest` path back into
this note or `docs/Kaggle_Notebook_logs.md`.

## Metric Improvement Expectations

Warm improvements should come mostly from:

- more users and sequence examples,
- longer autoregressive fine-tuning,
- larger recommender hidden size and depth.

Cold improvements should come mostly from:

- more items with complete text/image modalities,
- stronger behavior-aware alignment,
- better-trained independent RQ-VAE codebooks,
- more cross-modal MLM epochs.

If warm improves but cold remains near zero, keep the data scale fixed and run
ablation-focused retries:

```text
Phase 1 on/off
Phase 3 MLM on/off
text-only vs visual-only vs late-fused
RQ-VAE depth D in {2, 3, 4}
```

That will clarify whether cold-start weakness is caused by token quality, MLM
alignment, or the ranking/evaluation path.
