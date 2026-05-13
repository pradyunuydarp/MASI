# MASI Training Process: Latest Implemented Pipeline

This document explains the training process implemented by `notebooks/08_full_dataset_pipeline.ipynb` and the scripts it calls. It is based on the repository code, not only on the proposal text.

Primary implementation files:

- `notebooks/08_full_dataset_pipeline.ipynb`: run notebook for the full-dataset Kaggle workflow.
- `configs/Full_dataset.json`: base config from which the notebook derives runtime configs.
- `scripts/build_masi_tokens.py`: Phase 1 CLIP/alignment and Phase 2 RQ-VAE token generation.
- `scripts/run_masi_experiment.py`: Phase 3 MLM pretraining, autoregressive training, and evaluation.
- `src/masi/tokenization/masi_tokens.py`: CLIP feature extraction and fused semantic-ID export.
- `src/masi/alignment/behavior_alignment.py`: behavior-aware projection-head training.
- `src/masi/tokenization/rqvae.py`: independent text and vision RQ-VAE-style quantizers.
- `src/masi/recommender/sequence_data.py`: MLM and autoregressive dataset construction.
- `src/masi/recommender/mlm.py`: cross-modal masked-language model.
- `src/masi/recommender/generative.py`: autoregressive semantic-ID recommender.
- `src/masi/recommender/training.py`: Phase 3 training loops and MLM-to-recommender weight transfer.

## Executive Summary

The current pipeline is a staged multimodal recommendation system:

1. Extract text and image features with frozen pretrained CLIP.
2. Train small behavior-aware projection heads on top of CLIP embeddings.
3. Train two independent RQ-VAE-style quantizers: one for text, one for vision.
4. Convert each item into a late-fused semantic ID:

```text
[TXT] txt_c0_* txt_c1_* txt_c2_* [VIS] vis_c0_* vis_c1_* vis_c2_*
```

5. Train a Transformer MLM to reconstruct missing text or visual code spans.
6. Initialize the autoregressive recommender from the MLM weights and fine-tune it on chronological user histories.
7. Evaluate warm-start and cold-start ranking with `HR@10`, `NDCG@10`, `Coverage@10`, and latency.

The implementation is staged rather than end-to-end. Later stages consume frozen artifacts from earlier stages; gradients do not flow back from the recommender into CLIP, projection heads, or RQ-VAE codebooks.

## Model Classification

| Model/component | Purpose | Classification | What changes during training |
|---|---|---|---|
| `openai/clip-vit-base-patch32` CLIP text encoder | Converts product text metadata into dense vectors | Pretrained, used as-is | Nothing; loaded with `from_pretrained`, `eval()`, and `torch.no_grad()` |
| `openai/clip-vit-base-patch32` CLIP vision encoder | Converts item images into dense vectors | Pretrained, used as-is | Nothing |
| Text projection head | Maps CLIP text embeddings into behavior-aware space | From scratch, trained locally | Two-layer MLP text-head weights |
| Vision projection head | Maps CLIP image embeddings into behavior-aware space | From scratch, trained locally | Two-layer MLP vision-head weights |
| Behavior-aware alignment model | Holds the text and vision projection heads and trains them with collaborative InfoNCE | Transfer learning style adapter training | Only projection heads change; CLIP remains frozen |
| Text RQ-VAE | Quantizes aligned text embeddings into text code sequences | From scratch | Text encoder, text decoder, and text residual codebooks |
| Vision RQ-VAE | Quantizes aligned visual embeddings into visual code sequences | From scratch | Vision encoder, vision decoder, and vision residual codebooks |
| Text and vision residual codebooks | Discrete semantic identifier dictionaries | From scratch | Codebook vectors are trained, then optionally refit with residual k-means |
| Cross-modal MLM Transformer | Learns text-to-vision and vision-to-text token reconstruction | From scratch | Token embeddings, positional embeddings, Transformer layers, output head |
| Generative SID recommender | Generates/scores next-item semantic-ID tokens from user history | Transfer learning from internal MLM, then fine-tuned | Initialized from MLM weights when available, then trained autoregressively |

There is no pretrained BERT, T5, LLM, RecBole model, or pretrained recommender being loaded in Phase 3.

## Notation

The rest of this document uses these symbols:

| Symbol | Meaning |
|---|---|
| `U` | Number of users after filtering |
| `N` | Number of usable items after modality filtering |
| `B` | Batch size for the current stage |
| `D_clip` | CLIP output embedding size; `512` for `openai/clip-vit-base-patch32` |
| `d_p` | Behavior-aware projection dimension |
| `d_z` | RQ-VAE latent dimension |
| `D` | RQ-VAE residual depth, currently `3` |
| `K` | Codebook size per residual level, currently `128` in `Full_dataset.json` |
| `M` | Number of graph-hard negatives per anchor item |
| `V` | Recommender token vocabulary size |
| `H` | Recommender Transformer hidden dimension |
| `L` | Number of Transformer layers |
| `S_h` | Maximum serialized history length in tokens |
| `S_t` | Maximum target item length in tokens |
| `S_mlm` | Maximum MLM sequence length |

With the default late-fused depth `D = 3`, each item has:

```text
item_token_length = 2 modality markers + 3 text codes + 3 visual codes = 8
S_t = item_token_length + <EOS> = 9
S_mlm = <BOS> + item_token_length + <EOS> = 10
```

## End-To-End Run Order In `08_full_dataset_pipeline.ipynb`

The notebook is a run script around the repository modules. Its main sequence is:

1. Clone or locate the MASI repo.
2. Load `configs/Full_dataset.json`.
3. On Kaggle, derive a safe runtime config from the selected profile. The current default profile in the notebook is `long_safe`.
4. Optionally restore a previous run bundle and continue from checkpoints.
5. Preload/export CLIP into a local Hugging Face model directory.
6. Validate attached reviews, metadata, images, and manifests.
7. Write resolved configs for Phase 1/2 and Phase 3.
8. Run `scripts/build_masi_tokens.py`.
9. Run `scripts/run_masi_experiment.py`.
10. Write run summaries, checkpoint inventories, and export bundles.

The handoff artifact from Phase 1/2 into Phase 3 is:

```text
phase12_tokens/fused_semantic_ids.jsonl
```

Each JSONL row has:

```json
{
  "item_id": "B000...",
  "text_codes": ["txt_c0_17", "txt_c1_4", "txt_c2_63"],
  "visual_codes": ["vis_c0_2", "vis_c1_91", "vis_c2_8"]
}
```

## Phase 0: Dataset Inputs And Filtering

The notebook expects the prepared Amazon Reviews 2023 `Clothing_Shoes_and_Jewelry` bundle:

- `Clothing_Shoes_and_Jewelry.jsonl`
- `meta_Clothing_Shoes_and_Jewelry.jsonl`
- `images/`
- `image_download_manifest.json`
- `subset_manifest.json`

The data stage selects a bounded user/item slice and builds chronological user histories:

```text
user_histories: {user_id -> [item_1, item_2, ..., item_T]}
```

Filtering includes:

- minimum user interactions,
- minimum item interactions,
- max users/items/review records,
- optional review-record and rank offsets for chunked continuation runs,
- modality filtering so items used by MASI have the enabled text/image coverage.

The selected item set is later reduced to items that survive text/image feature extraction and user-history trimming.

## Phase 1A: Frozen CLIP Feature Extraction

Implemented by `encode_clip_embeddings()` in `src/masi/tokenization/masi_tokens.py`.

### Purpose

CLIP converts raw item content into dense text and image embeddings. This gives MASI content-based item representations so cold-start items can be encoded without interaction history.

### Inputs

This is a multi-input stage:

- text input: deterministic product text built from title, review text when present, descriptions, features, categories, brand, store, and selected details such as material/color;
- image input: one RGB product image per item.

### Model

```python
CLIPModel.from_pretrained(model_source)
CLIPProcessor.from_pretrained(model_source)
model.eval()
```

Feature extraction runs under `torch.no_grad()`.

### Dimensions

For a batch of `B` items:

```text
text strings                         -> CLIP processor -> CLIP text encoder  -> X_txt in R^{B x 512}
RGB images                           -> CLIP processor -> CLIP image encoder -> X_vis in R^{B x 512}
normalized per-item text embedding   -> x_txt,i in R^{512}
normalized per-item visual embedding -> x_vis,i in R^{512}
```

`D_clip = 512` because the configured CLIP variant is `openai/clip-vit-base-patch32`.

### What Changes

Nothing changes inside CLIP. The output dictionaries are cached in memory and passed to Phase 1B:

```text
text_embeddings:  {item_id -> R^{512}}
image_embeddings: {item_id -> R^{512}}
```

## Phase 1B: Behavior-Aware Contrastive Alignment

Implemented by `train_behavior_aware_alignment()` in `src/masi/alignment/behavior_alignment.py`.

### Purpose

CLIP is content-oriented. The proposal argues that item representations should also respect user behavior. This phase trains separate text and vision projection heads using collaborative positive pairs from user histories.

### Inputs

This is also a multi-input stage:

- text CLIP embeddings, one vector per item;
- image CLIP embeddings, one vector per item;
- user histories for graph-derived positives and negatives.

Positive item pairs are extracted from each user history using a forward window. If a user clicked:

```text
[i_1, i_2, i_3, i_4]
```

and `window_size = 2`, examples include `(i_1, i_2)`, `(i_1, i_3)`, `(i_2, i_3)`, `(i_2, i_4)`, etc.

Hard negatives come from popular non-neighbor items in the user-item graph, and in-batch positives also act as negatives.

### Model

There are two independent projection heads:

```text
Proj_txt: Linear(512, d_p) -> GELU -> Dropout -> Linear(d_p, d_p) -> L2 normalize
Proj_vis: Linear(512, d_p) -> GELU -> Dropout -> Linear(d_p, d_p) -> L2 normalize
```

The model container is `BehaviorAwareAlignmentModel`, which holds both heads.

### Dimensions

For a training batch of `B` positive item pairs and `M` hard negatives:

```text
A_txt: text anchors       R^{B x 512}
P_txt: text positives     R^{B x 512}
N_txt: text negatives     R^{B x M x 512}

A_vis: image anchors      R^{B x 512}
P_vis: image positives    R^{B x 512}
N_vis: image negatives    R^{B x M x 512}

Z_txt,a = Proj_txt(A_txt) R^{B x d_p}
Z_txt,p = Proj_txt(P_txt) R^{B x d_p}
Z_txt,n = Proj_txt(N_txt) R^{B x M x d_p}

Z_vis,a = Proj_vis(A_vis) R^{B x d_p}
Z_vis,p = Proj_vis(P_vis) R^{B x d_p}
Z_vis,n = Proj_vis(N_vis) R^{B x M x d_p}
```

### Loss

The implementation trains text and vision independently with the same collaborative supervision:

```text
L_beh = L_txt + L_vis
```

Each loss is InfoNCE with:

- one positive item,
- in-batch negatives,
- graph-hard negatives,
- temperature `tau`.

### What Changes

Only the projection heads are updated. CLIP is frozen, and the source CLIP embeddings are not updated.

After training, the stage writes aligned embeddings:

```text
aligned_text_embeddings:  {item_id -> R^{d_p}}
aligned_image_embeddings: {item_id -> R^{d_p}}
```

If behavior alignment is disabled, or if no positive pairs survive filtering, the implementation bypasses training and returns normalized input embeddings.

### Epoch Meaning

One alignment epoch is one shuffled pass over all collaborative positive item pairs. The number of optimizer steps is:

```text
ceil(num_positive_pairs / alignment.batch_size) * alignment.epochs
```

## Phase 2: Independent Text And Vision RQ-VAE Training

Implemented by `train_rqvae_model()` in `src/masi/tokenization/rqvae.py`.

### Purpose

This stage converts continuous aligned embeddings into discrete semantic identifiers. Text and vision are intentionally quantized separately to avoid early-fusion modality collapse.

### Inputs

This is not a multi-input model. It is run twice as two separate single-modality trainings:

```text
Text RQ-VAE input:   E_txt in R^{N x d_p}
Vision RQ-VAE input: E_vis in R^{N x d_p}
```

Each item embedding is one training example.

### Model

Each modality has its own model:

```text
Encoder:   Linear(d_p, d_z) -> GELU -> Linear(d_z, d_z)
Quantizer: D residual codebooks, each K vectors in R^{d_z}
Decoder:   Linear(d_z, d_z) -> GELU -> Linear(d_z, d_p)
```

There is no parameter sharing between the text and visual RQ-VAE models.

### Dimensions

For a mini-batch:

```text
X       R^{B x d_p}      aligned modality embedding
Z       R^{B x d_z}      encoder latent
Q       R^{B x d_z}      residual-quantized latent
X_hat   R^{B x d_p}      decoder reconstruction
C       {0..K-1}^{B x D} code indices
```

For the current `Full_dataset.json`, `d_p = 192`, `d_z = 192`, `D = 3`, `K = 128`.

### Residual Quantization

For each latent vector `z`:

1. Codebook level 0 selects the nearest vector to `z`.
2. The selected vector is subtracted to form a residual.
3. Codebook level 1 quantizes the residual.
4. The process repeats for `D` levels.

The item receives `D` text codes and `D` visual codes:

```text
S_txt,i = <ct_0, ct_1, ct_2>
S_vis,i = <cv_0, cv_1, cv_2>
```

### Loss

```text
L_rqvae = MSE(X_hat, X)
        + commitment_weight * (MSE(Z, stopgrad(Q)) + MSE(Q, stopgrad(Z)))
```

### What Changes

For each modality, the trainable parameters are:

- encoder weights,
- decoder weights,
- residual codebook vectors.

CLIP, projection heads, and the source aligned embeddings are not updated during RQ-VAE training.

### Residual K-Means Refit

The current `Full_dataset.json` sets:

```text
refit_codebooks_with_residual_kmeans = true
```

After neural RQ-VAE training, the implementation optionally refits residual codebooks over the learned latent vectors. This is a bounded-run stability decision to avoid degenerate assignments. It is not an extra external model.

### Epoch Meaning

One RQ-VAE epoch is one shuffled pass over all item embeddings for that modality. Text and vision RQ-VAEs have separate epoch loops.

## Phase 2 Output: Late-Fused Semantic IDs

Implemented by `build_fused_ids_from_quantized_codes()` in `src/masi/tokenization/masi_tokens.py`.

For each item:

```text
S_final,i = [TXT] S_txt,i [VIS] S_vis,i
```

With `D = 3`, this becomes:

```text
[TXT] txt_c0_a txt_c1_b txt_c2_c [VIS] vis_c0_x vis_c1_y vis_c2_z
```

The vocabulary starts with seven special tokens:

```text
<PAD>, <BOS>, <EOS>, <MASK>, <SEP>, [TXT], [VIS]
```

The maximum possible code-token vocabulary under late fusion is:

```text
V_code_max = 2 modalities * D levels * K codes
V_total_max = 7 + V_code_max
```

For `D = 3`, `K = 128`:

```text
V_total_max = 7 + 2 * 3 * 128 = 775
```

The actual `V` can be smaller if not every code appears in the selected subset.

## Phase 3A: Cross-Modal MLM Pretraining

Implemented by `CrossModalMLMDataset` and `CrossModalMLMPretrainer`.

### Purpose

Separate codebooks do not automatically teach the recommender that text and visual tokens describe the same item. The MLM stage trains a Transformer to reconstruct masked text tokens from visual context and masked visual tokens from text context.

### Inputs

This is an item-level token task, not a raw text/image task. For each item, the dataset creates two examples:

1. `text_to_visual`: text codes visible, visual codes masked.
2. `visual_to_text`: visual codes visible, text codes masked.

Modality markers stay visible.

For one item with `D = 3`:

```text
Original item tokens:
[TXT] t0 t1 t2 [VIS] v0 v1 v2

text_to_visual input:
<BOS> [TXT] t0 t1 t2 [VIS] <MASK> <MASK> <MASK> <EOS>

visual_to_text input:
<BOS> [TXT] <MASK> <MASK> <MASK> [VIS] v0 v1 v2 <EOS>
```

Labels are `-100` at ignored positions and the true token ID at masked positions.

### Model

```text
TokenEmbedding(V, H)
PositionEmbedding(S_mlm, H)
TransformerEncoder(L layers, H hidden size, num_heads heads, FFN size 4H)
LayerNorm(H)
Linear(H, V)
```

This is a custom PyTorch Transformer trained from scratch.

### Dimensions

```text
input_token_ids      N^{B x S_mlm}
token embeddings     R^{B x S_mlm x H}
encoded states       R^{B x S_mlm x H}
logits               R^{B x S_mlm x V}
labels               N^{B x S_mlm}, with -100 ignored
```

For the current default item length, `S_mlm = 10`. In the current `long_safe` notebook profile, `H = 256`, `L = 4`, and `num_heads = 8`.

### Loss

```text
CrossEntropy(logits.reshape(-1, V), labels.reshape(-1), ignore_index=-100)
```

### What Changes

The MLM model updates:

- token embeddings,
- positional embeddings,
- Transformer encoder weights,
- output layer norm,
- output projection.

The semantic IDs, RQ-VAE codebooks, projection heads, and CLIP remain unchanged.

### Epoch Meaning

One MLM epoch is one pass over the MLM examples. With both modalities enabled, the dataset normally has about `2N` MLM examples.

## Phase 3B: Autoregressive Generative Recommendation

Implemented by `GenerativeSequenceDataset`, `GenerativeSIDRecommender`, and `run_training_epochs()`.

### Purpose

This is the downstream recommender. It learns to predict the semantic-ID token sequence of the next item from a user's chronological interaction history.

### Inputs

This is a single integer-token stream model. It does not see raw text, images, or dense embeddings.

For each user history:

```text
[i_1, i_2, ..., i_T]
```

the dataset creates next-item examples for each prediction index:

```text
history items: [i_1, ..., i_t]
target item:   i_{t+1}
```

History serialization:

```text
<BOS> S_final,i1 <SEP> S_final,i2 <SEP> ... S_final,it <SEP>
```

Target serialization:

```text
S_final,target <EOS>
```

The training script concatenates them:

```text
model_input = [history_token_ids, target_token_ids]
labels      = [history_token_ids, target_token_ids]
```

The loss is shifted by one token, so position `t` predicts token `t+1`.

### Model

```text
TokenEmbedding(V, H)
PositionEmbedding(S_h + S_t, H)
Causal TransformerEncoder(L layers, H hidden size, num_heads heads, FFN size 4H)
LayerNorm(H)
Linear(H, V)
```

The implementation uses `nn.TransformerEncoder` with a causal mask, so it behaves like a decoder-only autoregressive model.

### Transfer From MLM

If no recommender checkpoint is restored and MLM is enabled, `initialize_generative_from_mlm()` copies:

- token embeddings,
- overlapping positional embeddings,
- compatible Transformer weights,
- output norm,
- output projection.

Then autoregressive fine-tuning continues from those copied weights.

If a `generative_recommender.pt` checkpoint is restored, the restored checkpoint takes priority over MLM initialization.

### Dimensions

```text
history_token_ids    N^{B x S_h}
target_token_ids     N^{B x S_t}
model_input          N^{B x (S_h + S_t)}
hidden states        R^{B x (S_h + S_t) x H}
logits               R^{B x (S_h + S_t) x V}
shifted logits       R^{B x (S_h + S_t - 1) x V}
shifted labels       N^{B x (S_h + S_t - 1)}
```

For the current `long_safe` notebook profile:

```text
S_h = 160
S_t = 9
S_h + S_t = 169
H = 256
L = 4
num_heads = 8
```

### Loss

```text
CrossEntropy(logits[:, :-1, :], labels[:, 1:], ignore_index=<PAD>)
```

### What Changes

The recommender updates:

- token embeddings,
- positional embeddings,
- causal Transformer weights,
- output layer norm,
- output projection.

The already generated semantic-ID table is not changed.

### Epoch Meaning

One autoregressive epoch is one pass over all next-item training examples created from the train histories.

## Evaluation Picture

The experiment script builds a deterministic leave-one-out split:

- the last item in each eligible user history becomes the evaluation target;
- a configured fraction of target items is assigned to the cold-start item set;
- cold items are removed from train prefixes;
- warm and cold examples are evaluated separately.

Ranking is implemented by scoring candidate item token sequences with the trained generative model. The full-dataset notebook uses candidate caps and candidate batches for tractable Kaggle runs.

Metrics:

- `HR@10`: whether the target item appears in the top 10.
- `NDCG@10`: hit quality with rank discount.
- `Coverage@10`: fraction of catalog items appearing in top-10 recommendations.
- `avg_latency_ms`: average ranking latency per evaluation example.

## What "Epoch" Means Across The Whole Pipeline

The same word refers to different datasets at different stages:

| Stage | Is it trained? | One epoch means |
|---|---:|---|
| CLIP encoding | No | No epoch; one feature extraction pass over items |
| Behavior alignment | Yes | One pass over collaborative positive item pairs |
| Text RQ-VAE | Yes | One pass over text-side item embeddings |
| Vision RQ-VAE | Yes | One pass over visual-side item embeddings |
| Cross-modal MLM | Yes | One pass over item-level masked examples, usually two per item |
| Autoregressive recommender | Yes | One pass over next-item examples from train histories |
| Evaluation | No | No epoch; one scoring pass over warm/cold evaluation examples |

## Base Hyperparameters In `configs/Full_dataset.json`

These are the base values before the Kaggle-safe notebook overrides them.

| Area | Hyperparameter | Value |
|---|---|---:|
| Global | `seed` | `7` |
| Runtime | `device` | `auto` |
| Runtime | `log_device_summary` | `true` |
| Runtime | `run_name` | `amazon_csj_full_dataset_train` |
| Runtime | `resume_if_artifacts_exist` | `true` |
| Dataset | `min_user_interactions` | `5` |
| Dataset | `min_item_interactions` | `5` |
| Dataset | `max_users` | `102400` |
| Dataset | `max_items` | `204800` |
| Dataset | `max_review_records` | `20000000` |
| Dataset | `review_record_offset` | `0` |
| Dataset | `user_rank_offset` | `0` |
| Dataset | `item_rank_offset` | `0` |
| Dataset | `collapse_consecutive_duplicates` | `false` |
| Assets | `download_missing_images` | `true` |
| Assets | `image_download_workers` | `16` |
| Assets | `image_download_retries` | `2` |
| Assets | `image_download_timeout_seconds` | `30` |
| Assets | `image_download_resume` | `true` |
| CLIP | `model_name` | `openai/clip-vit-base-patch32` |
| CLIP | `batch_size` | `32` |
| Alignment | `projection_dim` (`d_p`) | `192` |
| Alignment | `epochs` | `2` |
| Alignment | `batch_size` | `256` |
| Alignment | `learning_rate` | `0.001` |
| Alignment | `temperature` | `0.07` |
| Alignment | `hard_negative_count` (`M`) | `8` |
| Alignment | `window_size` | `2` |
| Alignment | `dropout` | `0.1` |
| Alignment | `keep_embeddings_on_device` | `true` |
| Tokenization | `latent_dim` (`d_z`) | `192` |
| Tokenization | `depth` (`D`) | `3` |
| Tokenization | `codebook_size` (`K`) | `128` |
| Tokenization | `epochs` | `8` |
| Tokenization | `batch_size` | `256` |
| Tokenization | `learning_rate` | `0.0008` |
| Tokenization | `commitment_weight` | `0.25` |
| Tokenization | `refit_codebooks_with_residual_kmeans` | `true` |
| Experiment | `history_max_tokens` (`S_h`) | `128` |
| Experiment | `target_max_tokens` | `null`, resolved to at least `9` |
| Experiment | `mlm_max_tokens` | `null`, resolved to at least `10` |
| Experiment | `batch_size` | `32` |
| Experiment | `learning_rate` | `0.0008` |
| Experiment | `hidden_dim` (`H`) | `128` |
| Experiment | `num_heads` | `4` |
| Experiment | `num_layers` (`L`) | `3` |
| Experiment | `dropout` | `0.1` |
| Experiment | `mlm_epochs` | `2` |
| Experiment | `autoregressive_epochs` | `3` |
| Experiment | `top_k` | `10` |
| Experiment | `max_eval_candidates` | `2048` |
| Experiment | `eval_candidate_batch_size` | `128` |
| Experiment | `cold_start_ratio` | `0.2` |
| Experiment | `min_train_history` | `1` |
| Experiment | `min_sequence_items` | `2` |
| Checkpointing | `alignment_save_steps` | `25` |
| Checkpointing | `text_rqvae_save_steps` | `25` |
| Checkpointing | `vision_rqvae_save_steps` | `25` |
| Checkpointing | `mlm_save_steps` | `25` |
| Checkpointing | `autoregressive_save_steps` | `25` |
| Checkpointing | `keep_last` | `3` |

All current main method toggles in `Full_dataset.json` are enabled:

```text
use_behavior_alignment = true
use_text_modality = true
use_visual_modality = true
use_late_fusion = true
use_cross_modal_mlm = true
use_generative_finetuning = true
use_cold_start_evaluation = true
```

## Actual Current Notebook Defaults: `long_safe`

On Kaggle, `08_full_dataset_pipeline.ipynb` sets:

```text
USE_KAGGLE_SAFE_LIMITS = RUNNING_ON_KAGGLE
KAGGLE_SAFE_PROFILE = "long_safe"
CONTINUE_TRAINING_FROM_CHECKPOINTS = true
AUTO_RESTORE_RESUME_BUNDLE = true
KAGGLE_CHECKPOINT_SAVE_STEPS = 25
ADVANCE_DATA_CHUNK_EACH_RUN = true
DATA_CHUNK_BY = "user_rank"
```

When running on Kaggle, `long_safe` overrides the base config as follows:

| Area | Hyperparameter | `long_safe` value |
|---|---|---:|
| Runtime | `run_name` | `amazon_csj_full_dataset_kaggle_long_safe_train` |
| Dataset | `max_users` | `12288` |
| Dataset | `max_items` | `24576` |
| Dataset | `max_review_records` | `50000000` |
| CLIP | `batch_size` | `16` |
| Alignment | `batch_size` | `256` |
| Alignment | `epochs` | `10` |
| Alignment | `learning_rate` | `0.0004` |
| Alignment | `hard_negative_count` | `24` |
| Alignment | `window_size` | `4` |
| Tokenization | `batch_size` | `256` |
| Tokenization | `epochs` | `30` |
| Tokenization | `learning_rate` | `0.0004` |
| Experiment | `batch_size` | `32` |
| Experiment | `history_max_tokens` | `160` |
| Experiment | `mlm_epochs` | `10` |
| Experiment | `autoregressive_epochs` | `30` |
| Experiment | `learning_rate` | `0.00025` |
| Experiment | `hidden_dim` | `256` |
| Experiment | `num_heads` | `8` |
| Experiment | `num_layers` | `4` |
| Experiment | `max_eval_candidates` | `1024` |
| Experiment | `eval_candidate_batch_size` | `128` |
| Checkpointing | all stage save intervals | `25` |
| Checkpointing | `restore_from_checkpoints` | `true` |

The notebook also defines smaller profiles:

| Profile | Users | Items | Review scan cap | Alignment epochs | RQ-VAE epochs | MLM epochs | AR epochs | Hidden/layers/heads |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `smoke_safe` | `512` | `1024` | `2000000` | base `2` | `4` | `1` | `2` | `128/3/4` |
| `scaled_safe` | `4096` | `8192` | `20000000` | `5` | `15` | `5` | `15` | `256/4/8` |
| `long_safe` | `12288` | `24576` | `50000000` | `10` | `30` | `10` | `30` | `256/4/8` |

## Checkpoints And Continuation

The full-dataset notebook is designed for long Kaggle sessions where interruption is possible.

Final checkpoints:

```text
checkpoints/phase12_tokens/behavior_alignment.pt
checkpoints/phase12_tokens/text_rqvae.pt
checkpoints/phase12_tokens/vision_rqvae.pt
checkpoints/phase3_experiment/cross_modal_mlm.pt
checkpoints/phase3_experiment/generative_recommender.pt
```

Periodic checkpoints:

```text
checkpoints/phase12_tokens/behavior_alignment_steps/step_*.pt
checkpoints/phase12_tokens/text_rqvae_steps/step_*.pt
checkpoints/phase12_tokens/vision_rqvae_steps/step_*.pt
checkpoints/phase3_experiment/cross_modal_mlm_steps/step_*.pt
checkpoints/phase3_experiment/generative_recommender_steps/step_*.pt
```

Each periodic directory writes `latest.json`. When continuation is enabled, the scripts restore the newest usable final or periodic checkpoint and continue optimizer steps from the restored global step when optimizer state is available.

## Parameter Scale Under The Current Main Config

These are useful for understanding model size. They are derived from the implemented modules.

Alignment projection heads with `d_p = 192`:

```text
params = 2 * (512*d_p + d_p*d_p + 2*d_p)
       = 271,104 trainable parameters
```

One RQ-VAE with `d_p = d_z = 192`, `D = 3`, `K = 128`:

```text
params = 4*d_p^2 + D*K*d_p + 4*d_p
       = 221,952 trainable parameters per modality
```

Both text and vision RQ-VAEs together:

```text
443,904 trainable parameters
```

Maximum recommender vocabulary with `D = 3`, `K = 128`:

```text
V_total_max = 775
```

For `long_safe` with `H = 256`, `L = 4`, `S_mlm = 10`, and `S_gen = 169`, the MLM and generative Transformers are still small research models, not LLMs. Their parameter counts are in the low millions, dominated by Transformer layers and vocabulary projections.

## Practical Interpretation

The current implemented pipeline is best understood as:

- pretrained CLIP gives general image/text understanding;
- projection heads adapt those features to collaborative behavior;
- separate RQ-VAEs discretize text and vision without letting one modality dominate the other;
- late-fused semantic IDs turn each item into a short token sequence;
- MLM pretraining teaches the recommender how text and visual code spans correspond;
- autoregressive fine-tuning teaches next-item sequence prediction;
- evaluation checks whether the resulting token model ranks held-out warm and cold items.

The key engineering choice is modularity. The pipeline trades end-to-end gradient flow for reproducible checkpoints, inspectable intermediate artifacts, and the ability to resume each stage independently in Kaggle sessions.
