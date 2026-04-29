# MASI Training Process

This note explains the implemented MASI training pipeline as it exists in this repository. It answers the main model-training questions directly: which models are pretrained, which models are trained locally, what each stage receives as input, what parameters change, and how the configured epochs and hyperparameters fit together.

Primary implementation files:

- `scripts/train_masi.py`: one-click orchestrator for Phase 1/2 token building and Phase 3 recommendation.
- `scripts/build_masi_tokens.py`: CLIP embedding extraction, behavior-aware projection training, RQ-VAE training, and fused semantic-ID export.
- `scripts/run_masi_experiment.py`: cross-modal MLM pretraining, autoregressive fine-tuning, and ranking evaluation.
- `src/masi/tokenization/masi_tokens.py`: CLIP encoding and late-fused token serialization.
- `src/masi/alignment/behavior_alignment.py`: behavior-aware contrastive projection heads.
- `src/masi/tokenization/rqvae.py`: independent text and image RQ-VAE-style quantizers.
- `src/masi/recommender/mlm.py`: cross-modal masked-token pretrainer.
- `src/masi/recommender/generative.py`: autoregressive semantic-ID recommender.

## Short Answer

The repository uses **pretrained frozen CLIP** weights for both text and image encoding. It does not train CLIP from scratch and does not fine-tune CLIP. The configured model is loaded with `CLIPModel.from_pretrained("openai/clip-vit-base-patch32")`, which downloads the pretrained Hugging Face model into the local Hugging Face cache if it is not already cached. The weights are then used under `torch.no_grad()` in evaluation mode.

The repository **does train** the behavior-aware projection heads on top of CLIP embeddings. These heads are small two-layer MLPs, one for text and one for images. They are the only trainable part of Phase 1.

The repository **does train** separate RQ-VAE-style models for text and images. These are not pretrained external models. There is one text quantizer and one vision quantizer, each with its own encoder, residual codebooks, and decoder.

The final MLM and generative recommendation models are **custom PyTorch Transformers trained from scratch** in this repository. There is no pretrained BERT, T5, LLM, or RecBole model being loaded for Phase 3. If cross-modal MLM is enabled, its learned weights are copied into the generative recommender before autoregressive fine-tuning.

## End-To-End Training Order

The one-click launcher runs the implemented phases in this order:

1. Resolve dataset paths, image cache paths, output directories, and checkpoint directories.
2. Build MASI tokens with `scripts/build_masi_tokens.py`.
3. Run the recommendation experiment with `scripts/run_masi_experiment.py`.
4. Write resolved configs, summaries, checkpoints, and a `run_manifest.json`.

The main handoff artifact between Phase 1/2 and Phase 3 is:

```text
fused_semantic_ids.jsonl
```

Each row contains:

```json
{
  "item_id": "...",
  "text_codes": ["txt_c0_...", "txt_c1_...", "txt_c2_..."],
  "visual_codes": ["vis_c0_...", "vis_c1_...", "vis_c2_..."]
}
```

The recommender renders that as:

```text
[TXT] txt_c0_* txt_c1_* txt_c2_* [VIS] vis_c0_* vis_c1_* vis_c2_*
```

For the current default depth `D = 3`, one item has `2D + 2 = 8` fused tokens before `<EOS>` is added for target generation.

## Phase 0: Dataset And Inputs

The configured dataset is Amazon Reviews 2023 `Clothing_Shoes_and_Jewelry`. The bounded configs use a subset-first contract: reviews, metadata, and images are prepared before training where possible.

Standard raw inputs:

- review JSONL: chronological user-item interactions with `user_id`, `parent_asin`, `timestamp`, review text, and sometimes review images.
- metadata JSONL: product-side title, description, features, categories, brand/store/details, and image URLs.
- images directory or image cache: one usable image path per selected item.

The data code applies:

- user/item k-core filtering,
- configured caps such as `max_users`, `max_items`, and `max_review_records`,
- item filtering to keep records with the required text/image modality coverage.

Notation used below:

- `N`: number of usable items after filtering.
- `U`: number of users after filtering.
- `B`: batch size.
- `D_clip = 512`: CLIP projected embedding dimension for `openai/clip-vit-base-patch32`.
- `d_p`: behavior-aware projection dimension.
- `d_z`: RQ-VAE latent dimension.
- `D`: residual codebook depth.
- `K`: codebook size per residual level.
- `V`: recommender vocabulary size.
- `H`: recommender hidden dimension.
- `L`: number of Transformer layers.
- `S_h`: maximum serialized history length in tokens.
- `S_t`: maximum target length in tokens.

## Phase 1A: Frozen CLIP Encoding

Implemented in `encode_clip_embeddings()` in `src/masi/tokenization/masi_tokens.py`.

The repository obtains CLIP from the configured pretrained model name, not from a repo-local checkpoint. If the model is not already present in the Hugging Face cache, Transformers downloads it on first use. The code path is:

```python
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()
```

For all text and image batches, the code runs inside:

```python
with torch.no_grad():
    ...
```

So CLIP receives inputs and produces embeddings, but CLIP weights are not updated.

Text input construction:

- concatenates title, review text if present, first description fragments, first features, first categories, selected details such as brand/material/color, brand, and store;
- uses deterministic formatting so the same item record gives the same CLIP text input.

Text tensor flow:

- raw strings: `B` item descriptions,
- CLIP processor output: token IDs and masks, internally padded/truncated by the CLIP processor,
- CLIP text output: `X_txt in R^{B x 512}`,
- normalized output stored per item: `x_txt,i in R^{512}`.

Image tensor flow:

- raw PIL RGB images: `B` images,
- CLIP processor output: pixel tensors resized/cropped according to CLIP processor defaults,
- CLIP image output: `X_vis in R^{B x 512}`,
- normalized output stored per item: `x_vis,i in R^{512}`.

CLIP model specifications from the configured Hugging Face model:

- model: `openai/clip-vit-base-patch32`,
- output projection dimension: `512`,
- text Transformer: hidden size `512`, `12` layers, `8` heads, max positions `77`, vocab size `49,408`,
- vision Transformer: hidden size `768`, `12` layers, `12` heads, image size `224`, patch size `32`,
- approximate parameter count: `151M`,
- trainable in this repo: `0` CLIP parameters.

## Phase 1B: Behavior-Aware Contrastive Alignment

Implemented in `src/masi/alignment/behavior_alignment.py`.

Inputs:

- frozen CLIP text embeddings: `{item_id: x_txt,i}`, each `x_txt,i in R^{512}`,
- frozen CLIP image embeddings: `{item_id: x_vis,i}`, each `x_vis,i in R^{512}`,
- chronological user histories: `{user_id: [item_1, item_2, ...]}`.

Positive pairs:

- `build_positive_item_pairs()` walks each user history with a forward window.
- If `window_size = 2`, item `i_t` is paired with `i_{t+1}` and `i_{t+2}` when available.
- These pairs approximate collaborative co-interest.

Hard negative pool:

- graph neighbors are items co-occurring in a user's history;
- hard negatives are frequent items that are not graph neighbors of the anchor item;
- in-batch negatives are also used through the InfoNCE batch matrix.

Trainable model:

```text
Proj_txt: Linear(512, d_p) -> GELU -> Dropout -> Linear(d_p, d_p) -> L2 normalize
Proj_vis: Linear(512, d_p) -> GELU -> Dropout -> Linear(d_p, d_p) -> L2 normalize
```

Forward dimensions:

- text anchor batch: `A_txt in R^{B x 512}`,
- text positive batch: `P_txt in R^{B x 512}`,
- text hard negatives: `N_txt in R^{B x M x 512}`,
- projected anchor: `Z_txt,a in R^{B x d_p}`,
- projected positive: `Z_txt,p in R^{B x d_p}`,
- projected negatives: `Z_txt,n in R^{B x M x d_p}`,
- same shape pattern for image.

Loss:

```text
L_beh = L_txt + L_vis
```

Each loss is InfoNCE over one positive, in-batch negatives, and graph-hard negatives, with temperature `tau`.

What changes during training:

- trainable: text projection head weights, image projection head weights;
- frozen: CLIP text encoder, CLIP vision encoder, raw extracted CLIP embeddings, dataset records.

Parameter count formula:

```text
params_alignment = 2 * (512*d_p + d_p*d_p + 2*d_p)
```

Examples:

- smoke/demo `d_p = 128`: `164,352` trainable parameters.
- subset/full-dataset bounded configs `d_p = 192`: `271,104` trainable parameters.
- deferred raw full config `d_p = 256`: `394,240` trainable parameters.

Epoch behavior:

- one alignment epoch means one full pass over the shuffled collaborative positive-pair list;
- each optimization step processes up to `alignment.batch_size` positive item pairs;
- if no positive pairs survive filtering, the stage is skipped and normalized CLIP embeddings are passed forward.

## Phase 2: Independent Text And Image RQ-VAE Training

Implemented in `src/masi/tokenization/rqvae.py`.

The text and image RQ-VAE-style models are trained locally. They are not downloaded pretrained models and they are not frozen CLIP-like backbones.

There are two independent models:

```text
RQ-VAE_txt trained on aligned text embeddings
RQ-VAE_vis trained on aligned image embeddings
```

Each model has:

```text
Encoder: Linear(d_p, d_z) -> GELU -> Linear(d_z, d_z)
Residual quantizer: D codebooks, each with K vectors in R^{d_z}
Decoder: Linear(d_z, d_z) -> GELU -> Linear(d_z, d_p)
```

Input/output dimensions for one modality:

- aligned embeddings: `E in R^{N x d_p}`,
- mini-batch: `X in R^{B x d_p}`,
- latent vectors: `Z in R^{B x d_z}`,
- residual quantized latent: `Q in R^{B x d_z}`,
- reconstructed embeddings: `X_hat in R^{B x d_p}`,
- code indices: `C in {0, ..., K-1}^{B x D}`.

Residual quantization:

1. Level `0` chooses nearest code vector for `Z`.
2. Subtract that vector to form a residual.
3. Level `1` quantizes the residual.
4. Repeat until `D` code indices are produced.

Training loss:

```text
L_rqvae = MSE(X_hat, X) + commitment_weight * (
    MSE(Z, stopgrad(Q)) + MSE(Q, stopgrad(Z))
)
```

What changes during training:

- trainable: encoder weights, decoder weights, residual codebook vectors;
- separate text and visual codebooks never share parameters;
- frozen: CLIP encoders and already-produced projected embeddings.

Parameter count formula for one modality:

```text
params_rqvae_one = (d_p*d_z + d_z)
                 + (d_z*d_z + d_z)
                 + (D*K*d_z)
                 + (d_z*d_z + d_z)
                 + (d_z*d_p + d_p)
```

When `d_p = d_z`, this simplifies to:

```text
params_rqvae_one = 4*d_p^2 + D*K*d_p + 4*d_p
```

Examples:

- smoke/demo `d_p=d_z=128`, `D=3`, `K=64`: `90,624` parameters per modality, `181,248` for both.
- subset/full-dataset bounded configs `d_p=d_z=192`, `D=3`, `K=128`: `221,952` per modality, `443,904` for both.
- deferred raw full config `d_p=d_z=256`, `D=3`, `K=256`: `459,776` per modality, `919,552` for both.

Epoch behavior:

- one RQ-VAE epoch means one full shuffled pass over all usable item embeddings for one modality;
- text RQ-VAE and image RQ-VAE are trained separately;
- each mini-batch contains up to `tokenization.batch_size` item embeddings.

Important bounded-run decision:

- bounded configs set `refit_codebooks_with_residual_kmeans = true`;
- after neural autoencoder training, the code refits residual codebooks with simple residual k-means over latent vectors;
- this was added because tiny bounded subsets produced degenerate identical code assignments with the pure trainable-codebook path;
- the deferred raw full config sets this to `false`, which is closer to the proposal's default trainable RQ-VAE path.

## Phase 2 Output: Late-Fused Semantic IDs

Implemented in `build_fused_ids_from_quantized_codes()` in `src/masi/tokenization/masi_tokens.py`.

For item `i`, the trained quantizers produce:

```text
S_txt,i = <ct_1, ct_2, ..., ct_D>
S_vis,i = <cv_1, cv_2, ..., cv_D>
```

The final MASI ID is:

```text
S_final,i = [TXT] S_txt,i [VIS] S_vis,i
```

The concrete token strings include modality and residual level:

```text
txt_c0_17, txt_c1_4, txt_c2_63
vis_c0_2,  vis_c1_91, vis_c2_8
```

Maximum possible code-token vocabulary from the codebooks is:

```text
V_code_max = 2 * D * K
V_total_max = 7 special tokens + V_code_max
```

The seven special tokens are:

```text
<PAD>, <BOS>, <EOS>, <MASK>, <SEP>, [TXT], [VIS]
```

For `D = 3`, `K = 128`, `V_total_max = 775`, although the actual vocabulary can be smaller if not every code appears in the bounded item subset.

## Phase 3A: Cross-Modal MLM Pretraining

Implemented in `src/masi/recommender/mlm.py` and `CrossModalMLMDataset` in `src/masi/recommender/sequence_data.py`.

This model is trained from scratch. It is a local Transformer encoder, not a pretrained BERT or LLM.

Model:

```text
TokenEmbedding(V, H)
PositionEmbedding(S_mlm, H)
TransformerEncoder with L layers, H hidden size, num_heads heads, FFN size 4H
LayerNorm(H)
Linear(H, V)
```

MLM examples:

- two deterministic examples are created per item when both text and image modalities are enabled;
- `text_to_visual`: text tokens remain visible, visual code tokens are replaced with `<MASK>`, labels are only visual tokens;
- `visual_to_text`: visual tokens remain visible, text code tokens are replaced with `<MASK>`, labels are only text tokens;
- modality markers `[TXT]` and `[VIS]` remain visible.

For `D = 3`, the item token length is `8`, and MLM sequence length resolves to:

```text
S_mlm = item_token_length + 2 = 10
```

Tensor dimensions:

- MLM input IDs: `I_mlm in N^{B x S_mlm}`,
- token embeddings: `R^{B x S_mlm x H}`,
- output logits: `R^{B x S_mlm x V}`,
- labels: `N^{B x S_mlm}`, with `-100` for ignored positions.

Loss:

```text
CrossEntropy(logits, labels), ignore_index = -100
```

What changes during MLM:

- trainable: MLM token embeddings, positional embeddings, Transformer layers, output norm, output projection;
- unchanged: fused semantic IDs, RQ-VAE codebooks, CLIP, projection heads.

After MLM, `initialize_generative_from_mlm()` copies shared weights into the generative recommender:

- token embeddings,
- position embeddings where lengths overlap,
- compatible Transformer layer weights,
- output norm,
- output projection.

If `use_cross_modal_mlm = false` or no MLM examples exist, this initialization is skipped.

## Phase 3B: Autoregressive Generative Fine-Tuning

Implemented in `src/masi/recommender/generative.py` and `GenerativeSequenceDataset` in `src/masi/recommender/sequence_data.py`.

This model is also a custom PyTorch model trained in this repository. It uses `nn.TransformerEncoder` with a causal mask, giving decoder-like autoregressive behavior.

Model:

```text
TokenEmbedding(V, H)
PositionEmbedding(S_h + S_t, H)
Causal TransformerEncoder with L layers, H hidden size, num_heads heads, FFN size 4H
LayerNorm(H)
Linear(H, V)
```

Training examples:

- for each user history `[i_1, i_2, ..., i_T]`,
- each prediction index `t >= 1` creates:
  - history items `[i_1, ..., i_t]`,
  - target item `i_{t+1}`.

History serialization:

```text
<BOS> S_final,i1 <SEP> S_final,i2 <SEP> ... S_final,it <SEP>
```

Target serialization:

```text
S_final,target <EOS>
```

For `D = 3`, the minimum target budget resolves to:

```text
S_t = item_token_length + 1 = 9
```

The training script concatenates history and target:

```text
input = [history_token_ids, target_token_ids]
labels = same sequence
```

Then the loss shifts by one token:

```text
logits[:, :-1, :] predict labels[:, 1:]
```

Tensor dimensions:

- generative input IDs: `I_gen in N^{B x (S_h + S_t)}`,
- hidden states: `R^{B x (S_h + S_t) x H}`,
- output logits: `R^{B x (S_h + S_t) x V}`,
- shifted loss logits: `R^{B x (S_h + S_t - 1) x V}`.

What changes during fine-tuning:

- trainable: generative token embeddings, positional embeddings, Transformer layers, output norm, output projection;
- unchanged: generated semantic-ID table, CLIP, projection heads, RQ-VAE quantizers.

Inference/evaluation:

- ranking uses likelihood scoring over candidate item token sequences;
- the current bounded evaluator exhaustively scores the bounded candidate catalog;
- metrics are `HR@10`, `NDCG@10`, `Coverage@10`, and average latency.

## Recommender Parameter Counts

The MLM and generative models have the same parameter structure except for positional embedding length. For the implemented PyTorch Transformer layer, an approximate exact formula is:

```text
params_transformer_model =
    2*V*H + V                    # token embedding plus output projection
  + S*H                          # positional embedding
  + 2*H                          # output LayerNorm
  + L*(12*H^2 + 13*H)            # TransformerEncoder layers with FFN size 4H
```

Here `S = S_mlm` for the MLM model and `S = S_h + S_t` for the generative model.

Upper-bound examples using the maximum code vocabulary:

| Config family | `D,K` | `V_total_max` | `H,L` | MLM params | Generative params |
|---|---:|---:|---:|---:|---:|
| smoke/demo | `3,64` | `391` | `64,2` | about `151K` | about `155K` with `S_h=64` |
| subset medium/large/full_dataset | `3,128` | `775` | `128,3` | about `796K` | about `812K` with `S_h=128` |
| deferred raw full | `3,256` | `1543` | `256,4` | about `3.95M` | about `4.02M` with `S_h=256` |

The actual parameter count can be slightly lower only if the active vocabulary is smaller than `V_total_max`.

## Configured Hyperparameters

The most important configs are:

- `configs/masi_train_csj_subset_medium.json`: faster bounded prepared-subset run.
- `configs/masi_train_csj_subset_large.json`: canonical bounded prepared-subset run.
- `configs/Full_dataset.json`: larger local prepared-subset run under `data/full_dataset`.
- `configs/masi_train_csj_full.json`: deferred raw full-CSJ reference path.
- `configs/masi_train_csj_smoke.json`: smallest integration smoke path.

### Smoke Config

From `configs/masi_train_csj_smoke.json`:

- seed: `7`
- dataset: `min_user_interactions=5`, `min_item_interactions=5`, `max_users=64`, `max_items=128`, `max_review_records=50000`
- CLIP: `openai/clip-vit-base-patch32`, batch size `8`
- alignment: `d_p=128`, epochs `2`, batch size `64`, learning rate `0.001`, temperature `0.07`, hard negatives `4`, window size `2`, dropout `0.1`
- tokenization: `d_z=128`, `D=3`, `K=64`, epochs `8`, batch size `64`, learning rate `0.002`, commitment weight `0.25`, residual k-means refit enabled
- recommender: `S_h=64`, batch size `8`, learning rate `0.001`, `H=64`, heads `4`, layers `2`, dropout `0.1`, MLM epochs `2`, autoregressive epochs `3`, `top_k=10`, cold-start ratio `0.25`

### Subset Medium Config

From `configs/masi_train_csj_subset_medium.json`:

- seed: `7`
- dataset: `min_user_interactions=5`, `min_item_interactions=5`, `max_users=256`, `max_items=512`, `max_review_records=150000`
- CLIP: `openai/clip-vit-base-patch32`, batch size `16`
- alignment: `d_p=192`, epochs `2`, batch size `128`, learning rate `0.001`, temperature `0.07`, hard negatives `8`, window size `2`, dropout `0.1`
- tokenization: `d_z=192`, `D=3`, `K=128`, epochs `6`, batch size `128`, learning rate `0.0008`, commitment weight `0.25`, residual k-means refit enabled
- recommender: `S_h=128`, batch size `16`, learning rate `0.0008`, `H=128`, heads `4`, layers `3`, dropout `0.1`, MLM epochs `2`, autoregressive epochs `2`, `top_k=10`, cold-start ratio `0.2`
- checkpoints: every `25` steps for alignment, both RQ-VAEs, MLM, and autoregressive fine-tuning; keep last `3`

### Subset Large Config

From `configs/masi_train_csj_subset_large.json`:

- seed: `7`
- dataset: `min_user_interactions=5`, `min_item_interactions=5`, `max_users=512`, `max_items=1024`, `max_review_records=400000`
- CLIP: `openai/clip-vit-base-patch32`, batch size `16`
- alignment: `d_p=192`, epochs `2`, batch size `128`, learning rate `0.001`, temperature `0.07`, hard negatives `8`, window size `2`, dropout `0.1`
- tokenization: `d_z=192`, `D=3`, `K=128`, epochs `8`, batch size `128`, learning rate `0.0008`, commitment weight `0.25`, residual k-means refit enabled
- recommender: `S_h=128`, batch size `16`, learning rate `0.0008`, `H=128`, heads `4`, layers `3`, dropout `0.1`, MLM epochs `2`, autoregressive epochs `3`, `top_k=10`, cold-start ratio `0.2`
- checkpoints: every `25` steps; keep last `3`

### Full Dataset Prepared-Subset Config

From `configs/Full_dataset.json`:

- seed: `7`
- dataset: `max_users=102400`, `max_items=204800`, `max_review_records=20000000`
- CLIP: `openai/clip-vit-base-patch32`, batch size `32`
- alignment: `d_p=192`, epochs `2`, batch size `256`, learning rate `0.001`, temperature `0.07`, hard negatives `8`, window size `2`, dropout `0.1`
- tokenization: `d_z=192`, `D=3`, `K=128`, epochs `8`, batch size `256`, learning rate `0.0008`, commitment weight `0.25`, residual k-means refit enabled
- recommender: `S_h=128`, batch size `32`, learning rate `0.0008`, `H=128`, heads `4`, layers `3`, dropout `0.1`, MLM epochs `2`, autoregressive epochs `3`, `top_k=10`, cold-start ratio `0.2`
- checkpoints: every `100` steps; keep last `3`

### Deferred Raw Full-CSJ Config

From `configs/masi_train_csj_full.json`:

- seed: `7`
- dataset: no configured user/item/review caps; raw review and metadata downloads enabled if missing
- CLIP: `openai/clip-vit-base-patch32`, batch size `32`
- alignment: `d_p=256`, epochs `4`, batch size `256`, learning rate `0.001`, temperature `0.07`, hard negatives `16`, window size `2`, dropout `0.1`
- tokenization: `d_z=256`, `D=3`, `K=256`, epochs `20`, batch size `256`, learning rate `0.0002`, commitment weight `0.25`, residual k-means refit disabled
- recommender: `S_h=256`, batch size `64`, learning rate `0.0005`, `H=256`, heads `8`, layers `4`, dropout `0.1`, MLM epochs `5`, autoregressive epochs `10`, `top_k=10`, cold-start ratio `0.2`

## Method Toggles

The training configs expose ablation switches through `method_toggles`:

- `use_behavior_alignment`: train or bypass Phase 1 projection heads.
- `use_text_modality`: include or drop text codes.
- `use_visual_modality`: include or drop visual codes.
- `use_late_fusion`: include or omit `[TXT]` and `[VIS]` markers.
- `use_cross_modal_mlm`: run or skip MLM pretraining.
- `use_generative_finetuning`: run or skip autoregressive fine-tuning.
- `use_cold_start_evaluation`: create or skip zero-shot cold-start split.

All current main training configs set these to `true`.

## What "Epoch" Means In Each Stage

The word epoch refers to a different dataset depending on the stage:

- CLIP encoding has no training epoch. It is a feature extraction pass over item batches.
- Alignment epoch: one pass over collaborative positive item pairs from user histories.
- Text RQ-VAE epoch: one pass over text-side item embeddings.
- Image RQ-VAE epoch: one pass over image-side item embeddings.
- MLM epoch: one pass over item-level cross-modal mask examples, usually two examples per item.
- Autoregressive epoch: one pass over next-item training examples generated from chronological user histories.

The stages are sequential. Later stages consume frozen artifacts from earlier stages. The current implementation does not jointly update CLIP, projection heads, RQ-VAE codebooks, and recommender weights in one end-to-end backpropagation graph.

## Comparison With The Proposal PDF And Design Docs

The proposal defines three main phases:

1. frozen CLIP encoders plus trainable modality-specific projection heads,
2. independent text and visual RQ-VAE codebooks with late fusion,
3. cross-modal MLM pretraining followed by sequential fine-tuning.

The repository implements that structure:

- CLIP is frozen and loaded from pretrained `openai/clip-vit-base-patch32`;
- text and image projection heads are trained with behavior-aware InfoNCE losses;
- text and image RQ-VAE-style quantizers are trained independently;
- semantic IDs are late-fused with `[TXT]` and `[VIS]`;
- cross-modal MLM and autoregressive fine-tuning are implemented with local Transformer modules;
- warm-start and zero-shot cold-start leave-one-out evaluation exists for bounded catalogs.

Important implementation decisions and deviations:

- The proposal mentioned implementing Phase 3 within RecBole. The repository uses custom PyTorch modules first. This is documented in `docs/technical_design.md`; RecBole is treated as an optional future adapter.
- The proposal targets roughly `100,000` active users and `50,000` items with valid high-resolution images. The active workflow is currently subset-first and bounded because raw CSJ data, metadata, images, and checkpoints are too large for a simple ephemeral/local workflow.
- The bounded configs use residual k-means refitting after RQ-VAE training to avoid degenerate assignments on small subsets. This is a practical bounded-run fix and not the default conceptual RQ-VAE path in the proposal.
- The current full-catalog evaluation is not yet production-scale. Ranking exhaustively scores bounded candidate catalogs, and `TODO_TASKS.md` still tracks scalable retrieval as a blocker for full Amazon-scale evaluation.
- The target baselines `SASRec`, `CEMG`, `MGR-LF++`, and `DIGER` are not fully reproduced yet. The repo contains a SASRec-style module, while the multimodal baselines remain pending.

The main conclusion is that the repository has implemented the proposal's core model-training architecture, but the current verified workflow is a bounded research pipeline rather than the final full-scale experimental run promised by the report.
