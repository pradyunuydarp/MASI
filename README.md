# MASI

MASI stands for **Multimodal-Augmented Semantic Identifiers for Cold-Start Discovery in Generative Recommendation**. This repository is the implementation workspace for the research proposal in [Multimodal_Augmented_Semantic_Identifiers_for_Cold_Start_Discovery_in_Generative_Recommendation__Research_Proposal.pdf](/Users/pradyundevarakonda/Developer/MASI/Multimodal_Augmented_Semantic_Identifiers_for_Cold_Start_Discovery_in_Generative_Recommendation__Research_Proposal.pdf).

The project targets a core limitation in modern generative recommendation: text-only semantic identifiers underperform in visual-heavy domains such as fashion, short-form video, and e-commerce. MASI proposes a multimodal, behavior-aware alternative that preserves visual and textual signal separately during quantization, then aligns them for sequential generation.

## Research Goal

Build and evaluate a generative recommendation pipeline that improves **zero-shot cold-start discovery** by:

- learning behavior-aware text and image representations with frozen CLIP encoders and trainable projection heads,
- quantizing text and vision embeddings through **independent RQ-VAE codebooks**,
- forming a late-fused semantic identifier sequence with modality markers,
- pretraining a recommender Transformer to align cross-modal token structure before sequential fine-tuning.

The intended outcome is stronger cold-start hit rate and broader recommendation coverage than text-only or early-fusion baselines.

## Proposed MASI Pipeline

The proposal defines three phases:

1. **Behavior-Aware Contrastive Alignment**
   Learn separate projected text and image embeddings using collaborative positives from the user-item graph and mixed negative sampling.
2. **Independent Modality Quantization**
   Train separate text and vision RQ-VAE stacks and build final semantic IDs as:
   `[TXT] <text codes> [VIS] <vision codes>`
3. **Recommender-Side Cross-Modal Alignment**
   Pretrain the sequence model with cross-modal masked token reconstruction, then fine-tune on chronological user interaction sequences.

## Planned Dataset And Evaluation

- Dataset: Amazon Reviews 2023, Clothing/Shoes/Jewelry subset, filtered with a 5-core strategy
- Primary setting: leave-one-out evaluation
- Splits: warm-start and zero-shot cold-start
- Metrics: `HR@10`, `NDCG@10`, `Coverage@10`, and inference latency

## Expected Baselines

The proposal compares MASI against:

- `SASRec`
- `CEMG`
- `MGR-LF++`
- `DIGER`

These baselines define the evaluation bar for cold-start hit rate, coverage, and efficiency.

## Repository Documents

- [AGENTS.md](/Users/pradyundevarakonda/Developer/MASI/AGENTS.md): operating instructions for AI agents working in this repository
- [TODO_TASKS.md](/Users/pradyundevarakonda/Developer/MASI/TODO_TASKS.md): living project tracker maintained by AI agents
- [Multimodal_Augmented_Semantic_Identifiers_for_Cold_Start_Discovery_in_Generative_Recommendation__Research_Proposal.pdf](/Users/pradyundevarakonda/Developer/MASI/Multimodal_Augmented_Semantic_Identifiers_for_Cold_Start_Discovery_in_Generative_Recommendation__Research_Proposal.pdf): source proposal
- [docs/technical_design.md](/Users/pradyundevarakonda/Developer/MASI/docs/technical_design.md): proposal-to-code translation with module boundaries and artifact contracts
- [docs/reference_repos.md](/Users/pradyundevarakonda/Developer/MASI/docs/reference_repos.md): working map of foundational papers and public repositories relevant to MASI
- [docs/masi_implementation_note.tex](/Users/pradyundevarakonda/Developer/MASI/docs/masi_implementation_note.tex): LaTeX implementation note covering completed work, rationale, and engineering challenges

## Current Repository Structure

- `configs/`: JSON configs for reproducible preprocessing and, later, training runs
- `data/raw/`: expected location for source datasets and local demo inputs
- `data/processed/`: filtered outputs ready for later CLIP extraction and graph construction
- `docs/`: technical notes and future deviation logs
- `notebooks/`: demo notebooks that exercise repository workflows
- `outputs/`: manifests, summaries, and future experiment results
- `scripts/`: CLI entrypoints that wrap reusable library code
- `src/masi/`: library modules for data, alignment, tokenization, and recommender stages

## Hardware Assumptions

- CLIP extraction and RQ-VAE training run on `cuda`, `mps`, or `cpu`, but the bounded subset configs are GPU-preferred for practical turnaround.
- The later-stage Phase 3 modules now use `cuda` when available, otherwise `mps`, otherwise `cpu`.
- On Apple Silicon, the Phase 3 Transformer stack disables PyTorch's nested-tensor fast path so MLM pretraining and generative ranking can run on `mps` without falling back to CPU.
- The canonical workflow is now subset-first: prepare a bounded CSJ dataset locally, download images locally on CPU, upload that prepared dataset to Kaggle, and train from the prepared slice without redownloading the full raw files or image set.
- Kaggle remains a bounded-run target rather than a raw full-CSJ target because the raw review dump, raw metadata dump, and full image cache do not fit safely inside one ephemeral session.

## Subset-First CSJ Training

The repository now treats a prepared sliced Amazon Reviews 2023 `Clothing_Shoes_and_Jewelry` dataset as the main training input for bounded MASI runs.

Main artifacts:

- [scripts/prepare_amazon_csj_subset.py](/home/dheerajKDE/Documents/College/sem8/Rec_sys/MASI/scripts/prepare_amazon_csj_subset.py)
- [scripts/download_amazon_csj_subset_images.py](/home/dheerajKDE/Documents/College/sem8/Rec_sys/MASI/scripts/download_amazon_csj_subset_images.py)
- [scripts/train_masi.py](/Users/pradyundevarakonda/Developer/MASI/scripts/train_masi.py)
- [scripts/download_amazon_csj_images.py](/Users/pradyundevarakonda/Developer/MASI/scripts/download_amazon_csj_images.py)
- [configs/masi_train_csj_subset_medium.json](/home/dheerajKDE/Documents/College/sem8/Rec_sys/MASI/configs/masi_train_csj_subset_medium.json)
- [configs/masi_train_csj_subset_large.json](/home/dheerajKDE/Documents/College/sem8/Rec_sys/MASI/configs/masi_train_csj_subset_large.json)
- [configs/masi_train_csj_smoke.json](/Users/pradyundevarakonda/Developer/MASI/configs/masi_train_csj_smoke.json)
- [configs/masi_train_csj_medium_colab.json](/Users/pradyundevarakonda/Developer/MASI/configs/masi_train_csj_medium_colab.json)

Canonical workflow:

1. download raw CSJ reviews and metadata locally,
2. prepare a deterministic bounded subset locally,
3. download one validated image per selected item locally on CPU,
4. upload the prepared subset dataset to Kaggle,
5. run `train_masi.py` against the prepared dataset and reuse attached images directly.

What the launcher does:

- resolves a storage root for local, Kaggle, or Colab runs,
- discovers prepared subset datasets under `/kaggle/input/<slug>` or `/kaggle/input/datasets/<user>/<slug>`,
- falls back to local configured dataset paths when running outside Kaggle,
- reuses preloaded read-only images from the attached prepared dataset and only downloads missing items into the writable cache,
- runs Phase 1 alignment, Phase 2 dual quantization, and Phase 3 experiment training in order,
- writes resolved stage configs, periodic step checkpoints when configured, final checkpoints, summaries, and a top-level run manifest.

Recommended bounded progression:

- `smoke`: fastest integration check, may skip alignment and fine-tuning if too few multimodal items survive
- `subset_medium`: faster bounded iteration target for Kaggle and local regression checks
- `subset_large`: canonical bounded training target sized for a single Kaggle session with resume support available
- `medium_colab`: older bounded Colab path retained for comparison and non-Kaggle experimentation

Verified launcher artifact:

- [outputs/amazon_csj_smoke_train/run_manifest.json](/Users/pradyundevarakonda/Developer/MASI/outputs/amazon_csj_smoke_train/run_manifest.json)

## Kaggle Prepared Dataset Contract

The standard prepared subset dataset layout is:

- `Clothing_Shoes_and_Jewelry.jsonl`
- `meta_Clothing_Shoes_and_Jewelry.jsonl`
- `images/`
- `subset_manifest.json`
- `image_download_manifest.json`

Recommended Kaggle Dataset names:

- `masi-amazon-csj-subset-medium`
- `masi-amazon-csj-subset-large`

What the Kaggle path does:

- discovers the prepared subset dataset by slug across Kaggle's direct and nested dataset mount layouts,
- bootstraps a fresh writable repo checkout from `https://github.com/pradyunuydarp/MASI.git` into `/kaggle/working/MASI`,
- validates preloaded images from the attached dataset and downloads only missing ones into `/kaggle/working/masi_artifacts/data/processed/...`,
- runs the same `train_masi.py` launcher against the prepared subset config,
- packages the resulting manifests, checkpoints, and any writable-cache artifacts into one zip bundle that can be reattached to resume a later session.

Where Kaggle checkpoints go:

- the default storage root on Kaggle is `/kaggle/working/masi_artifacts`,
- a medium run writes checkpoints under `/kaggle/working/masi_artifacts/outputs/amazon_csj_subset_medium_train/checkpoints/`,
- a large run writes checkpoints under `/kaggle/working/masi_artifacts/outputs/amazon_csj_subset_large_train/checkpoints/`,
- final stage checkpoints stay at the phase root, such as `phase12_tokens/behavior_alignment.pt` and `phase3_experiment/generative_recommender.pt`,
- periodic step checkpoints are retained in sibling directories such as `phase12_tokens/behavior_alignment_steps/step_0000025.pt` and `phase3_experiment/generative_recommender_steps/step_0000025.pt`.

How not to lose them after the session ends:

- anything under `/kaggle/working` is ephemeral until you explicitly persist it,
- use the notebook's zip-bundle packaging step and either `Save Version` on the Kaggle notebook or publish the bundle as a private Kaggle Dataset,
- on the next session, reattach that resume dataset and unpack it back into `/kaggle/working/masi_artifacts` before rerunning the launcher.

## Deferred Full-Corpus Path

The raw full-CSJ launcher and configs are retained only as a deferred reference path. They are no longer the main documented workflow because full-corpus MASI still requires additional infrastructure beyond a bounded single-machine or Kaggle session setup.

Minimum practical requirements for the deferred full path:

- sharded storage outside git for raw reviews, metadata, image caches, embeddings, checkpoints, and run artifacts,
- resumable preprocessing and training stages with manifest-based progress tracking,
- deterministic split generation and seed handling across shards,
- batched or indexed retrieval for evaluation instead of the current exhaustive per-item scoring path,
- a GPU-preferred training environment with enough disk budget for cached multimodal features.

## Implemented First Slice

The repository now contains a runnable Phase 1 preparation slice that is aligned with the proposal's first dependency chain:

- typed dataset and path contracts,
- raw metadata and interaction validation,
- iterative k-core filtering,
- deterministic dataset manifest generation,
- a synthetic demo path that works without downloading Amazon Reviews 2023 yet,
- a Jupyter notebook that demonstrates the same flow.

Produced artifacts after running the demo:

- `data/raw/amazon_reviews_2023/demo_metadata.jsonl`
- `data/raw/amazon_reviews_2023/demo_reviews.jsonl`
- `data/processed/demo/metadata.filtered.jsonl`
- `data/processed/demo/interactions.filtered.jsonl`
- `outputs/demo/demo_summary.json`
- `outputs/demo/dataset_manifest.json`

## Recommender Foundation

The repository now also contains the first recommender-side foundation for MASI, grounded in the public baseline ecosystem around `SASRec`, semantic-ID generation, and autoregressive generative recommendation.

Implemented artifacts:

- [src/masi/recommender/vocabulary.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/vocabulary.py): late-fused semantic-ID token contract with `[TXT]` and `[VIS]`
- [src/masi/recommender/sequence_data.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/sequence_data.py): user-history serialization and cross-modal MLM datasets
- [src/masi/recommender/sasrec.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/sasrec.py): SASRec-style sequential baseline
- [src/masi/recommender/generative.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/generative.py): decoder-style semantic-ID generator inspired by TIGER
- [src/masi/recommender/mlm.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/mlm.py): cross-modal masked-token pretraining model
- [src/masi/recommender/training.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/training.py): training helpers for demo runs and early regression checks
- [scripts/demo_recommender_foundation.py](/Users/pradyundevarakonda/Developer/MASI/scripts/demo_recommender_foundation.py): synthetic end-to-end recommender demo
- [notebooks/02_recommender_foundation_demo.ipynb](/Users/pradyundevarakonda/Developer/MASI/notebooks/02_recommender_foundation_demo.ipynb): demo notebook for the recommender stack

Verified demo artifact:

- [outputs/recommender_demo/recommender_demo_summary.json](/Users/pradyundevarakonda/Developer/MASI/outputs/recommender_demo/recommender_demo_summary.json)

## Phase 1 + Phase 2 MASI Tokens

The repository now has an actual bounded implementation of the proposal's first two modeling stages:

1. frozen CLIP text and vision encoders,
2. behavior-aware contrastive alignment with separate text and image projection heads,
3. independent modality quantization with separate residual codebooks,
4. late fusion into `[TXT] <text codes> [VIS] <vision codes>`.

Main artifacts:

- [src/masi/alignment/behavior_alignment.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/alignment/behavior_alignment.py)
- [src/masi/tokenization/rqvae.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/tokenization/rqvae.py)
- [src/masi/tokenization/masi_tokens.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/tokenization/masi_tokens.py)
- [scripts/build_masi_tokens.py](/Users/pradyundevarakonda/Developer/MASI/scripts/build_masi_tokens.py)
- [configs/masi_tokens_amazon_csj_demo.json](/Users/pradyundevarakonda/Developer/MASI/configs/masi_tokens_amazon_csj_demo.json)

Current token artifact:

- [outputs/masi_tokens_amazon_csj_demo/fused_semantic_ids.jsonl](/Users/pradyundevarakonda/Developer/MASI/outputs/masi_tokens_amazon_csj_demo/fused_semantic_ids.jsonl)
- [outputs/masi_tokens_amazon_csj_demo/masi_token_summary.json](/Users/pradyundevarakonda/Developer/MASI/outputs/masi_tokens_amazon_csj_demo/masi_token_summary.json)

## Imported Amazon Data

The recommender demo no longer uses hardcoded interaction histories. It now reads the official Amazon Reviews 2023 `Clothing_Shoes_and_Jewelry` review schema from:

- `user_id`
- `parent_asin`
- `timestamp`
- review-side `title`, `text`, and `images`

Current local constraint:

- the raw review file served by the official endpoint is about `25.9 GiB`
- this machine does not have enough free space to store the full raw review dump end to end
- the repo therefore uses a bounded imported prefix of the official review file for local development

Current real-data demo artifact:

- [outputs/recommender_amazon_csj_demo/recommender_demo_summary.json](/Users/pradyundevarakonda/Developer/MASI/outputs/recommender_amazon_csj_demo/recommender_demo_summary.json)

Current modeling limitation:

- the full one-click launcher now prefers the local CSJ metadata file and uses review-side multimodal records only as a bounded fallback
- the current local smoke artifact still runs on a truncated review file, so its token coverage is intentionally much lower than a real full-CSJ run
- the recommender now prefers the generated MASI fused-ID artifact when it exists and only falls back to proxy tokens if that artifact is missing

## Later-Stage Experiment Runner

The repository now includes a bounded but complete later-stage experiment path for the proposal:

1. deterministic warm-start and zero-shot cold-start leave-one-out splitting,
2. optional cross-modal MLM pretraining on the same Transformer backbone,
3. chronological fine-tuning on user histories,
4. token-sequence retrieval against the bounded item catalog,
5. `HR@10`, `NDCG@10`, `Coverage@10`, and latency reporting.

Main artifacts:

- [src/masi/common/toggles.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/common/toggles.py)
- [src/masi/recommender/evaluation.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/evaluation.py)
- [src/masi/recommender/retrieval.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/retrieval.py)
- [scripts/run_masi_experiment.py](/Users/pradyundevarakonda/Developer/MASI/scripts/run_masi_experiment.py)
- [scripts/train_masi.py](/Users/pradyundevarakonda/Developer/MASI/scripts/train_masi.py)
- [configs/masi_experiment_amazon_csj_demo.json](/Users/pradyundevarakonda/Developer/MASI/configs/masi_experiment_amazon_csj_demo.json)
- [configs/masi_train_csj_subset_large.json](/home/dheerajKDE/Documents/College/sem8/Rec_sys/MASI/configs/masi_train_csj_subset_large.json)
- [configs/masi_train_csj_full.json](/Users/pradyundevarakonda/Developer/MASI/configs/masi_train_csj_full.json)

Verified experiment artifact:

- [outputs/masi_experiment_amazon_csj_demo/experiment_summary.json](/Users/pradyundevarakonda/Developer/MASI/outputs/masi_experiment_amazon_csj_demo/experiment_summary.json)
- [outputs/amazon_csj_smoke_train/phase3_experiment/experiment_summary.json](/Users/pradyundevarakonda/Developer/MASI/outputs/amazon_csj_smoke_train/phase3_experiment/experiment_summary.json)

The experiment config exposes proposal-aligned booleans so ablations can switch methods on or off without editing code:

- `use_behavior_alignment`
- `use_text_modality`
- `use_visual_modality`
- `use_late_fusion`
- `use_cross_modal_mlm`
- `use_generative_finetuning`
- `use_cold_start_evaluation`

## Setup

The minimal preprocessing-only demo path is still standard-library only.

Optional package install for future work:

```bash
python3 -m pip install -e .
```

For the full MASI training path, create the local virtual environment used for the verified smoke run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[recommender]"
```

The `recommender` extra now includes the packages required by the one-click launcher and the Phase 1/2 token builder, including `torch`, `transformers`, and `Pillow`.

If you want to open the notebooks locally, install Jupyter inside that environment:

```bash
.venv/bin/python -m pip install jupyter
```

## Usage

Run the proposal-aligned smoke pipeline end to end:

```bash
make train-smoke
```

Run the medium-scale bounded Colab config:

```bash
PYTHONPATH=src python scripts/train_masi.py \
  --config configs/masi_train_csj_medium_colab.json \
  --storage-root /content/MASI
```

Run the canonical bounded large subset config:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_masi.py \
  --config configs/masi_train_csj_subset_large.json \
  --storage-root /path/to/masi_storage
```

Run the faster bounded medium subset config:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_masi.py \
  --config configs/masi_train_csj_subset_medium.json \
  --storage-root /path/to/masi_storage
```

Run the synthetic preprocessing demo:

```bash
PYTHONPATH=src python3 scripts/demo_phase1_prep.py --config configs/data_prep_demo.json
```

Generate the filtered dataset manifest:

```bash
PYTHONPATH=src python3 scripts/build_dataset_manifest.py --config configs/data_prep_demo.json
```

Run the recommender foundation demo:

```bash
PYTHONPATH=src .venv/bin/python scripts/demo_recommender_foundation.py --config configs/recommender_amazon_csj_demo.json
```

Download a bounded prefix of the official Amazon CSJ reviews file for local development:

```bash
PYTHONPATH=src python3 scripts/download_amazon_csj_dataset.py
```

Download the full CSJ reviews file and metadata file:

```bash
PYTHONPATH=src python3 scripts/download_amazon_csj_dataset.py --full-reviews --download-metadata
```

Prepare the canonical bounded large subset locally:

```bash
PYTHONPATH=src python scripts/prepare_amazon_csj_subset.py \
  --reviews-path /path/to/raw/Clothing_Shoes_and_Jewelry.jsonl \
  --metadata-path /path/to/raw/meta_Clothing_Shoes_and_Jewelry.jsonl \
  --output-dir /path/to/masi-amazon-csj-subset-large \
  --preset large
```

Prepare the faster bounded medium subset locally:

```bash
PYTHONPATH=src python scripts/prepare_amazon_csj_subset.py \
  --reviews-path /path/to/raw/Clothing_Shoes_and_Jewelry.jsonl \
  --metadata-path /path/to/raw/meta_Clothing_Shoes_and_Jewelry.jsonl \
  --output-dir /path/to/masi-amazon-csj-subset-medium \
  --preset medium
```

Download validated subset images locally into the prepared dataset:

```bash
PYTHONPATH=src python scripts/download_amazon_csj_subset_images.py \
  --metadata-path /path/to/masi-amazon-csj-subset-large/meta_Clothing_Shoes_and_Jewelry.jsonl \
  --output-dir /path/to/masi-amazon-csj-subset-large \
  --workers 8 \
  --retries 2 \
  --resume
```

Run the subset-medium config on Kaggle after attaching the prepared subset dataset `masi-amazon-csj-subset-medium`:

```bash
PYTHONPATH=src python scripts/train_masi.py \
  --config configs/masi_train_csj_subset_medium.json \
  --storage-root /kaggle/working/masi_artifacts
```

Validate preloaded subset images on Kaggle and download only missing ones into the writable cache:

```bash
PYTHONPATH=src python scripts/download_amazon_csj_images.py \
  --config configs/masi_train_csj_subset_medium.json \
  --storage-root /kaggle/working/masi_artifacts \
  --workers 8 \
  --retries 2 \
  --resume
```

For later Kaggle-session resume, the notebooks export a dataset-ready folder that mirrors the writable `storage_root` layout:

- `outputs/amazon_csj_subset_medium_train/`
- `data/processed/amazon_csj_subset_medium/`

Use [notebooks/07_kaggle_github_bootstrap_medium_run.ipynb](/home/dheerajKDE/Documents/College/sem8/Rec_sys/MASI/notebooks/07_kaggle_github_bootstrap_medium_run.ipynb) as the default Kaggle entrypoint for the prepared subset medium run. It validates the attached prepared dataset first, clones the repo from GitHub into `/kaggle/working/MASI`, installs `.[recommender]`, optionally restores a resume bundle, optionally validates and backfills missing images, runs `train_masi.py`, verifies the required manifests, and exports a dataset-ready bundle. Keep [notebooks/06_kaggle_medium_smoke_test.ipynb](/home/dheerajKDE/Documents/College/sem8/Rec_sys/MASI/notebooks/06_kaggle_medium_smoke_test.ipynb) and [notebooks/05_kaggle_full_workflow.ipynb](/home/dheerajKDE/Documents/College/sem8/Rec_sys/MASI/notebooks/05_kaggle_full_workflow.ipynb) only as older Kaggle variants.

The raw full-CSJ path is still retained as a deferred reference config, but it is no longer the main README workflow.

Build Phase 1 + Phase 2 MASI fused tokens:

```bash
make masi-tokens
```

Open the demo notebook:

- [notebooks/01_dataset_and_feature_prep_demo.ipynb](/Users/pradyundevarakonda/Developer/MASI/notebooks/01_dataset_and_feature_prep_demo.ipynb)
- [notebooks/02_recommender_foundation_demo.ipynb](/Users/pradyundevarakonda/Developer/MASI/notebooks/02_recommender_foundation_demo.ipynb)
- [notebooks/03_colab_smoke_test.ipynb](/Users/pradyundevarakonda/Developer/MASI/notebooks/03_colab_smoke_test.ipynb)
- [notebooks/04_colab_smoke_test_fresh_clone.ipynb](/Users/pradyundevarakonda/Developer/MASI/notebooks/04_colab_smoke_test_fresh_clone.ipynb)
- [notebooks/07_kaggle_github_bootstrap_medium_run.ipynb](/home/dheerajKDE/Documents/College/sem8/Rec_sys/MASI/notebooks/07_kaggle_github_bootstrap_medium_run.ipynb)
- [notebooks/05_kaggle_full_workflow.ipynb](/home/dheerajKDE/Documents/College/sem8/Rec_sys/MASI/notebooks/05_kaggle_full_workflow.ipynb)
- [notebooks/06_kaggle_medium_smoke_test.ipynb](/home/dheerajKDE/Documents/College/sem8/Rec_sys/MASI/notebooks/06_kaggle_medium_smoke_test.ipynb)

Run the later-stage bounded experiment:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_masi_experiment.py --config configs/masi_experiment_amazon_csj_demo.json
```

## Full-Corpus Next Steps

Before attempting the entire Amazon Reviews 2023 dataset, the recommended order is:

1. define storage, checkpoint, and shard contracts,
2. add sharded review and metadata ingestion,
3. cache CLIP embeddings and image assets,
4. add scalable retrieval/evaluation,
5. run full Phase 1 -> Phase 3 training,
6. run baselines and ablations on the same full-scale splits.

## Initial Execution Plan

The proposal provides an 8-week implementation arc:

- Weeks 1-2: dataset ETL and CLIP feature extraction
- Weeks 3-4: user-item graph construction, mixed negative sampling, projection-head training
- Weeks 5-6: dual RQ-VAE training and recommender MLM pretraining
- Weeks 7-8: generative fine-tuning, ablations, evaluation, and paper finalization

## Working Principle

This repository is intended to be advanced incrementally. AI agents should treat the proposal as the source specification, keep documentation current, and update [TODO_TASKS.md](/Users/pradyundevarakonda/Developer/MASI/TODO_TASKS.md) whenever they complete, refine, or discover work.
