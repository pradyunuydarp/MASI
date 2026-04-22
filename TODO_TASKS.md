# TODO_TASKS.md

This file is the living progress ledger for AI agents working on MASI. Update it whenever work starts, completes, is blocked, or changes scope.

## Status Legend

- `TODO`: not started
- `IN_PROGRESS`: actively being worked on
- `BLOCKED`: cannot proceed without a dependency, decision, or resource
- `DONE`: completed with a repository artifact or documented outcome

## Current Project State

- `DONE` Bootstrap repository documentation from the proposal.
  Artifact: [README.md](/Users/pradyundevarakonda/Developer/MASI/README.md), [AGENTS.md](/Users/pradyundevarakonda/Developer/MASI/AGENTS.md), [TODO_TASKS.md](/Users/pradyundevarakonda/Developer/MASI/TODO_TASKS.md)
- `DONE` Create the implementation scaffold for data, configs, source code, scripts, notebooks, and outputs.
  Artifact: `configs/`, `data/`, `docs/`, `notebooks/`, `outputs/`, `scripts/`, `src/masi/`, [pyproject.toml](/Users/pradyundevarakonda/Developer/MASI/pyproject.toml)
- `DONE` Add environment setup instructions and dependency management for the initial preprocessing slice.
  Artifact: [README.md](/Users/pradyundevarakonda/Developer/MASI/README.md), [pyproject.toml](/Users/pradyundevarakonda/Developer/MASI/pyproject.toml)
- `DONE` Add a demo-able preprocessing notebook and synthetic data workflow.
  Artifact: [notebooks/01_dataset_and_feature_prep_demo.ipynb](/Users/pradyundevarakonda/Developer/MASI/notebooks/01_dataset_and_feature_prep_demo.ipynb), [scripts/demo_phase1_prep.py](/Users/pradyundevarakonda/Developer/MASI/scripts/demo_phase1_prep.py)
- `DONE` Add a Colab bootstrap notebook for the one-click MASI smoke pipeline.
  Artifact: [notebooks/03_colab_smoke_test.ipynb](/Users/pradyundevarakonda/Developer/MASI/notebooks/03_colab_smoke_test.ipynb)
- `DONE` Add a fresh-clone Colab smoke notebook that always replaces any old `/content/MASI` checkout before running.
  Artifact: [notebooks/04_colab_smoke_test_fresh_clone.ipynb](/Users/pradyundevarakonda/Developer/MASI/notebooks/04_colab_smoke_test_fresh_clone.ipynb)
- `DONE` Add a Kaggle-safe bounded workflow with a dedicated notebook, bounded config, and resumable image downloader.
  Artifact: `configs/masi_train_csj_medium_kaggle.json`, `scripts/download_amazon_csj_images.py`, `notebooks/05_kaggle_full_workflow.ipynb`, [src/masi/data/amazon_csj_assets.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/data/amazon_csj_assets.py)
- `DONE` Add periodic step-based checkpoint emission for bounded Phase 1, Phase 2, and Phase 3 training so Kaggle runs retain recent weights before the session ends.
  Artifact: [src/masi/common/checkpoints.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/common/checkpoints.py), [src/masi/alignment/behavior_alignment.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/alignment/behavior_alignment.py), [src/masi/tokenization/rqvae.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/tokenization/rqvae.py), [src/masi/recommender/training.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/training.py), [scripts/build_masi_tokens.py](/Users/pradyundevarakonda/Developer/MASI/scripts/build_masi_tokens.py), [scripts/run_masi_experiment.py](/Users/pradyundevarakonda/Developer/MASI/scripts/run_masi_experiment.py), [configs/masi_train_csj_medium_kaggle.json](/Users/pradyundevarakonda/Developer/MASI/configs/masi_train_csj_medium_kaggle.json), [README.md](/Users/pradyundevarakonda/Developer/MASI/README.md)
- `DONE` Add a local helper script for downloading the bounded raw CSJ files before uploading them as a private Kaggle Dataset.
  Artifact: `scripts/download_kaggle_safe_csj_local.py`
- `DONE` Switch the Kaggle bounded workflow to read raw files directly from the attached Dataset input `masi-amazon-csj-raw` and restore prior run artifacts from a second resume dataset.
  Artifact: `configs/masi_train_csj_medium_kaggle.json`, `notebooks/05_kaggle_full_workflow.ipynb`, [README.md](/Users/pradyundevarakonda/Developer/MASI/README.md)
- `DONE` Add a web-sourced reference map for foundational papers and public repositories relevant to MASI.
  Artifact: [docs/reference_repos.md](/Users/pradyundevarakonda/Developer/MASI/docs/reference_repos.md)
- `DONE` Add the first recommender-side implementation foundation and demo workflow.
  Artifact: [src/masi/recommender/__init__.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/__init__.py), [scripts/demo_recommender_foundation.py](/Users/pradyundevarakonda/Developer/MASI/scripts/demo_recommender_foundation.py), [notebooks/02_recommender_foundation_demo.ipynb](/Users/pradyundevarakonda/Developer/MASI/notebooks/02_recommender_foundation_demo.ipynb)
- `DONE` Add a LaTeX implementation note documenting completed MASI work, rationale, and engineering challenges.
  Artifact: [docs/masi_implementation_note.tex](/Users/pradyundevarakonda/Developer/MASI/docs/masi_implementation_note.tex)
- `DONE` Add a one-click MASI launcher for the proposal's primary CSJ benchmark, with resolved configs, checkpoints, and run manifests for local, Kaggle, and Colab runs.
  Artifact: [scripts/train_masi.py](/Users/pradyundevarakonda/Developer/MASI/scripts/train_masi.py), [configs/masi_train_csj_full.json](/Users/pradyundevarakonda/Developer/MASI/configs/masi_train_csj_full.json), [configs/masi_train_csj_smoke.json](/Users/pradyundevarakonda/Developer/MASI/configs/masi_train_csj_smoke.json), [outputs/amazon_csj_smoke_train/run_manifest.json](/Users/pradyundevarakonda/Developer/MASI/outputs/amazon_csj_smoke_train/run_manifest.json)
- `DONE` Reserve a repo-local `Colab_interactions/` workspace for transient Colab-side files and keep all contents untracked.
  Artifact: `.gitignore`, `Colab_interactions/`

## Phase 0: Proposal-To-Codebase Translation

- `DONE` Convert the proposal into a technical design note with module boundaries, training stages, and data contracts.
  Artifact: [docs/technical_design.md](/Users/pradyundevarakonda/Developer/MASI/docs/technical_design.md)
- `DONE` Decide the primary framework stack for recommendation training, including whether RecBole will be extended directly or wrapped.
  Decision: use a custom PyTorch foundation for MASI recommender modules first, with RecBole left as an optional future adapter for baselines and evaluation. Artifact: [docs/technical_design.md](/Users/pradyundevarakonda/Developer/MASI/docs/technical_design.md), [src/masi/recommender/](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/)
- `DONE` Document expected hardware assumptions for CLIP extraction, RQ-VAE training, and Transformer fine-tuning.
  Artifact: [README.md](/Users/pradyundevarakonda/Developer/MASI/README.md)

## Phase 1: Data And Feature Preparation

- `IN_PROGRESS` Acquire and register the Amazon Reviews 2023 Clothing/Shoes/Jewelry subset.
  Current artifact: official imported review file prefix in `data/raw/amazon_reviews_2023/Clothing_Shoes_and_Jewelry.jsonl`; full raw review file is larger than currently available disk space.
- `DONE` Implement 5-core filtering for users and items.
  Artifact: [src/masi/data/amazon2023.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/data/amazon2023.py)
- `DONE` Build ETL scaffolding for item text, metadata, image references, and user interaction sequences.
  Artifact: [src/masi/data/contracts.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/data/contracts.py), [src/masi/data/amazon2023.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/data/amazon2023.py), [scripts/build_dataset_manifest.py](/Users/pradyundevarakonda/Developer/MASI/scripts/build_dataset_manifest.py)
- `DONE` Validate image availability and drop missing or invalid image assets before visual-token training.
  Artifact: [src/masi/data/amazon_csj_assets.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/data/amazon_csj_assets.py), [scripts/build_masi_tokens.py](/Users/pradyundevarakonda/Developer/MASI/scripts/build_masi_tokens.py)
- `DONE` Implement CLIP text embedding extraction for the bounded MASI token pipeline.
  Artifact: [src/masi/tokenization/masi_tokens.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/tokenization/masi_tokens.py), [scripts/build_masi_tokens.py](/Users/pradyundevarakonda/Developer/MASI/scripts/build_masi_tokens.py)
- `DONE` Implement CLIP image embedding extraction for the bounded MASI token pipeline.
  Artifact: [src/masi/tokenization/masi_tokens.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/tokenization/masi_tokens.py), [src/masi/data/amazon_csj_assets.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/data/amazon_csj_assets.py), [scripts/build_masi_tokens.py](/Users/pradyundevarakonda/Developer/MASI/scripts/build_masi_tokens.py)
- `IN_PROGRESS` Persist extracted preprocessing artifacts in a reproducible format with split metadata.
  Current artifact: filtered JSONL tables and dataset manifest for the demo run under `data/processed/demo/` and `outputs/demo/`.
- `DONE` Add real Amazon CSJ training configs for smoke and full proposal-aligned runs.
  Artifact: [configs/masi_train_csj_smoke.json](/Users/pradyundevarakonda/Developer/MASI/configs/masi_train_csj_smoke.json), [configs/masi_train_csj_full.json](/Users/pradyundevarakonda/Developer/MASI/configs/masi_train_csj_full.json)
- `DONE` Add a medium-scale Colab config to cover more of the MASI pipeline than the smoke path without requiring the full CSJ run.
  Artifact: [configs/masi_train_csj_medium_colab.json](/Users/pradyundevarakonda/Developer/MASI/configs/masi_train_csj_medium_colab.json)
- `DONE` Add a bounded Kaggle config that prefers local metadata slices and threaded resumable image downloads over remote metadata streaming.
  Artifact: `configs/masi_train_csj_medium_kaggle.json`, `scripts/download_amazon_csj_images.py`
- `DONE` Add deterministic leave-one-out split generation for warm-start and zero-shot cold-start evaluation.
  Artifact: [src/masi/recommender/evaluation.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/evaluation.py), [scripts/run_masi_experiment.py](/Users/pradyundevarakonda/Developer/MASI/scripts/run_masi_experiment.py), [outputs/masi_experiment_amazon_csj_demo/experiment_summary.json](/Users/pradyundevarakonda/Developer/MASI/outputs/masi_experiment_amazon_csj_demo/experiment_summary.json)
- `TODO` Run the bounded Kaggle workflow and capture the first Kaggle-produced checkpoint bundle and run manifest.
- `TODO` Run the full one-click CSJ launcher on a GPU machine with the complete review and metadata files to capture the first non-smoke benchmark artifact.
  Note: full CSJ still requires more storage than Kaggle's ephemeral disk once the raw files and full image cache are combined.

## Phase 2: Behavior-Aware Contrastive Alignment

- `DONE` Construct the user-item bipartite graph from filtered interactions for the bounded Amazon MASI token build.
  Artifact: [src/masi/alignment/behavior_alignment.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/alignment/behavior_alignment.py)
- `DONE` Implement positive pair generation from collaborative signals.
  Artifact: [src/masi/alignment/behavior_alignment.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/alignment/behavior_alignment.py)
- `DONE` Implement mixed negative sampling with in-batch and hard graph negatives.
  Artifact: [src/masi/alignment/behavior_alignment.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/alignment/behavior_alignment.py)
- `DONE` Implement frozen CLIP backbones with trainable text and vision projection heads.
  Artifact: [src/masi/alignment/behavior_alignment.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/alignment/behavior_alignment.py), [src/masi/tokenization/masi_tokens.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/tokenization/masi_tokens.py)
- `DONE` Train the projection heads with separate `Lvis` and `Ltxt` InfoNCE losses.
  Artifact: [scripts/build_masi_tokens.py](/Users/pradyundevarakonda/Developer/MASI/scripts/build_masi_tokens.py), [outputs/masi_tokens_amazon_csj_demo/masi_token_summary.json](/Users/pradyundevarakonda/Developer/MASI/outputs/masi_tokens_amazon_csj_demo/masi_token_summary.json)
- `TODO` Evaluate whether projected embeddings improve behavioral clustering for warm items.
- `TODO` Define the cold-start inference path for items without interaction edges.

## Phase 3: Independent Modality Quantization

- `DONE` Implement the text-side RQ-VAE-style independent quantization pipeline.
  Artifact: [src/masi/tokenization/rqvae.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/tokenization/rqvae.py)
- `DONE` Implement the vision-side RQ-VAE-style independent quantization pipeline.
  Artifact: [src/masi/tokenization/rqvae.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/tokenization/rqvae.py)
- `DONE` Choose and document the default codebook depth and size for proposal-aligned full training.
  Decision: the current full-train default is `D=3` with `K=256` per modality, with depth sweeps still tracked as ablations. Artifact: [configs/masi_train_csj_full.json](/Users/pradyundevarakonda/Developer/MASI/configs/masi_train_csj_full.json)
- `DONE` Generate text semantic IDs from projected text embeddings for the bounded Amazon MASI token build.
  Artifact: [outputs/masi_tokens_amazon_csj_demo/fused_semantic_ids.jsonl](/Users/pradyundevarakonda/Developer/MASI/outputs/masi_tokens_amazon_csj_demo/fused_semantic_ids.jsonl)
- `DONE` Generate visual semantic IDs from projected visual embeddings for the bounded Amazon MASI token build.
  Artifact: [outputs/masi_tokens_amazon_csj_demo/fused_semantic_ids.jsonl](/Users/pradyundevarakonda/Developer/MASI/outputs/masi_tokens_amazon_csj_demo/fused_semantic_ids.jsonl)
- `DONE` Implement late fusion with `[TXT]` and `[VIS]` modality markers.
  Artifact: [src/masi/tokenization/masi_tokens.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/tokenization/masi_tokens.py), [outputs/masi_tokens_amazon_csj_demo/fused_semantic_ids.jsonl](/Users/pradyundevarakonda/Developer/MASI/outputs/masi_tokens_amazon_csj_demo/fused_semantic_ids.jsonl)
- `TODO` Validate that modality-specific codebooks reduce collisions for visually distinct items with similar text.

## Phase 4: Recommender-Side Cross-Modal Alignment

- `DONE` Choose the Transformer backbone for token-sequence recommendation.
  Artifact: [src/masi/recommender/sasrec.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/sasrec.py), [src/masi/recommender/generative.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/generative.py)
- `DONE` Implement cross-modal MLM pretraining to predict visual tokens from text tokens and vice versa.
  Artifact: [src/masi/recommender/mlm.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/mlm.py), [src/masi/recommender/sequence_data.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/sequence_data.py), [outputs/recommender_demo/recommender_demo_summary.json](/Users/pradyundevarakonda/Developer/MASI/outputs/recommender_demo/recommender_demo_summary.json)
- `DONE` Fine-tune the recommender on chronological user interaction sequences for the bounded Amazon experiment path.
  Artifact: [src/masi/recommender/training.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/training.py), [scripts/run_masi_experiment.py](/Users/pradyundevarakonda/Developer/MASI/scripts/run_masi_experiment.py), [outputs/masi_experiment_amazon_csj_demo/experiment_summary.json](/Users/pradyundevarakonda/Developer/MASI/outputs/masi_experiment_amazon_csj_demo/experiment_summary.json)
- `DONE` Define decoding and retrieval from generated semantic identifier sequences.
  Artifact: [src/masi/recommender/retrieval.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/retrieval.py), [scripts/run_masi_experiment.py](/Users/pradyundevarakonda/Developer/MASI/scripts/run_masi_experiment.py)
- `DONE` Measure token-level reconstruction and downstream recommendation readiness after MLM pretraining.
  Artifact: MLM and autoregressive loss curves plus ranking metrics in [outputs/masi_experiment_amazon_csj_demo/experiment_summary.json](/Users/pradyundevarakonda/Developer/MASI/outputs/masi_experiment_amazon_csj_demo/experiment_summary.json)
- `DONE` Enable Apple Silicon GPU acceleration for Phase 3 MLM pretraining and generative ranking.
  Artifact: [src/masi/recommender/mlm.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/mlm.py), [src/masi/recommender/generative.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/generative.py), [scripts/run_masi_experiment.py](/Users/pradyundevarakonda/Developer/MASI/scripts/run_masi_experiment.py), [outputs/masi_experiment_amazon_csj_demo/experiment_summary.json](/Users/pradyundevarakonda/Developer/MASI/outputs/masi_experiment_amazon_csj_demo/experiment_summary.json)
- `DONE` Replace synthetic fused semantic IDs with real Phase 2 outputs for the bounded Amazon MASI token build.
  Artifact: [outputs/masi_tokens_amazon_csj_demo/fused_semantic_ids.jsonl](/Users/pradyundevarakonda/Developer/MASI/outputs/masi_tokens_amazon_csj_demo/fused_semantic_ids.jsonl), [configs/recommender_amazon_csj_demo.json](/Users/pradyundevarakonda/Developer/MASI/configs/recommender_amazon_csj_demo.json)
- `DONE` Wire leave-one-out warm-start and zero-shot splits into the recommender datasets and training loop.
  Artifact: [src/masi/recommender/evaluation.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/evaluation.py), [scripts/run_masi_experiment.py](/Users/pradyundevarakonda/Developer/MASI/scripts/run_masi_experiment.py)
- `DONE` Add checkpoint emission and automatic token-budget resolution for proposal-aligned Phase 3 runs.
  Artifact: [scripts/run_masi_experiment.py](/Users/pradyundevarakonda/Developer/MASI/scripts/run_masi_experiment.py), [outputs/amazon_csj_smoke_train/phase3_experiment/experiment_summary.json](/Users/pradyundevarakonda/Developer/MASI/outputs/amazon_csj_smoke_train/phase3_experiment/experiment_summary.json)

## Phase 5: Evaluation And Ablations

- `DONE` Implement leave-one-out evaluation for warm-start and zero-shot cold-start splits.
  Artifact: [src/masi/recommender/evaluation.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/evaluation.py), [scripts/run_masi_experiment.py](/Users/pradyundevarakonda/Developer/MASI/scripts/run_masi_experiment.py)
- `DONE` Compute `HR@10`, `NDCG@10`, `Coverage@10`, and inference latency.
  Artifact: [src/masi/recommender/evaluation.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/evaluation.py), [outputs/masi_experiment_amazon_csj_demo/experiment_summary.json](/Users/pradyundevarakonda/Developer/MASI/outputs/masi_experiment_amazon_csj_demo/experiment_summary.json)
- `IN_PROGRESS` Add or reproduce the proposal baselines: `SASRec`, `CEMG`, `MGR-LF++`, and `DIGER`.
  Current artifact: SASRec-style baseline module in [src/masi/recommender/sasrec.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/sasrec.py); multimodal baselines remain pending due missing public code and earlier pipeline dependencies.
- `DONE` Implement parameterized method toggles so Phase 1, modality, late-fusion, Phase 3 MLM, fine-tuning, and cold-start evaluation can be switched on or off without code edits.
  Artifact: [src/masi/common/toggles.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/common/toggles.py), [scripts/build_masi_tokens.py](/Users/pradyundevarakonda/Developer/MASI/scripts/build_masi_tokens.py), [scripts/run_masi_experiment.py](/Users/pradyundevarakonda/Developer/MASI/scripts/run_masi_experiment.py), [configs/masi_experiment_amazon_csj_demo.json](/Users/pradyundevarakonda/Developer/MASI/configs/masi_experiment_amazon_csj_demo.json)
- `TODO` Run ablation for Phase 1 behavior alignment on vs. off using the new toggleable token-build and experiment configs.
- `TODO` Run ablation for Phase 3 cross-modal MLM on vs. off using the new toggleable experiment configs.
- `TODO` Run ablation over RQ-VAE depth `D in {2,3,4}`.
- `TODO` Summarize whether Phase 1 and Phase 3 are complementary or redundant.

## Phase 6: Reporting

- `DONE` Create reproducible experiment config and summary templates for runs, metrics, and artifacts.
  Artifact: [configs/masi_experiment_amazon_csj_demo.json](/Users/pradyundevarakonda/Developer/MASI/configs/masi_experiment_amazon_csj_demo.json), [outputs/masi_experiment_amazon_csj_demo/experiment_summary.json](/Users/pradyundevarakonda/Developer/MASI/outputs/masi_experiment_amazon_csj_demo/experiment_summary.json)
- `TODO` Add result tables matching the proposal metrics and baselines.
- `TODO` Maintain a concise implementation log for proposal deviations and design choices.
- `TODO` Prepare manuscript-ready figures for the MASI pipeline, evaluation setup, and ablation outcomes.

## Open Questions

- `DONE` Confirm whether the repository will build directly on RecBole or use custom training loops around selected components.
  Answer: custom PyTorch training loops first; RecBole is optional for later integration.
- `DONE` Decide the concrete CLIP variant to use for text and image embedding extraction.
  Decision: `openai/clip-vit-base-patch32` is the current default for full-CSJ runs; larger variants remain future ablations. Artifact: [configs/masi_train_csj_full.json](/Users/pradyundevarakonda/Developer/MASI/configs/masi_train_csj_full.json)
- `DONE` Decide where large datasets, extracted embeddings, and checkpoints will live relative to the repository.
  Decision: the one-click launcher resolves a storage root via `--storage-root`, `MASI_STORAGE_ROOT`, or environment defaults, and writes run artifacts beneath that root. Artifact: [scripts/train_masi.py](/Users/pradyundevarakonda/Developer/MASI/scripts/train_masi.py)
- `DONE` Define the exact reproducibility contract for random seeds, splits, and checkpoint naming.
  Artifact: fixed seeds and deterministic split inputs in [scripts/build_masi_tokens.py](/Users/pradyundevarakonda/Developer/MASI/scripts/build_masi_tokens.py) and [scripts/run_masi_experiment.py](/Users/pradyundevarakonda/Developer/MASI/scripts/run_masi_experiment.py), plus resolved-config and checkpoint manifests in [outputs/amazon_csj_smoke_train/run_manifest.json](/Users/pradyundevarakonda/Developer/MASI/outputs/amazon_csj_smoke_train/run_manifest.json)

## Full Amazon Reviews 2023 Scale-Up

- `TODO` Define the full-scale storage contract for the entire Amazon Reviews 2023 corpus, including raw reviews, metadata, image caches, embeddings, checkpoints, logs, and run artifacts.
- `TODO` Document hardware requirements for full-corpus runs, including disk budget, RAM, and recommended GPU class beyond local Apple Silicon development.
- `TODO` Add manifest-based run tracking so each long-running stage records config hash, shard range, input lineage, output lineage, and completion state.
- `TODO` Add checkpoint and resume support for every expensive script used in full-scale preprocessing and training.

### Full-Scale Data Pipeline

- `TODO` Implement sharded ingestion for the full Amazon Reviews 2023 review corpus rather than relying on one bounded local file.
- `TODO` Implement sharded ingestion for the full Amazon Reviews 2023 metadata corpus and align `parent_asin` records across categories.
- `TODO` Add global 5-core filtering at full scale for users and items across the selected training universe.
- `TODO` Add corpus-scale image validation and a fallback policy for missing, invalid, or rate-limited image assets.
- `TODO` Persist processed user histories and item metadata in shard-safe artifacts that can be resumed and merged.
- `TODO` Generate deterministic warm-start and zero-shot cold-start splits at full scale with stable seed and manifest metadata.
- `TODO` Add dataset statistics reports for each full-scale preprocessing stage.

### Full-Scale Feature Extraction

- `TODO` Add sharded CLIP text embedding extraction with resumable caches.
- `TODO` Add sharded CLIP image embedding extraction with resumable caches.
- `TODO` Add merge/index steps that consolidate embedding shards into Phase 1-ready artifacts.
- `TODO` Benchmark CLIP throughput on target hardware and choose production batch sizes for full runs.

### Full-Scale Modeling

- `TODO` Add checkpointed full-scale Phase 1 behavior-alignment training over the warm-item graph.
- `TODO` Add checkpointed full-scale Phase 2 dual RQ-VAE training and semantic-ID export.
- `TODO` Add checkpointed full-scale Phase 3 MLM pretraining and sequential fine-tuning.
- `TODO` Add production config variants for RQ-VAE depth sweeps `D in {2,3,4}` and for method-toggle ablations.

### Full-Scale Evaluation

- `BLOCKED` Replace the current exhaustive candidate scoring path before attempting full-catalog evaluation at Amazon scale.
- `TODO` Implement batched or indexed retrieval for full-catalog ranking evaluation.
- `TODO` Compute full-scale `HR@10`, `NDCG@10`, `Coverage@10`, and latency on warm-start and zero-shot splits.
- `TODO` Add prediction dumps and error-analysis artifacts for large-scale evaluation runs.

### Full-Scale Baselines And Ablations

- `TODO` Reproduce `SASRec` on the same full-scale splits used for MASI.
- `TODO` Reproduce or document feasible approximations for `CEMG`, `MGR-LF++`, and `DIGER` on the same evaluation protocol.
- `TODO` Run full-scale ablations for Phase 1 on/off, Phase 3 MLM on/off, and modality/codebook depth variants.
- `TODO` Summarize whether Phase 1 and Phase 3 remain complementary at full scale.
