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
- `DONE` Add a web-sourced reference map for foundational papers and public repositories relevant to MASI.
  Artifact: [docs/reference_repos.md](/Users/pradyundevarakonda/Developer/MASI/docs/reference_repos.md)
- `DONE` Add the first recommender-side implementation foundation and demo workflow.
  Artifact: [src/masi/recommender/__init__.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/__init__.py), [scripts/demo_recommender_foundation.py](/Users/pradyundevarakonda/Developer/MASI/scripts/demo_recommender_foundation.py), [notebooks/02_recommender_foundation_demo.ipynb](/Users/pradyundevarakonda/Developer/MASI/notebooks/02_recommender_foundation_demo.ipynb)

## Phase 0: Proposal-To-Codebase Translation

- `DONE` Convert the proposal into a technical design note with module boundaries, training stages, and data contracts.
  Artifact: [docs/technical_design.md](/Users/pradyundevarakonda/Developer/MASI/docs/technical_design.md)
- `DONE` Decide the primary framework stack for recommendation training, including whether RecBole will be extended directly or wrapped.
  Decision: use a custom PyTorch foundation for MASI recommender modules first, with RecBole left as an optional future adapter for baselines and evaluation. Artifact: [docs/technical_design.md](/Users/pradyundevarakonda/Developer/MASI/docs/technical_design.md), [src/masi/recommender/](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/)
- `TODO` Document expected hardware assumptions for CLIP extraction, RQ-VAE training, and Transformer fine-tuning.

## Phase 1: Data And Feature Preparation

- `TODO` Acquire and register the Amazon Reviews 2023 Clothing/Shoes/Jewelry subset.
- `DONE` Implement 5-core filtering for users and items.
  Artifact: [src/masi/data/amazon2023.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/data/amazon2023.py)
- `DONE` Build ETL scaffolding for item text, metadata, image references, and user interaction sequences.
  Artifact: [src/masi/data/contracts.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/data/contracts.py), [src/masi/data/amazon2023.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/data/amazon2023.py), [scripts/build_dataset_manifest.py](/Users/pradyundevarakonda/Developer/MASI/scripts/build_dataset_manifest.py)
- `TODO` Validate image availability and define a fallback policy for missing or invalid images.
- `TODO` Implement CLIP text embedding extraction.
- `TODO` Implement CLIP image embedding extraction.
- `IN_PROGRESS` Persist extracted preprocessing artifacts in a reproducible format with split metadata.
  Current artifact: filtered JSONL tables and dataset manifest for the demo run under `data/processed/demo/` and `outputs/demo/`.
- `TODO` Add the real Amazon subset config once raw source files are available locally.
- `TODO` Add deterministic train/validation/test split generation for warm-start and zero-shot cold-start evaluation.

## Phase 2: Behavior-Aware Contrastive Alignment

- `TODO` Construct the user-item bipartite graph from filtered interactions.
- `TODO` Implement positive pair generation from collaborative signals.
- `TODO` Implement mixed negative sampling with in-batch and hard graph negatives.
- `TODO` Implement frozen CLIP backbones with trainable text and vision projection heads.
- `TODO` Train the projection heads with separate `Lvis` and `Ltxt` InfoNCE losses.
- `TODO` Evaluate whether projected embeddings improve behavioral clustering for warm items.
- `TODO` Define the cold-start inference path for items without interaction edges.

## Phase 3: Independent Modality Quantization

- `TODO` Implement the text-side RQ-VAE pipeline.
- `TODO` Implement the vision-side RQ-VAE pipeline.
- `TODO` Choose and document codebook depth `D` and vocabulary size per level.
- `TODO` Generate text semantic IDs from projected text embeddings.
- `TODO` Generate visual semantic IDs from projected visual embeddings.
- `TODO` Implement late fusion with `[TXT]` and `[VIS]` modality markers.
- `TODO` Validate that modality-specific codebooks reduce collisions for visually distinct items with similar text.

## Phase 4: Recommender-Side Cross-Modal Alignment

- `DONE` Choose the Transformer backbone for token-sequence recommendation.
  Artifact: [src/masi/recommender/sasrec.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/sasrec.py), [src/masi/recommender/generative.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/generative.py)
- `DONE` Implement cross-modal MLM pretraining to predict visual tokens from text tokens and vice versa.
  Artifact: [src/masi/recommender/mlm.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/mlm.py), [src/masi/recommender/sequence_data.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/sequence_data.py), [outputs/recommender_demo/recommender_demo_summary.json](/Users/pradyundevarakonda/Developer/MASI/outputs/recommender_demo/recommender_demo_summary.json)
- `IN_PROGRESS` Fine-tune the recommender on chronological user interaction sequences.
  Current artifact: synthetic sequence serialization and one-step optimization demo in [scripts/demo_recommender_foundation.py](/Users/pradyundevarakonda/Developer/MASI/scripts/demo_recommender_foundation.py)
- `TODO` Define decoding and retrieval from generated semantic identifier sequences.
- `IN_PROGRESS` Measure token-level reconstruction and downstream recommendation readiness after MLM pretraining.
  Current artifact: demo losses and tensor-shape verification in [outputs/recommender_demo/recommender_demo_summary.json](/Users/pradyundevarakonda/Developer/MASI/outputs/recommender_demo/recommender_demo_summary.json)
- `TODO` Replace synthetic fused semantic IDs with real Phase 2 outputs once RQ-VAE tokenization is implemented.
- `TODO` Wire leave-one-out warm-start and zero-shot splits into the recommender datasets and training loop.

## Phase 5: Evaluation And Ablations

- `TODO` Implement leave-one-out evaluation for warm-start and zero-shot cold-start splits.
- `TODO` Compute `HR@10`, `NDCG@10`, `Coverage@10`, and inference latency.
- `IN_PROGRESS` Add or reproduce the proposal baselines: `SASRec`, `CEMG`, `MGR-LF++`, and `DIGER`.
  Current artifact: SASRec-style baseline module in [src/masi/recommender/sasrec.py](/Users/pradyundevarakonda/Developer/MASI/src/masi/recommender/sasrec.py); multimodal baselines remain pending due missing public code and earlier pipeline dependencies.
- `TODO` Run ablation for Phase 1 behavior alignment on vs. off.
- `TODO` Run ablation for Phase 3 cross-modal MLM on vs. off.
- `TODO` Run ablation over RQ-VAE depth `D in {2,3,4}`.
- `TODO` Summarize whether Phase 1 and Phase 3 are complementary or redundant.

## Phase 6: Reporting

- `TODO` Create experiment tracking templates for runs, metrics, and artifacts.
- `TODO` Add result tables matching the proposal metrics and baselines.
- `TODO` Maintain a concise implementation log for proposal deviations and design choices.
- `TODO` Prepare manuscript-ready figures for the MASI pipeline, evaluation setup, and ablation outcomes.

## Open Questions

- `DONE` Confirm whether the repository will build directly on RecBole or use custom training loops around selected components.
  Answer: custom PyTorch training loops first; RecBole is optional for later integration.
- `TODO` Decide the concrete CLIP variant to use for text and image embedding extraction.
- `TODO` Decide where large datasets, extracted embeddings, and checkpoints will live relative to the repository.
- `TODO` Define the exact reproducibility contract for random seeds, splits, and checkpoint naming.
