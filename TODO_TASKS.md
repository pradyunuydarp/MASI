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
- `TODO` Create the implementation scaffold for data, configs, source code, scripts, and outputs.
- `TODO` Add environment setup instructions and dependency management once the stack is chosen.

## Phase 0: Proposal-To-Codebase Translation

- `TODO` Convert the proposal into a technical design note with module boundaries, training stages, and data contracts.
- `TODO` Decide the primary framework stack for recommendation training, including whether RecBole will be extended directly or wrapped.
- `TODO` Document expected hardware assumptions for CLIP extraction, RQ-VAE training, and Transformer fine-tuning.

## Phase 1: Data And Feature Preparation

- `TODO` Acquire and register the Amazon Reviews 2023 Clothing/Shoes/Jewelry subset.
- `TODO` Implement 5-core filtering for users and items.
- `TODO` Build ETL for item text, metadata, image references, and user interaction sequences.
- `TODO` Validate image availability and define a fallback policy for missing or invalid images.
- `TODO` Implement CLIP text embedding extraction.
- `TODO` Implement CLIP image embedding extraction.
- `TODO` Persist extracted features in a reproducible format with split metadata.

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

- `TODO` Choose the Transformer backbone for token-sequence recommendation.
- `TODO` Implement cross-modal MLM pretraining to predict visual tokens from text tokens and vice versa.
- `TODO` Fine-tune the recommender on chronological user interaction sequences.
- `TODO` Define decoding and retrieval from generated semantic identifier sequences.
- `TODO` Measure token-level reconstruction and downstream recommendation readiness after MLM pretraining.

## Phase 5: Evaluation And Ablations

- `TODO` Implement leave-one-out evaluation for warm-start and zero-shot cold-start splits.
- `TODO` Compute `HR@10`, `NDCG@10`, `Coverage@10`, and inference latency.
- `TODO` Add or reproduce the proposal baselines: `SASRec`, `CEMG`, `MGR-LF++`, and `DIGER`.
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

- `TODO` Confirm whether the repository will build directly on RecBole or use custom training loops around selected components.
- `TODO` Decide the concrete CLIP variant to use for text and image embedding extraction.
- `TODO` Decide where large datasets, extracted embeddings, and checkpoints will live relative to the repository.
- `TODO` Define the exact reproducibility contract for random seeds, splits, and checkpoint naming.
