# MASI Technical Design Note

This document translates the proposal into an implementation-oriented plan with explicit module boundaries, data contracts, and early assumptions. It is the first proposal-to-code artifact in the repository and should be revised as the codebase grows.

## Design Goals

- Preserve the proposal's phase ordering unless a dependency forces a reorder.
- Keep each stage reproducible through configuration files, deterministic outputs, and stable artifact paths.
- Avoid silent architectural drift from the proposal.

## Selected Early Stack

- Language: Python 3.10+
- Core data handling: Python standard-library records for the initial preprocessing slice
- Configuration: JSON files parsed into typed dataclasses
- Notebook support: Jupyter
- Modeling stack: custom PyTorch foundation for recommender modules and demos, with RecBole kept as an optional future adapter for baseline parity

## Repository Layout

- `configs/`: JSON configs for data prep and, later, training/evaluation runs
- `data/raw/`: expected home for unmodified source datasets
- `data/processed/`: filtered interactions and item metadata
- `docs/`: design notes and future deviation logs
- `notebooks/`: demo and analysis notebooks
- `outputs/`: JSON manifests, summaries, and future experiment artifacts
- `scripts/`: executable entrypoints that wrap the reusable library code
- `src/masi/data/`: dataset ingestion, validation, and filtering
- `src/masi/alignment/`: Phase 1 behavior-aware contrastive alignment
- `src/masi/tokenization/`: Phase 2 dual RQ-VAE modules
- `src/masi/recommender/`: Phase 3 recommender-side pretraining and fine-tuning
- `scripts/train_masi.py`: bounded-run launcher that resolves prepared subset inputs, storage, checkpoints, and stage execution

## Phase Mapping

### Phase 1: Data Ingestion And Feature Preparation

Current implementation artifacts:

- `src/masi/data/contracts.py`
- `src/masi/data/amazon2023.py`
- `scripts/build_dataset_manifest.py`
- `scripts/demo_phase1_prep.py`
- `configs/data_prep_demo.json`

Responsibilities:

- validate raw metadata and interaction schemas,
- apply iterative user-item k-core filtering before any bounded subset selection,
- standardize concatenated item text fields for future CLIP text encoding,
- summarize missing text and image coverage before feature extraction.

Expected outputs:

- `data/processed/<run>/metadata.filtered.jsonl`
- `data/processed/<run>/interactions.filtered.jsonl`
- `outputs/<run>/dataset_manifest.json`

### Phase 2: Behavior-Aware Contrastive Alignment

Current implementation artifacts:

- `src/masi/alignment/behavior_alignment.py`
- `src/masi/tokenization/masi_tokens.py`
- `scripts/build_masi_tokens.py`

Implemented foundation:

- frozen CLIP embedding extraction for text and images,
- user-history-derived positive item pairs,
- mixed negative sampling with in-batch negatives plus graph-hard negatives,
- trainable text and vision projection heads for behavior-aware alignment.

Expected outputs:

- projected text embeddings
- projected visual embeddings
- graph sampling caches
- alignment checkpoints and training metrics

### Phase 3: Independent Modality Quantization

Current implementation artifacts:

- `src/masi/tokenization/rqvae.py`
- `src/masi/tokenization/masi_tokens.py`
- `outputs/masi_tokens_amazon_csj_demo/fused_semantic_ids.jsonl`
- `outputs/amazon_csj_smoke_train/phase12_tokens/fused_semantic_ids.jsonl`

Implemented foundation:

- separate residual codebooks for text and image modalities,
- independent quantization over aligned embeddings,
- late fusion with `[TXT]` and `[VIS]` markers.

Expected outputs:

- text semantic ID tables
- visual semantic ID tables
- fused semantic IDs for recommendation training

### Phase 4: Recommender-Side Cross-Modal Alignment

Current implementation artifacts:

- `src/masi/recommender/vocabulary.py`
- `src/masi/recommender/sequence_data.py`
- `src/masi/recommender/amazon_data.py`
- `src/masi/recommender/sasrec.py`
- `src/masi/recommender/generative.py`
- `src/masi/recommender/mlm.py`
- `src/masi/recommender/evaluation.py`
- `src/masi/recommender/retrieval.py`
- `src/masi/recommender/training.py`
- `scripts/demo_recommender_foundation.py`
- `scripts/run_masi_experiment.py`
- `scripts/train_masi.py`
- `scripts/download_amazon_csj_dataset.py`
- `scripts/prepare_amazon_csj_subset.py`
- `scripts/download_amazon_csj_subset_images.py`
- `configs/recommender_demo.json`
- `configs/recommender_amazon_csj_demo.json`
- `configs/masi_experiment_amazon_csj_demo.json`
- `configs/masi_train_csj_full.json`
- `configs/masi_train_csj_smoke.json`
- `configs/masi_train_csj_subset_medium.json`
- `configs/masi_train_csj_subset_large.json`

Implemented foundation:

- fused semantic-ID vocabulary with modality markers,
- serialized user-history token datasets for generative recommendation,
- item-level cross-modal MLM datasets for text-to-visual and visual-to-text reconstruction,
- real Amazon review sequence import over `user_id`, `parent_asin`, and `timestamp`,
- SASRec-style sequential baseline,
- decoder-style semantic-ID generator for next-item token prediction.
- deterministic leave-one-out warm-start and zero-shot cold-start splitting,
- likelihood-based retrieval against the bounded item catalog,
- proposal-aligned ranking metrics and ablation toggles,
- automatic token-budget resolution for late-fused IDs,
- checkpoint emission for the MLM and generative fine-tuning stages,
- a one-click launcher that writes resolved configs and a run manifest.

Remaining modules:

- broader baseline reproduction beyond the existing SASRec implementation,
- larger-scale evaluation on a fuller Amazon subset,
- experiment-sweep automation for the remaining ablation matrix.

Expected outputs:

- MLM checkpoints
- sequential recommendation checkpoints
- evaluation predictions and metrics
- run manifests and resolved per-stage configs

## Assumptions

- The repository will use a custom PyTorch recommender foundation first, then optionally add RecBole adapters where they simplify baseline comparison or evaluation.
- `openai/clip-vit-base-patch32` is the current default CLIP variant for reproducible full-CSJ runs; larger variants remain future ablations.
- Large raw datasets and checkpoints will remain outside git and be referenced by config paths.

## Known Deviations From The Proposal

One implementation choice now differs from the proposal's tentative mention of leveraging RecBole during Phase 3:

- What changed: the initial recommender foundation is implemented as custom PyTorch modules in-repo rather than starting inside RecBole.
- Why it changed: MASI needs explicit control over fused semantic-ID token serialization, cross-modal MLM masking, and decoder-style token generation, which are easier to prototype faithfully in a small local stack before library integration.
- Relaxed assumption: RecBole is no longer assumed to be the primary first implementation surface; it is now treated as an optional integration path for baselines and evaluation.

There is a second temporary deviation forced by local resource and ingestion constraints:

- What changed: the smoke path can still fall back to review-side multimodal fields when the full metadata file is unavailable locally.
- Why it changed: the local development workspace does not always keep the full metadata file alongside the bounded review prefix.
- Relaxed assumption: the proposal-aligned full-CSJ launcher prefers the local metadata file, but the smoke path remains allowed to run on review-side text/image records so stage integration can still be regression-tested in low-storage environments.

There is a third intentional deviation in the current bounded training contract:

- What changed: the repository now treats prepared sliced CSJ datasets as the canonical training input for Kaggle and other bounded runs.
- Why it changed: re-downloading raw reviews, raw metadata, and product images inside ephemeral Kaggle sessions was the dominant operational bottleneck and reduced reproducibility.
- Relaxed assumption: the raw full-CSJ path remains a deferred reference target, while the actively maintained workflow optimizes for deterministic subset preparation, uploaded prepared datasets, and image reuse from read-only inputs.
