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
- apply iterative k-core filtering,
- standardize concatenated item text fields for future CLIP text encoding,
- summarize missing text and image coverage before feature extraction.

Expected outputs:

- `data/processed/<run>/metadata.filtered.jsonl`
- `data/processed/<run>/interactions.filtered.jsonl`
- `outputs/<run>/dataset_manifest.json`

### Phase 2: Behavior-Aware Contrastive Alignment

Planned modules:

- CLIP embedding extraction wrappers for text and images
- user-item bipartite graph construction
- mixed negative sampler with in-batch and hard graph negatives
- trainable projection heads on top of frozen CLIP encoders

Expected outputs:

- projected text embeddings
- projected visual embeddings
- graph sampling caches
- alignment checkpoints and training metrics

### Phase 3: Independent Modality Quantization

Planned modules:

- text RQ-VAE encoder/decoder and residual codebooks
- vision RQ-VAE encoder/decoder and residual codebooks
- late-fusion semantic ID builder with `[TXT]` and `[VIS]` markers

Expected outputs:

- text semantic ID tables
- visual semantic ID tables
- fused semantic IDs for recommendation training

### Phase 4: Recommender-Side Cross-Modal Alignment

Current implementation artifacts:

- `src/masi/recommender/vocabulary.py`
- `src/masi/recommender/sequence_data.py`
- `src/masi/recommender/sasrec.py`
- `src/masi/recommender/generative.py`
- `src/masi/recommender/mlm.py`
- `src/masi/recommender/training.py`
- `scripts/demo_recommender_foundation.py`
- `configs/recommender_demo.json`

Implemented foundation:

- fused semantic-ID vocabulary with modality markers,
- serialized user-history token datasets for generative recommendation,
- item-level cross-modal MLM datasets for text-to-visual and visual-to-text reconstruction,
- SASRec-style sequential baseline,
- decoder-style semantic-ID generator for next-item token prediction.

Remaining modules:

- chronological fine-tuning on real user sequences,
- decoding and retrieval against the full item catalog,
- evaluation hooks for token-level reconstruction and ranking metrics.

Expected outputs:

- MLM checkpoints
- sequential recommendation checkpoints
- evaluation predictions and metrics

## Assumptions

- The repository will use a custom PyTorch recommender foundation first, then optionally add RecBole adapters where they simplify baseline comparison or evaluation.
- The exact CLIP variant remains undecided and is tracked in `TODO_TASKS.md`.
- Large raw datasets and checkpoints will remain outside git and be referenced by config paths.

## Known Deviations From The Proposal

One implementation choice now differs from the proposal's tentative mention of leveraging RecBole during Phase 3:

- What changed: the initial recommender foundation is implemented as custom PyTorch modules in-repo rather than starting inside RecBole.
- Why it changed: MASI needs explicit control over fused semantic-ID token serialization, cross-modal MLM masking, and decoder-style token generation, which are easier to prototype faithfully in a small local stack before library integration.
- Relaxed assumption: RecBole is no longer assumed to be the primary first implementation surface; it is now treated as an optional integration path for baselines and evaluation.
