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

## Current Repository Structure

- `configs/`: JSON configs for reproducible preprocessing and, later, training runs
- `data/raw/`: expected location for source datasets and local demo inputs
- `data/processed/`: filtered outputs ready for later CLIP extraction and graph construction
- `docs/`: technical notes and future deviation logs
- `notebooks/`: demo notebooks that exercise repository workflows
- `outputs/`: manifests, summaries, and future experiment results
- `scripts/`: CLI entrypoints that wrap reusable library code
- `src/masi/`: library modules for data, alignment, tokenization, and recommender stages

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

## Setup

The current preprocessing slice is standard-library only. No third-party Python packages are required for the verified CLI workflow.

Optional package install for future work:

```bash
python3 -m pip install -e .
```

For recommender development, create the local virtual environment used for the verified demo:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[recommender]"
```

If you want to open the notebooks locally, install Jupyter inside that environment:

```bash
.venv/bin/python -m pip install jupyter
```

## Usage

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
PYTHONPATH=src .venv/bin/python scripts/demo_recommender_foundation.py --config configs/recommender_demo.json
```

Open the demo notebook:

- [notebooks/01_dataset_and_feature_prep_demo.ipynb](/Users/pradyundevarakonda/Developer/MASI/notebooks/01_dataset_and_feature_prep_demo.ipynb)
- [notebooks/02_recommender_foundation_demo.ipynb](/Users/pradyundevarakonda/Developer/MASI/notebooks/02_recommender_foundation_demo.ipynb)

## Initial Execution Plan

The proposal provides an 8-week implementation arc:

- Weeks 1-2: dataset ETL and CLIP feature extraction
- Weeks 3-4: user-item graph construction, mixed negative sampling, projection-head training
- Weeks 5-6: dual RQ-VAE training and recommender MLM pretraining
- Weeks 7-8: generative fine-tuning, ablations, evaluation, and paper finalization

## Working Principle

This repository is intended to be advanced incrementally. AI agents should treat the proposal as the source specification, keep documentation current, and update [TODO_TASKS.md](/Users/pradyundevarakonda/Developer/MASI/TODO_TASKS.md) whenever they complete, refine, or discover work.
