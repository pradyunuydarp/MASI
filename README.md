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

## Initial Execution Plan

The proposal provides an 8-week implementation arc:

- Weeks 1-2: dataset ETL and CLIP feature extraction
- Weeks 3-4: user-item graph construction, mixed negative sampling, projection-head training
- Weeks 5-6: dual RQ-VAE training and recommender MLM pretraining
- Weeks 7-8: generative fine-tuning, ablations, evaluation, and paper finalization

## Working Principle

This repository is intended to be advanced incrementally. AI agents should treat the proposal as the source specification, keep documentation current, and update [TODO_TASKS.md](/Users/pradyundevarakonda/Developer/MASI/TODO_TASKS.md) whenever they complete, refine, or discover work.
