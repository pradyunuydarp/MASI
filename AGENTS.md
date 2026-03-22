# AGENTS.md

## Purpose

This repository implements the research proposal for **MASI: Multimodal-Augmented Semantic Identifiers for Cold-Start Discovery in Generative Recommendation**. All AI agents working here must treat the proposal PDF as the primary specification and keep repository documentation synchronized with implementation progress.

Source proposal:
- [Multimodal_Augmented_Semantic_Identifiers_for_Cold_Start_Discovery_in_Generative_Recommendation__Research_Proposal.pdf](/Users/pradyundevarakonda/Developer/MASI/Multimodal_Augmented_Semantic_Identifiers_for_Cold_Start_Discovery_in_Generative_Recommendation__Research_Proposal.pdf)

## What MASI Is

MASI is a multimodal generative recommendation framework designed to improve cold-start discovery in visual-heavy domains. The architecture proposed in the PDF has three main phases:

1. Behavior-aware contrastive alignment over frozen CLIP text and vision encoders with trainable projection heads.
2. Independent modality quantization with separate text and visual RQ-VAE codebooks.
3. Recommender-side cross-modal token alignment followed by sequential fine-tuning.

The research claim is that late-fused dual codebooks plus behavioral alignment should reduce modality collapse and improve zero-shot recommendation quality.

## Agent Mission

When you work in this repository, your job is to convert the proposal into a reproducible research codebase and keep the project legible for future agents and collaborators.

You must optimize for:

- faithfulness to the proposal,
- reproducibility,
- measurable progress,
- explicit assumptions,
- minimal undocumented work.

## Required Operating Rules

All AI agents must follow these instructions:

1. Read the proposal PDF before making architectural decisions.
2. Prefer implementing work in proposal order unless a dependency forces a different sequence.
3. Record meaningful progress in [TODO_TASKS.md](/Users/pradyundevarakonda/Developer/MASI/TODO_TASKS.md) during the same task where the work is done.
4. Update [README.md](/Users/pradyundevarakonda/Developer/MASI/README.md) whenever the repository structure, setup flow, or research status materially changes.
5. Do not silently diverge from the proposal. If you intentionally change the design, document:
   - what changed,
   - why it changed,
   - what proposal assumption was relaxed.
6. Keep experiments reproducible. New training or evaluation code should come with configuration, expected inputs, and output locations.
7. Treat `TODO_TASKS.md` as the canonical short-form project ledger for agent handoff.
8. When adding code, include enough documentation that another agent can continue without re-deriving context from scratch.

## TODO_TASKS.md Maintenance Contract

AI agents are responsible for maintaining [TODO_TASKS.md](/Users/pradyundevarakonda/Developer/MASI/TODO_TASKS.md).

Required behavior:

- Mark tasks as `TODO`, `IN_PROGRESS`, `BLOCKED`, or `DONE`.
- Add newly discovered tasks when implementation reveals missing prerequisites.
- Keep task wording concrete and action-oriented.
- When closing a task, note the artifact produced, such as a module, config, dataset script, experiment result, or document.
- If a task becomes obsolete, do not delete it silently; mark it as superseded and explain briefly.
- Keep the list concise enough for fast handoff, but complete enough that another agent can continue immediately.

Suggested update pattern:

- move completed items to `DONE`,
- add follow-up tasks directly below the parent area,
- refresh blockers and assumptions,
- keep progress aligned to proposal phases and evaluation milestones.

## Recommended Repository Shape

As the codebase grows, prefer a structure close to:

- `data/` for ETL and dataset preparation
- `configs/` for experiment and training configuration
- `src/` for core implementation
- `src/alignment/` for Phase 1 behavior-aware contrastive learning
- `src/tokenization/` for Phase 2 dual RQ-VAE codebooks
- `src/recommender/` for Phase 3 MLM pretraining and sequential fine-tuning
- `scripts/` for executable workflows
- `experiments/` or `outputs/` for tracked results and generated artifacts
- `docs/` for design notes, ablations, and implementation decisions

This is guidance, not a hard constraint. If you choose a different layout, document it in the README.

## Implementation Priorities

Prioritize work in this order unless blocked:

1. Dataset ingestion and filtering for Amazon Reviews 2023 Clothing/Shoes/Jewelry
2. CLIP feature extraction pipeline
3. User-item graph construction and mixed negative sampling
4. Projection-head training for behavior-aware alignment
5. Independent text and vision RQ-VAE training
6. Late-fused semantic ID generation
7. Cross-modal MLM pretraining on the recommender backbone
8. Sequential fine-tuning and evaluation
9. Ablations on Phase 1 vs Phase 3 and ID depth `D in {2,3,4}`
10. Result packaging for manuscript-ready reporting

## Definition Of Done

A work item should usually not be marked `DONE` unless:

- code or documentation exists in the repository,
- usage is discoverable,
- inputs and outputs are defined,
- obvious follow-up work is tracked in `TODO_TASKS.md`.

## Research Targets To Preserve

Agents should keep these proposal targets visible while implementing:

- improve cold-start `HR@10` over the stated baselines,
- improve `Coverage@10`,
- preserve manageable inference latency,
- validate that late fusion reduces modality interference,
- test whether Phase 1 and Phase 3 provide complementary gains.

## Handoff Standard

Before ending a task, update the repository so the next agent can answer these questions quickly:

- What was completed?
- What remains?
- What is blocked?
- What assumptions or deviations were introduced?
- Where are the outputs or artifacts?

If those answers are not obvious from the repo state, the task is not fully handed off.
