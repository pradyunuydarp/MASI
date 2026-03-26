# MASI Reference Papers And Repositories

This document collects the proposal references that are most relevant to MASI's recommender implementation, along with public repositories when they could be verified. It is not meant to be exhaustive for every citation in the proposal; it is the working map for the code currently being implemented.

## How This Is Used

- `SASRec` informs the sequential recommendation baseline.
- `TIGER` informs the generative semantic-ID training setup.
- `RQ-VAE` informs the discrete tokenization stage that feeds the recommender.
- `CLIP` informs the multimodal feature backbone used before quantization.
- `RecBole` remains a comparison/integration target, not the current primary implementation path.

## Verified Foundations

### SASRec

- Paper: *Self-Attentive Sequential Recommendation* (Kang and McAuley, 2018)
- Role in MASI: sequential baseline and reference for history encoding
- Public repo: [kang205/SASRec](https://github.com/kang205/SASRec)

### P5

- Paper: *Recommendation as Language Processing* (Geng et al., 2022)
- Role in MASI: historical context for prompt-based generative recommendation before semantic IDs
- Public repo: [jeykigung/P5](https://github.com/jeykigung/P5)

### TIGER

- Paper: *Recommender Systems with Generative Retrieval* (Rajput et al., 2023)
- Role in MASI: semantic-ID autoregressive generation logic
- Public repo status: no clearly official public repository was verified during this pass
- Practical fallback repo inspected for surrounding ecosystem context: [meta-recsys/generative-recommenders](https://github.com/meta-recsys/generative-recommenders)

### Better Generalization With Semantic IDs

- Paper: *Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations* (Singh et al., 2024)
- Role in MASI: rationale for semantic-ID generalization and cold-start transfer
- Public repo status: no verified official repository surfaced during this pass

### CLIP

- Paper: *Learning Transferable Visual Models From Natural Language Supervision* (Radford et al., 2021)
- Role in MASI: frozen text and vision encoders used before behavior-aware projection
- Public repo: [openai/CLIP](https://github.com/openai/CLIP)

### RQ-VAE

- Paper: *Autoregressive Image Generation using Residual Quantization* (Lee et al., 2022)
- Role in MASI: residual-quantization mechanism adapted for semantic-ID generation
- Public repo: [kakaobrain/rq-vae-transformer](https://github.com/kakaobrain/rq-vae-transformer)

### RecBole

- Project: *RecBole: A Unified Recommendation Library*
- Role in MASI: optional future baseline integration and evaluation harness
- Public repo: [RUCAIBox/RecBole](https://github.com/RUCAIBox/RecBole)

## Proposal-Era Multimodal SID Papers

The following proposal citations were searched for, but repository availability remains unclear from public sources reviewed during this pass:

- *Beyond Unimodal Boundaries: Generative Recommendation with Multimodal Semantics* (Zhu et al., 2025)
- *MMQ: Multimodal Mixture-of-Quantization Tokenization for Semantic ID Generation and User Behavioral Adaptation* (Xu et al., 2025)
- *CEMG: Collaborative-Enhanced Multimodal Generative Recommendation* (Lin et al., 2026)
- *Differentiable Semantic ID for Generative Recommendation* (Fu et al., 2026)
- *Unleash the Potential of Long Semantic IDs for Generative Recommendation* (Xia et al., 2026)

For MASI implementation, this means we can faithfully ground the recommender foundation on the earlier public baselines now, while treating the newer multimodal SID papers as paper-spec guidance until public code appears.
