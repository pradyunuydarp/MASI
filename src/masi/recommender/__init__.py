"""Phase 4 recommender-side modeling modules.

This package now contains the first functional recommender foundation for MASI:

- SASRec-style sequential encoding for baseline parity,
- semantic-ID token serialization utilities,
- decoder-only generative recommendation modeling,
- cross-modal MLM pretraining scaffolding for the proposal's Phase 4.
"""

from masi.recommender.generative import GenerativeSIDRecommender
from masi.recommender.mlm import CrossModalMLMPretrainer
from masi.recommender.sasrec import SASRecConfig, SASRecModel
from masi.recommender.sequence_data import (
    CrossModalMLMExample,
    CrossModalMLMDataset,
    GenerativeTrainingExample,
    GenerativeSequenceDataset,
)
from masi.recommender.vocabulary import FusedSemanticId, TokenVocabulary

__all__ = [
    "CrossModalMLMExample",
    "CrossModalMLMDataset",
    "CrossModalMLMPretrainer",
    "FusedSemanticId",
    "GenerativeSequenceDataset",
    "GenerativeSIDRecommender",
    "GenerativeTrainingExample",
    "SASRecConfig",
    "SASRecModel",
    "TokenVocabulary",
]
