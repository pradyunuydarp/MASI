"""Phase 4 recommender-side modeling modules.

This package now contains the first functional recommender foundation for MASI:

- SASRec-style sequential encoding for baseline parity,
- semantic-ID token serialization utilities,
- decoder-only generative recommendation modeling,
- cross-modal MLM pretraining scaffolding for the proposal's Phase 4.
"""

from masi.recommender.generative import GenerativeSIDRecommender
from masi.recommender.mlm import CrossModalMLMPretrainer
from masi.recommender.amazon_data import AmazonSequenceBuildResult, build_real_amazon_histories
from masi.recommender.evaluation import LeaveOneOutExample, LeaveOneOutSplit, build_leave_one_out_split, evaluate_generative_ranking
from masi.recommender.retrieval import RetrievalCandidate, RetrievalResult, build_item_token_lookup, rank_catalog_candidates
from masi.recommender.sasrec import SASRecConfig, SASRecModel
from masi.recommender.sequence_data import (
    CrossModalMLMExample,
    CrossModalMLMDataset,
    GenerativeTrainingExample,
    GenerativeSequenceDataset,
)
from masi.recommender.training import initialize_generative_from_mlm, run_training_epochs
from masi.recommender.vocabulary import FusedSemanticId, TokenVocabulary

__all__ = [
    "AmazonSequenceBuildResult",
    "CrossModalMLMExample",
    "CrossModalMLMDataset",
    "CrossModalMLMPretrainer",
    "FusedSemanticId",
    "LeaveOneOutExample",
    "LeaveOneOutSplit",
    "GenerativeSequenceDataset",
    "GenerativeSIDRecommender",
    "GenerativeTrainingExample",
    "RetrievalCandidate",
    "RetrievalResult",
    "SASRecConfig",
    "SASRecModel",
    "TokenVocabulary",
    "build_item_token_lookup",
    "build_leave_one_out_split",
    "build_real_amazon_histories",
    "evaluate_generative_ranking",
    "initialize_generative_from_mlm",
    "rank_catalog_candidates",
    "run_training_epochs",
]
