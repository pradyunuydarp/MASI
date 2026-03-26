"""Top-level package for the MASI research codebase.

The repository is intentionally being built in proposal order. The first
implementation slice focuses on reproducible data contracts, dataset
validation, and demo-friendly preprocessing utilities that later phases can
reuse without re-deriving assumptions from the proposal PDF.
"""

__all__ = ["common", "data", "alignment", "tokenization", "recommender"]
