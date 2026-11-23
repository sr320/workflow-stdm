"""Sparse CP decomposition (vendored and adapted from Barnacle; MIT licence)."""

from .als import SparseCP, als_lasso
from .fms import factor_match_score

__all__ = ["SparseCP", "als_lasso", "factor_match_score"]
