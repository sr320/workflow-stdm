"""Sparse Tensor Decomposition for Multi-species gene expression data."""

__version__ = "0.1.0"

from .decomposition import SparseTensorDecomposer
from .data_loader import GeneExpressionLoader

__all__ = ["SparseTensorDecomposer", "GeneExpressionLoader"]

