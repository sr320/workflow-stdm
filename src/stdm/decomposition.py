"""Sparse tensor decomposition module optimized for gene expression data."""

import numpy as np
from typing import Optional, Tuple, Dict, Any
import tensorly as tl
from tensorly.decomposition import parafac, tucker
from scipy import sparse


class SparseTensorDecomposer:
    """
    Sparse tensor decomposition for multi-species gene expression analysis.
    
    Optimized for handling gene expression data with dimensions:
    - Genes: ~10k
    - Species: 3
    - Time points: 4
    
    The tensor shape is (genes, species, time_points).
    """
    
    def __init__(self, rank: int = 5, method: str = "parafac", 
                 sparsity_threshold: float = 0.01, normalize: bool = True):
        """
        Initialize the sparse tensor decomposer.
        
        Parameters
        ----------
        rank : int, default=5
            Number of components/rank for the decomposition
        method : str, default="parafac"
            Decomposition method: "parafac" (CP decomposition) or "tucker"
        sparsity_threshold : float, default=0.01
            Threshold for sparsifying the tensor (values below this are set to 0)
        normalize : bool, default=True
            Whether to normalize the input tensor
        """
        self.rank = rank
        self.method = method.lower()
        self.sparsity_threshold = sparsity_threshold
        self.normalize = normalize
        self.factors_ = None
        self.core_ = None
        self.reconstruction_error_ = None
        
        if self.method not in ["parafac", "tucker"]:
            raise ValueError(f"Method must be 'parafac' or 'tucker', got {method}")
    
    def fit(self, tensor: np.ndarray, n_iter_max: int = 100, 
            tol: float = 1e-7, verbose: bool = False) -> "SparseTensorDecomposer":
        """
        Fit the sparse tensor decomposition model.
        
        Parameters
        ----------
        tensor : np.ndarray
            Input tensor of shape (genes, species, time_points)
        n_iter_max : int, default=100
            Maximum number of iterations
        tol : float, default=1e-7
            Convergence tolerance
        verbose : bool, default=False
            Whether to print progress
            
        Returns
        -------
        self : SparseTensorDecomposer
            Fitted decomposer
        """
        # Validate input shape
        if tensor.ndim != 3:
            raise ValueError(f"Expected 3D tensor, got {tensor.ndim}D")
        
        # Apply sparsity threshold
        sparse_tensor = tensor.copy()
        sparse_tensor[np.abs(sparse_tensor) < self.sparsity_threshold] = 0
        
        # Normalize if requested
        if self.normalize:
            sparse_tensor = self._normalize_tensor(sparse_tensor)
        
        # Perform decomposition
        if self.method == "parafac":
            self._fit_parafac(sparse_tensor, n_iter_max, tol, verbose)
        elif self.method == "tucker":
            self._fit_tucker(sparse_tensor, n_iter_max, tol, verbose)
        
        # Calculate reconstruction error
        reconstructed = self.reconstruct()
        self.reconstruction_error_ = np.linalg.norm(sparse_tensor - reconstructed) / np.linalg.norm(sparse_tensor)
        
        return self
    
    def _normalize_tensor(self, tensor: np.ndarray) -> np.ndarray:
        """Normalize tensor to have unit Frobenius norm."""
        norm = np.linalg.norm(tensor)
        if norm > 0:
            return tensor / norm
        return tensor
    
    def _fit_parafac(self, tensor: np.ndarray, n_iter_max: int, 
                     tol: float, verbose: bool) -> None:
        """Fit PARAFAC (CP) decomposition."""
        # Convert to tensorly format
        tl_tensor = tl.tensor(tensor)
        
        # Use init='random' for better memory efficiency with large tensors
        # Perform PARAFAC decomposition
        factors = parafac(tl_tensor, rank=self.rank, n_iter_max=n_iter_max, 
                         tol=tol, verbose=verbose, init='random')
        
        # Store factors
        self.factors_ = factors.factors
        self.core_ = None  # PARAFAC doesn't have a core tensor
    
    def _fit_tucker(self, tensor: np.ndarray, n_iter_max: int, 
                    tol: float, verbose: bool) -> None:
        """Fit Tucker decomposition."""
        # Convert to tensorly format
        tl_tensor = tl.tensor(tensor)
        
        # Define ranks for each mode
        ranks = [min(self.rank, s) for s in tensor.shape]
        
        # Use init='random' for better memory efficiency
        # Perform Tucker decomposition
        core, factors = tucker(tl_tensor, rank=ranks, n_iter_max=n_iter_max, 
                              tol=tol, verbose=verbose, init='random')
        
        # Store factors and core
        self.factors_ = factors
        self.core_ = core
    
    def reconstruct(self) -> np.ndarray:
        """
        Reconstruct the tensor from the decomposition.
        
        Returns
        -------
        np.ndarray
            Reconstructed tensor
        """
        if self.factors_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        if self.method == "parafac":
            # PARAFAC reconstruction: outer product of factors
            reconstructed = tl.cp_to_tensor((None, self.factors_))
        elif self.method == "tucker":
            # Tucker reconstruction: core tensor times factor matrices
            reconstructed = tl.tucker_to_tensor((self.core_, self.factors_))
        
        return reconstructed
    
    def get_gene_factors(self) -> np.ndarray:
        """
        Get the factor matrix for genes (first mode).
        
        Returns
        -------
        np.ndarray
            Gene factor matrix of shape (n_genes, rank)
        """
        if self.factors_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.factors_[0]
    
    def get_species_factors(self) -> np.ndarray:
        """
        Get the factor matrix for species (second mode).
        
        Returns
        -------
        np.ndarray
            Species factor matrix of shape (n_species, rank)
        """
        if self.factors_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.factors_[1]
    
    def get_time_factors(self) -> np.ndarray:
        """
        Get the factor matrix for time points (third mode).
        
        Returns
        -------
        np.ndarray
            Time factor matrix of shape (n_timepoints, rank)
        """
        if self.factors_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.factors_[2]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the decomposition results.
        
        Returns
        -------
        dict
            Summary statistics including reconstruction error and factor shapes
        """
        if self.factors_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        summary = {
            "method": self.method,
            "rank": self.rank,
            "reconstruction_error": self.reconstruction_error_,
            "gene_factors_shape": self.factors_[0].shape,
            "species_factors_shape": self.factors_[1].shape,
            "time_factors_shape": self.factors_[2].shape,
        }
        
        if self.method == "tucker" and self.core_ is not None:
            summary["core_shape"] = self.core_.shape
        
        return summary
