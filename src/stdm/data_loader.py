"""Data loading utilities for gene expression data."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from pathlib import Path


class GeneExpressionLoader:
    """
    Load and preprocess gene expression data for sparse tensor decomposition.
    
    Handles data in the format:
    - Genes: ~10k
    - Species: 3
    - Time points: 4
    """
    
    def __init__(self, n_genes: int = 10000, n_species: int = 3, 
                 n_timepoints: int = 4):
        """
        Initialize the gene expression loader.
        
        Parameters
        ----------
        n_genes : int, default=10000
            Number of genes
        n_species : int, default=3
            Number of species
        n_timepoints : int, default=4
            Number of time points
        """
        self.n_genes = n_genes
        self.n_species = n_species
        self.n_timepoints = n_timepoints
        self.gene_names: Optional[List[str]] = None
        self.species_names: Optional[List[str]] = None
        self.timepoint_names: Optional[List[str]] = None
    
    def load_from_csv(self, filepath: str, 
                     gene_col: str = "gene",
                     species_col: str = "species",
                     timepoint_col: str = "timepoint",
                     value_col: str = "expression") -> np.ndarray:
        """
        Load gene expression data from CSV file.
        
        Expected CSV format:
        gene,species,timepoint,expression
        gene1,species1,t0,0.5
        gene1,species1,t1,0.7
        ...
        
        Parameters
        ----------
        filepath : str
            Path to CSV file
        gene_col : str, default="gene"
            Column name for genes
        species_col : str, default="species"
            Column name for species
        timepoint_col : str, default="timepoint"
            Column name for timepoints
        value_col : str, default="expression"
            Column name for expression values
            
        Returns
        -------
        np.ndarray
            Tensor of shape (n_genes, n_species, n_timepoints)
        """
        df = pd.read_csv(filepath)
        
        # Extract unique values
        self.gene_names = sorted(df[gene_col].unique())
        self.species_names = sorted(df[species_col].unique())
        self.timepoint_names = sorted(df[timepoint_col].unique())
        
        # Update dimensions
        self.n_genes = len(self.gene_names)
        self.n_species = len(self.species_names)
        self.n_timepoints = len(self.timepoint_names)
        
        # Create tensor
        tensor = np.zeros((self.n_genes, self.n_species, self.n_timepoints))
        
        # Create index mappings
        gene_to_idx = {g: i for i, g in enumerate(self.gene_names)}
        species_to_idx = {s: i for i, s in enumerate(self.species_names)}
        timepoint_to_idx = {t: i for i, t in enumerate(self.timepoint_names)}
        
        # Fill tensor
        for _, row in df.iterrows():
            i = gene_to_idx[row[gene_col]]
            j = species_to_idx[row[species_col]]
            k = timepoint_to_idx[row[timepoint_col]]
            tensor[i, j, k] = row[value_col]
        
        return tensor
    
    def generate_synthetic_data(self, sparsity: float = 0.3, 
                               noise_level: float = 0.1,
                               seed: Optional[int] = None) -> np.ndarray:
        """
        Generate synthetic gene expression data for testing.
        
        Parameters
        ----------
        sparsity : float, default=0.3
            Fraction of zero entries (0.0 to 1.0)
        noise_level : float, default=0.1
            Standard deviation of Gaussian noise to add
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        np.ndarray
            Synthetic tensor of shape (n_genes, n_species, n_timepoints)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate gene names, species names, and timepoint names
        self.gene_names = [f"gene_{i+1}" for i in range(self.n_genes)]
        self.species_names = [f"species_{i+1}" for i in range(self.n_species)]
        self.timepoint_names = [f"t{i}" for i in range(self.n_timepoints)]
        
        # Create low-rank structure with rank 3
        rank = 3
        gene_factors = np.random.randn(self.n_genes, rank)
        species_factors = np.random.randn(self.n_species, rank)
        time_factors = np.random.randn(self.n_timepoints, rank)
        
        # Construct tensor from factors
        tensor = np.zeros((self.n_genes, self.n_species, self.n_timepoints))
        for r in range(rank):
            tensor += np.outer(gene_factors[:, r], 
                              np.outer(species_factors[:, r], 
                                      time_factors[:, r]).ravel()).reshape(
                self.n_genes, self.n_species, self.n_timepoints)
        
        # Make all values positive (gene expression is non-negative)
        tensor = np.abs(tensor)
        
        # Add noise
        if noise_level > 0:
            noise = np.random.randn(*tensor.shape) * noise_level
            tensor += noise
            tensor = np.maximum(tensor, 0)  # Ensure non-negative
        
        # Apply sparsity
        if sparsity > 0:
            mask = np.random.rand(*tensor.shape) < sparsity
            tensor[mask] = 0
        
        return tensor
    
    def save_to_csv(self, tensor: np.ndarray, filepath: str) -> None:
        """
        Save tensor to CSV file in long format.
        
        Parameters
        ----------
        tensor : np.ndarray
            Tensor of shape (n_genes, n_species, n_timepoints)
        filepath : str
            Path to save CSV file
        """
        if self.gene_names is None:
            self.gene_names = [f"gene_{i+1}" for i in range(tensor.shape[0])]
        if self.species_names is None:
            self.species_names = [f"species_{i+1}" for i in range(tensor.shape[1])]
        if self.timepoint_names is None:
            self.timepoint_names = [f"t{i}" for i in range(tensor.shape[2])]
        
        # Create long-format dataframe
        data = []
        for i, gene in enumerate(self.gene_names):
            for j, species in enumerate(self.species_names):
                for k, timepoint in enumerate(self.timepoint_names):
                    data.append({
                        "gene": gene,
                        "species": species,
                        "timepoint": timepoint,
                        "expression": tensor[i, j, k]
                    })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    def preprocess(self, tensor: np.ndarray, 
                  log_transform: bool = True,
                  standardize: bool = True) -> np.ndarray:
        """
        Preprocess gene expression tensor.
        
        Parameters
        ----------
        tensor : np.ndarray
            Raw expression tensor
        log_transform : bool, default=True
            Apply log2(x+1) transformation
        standardize : bool, default=True
            Standardize each gene to zero mean and unit variance
            
        Returns
        -------
        np.ndarray
            Preprocessed tensor
        """
        result = tensor.copy()
        
        # Log transformation
        if log_transform:
            result = np.log2(result + 1)
        
        # Standardize genes (across species and timepoints)
        if standardize:
            for i in range(result.shape[0]):
                gene_data = result[i, :, :].ravel()
                mean = np.mean(gene_data)
                std = np.std(gene_data)
                if std > 0:
                    result[i, :, :] = (result[i, :, :] - mean) / std
        
        return result
    
    def get_tensor_info(self, tensor: np.ndarray) -> Dict:
        """
        Get information about the tensor.
        
        Parameters
        ----------
        tensor : np.ndarray
            Input tensor
            
        Returns
        -------
        dict
            Dictionary with tensor statistics
        """
        sparsity = np.sum(tensor == 0) / tensor.size
        
        return {
            "shape": tensor.shape,
            "n_genes": tensor.shape[0],
            "n_species": tensor.shape[1],
            "n_timepoints": tensor.shape[2],
            "sparsity": sparsity,
            "min_value": np.min(tensor),
            "max_value": np.max(tensor),
            "mean_value": np.mean(tensor),
            "std_value": np.std(tensor),
        }
