"""Basic tests for the sparse tensor decomposition workflow."""

import numpy as np
from pathlib import Path
import tempfile
import os
import sys

from stdm import SparseTensorDecomposer, GeneExpressionLoader


class TestGeneExpressionLoader:
    """Test the GeneExpressionLoader class."""
    
    def test_initialization(self):
        """Test loader initialization."""
        loader = GeneExpressionLoader(n_genes=100, n_species=3, n_timepoints=4)
        assert loader.n_genes == 100
        assert loader.n_species == 3
        assert loader.n_timepoints == 4
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        loader = GeneExpressionLoader(n_genes=100, n_species=3, n_timepoints=4)
        tensor = loader.generate_synthetic_data(sparsity=0.3, seed=42)
        
        assert tensor.shape == (100, 3, 4)
        assert np.all(tensor >= 0)  # Gene expression should be non-negative
        
        # Check sparsity is approximately correct
        sparsity = np.sum(tensor == 0) / tensor.size
        assert 0.2 <= sparsity <= 0.4  # Allow some variance
    
    def test_preprocessing(self):
        """Test data preprocessing."""
        loader = GeneExpressionLoader(n_genes=100, n_species=3, n_timepoints=4)
        tensor = loader.generate_synthetic_data(seed=42)
        
        # Test log transformation
        processed = loader.preprocess(tensor, log_transform=True, standardize=False)
        assert np.all(processed >= 0)  # log2(x+1) should be non-negative
        
        # Test standardization
        processed = loader.preprocess(tensor, log_transform=False, standardize=True)
        # Each gene should have approximately zero mean (allowing for sparse data)
        gene_means = np.mean(processed, axis=(1, 2))
        assert np.abs(np.mean(gene_means)) < 0.5
    
    def test_csv_io(self):
        """Test CSV reading and writing."""
        loader = GeneExpressionLoader(n_genes=10, n_species=2, n_timepoints=3)
        tensor = loader.generate_synthetic_data(seed=42)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save
            loader.save_to_csv(tensor, temp_path)
            assert os.path.exists(temp_path)
            
            # Load
            loader2 = GeneExpressionLoader()
            tensor2 = loader2.load_from_csv(temp_path)
            
            assert tensor2.shape == tensor.shape
            np.testing.assert_array_almost_equal(tensor, tensor2)
        finally:
            os.unlink(temp_path)
    
    def test_tensor_info(self):
        """Test tensor info retrieval."""
        loader = GeneExpressionLoader(n_genes=100, n_species=3, n_timepoints=4)
        tensor = loader.generate_synthetic_data(seed=42)
        
        info = loader.get_tensor_info(tensor)
        
        assert info['shape'] == (100, 3, 4)
        assert info['n_genes'] == 100
        assert info['n_species'] == 3
        assert info['n_timepoints'] == 4
        assert 'sparsity' in info
        assert 'min_value' in info
        assert 'max_value' in info


class TestSparseTensorDecomposer:
    """Test the SparseTensorDecomposer class."""
    
    def test_initialization(self):
        """Test decomposer initialization."""
        decomposer = SparseTensorDecomposer(rank=5, method="parafac")
        assert decomposer.rank == 5
        assert decomposer.method == "parafac"
        
        # Test invalid method
        try:
            SparseTensorDecomposer(rank=5, method="invalid")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
    
    def test_parafac_decomposition(self):
        """Test PARAFAC decomposition."""
        loader = GeneExpressionLoader(n_genes=100, n_species=3, n_timepoints=4)
        tensor = loader.generate_synthetic_data(seed=42)
        tensor = loader.preprocess(tensor, log_transform=True, standardize=True)
        
        decomposer = SparseTensorDecomposer(rank=5, method="parafac")
        decomposer.fit(tensor, n_iter_max=20)
        
        # Check factors exist and have correct shapes
        assert decomposer.factors_ is not None
        assert decomposer.get_gene_factors().shape == (100, 5)
        assert decomposer.get_species_factors().shape == (3, 5)
        assert decomposer.get_time_factors().shape == (4, 5)
        
        # Check reconstruction error is reasonable
        assert 0 <= decomposer.reconstruction_error_ <= 1
    
    def test_tucker_decomposition(self):
        """Test Tucker decomposition."""
        loader = GeneExpressionLoader(n_genes=100, n_species=3, n_timepoints=4)
        tensor = loader.generate_synthetic_data(seed=42)
        tensor = loader.preprocess(tensor, log_transform=True, standardize=True)
        
        decomposer = SparseTensorDecomposer(rank=5, method="tucker")
        decomposer.fit(tensor, n_iter_max=20)
        
        # Check factors and core exist
        assert decomposer.factors_ is not None
        assert decomposer.core_ is not None
        
        # Tucker may have different factor shapes for species and time
        # (limited by dimension size)
        gene_factors = decomposer.get_gene_factors()
        assert gene_factors.shape[0] == 100
        assert gene_factors.shape[1] == 5
        
        # Check reconstruction
        reconstructed = decomposer.reconstruct()
        assert reconstructed.shape == tensor.shape
    
    def test_reconstruction(self):
        """Test tensor reconstruction."""
        loader = GeneExpressionLoader(n_genes=50, n_species=3, n_timepoints=4)
        tensor = loader.generate_synthetic_data(seed=42)
        tensor = loader.preprocess(tensor, log_transform=True, standardize=True)
        
        decomposer = SparseTensorDecomposer(rank=3, method="parafac")
        decomposer.fit(tensor, n_iter_max=50)
        
        reconstructed = decomposer.reconstruct()
        
        # Reconstructed tensor should have same shape
        assert reconstructed.shape == tensor.shape
        
        # Reconstruction error should match calculated error
        manual_error = np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)
        np.testing.assert_almost_equal(manual_error, decomposer.reconstruction_error_, decimal=6)
    
    def test_summary(self):
        """Test summary generation."""
        loader = GeneExpressionLoader(n_genes=50, n_species=3, n_timepoints=4)
        tensor = loader.generate_synthetic_data(seed=42)
        
        decomposer = SparseTensorDecomposer(rank=3, method="parafac")
        decomposer.fit(tensor, n_iter_max=10)
        
        summary = decomposer.get_summary()
        
        assert 'method' in summary
        assert 'rank' in summary
        assert 'reconstruction_error' in summary
        assert summary['method'] == 'parafac'
        assert summary['rank'] == 3
    
    def test_fit_before_methods(self):
        """Test that methods fail before fitting."""
        decomposer = SparseTensorDecomposer(rank=3)
        
        try:
            decomposer.get_gene_factors()
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        try:
            decomposer.reconstruct()
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        try:
            decomposer.get_summary()
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_full_workflow_small(self):
        """Test complete workflow with small dataset."""
        # Generate data
        loader = GeneExpressionLoader(n_genes=100, n_species=3, n_timepoints=4)
        tensor = loader.generate_synthetic_data(sparsity=0.3, seed=42)
        
        # Preprocess
        tensor = loader.preprocess(tensor, log_transform=True, standardize=True)
        
        # Decompose with PARAFAC
        decomposer = SparseTensorDecomposer(rank=5, method="parafac")
        decomposer.fit(tensor, n_iter_max=20)
        
        # Verify results
        assert decomposer.reconstruction_error_ < 1.0
        gene_factors = decomposer.get_gene_factors()
        assert not np.isnan(gene_factors).any()
        assert not np.isinf(gene_factors).any()
    
    def test_full_workflow_specs(self):
        """Test with exact specifications: 10k genes, 3 species, 4 time points."""
        # This is the key integration test for the requirements
        loader = GeneExpressionLoader(n_genes=10000, n_species=3, n_timepoints=4)
        tensor = loader.generate_synthetic_data(sparsity=0.3, seed=42)
        
        # Verify tensor shape
        assert tensor.shape == (10000, 3, 4)
        
        # Preprocess
        tensor = loader.preprocess(tensor, log_transform=True, standardize=True)
        
        # Decompose
        decomposer = SparseTensorDecomposer(rank=5, method="parafac")
        decomposer.fit(tensor, n_iter_max=10)  # Use fewer iterations for testing
        
        # Verify factors
        assert decomposer.get_gene_factors().shape == (10000, 5)
        assert decomposer.get_species_factors().shape == (3, 5)
        assert decomposer.get_time_factors().shape == (4, 5)
        
        # Verify reconstruction works
        reconstructed = decomposer.reconstruct()
        assert reconstructed.shape == (10000, 3, 4)
        
        print(f"✓ Successfully handled 10k genes, 3 species, 4 time points")
        print(f"  Reconstruction error: {decomposer.reconstruction_error_:.6f}")


if __name__ == "__main__":
    # Run basic tests without pytest
    print("Running basic tests...")
    
    # Test data generation
    print("\n1. Testing data generation...")
    test_loader = TestGeneExpressionLoader()
    test_loader.test_synthetic_data_generation()
    print("   ✓ Data generation works")
    
    # Test decomposition
    print("\n2. Testing PARAFAC decomposition...")
    test_decomp = TestSparseTensorDecomposer()
    test_decomp.test_parafac_decomposition()
    print("   ✓ PARAFAC decomposition works")
    
    # Test full workflow
    print("\n3. Testing full workflow with specs...")
    test_integration = TestIntegration()
    test_integration.test_full_workflow_specs()
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
