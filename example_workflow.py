#!/usr/bin/env python
"""
Example workflow for sparse tensor decomposition of gene expression data.

This script demonstrates the complete workflow:
1. Generate synthetic gene expression data
2. Preprocess the data
3. Perform tensor decomposition
4. Analyze results
"""

import numpy as np
from pathlib import Path

from stdm import SparseTensorDecomposer, GeneExpressionLoader


def main():
    print("=" * 70)
    print("Sparse Tensor Decomposition Workflow for Gene Expression Data")
    print("=" * 70)
    
    # Configuration
    n_genes = 10000
    n_species = 3
    n_timepoints = 4
    rank = 5
    sparsity = 0.3
    noise_level = 0.1
    seed = 42
    
    print("\nConfiguration:")
    print(f"  Genes: {n_genes}")
    print(f"  Species: {n_species}")
    print(f"  Time points: {n_timepoints}")
    print(f"  Decomposition rank: {rank}")
    print(f"  Sparsity: {sparsity}")
    print(f"  Noise level: {noise_level}")
    
    # Step 1: Generate synthetic data
    print("\n" + "=" * 70)
    print("Step 1: Generating synthetic gene expression data")
    print("=" * 70)
    
    loader = GeneExpressionLoader(
        n_genes=n_genes,
        n_species=n_species,
        n_timepoints=n_timepoints
    )
    
    tensor = loader.generate_synthetic_data(
        sparsity=sparsity,
        noise_level=noise_level,
        seed=seed
    )
    
    info = loader.get_tensor_info(tensor)
    print(f"\nGenerated tensor:")
    print(f"  Shape: {info['shape']}")
    print(f"  Sparsity: {info['sparsity']:.2%}")
    print(f"  Value range: [{info['min_value']:.4f}, {info['max_value']:.4f}]")
    print(f"  Mean: {info['mean_value']:.4f} ± {info['std_value']:.4f}")
    
    # Step 2: Preprocess data
    print("\n" + "=" * 70)
    print("Step 2: Preprocessing data")
    print("=" * 70)
    
    print("\nApplying log2(x+1) transformation and standardization...")
    preprocessed_tensor = loader.preprocess(
        tensor,
        log_transform=True,
        standardize=True
    )
    
    info_preprocessed = loader.get_tensor_info(preprocessed_tensor)
    print(f"\nPreprocessed tensor:")
    print(f"  Value range: [{info_preprocessed['min_value']:.4f}, {info_preprocessed['max_value']:.4f}]")
    print(f"  Mean: {info_preprocessed['mean_value']:.4f} ± {info_preprocessed['std_value']:.4f}")
    
    # Step 3: Perform PARAFAC decomposition
    print("\n" + "=" * 70)
    print("Step 3: Performing PARAFAC (CP) decomposition")
    print("=" * 70)
    
    decomposer_parafac = SparseTensorDecomposer(
        rank=rank,
        method="parafac",
        sparsity_threshold=0.01,
        normalize=True
    )
    
    print(f"\nFitting PARAFAC model with rank {rank}...")
    decomposer_parafac.fit(preprocessed_tensor, verbose=False)
    
    summary_parafac = decomposer_parafac.get_summary()
    print(f"\nPARAFAC Results:")
    print(f"  Reconstruction error: {summary_parafac['reconstruction_error']:.6f}")
    print(f"  Gene factors shape: {summary_parafac['gene_factors_shape']}")
    print(f"  Species factors shape: {summary_parafac['species_factors_shape']}")
    print(f"  Time factors shape: {summary_parafac['time_factors_shape']}")
    
    # Step 4: Perform Tucker decomposition
    print("\n" + "=" * 70)
    print("Step 4: Performing Tucker decomposition")
    print("=" * 70)
    
    decomposer_tucker = SparseTensorDecomposer(
        rank=rank,
        method="tucker",
        sparsity_threshold=0.01,
        normalize=True
    )
    
    print(f"\nFitting Tucker model with rank {rank}...")
    decomposer_tucker.fit(preprocessed_tensor, verbose=False)
    
    summary_tucker = decomposer_tucker.get_summary()
    print(f"\nTucker Results:")
    print(f"  Reconstruction error: {summary_tucker['reconstruction_error']:.6f}")
    print(f"  Core tensor shape: {summary_tucker['core_shape']}")
    print(f"  Gene factors shape: {summary_tucker['gene_factors_shape']}")
    print(f"  Species factors shape: {summary_tucker['species_factors_shape']}")
    print(f"  Time factors shape: {summary_tucker['time_factors_shape']}")
    
    # Step 5: Analyze factors
    print("\n" + "=" * 70)
    print("Step 5: Analyzing decomposition factors")
    print("=" * 70)
    
    gene_factors = decomposer_parafac.get_gene_factors()
    species_factors = decomposer_parafac.get_species_factors()
    time_factors = decomposer_parafac.get_time_factors()
    
    print("\nPARAFAC Factor Statistics:")
    print(f"  Gene factors:")
    print(f"    Shape: {gene_factors.shape}")
    print(f"    Mean: {np.mean(gene_factors):.4f}")
    print(f"    Std: {np.std(gene_factors):.4f}")
    print(f"  Species factors:")
    print(f"    Shape: {species_factors.shape}")
    print(f"    Mean: {np.mean(species_factors):.4f}")
    print(f"    Std: {np.std(species_factors):.4f}")
    print(f"  Time factors:")
    print(f"    Shape: {time_factors.shape}")
    print(f"    Mean: {np.mean(time_factors):.4f}")
    print(f"    Std: {np.std(time_factors):.4f}")
    
    # Show top contributing genes for each component
    print("\nTop 5 genes for each component (by absolute factor value):")
    for component in range(min(3, rank)):  # Show first 3 components
        top_genes_idx = np.argsort(np.abs(gene_factors[:, component]))[-5:][::-1]
        print(f"\n  Component {component + 1}:")
        for idx in top_genes_idx:
            gene_name = loader.gene_names[idx] if loader.gene_names else f"gene_{idx}"
            print(f"    {gene_name}: {gene_factors[idx, component]:.4f}")
    
    # Step 6: Save results
    print("\n" + "=" * 70)
    print("Step 6: Saving results")
    print("=" * 70)
    
    output_dir = Path("example_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save data
    loader.save_to_csv(tensor, output_dir / "original_data.csv")
    np.save(output_dir / "preprocessed_tensor.npy", preprocessed_tensor)
    
    # Save PARAFAC results
    parafac_dir = output_dir / "parafac"
    parafac_dir.mkdir(exist_ok=True)
    np.save(parafac_dir / "gene_factors.npy", decomposer_parafac.get_gene_factors())
    np.save(parafac_dir / "species_factors.npy", decomposer_parafac.get_species_factors())
    np.save(parafac_dir / "time_factors.npy", decomposer_parafac.get_time_factors())
    np.save(parafac_dir / "reconstructed.npy", decomposer_parafac.reconstruct())
    
    # Save Tucker results
    tucker_dir = output_dir / "tucker"
    tucker_dir.mkdir(exist_ok=True)
    np.save(tucker_dir / "gene_factors.npy", decomposer_tucker.get_gene_factors())
    np.save(tucker_dir / "species_factors.npy", decomposer_tucker.get_species_factors())
    np.save(tucker_dir / "time_factors.npy", decomposer_tucker.get_time_factors())
    np.save(tucker_dir / "reconstructed.npy", decomposer_tucker.reconstruct())
    
    print(f"\nResults saved to: {output_dir}/")
    print("  original_data.csv - Original synthetic data in CSV format")
    print("  preprocessed_tensor.npy - Preprocessed tensor")
    print("  parafac/ - PARAFAC decomposition results")
    print("  tucker/ - Tucker decomposition results")
    
    # Step 7: Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    print(f"\nWorkflow completed successfully!")
    print(f"\nComparison of methods:")
    print(f"  PARAFAC reconstruction error: {summary_parafac['reconstruction_error']:.6f}")
    print(f"  Tucker reconstruction error: {summary_tucker['reconstruction_error']:.6f}")
    
    if summary_tucker['reconstruction_error'] < summary_parafac['reconstruction_error']:
        print(f"\nTucker decomposition achieved lower reconstruction error.")
    else:
        print(f"\nPARAFAC decomposition achieved lower reconstruction error.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
