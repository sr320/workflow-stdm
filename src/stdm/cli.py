"""Command-line interface for sparse tensor decomposition workflow."""

import argparse
import json
from pathlib import Path
import numpy as np

from .decomposition import SparseTensorDecomposer
from .data_loader import GeneExpressionLoader


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Sparse Tensor Decomposition for Multi-species Gene Expression Data"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate synthetic data command
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic gene expression data")
    gen_parser.add_argument("--output", "-o", required=True, help="Output CSV file path")
    gen_parser.add_argument("--genes", "-g", type=int, default=10000, help="Number of genes")
    gen_parser.add_argument("--species", "-s", type=int, default=3, help="Number of species")
    gen_parser.add_argument("--timepoints", "-t", type=int, default=4, help="Number of time points")
    gen_parser.add_argument("--sparsity", type=float, default=0.3, help="Sparsity level (0-1)")
    gen_parser.add_argument("--noise", type=float, default=0.1, help="Noise level")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Decompose command
    decompose_parser = subparsers.add_parser("decompose", help="Perform tensor decomposition")
    decompose_parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    decompose_parser.add_argument("--output", "-o", required=True, help="Output directory for results")
    decompose_parser.add_argument("--rank", "-r", type=int, default=5, help="Decomposition rank")
    decompose_parser.add_argument("--method", "-m", choices=["parafac", "tucker"], 
                                 default="parafac", help="Decomposition method")
    decompose_parser.add_argument("--sparsity-threshold", type=float, default=0.01,
                                 help="Sparsity threshold")
    decompose_parser.add_argument("--normalize", action="store_true", help="Normalize tensor")
    decompose_parser.add_argument("--log-transform", action="store_true", 
                                 help="Apply log transformation")
    decompose_parser.add_argument("--standardize", action="store_true", 
                                 help="Standardize gene expression")
    decompose_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get information about tensor data")
    info_parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        run_generate(args)
    elif args.command == "decompose":
        run_decompose(args)
    elif args.command == "info":
        run_info(args)
    else:
        parser.print_help()


def run_generate(args):
    """Generate synthetic gene expression data."""
    print(f"Generating synthetic gene expression data...")
    print(f"  Genes: {args.genes}")
    print(f"  Species: {args.species}")
    print(f"  Timepoints: {args.timepoints}")
    print(f"  Sparsity: {args.sparsity}")
    print(f"  Noise level: {args.noise}")
    print(f"  Seed: {args.seed}")
    
    loader = GeneExpressionLoader(
        n_genes=args.genes,
        n_species=args.species,
        n_timepoints=args.timepoints
    )
    
    tensor = loader.generate_synthetic_data(
        sparsity=args.sparsity,
        noise_level=args.noise,
        seed=args.seed
    )
    
    loader.save_to_csv(tensor, args.output)
    
    info = loader.get_tensor_info(tensor)
    print(f"\nGenerated tensor:")
    print(f"  Shape: {info['shape']}")
    print(f"  Sparsity: {info['sparsity']:.2%}")
    print(f"  Value range: [{info['min_value']:.4f}, {info['max_value']:.4f}]")
    print(f"  Mean: {info['mean_value']:.4f}")
    print(f"\nSaved to: {args.output}")


def run_decompose(args):
    """Perform tensor decomposition."""
    print(f"Loading data from: {args.input}")
    
    loader = GeneExpressionLoader()
    tensor = loader.load_from_csv(args.input)
    
    print(f"Loaded tensor shape: {tensor.shape}")
    
    # Preprocess
    if args.log_transform or args.standardize:
        print("Preprocessing tensor...")
        tensor = loader.preprocess(
            tensor,
            log_transform=args.log_transform,
            standardize=args.standardize
        )
    
    # Decompose
    print(f"\nPerforming {args.method.upper()} decomposition...")
    print(f"  Rank: {args.rank}")
    print(f"  Sparsity threshold: {args.sparsity_threshold}")
    print(f"  Normalize: {args.normalize}")
    
    decomposer = SparseTensorDecomposer(
        rank=args.rank,
        method=args.method,
        sparsity_threshold=args.sparsity_threshold,
        normalize=args.normalize
    )
    
    decomposer.fit(tensor, verbose=args.verbose)
    
    # Get summary
    summary = decomposer.get_summary()
    print(f"\nDecomposition complete!")
    print(f"  Reconstruction error: {summary['reconstruction_error']:.6f}")
    print(f"  Gene factors shape: {summary['gene_factors_shape']}")
    print(f"  Species factors shape: {summary['species_factors_shape']}")
    print(f"  Time factors shape: {summary['time_factors_shape']}")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving results to: {output_dir}")
    
    # Save factors
    np.save(output_dir / "gene_factors.npy", decomposer.get_gene_factors())
    np.save(output_dir / "species_factors.npy", decomposer.get_species_factors())
    np.save(output_dir / "time_factors.npy", decomposer.get_time_factors())
    
    # Save reconstructed tensor
    reconstructed = decomposer.reconstruct()
    np.save(output_dir / "reconstructed_tensor.npy", reconstructed)
    
    # Save summary as JSON
    with open(output_dir / "summary.json", "w") as f:
        # Convert numpy types to Python types for JSON serialization
        json_summary = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in summary.items()
        }
        json.dump(json_summary, f, indent=2)
    
    print(f"  gene_factors.npy")
    print(f"  species_factors.npy")
    print(f"  time_factors.npy")
    print(f"  reconstructed_tensor.npy")
    print(f"  summary.json")
    print("\nDone!")


def run_info(args):
    """Get information about tensor data."""
    print(f"Loading data from: {args.input}")
    
    loader = GeneExpressionLoader()
    tensor = loader.load_from_csv(args.input)
    
    info = loader.get_tensor_info(tensor)
    
    print("\nTensor Information:")
    print(f"  Shape: {info['shape']}")
    print(f"  Number of genes: {info['n_genes']}")
    print(f"  Number of species: {info['n_species']}")
    print(f"  Number of timepoints: {info['n_timepoints']}")
    print(f"  Sparsity: {info['sparsity']:.2%}")
    print(f"  Value range: [{info['min_value']:.4f}, {info['max_value']:.4f}]")
    print(f"  Mean: {info['mean_value']:.4f}")
    print(f"  Std: {info['std_value']:.4f}")


if __name__ == "__main__":
    main()
