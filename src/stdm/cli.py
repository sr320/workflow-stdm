"""Command-line interface for sparse tensor decomposition workflow."""

import argparse
import json
from pathlib import Path
import numpy as np
from datetime import datetime

from .decomposition import SparseTensorDecomposer
from .data_loader import GeneExpressionLoader


def generate_readme_report(output_dir, input_file, args, summary, info, timestamp):
    """Generate a comprehensive README report for the analysis results.
    
    Args:
        output_dir: Path to output directory
        input_file: Path to input data file
        args: Command-line arguments
        summary: Decomposition summary dictionary
        info: Input tensor information dictionary
        timestamp: Run timestamp string
    """
    readme_path = output_dir / "README.md"
    
    # Determine optimal parameters based on reconstruction error
    rec_error = summary['reconstruction_error']
    if rec_error < 0.1:
        quality = "Excellent"
        recommendation = "The fit is excellent. Model may be capturing all important patterns, but check for potential overfitting."
    elif rec_error < 0.3:
        quality = "Good"
        recommendation = "The fit is good. The model captures most of the variation in the data."
    elif rec_error < 0.5:
        quality = "Acceptable"
        recommendation = "The fit is acceptable. Consider increasing the rank to capture more variation."
    else:
        quality = "Poor"
        recommendation = "The fit is poor. Try increasing the rank or checking data quality."
    
    # Rank recommendations
    current_rank = args.rank
    if rec_error > 0.5:
        rank_suggestion = f"Try increasing rank to {current_rank + 5} or {current_rank + 10}"
    elif rec_error < 0.1:
        rank_suggestion = f"Consider decreasing rank to {max(2, current_rank - 2)} to reduce overfitting"
    else:
        rank_suggestion = f"Current rank ({current_rank}) appears appropriate"
    
    readme_content = f"""# Sparse Tensor Decomposition Analysis Report

## Run Information

- **Timestamp**: {timestamp}
- **Input File**: {input_file}
- **Output Directory**: {output_dir}
- **Method**: {args.method.upper()}
- **Rank**: {args.rank}

## Input Data Summary

- **Shape**: {info['shape']} (genes × species × timepoints)
- **Number of Genes**: {info['n_genes']:,}
- **Number of Species**: {info['n_species']}
- **Number of Timepoints**: {info['n_timepoints']}
- **Sparsity**: {info['sparsity']:.2%}
- **Value Range**: [{info['min_value']:.4f}, {info['max_value']:.4f}]
- **Mean Value**: {info['mean_value']:.4f}
- **Std Deviation**: {info['std_value']:.4f}

## Analysis Parameters

- **Decomposition Method**: {args.method.upper()}
- **Rank**: {args.rank}
- **Sparsity Threshold**: {args.sparsity_threshold}
- **Normalization**: {args.normalize}
- **Log Transform**: {args.log_transform}
- **Standardization**: {args.standardize}

## Results Summary

- **Reconstruction Error**: {rec_error:.6f}
- **Quality Assessment**: {quality}
- **Gene Factors Shape**: {summary['gene_factors_shape']}
- **Species Factors Shape**: {summary['species_factors_shape']}
- **Time Factors Shape**: {summary['time_factors_shape']}

## Output Files

This directory contains the following files:

### 1. **gene_factors.npy**
- **Description**: Gene factor matrix ({info['n_genes']:,} × {args.rank})
- **Usage**: Each row represents a gene's loading on each component
- **Interpretation**: Higher absolute values indicate stronger association with that component
- **Load in Python**: `gene_factors = np.load('gene_factors.npy')`

### 2. **species_factors.npy**
- **Description**: Species factor matrix ({info['n_species']} × {args.rank})
- **Usage**: Shows how each species contributes to each component
- **Interpretation**: Components with high loading in one species indicate species-specific patterns
- **Load in Python**: `species_factors = np.load('species_factors.npy')`

### 3. **time_factors.npy**
- **Description**: Time point factor matrix ({info['n_timepoints']} × {args.rank})
- **Usage**: Shows temporal patterns for each component
- **Interpretation**: Positive/negative trends indicate up/down regulation over time
- **Load in Python**: `time_factors = np.load('time_factors.npy')`

### 4. **reconstructed_tensor.npy**
- **Description**: Reconstructed tensor from decomposition ({info['n_genes']:,} × {info['n_species']} × {info['n_timepoints']})
- **Usage**: Can be compared with original data for validation
- **Load in Python**: `reconstructed = np.load('reconstructed_tensor.npy')`

### 5. **summary.json**
- **Description**: JSON file containing decomposition statistics
- **Contents**: Method, rank, reconstruction error, and factor shapes
- **Load in Python**: 
  ```python
  import json
  with open('summary.json') as f:
      summary = json.load(f)
  ```

### 6. **README.md** (this file)
- **Description**: This comprehensive report documenting the analysis

## Quality Assessment

**Reconstruction Error**: {rec_error:.6f} ({quality})

{recommendation}

### Interpretation Guide

#### Reconstruction Error Ranges:
- **0.0 - 0.1**: Excellent fit (may be overfitting)
- **0.1 - 0.3**: Good fit
- **0.3 - 0.5**: Acceptable fit
- **> 0.5**: Poor fit (increase rank or check data quality)

## Recommendations

### Parameter Optimization

**Rank Selection**: {rank_suggestion}

**Method Comparison**:
- **PARAFAC/CP**: Best for identifying simple, interpretable patterns
- **Tucker**: Better for complex, hierarchical patterns (but more parameters)

### Next Steps

1. **Examine Factor Matrices**:
   ```python
   import numpy as np
   
   # Load factors
   gene_factors = np.load('gene_factors.npy')
   species_factors = np.load('species_factors.npy')
   time_factors = np.load('time_factors.npy')
   
   # Identify top genes for each component
   for component in range({args.rank}):
       top_genes_idx = np.argsort(np.abs(gene_factors[:, component]))[-10:][::-1]
       print(f"Component {{component + 1}} - Top 10 genes: {{top_genes_idx}}")
   ```

2. **Validate Results**:
   ```python
   # Compare original vs reconstructed
   original = np.load('path/to/original_tensor.npy')
   reconstructed = np.load('reconstructed_tensor.npy')
   
   diff = np.abs(original - reconstructed)
   print(f"Mean absolute difference: {{np.mean(diff):.6f}}")
   print(f"Max absolute difference: {{np.max(diff):.6f}}")
   ```

3. **Biological Interpretation**:
   - Look for gene modules with similar factor profiles (co-regulated genes)
   - Identify species-specific regulatory patterns
   - Examine temporal dynamics (up/down regulation over time)

## Additional Analysis Ideas

- **Gene Set Enrichment**: Use top genes from each component for pathway analysis
- **Clustering**: Cluster genes based on their factor profiles
- **Visualization**: Create heatmaps of factor matrices to identify patterns
- **Cross-validation**: Run decomposition multiple times with different parameters

## How to Load and Use Results

```python
import numpy as np
import json

# Load all results
gene_factors = np.load('gene_factors.npy')
species_factors = np.load('species_factors.npy')
time_factors = np.load('time_factors.npy')
reconstructed = np.load('reconstructed_tensor.npy')

with open('summary.json') as f:
    summary = json.load(f)

print(f"Reconstruction error: {{summary['reconstruction_error']:.6f}}")

# Example: Get top 20 genes for component 1
component_idx = 0
top_genes = np.argsort(np.abs(gene_factors[:, component_idx]))[-20:][::-1]
print(f"Top 20 genes for component {{component_idx + 1}}: {{top_genes}}")

# Example: Examine temporal pattern for a component
import matplotlib.pyplot as plt
plt.plot(time_factors[:, component_idx])
plt.xlabel('Time Point')
plt.ylabel('Factor Value')
plt.title(f'Temporal Pattern - Component {{component_idx + 1}}')
plt.show()
```

## Citation

If you use these results in a publication, please cite the workflow-stdm package.

---

*Report generated automatically by workflow-stdm*
*Timestamp: {timestamp}*
"""
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    return readme_path


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
    
    # Run command - simplified workflow using input-data
    run_parser = subparsers.add_parser("run", help="Run workflow on data from input-data directory")
    run_parser.add_argument("--input", "-i", help="Input CSV file (default: gene_expression_data.csv from input-data/)")
    run_parser.add_argument("--output", "-o", default="results", help="Output directory for results")
    run_parser.add_argument("--rank", "-r", type=int, default=5, help="Decomposition rank")
    run_parser.add_argument("--method", "-m", choices=["parafac", "tucker"], 
                           default="parafac", help="Decomposition method")
    run_parser.add_argument("--sparsity-threshold", type=float, default=0.01,
                           help="Sparsity threshold")
    run_parser.add_argument("--normalize", action="store_true", help="Normalize tensor")
    run_parser.add_argument("--log-transform", action="store_true", default=True,
                           help="Apply log transformation (default: True)")
    run_parser.add_argument("--no-log-transform", dest="log_transform", action="store_false",
                           help="Skip log transformation")
    run_parser.add_argument("--standardize", action="store_true", default=True,
                           help="Standardize gene expression (default: True)")
    run_parser.add_argument("--no-standardize", dest="standardize", action="store_false",
                           help="Skip standardization")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # MOSAiC data commands
    fetch_parser = subparsers.add_parser(
        "fetch", help="Download MOSAiC data files listed in data/manifest.csv")
    fetch_parser.add_argument("--species", nargs="+", choices=["apul", "peve", "ptua", "multi"])
    fetch_parser.add_argument("--data-type", nargs="+", dest="data_types",
                              help="e.g. gene mirna lncrna gene_meth orthologs physiology")
    fetch_parser.add_argument("--key", nargs="+", dest="keys", help="Manifest keys, e.g. apul_gene")
    fetch_parser.add_argument("--include-large", action="store_true",
                              help="Also fetch ~1 GB CpG-level methylation files")
    fetch_parser.add_argument("--force", action="store_true", help="Re-download cached files")
    fetch_parser.add_argument("--update-checksums", action="store_true",
                              help="Accept changed upstream files into data/checksums.csv")
    fetch_parser.add_argument("--list", action="store_true", help="List manifest entries and exit")

    samples_parser = subparsers.add_parser(
        "samples", help="Build the sample metadata table from cached MOSAiC files")
    samples_parser.add_argument("--output", "-o", help="Output CSV (default: data/samples.csv)")

    tensor_parser = subparsers.add_parser(
        "build-tensor", help="Build the ortholog x colony x timepoint tensor from MOSAiC counts")
    tensor_parser.add_argument("--normalization", choices=["log2-mor", "vst-joint"],
                               default="log2-mor",
                               help="log2-mor: per-species median-of-ratios + log2 (default); "
                                    "vst-joint: DESeq2 VST across species, as in the 2026 talk")
    tensor_parser.add_argument("--center", choices=["none", "species"], default="none",
                               help="Subtract each group's mean within each species")
    tensor_parser.add_argument("--min-count", type=float, default=10,
                               help="Normalised count threshold for keeping a group (default 10)")
    tensor_parser.add_argument("--min-fraction", type=float, default=0.5,
                               help="Fraction of samples per species that must reach --min-count")
    tensor_parser.add_argument("--output", "-o",
                               help="Output directory (default: data/processed/tensor-<normalization>"
                                    "[-centered])")

    tensor_help = ("'talk' for the talk's VST matrix (missing samples masked) or a directory "
                   "written by build-tensor")

    repro_parser = subparsers.add_parser(
        "reproduce-talk", help="Refit the talk's rank-35 Barnacle model and compare to published factors")
    repro_parser.add_argument("--output", "-o", default="results/reproduce-talk")
    repro_parser.add_argument("--n-jobs", type=int, default=3, help="Parallel initialisations")

    rank_parser = subparsers.add_parser(
        "select-rank", help="Stage 1: choose rank by held-out-sample CV (1-SE rule)")
    rank_parser.add_argument("--tensor", default="talk", help=tensor_help)
    rank_parser.add_argument("--ranks", type=int, nargs="+",
                             default=[5, 10, 15, 20, 25, 30, 35, 40])
    rank_parser.add_argument("--splits", type=int, default=5)
    rank_parser.add_argument("--test-fraction", type=float, default=0.5)
    rank_parser.add_argument("--max-iter", type=int, default=1000)
    rank_parser.add_argument("--tol", type=float, default=1e-5)
    rank_parser.add_argument("--seed", type=int, default=0)
    rank_parser.add_argument("--workers", type=int, help="Parallel fits (default: cores - 1)")
    rank_parser.add_argument("--output", "-o", default="results/select-rank")

    lam_parser = subparsers.add_parser(
        "select-lambda", help="Stage 2: choose gene sparsity by split-half FMS (1-SE rule)")
    lam_parser.add_argument("--tensor", default="talk", help=tensor_help)
    lam_parser.add_argument("--rank", type=int, required=True)
    lam_parser.add_argument("--grid", type=float, nargs="+",
                            default=[0.0, 0.2, 1, 5, 25, 100, 500])
    lam_parser.add_argument("--lambda-colony", type=float, default=0.1)
    lam_parser.add_argument("--lambda-time", type=float, default=0.05)
    lam_parser.add_argument("--splits", type=int, default=4)
    lam_parser.add_argument("--max-iter", type=int, default=1000)
    lam_parser.add_argument("--tol", type=float, default=1e-5)
    lam_parser.add_argument("--seed", type=int, default=0)
    lam_parser.add_argument("--workers", type=int, help="Parallel fits (default: cores - 1)")
    lam_parser.add_argument("--output", "-o", default="results/select-lambda")

    fit_parser = subparsers.add_parser(
        "sparse-cp", help="Fit sparse CP (Barnacle) to a tensor, masking missing samples")
    fit_parser.add_argument("--tensor", default="talk", help=tensor_help)
    fit_parser.add_argument("--rank", type=int, required=True)
    fit_parser.add_argument("--lambdas", type=float, nargs=3, metavar=("GENE", "COLONY", "TIME"),
                            default=[0.2, 0.1, 0.05])
    fit_parser.add_argument("--n-init", type=int, default=3)
    fit_parser.add_argument("--n-jobs", type=int, default=3)
    fit_parser.add_argument("--max-iter", type=int, default=2000)
    fit_parser.add_argument("--tol", type=float, default=1e-5)
    fit_parser.add_argument("--seed", type=int, default=81)
    fit_parser.add_argument("--output", "-o", required=True)

    args = parser.parse_args()

    if args.command == "reproduce-talk":
        run_reproduce_talk(args)
    elif args.command == "select-rank":
        run_select_rank(args)
    elif args.command == "select-lambda":
        run_select_lambda(args)
    elif args.command == "sparse-cp":
        run_sparse_cp(args)
    elif args.command == "build-tensor":
        run_build_tensor(args)
    elif args.command == "fetch":
        run_fetch(args)
    elif args.command == "samples":
        run_samples(args)
    elif args.command == "generate":
        run_generate(args)
    elif args.command == "decompose":
        run_decompose(args)
    elif args.command == "info":
        run_info(args)
    elif args.command == "run":
        run_workflow(args)
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
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Loading data from: {args.input}")
    
    loader = GeneExpressionLoader()
    tensor = loader.load_from_csv(args.input)
    
    print(f"Loaded tensor shape: {tensor.shape}")
    
    # Get initial tensor info
    info = loader.get_tensor_info(tensor)
    
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
    
    # Add timestamp to output directory
    base_output = Path(args.output)
    output_dir = base_output / timestamp
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
    
    # Generate comprehensive README report
    print(f"\nGenerating analysis report...")
    readme_path = generate_readme_report(
        output_dir=output_dir,
        input_file=Path(args.input),
        args=args,
        summary=summary,
        info=info,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    print(f"  gene_factors.npy")
    print(f"  species_factors.npy")
    print(f"  time_factors.npy")
    print(f"  reconstructed_tensor.npy")
    print(f"  summary.json")
    print(f"  README.md (analysis report)")
    print("\nDone!")
    print(f"\nResults saved to: {output_dir}/")
    print(f"Read {output_dir / 'README.md'} for detailed analysis report")


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


def run_workflow(args):
    """Run complete workflow using data from input-data directory."""
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine input file
    if args.input is None:
        input_file = Path("input-data") / "gene_expression_data.csv"
    else:
        # If user provides just a filename, look in input-data directory first
        input_path = Path(args.input)
        if not input_path.exists() and not input_path.is_absolute():
            input_data_path = Path("input-data") / args.input
            if input_data_path.exists():
                input_file = input_data_path
            else:
                input_file = input_path
        else:
            input_file = input_path
    
    # Check if file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        print(f"\nAvailable files in input-data/:")
        input_data_dir = Path("input-data")
        if input_data_dir.exists():
            for f in input_data_dir.glob("*.csv"):
                print(f"  - {f.name}")
        else:
            print("  (input-data directory not found)")
        return
    
    # Add timestamp to output directory
    base_output = Path(args.output)
    output_dir = base_output / timestamp
    
    print("=" * 70)
    print("Sparse Tensor Decomposition Workflow")
    print("=" * 70)
    print(f"\nRun Timestamp: {timestamp}")
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}/")
    print(f"Method: {args.method.upper()}")
    print(f"Rank: {args.rank}")
    
    # Load data
    print(f"\nStep 1: Loading data...")
    loader = GeneExpressionLoader()
    tensor = loader.load_from_csv(str(input_file))
    
    info = loader.get_tensor_info(tensor)
    print(f"  Shape: {info['shape']}")
    print(f"  Sparsity: {info['sparsity']:.2%}")
    print(f"  Value range: [{info['min_value']:.4f}, {info['max_value']:.4f}]")
    
    # Preprocess
    if args.log_transform or args.standardize:
        print(f"\nStep 2: Preprocessing...")
        if args.log_transform:
            print("  - Applying log2(x+1) transformation")
        if args.standardize:
            print("  - Standardizing gene expression")
        
        tensor = loader.preprocess(
            tensor,
            log_transform=args.log_transform,
            standardize=args.standardize
        )
    else:
        print(f"\nStep 2: Preprocessing... (skipped)")
    
    # Decompose
    print(f"\nStep 3: Performing {args.method.upper()} decomposition...")
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
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStep 4: Saving results to: {output_dir}/")
    
    # Save factors
    np.save(output_dir / "gene_factors.npy", decomposer.get_gene_factors())
    np.save(output_dir / "species_factors.npy", decomposer.get_species_factors())
    np.save(output_dir / "time_factors.npy", decomposer.get_time_factors())
    
    # Save reconstructed tensor
    reconstructed = decomposer.reconstruct()
    np.save(output_dir / "reconstructed_tensor.npy", reconstructed)
    
    # Save summary as JSON
    with open(output_dir / "summary.json", "w") as f:
        json_summary = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in summary.items()
        }
        json.dump(json_summary, f, indent=2)
    
    # Generate comprehensive README report
    print(f"\nStep 5: Generating analysis report...")
    readme_path = generate_readme_report(
        output_dir=output_dir,
        input_file=input_file,
        args=args,
        summary=summary,
        info=info,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    print(f"  - gene_factors.npy")
    print(f"  - species_factors.npy")
    print(f"  - time_factors.npy")
    print(f"  - reconstructed_tensor.npy")
    print(f"  - summary.json")
    print(f"  - README.md (analysis report)")
    
    print("\n" + "=" * 70)
    print("Workflow completed successfully!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}/")
    print(f"Read {output_dir / 'README.md'} for detailed analysis report")



if __name__ == "__main__":
    main()


def run_fetch(args):
    """Download MOSAiC files into data/raw/."""
    from . import mosaic

    manifest = mosaic.load_manifest()
    if args.list:
        for e in manifest.values():
            cached = "cached" if mosaic.raw_path(e).exists() else ""
            large = "large" if e.large else ""
            print(f"{e.key:24s} {e.kind:6s} {large:5s} {cached:6s} {e.description}")
        return
    entries = mosaic.select(manifest, species=args.species, data_types=args.data_types,
                            keys=args.keys, include_large=args.include_large)
    if not entries:
        print("Nothing matched.")
        return
    mosaic.fetch(entries, force=args.force, update_checksums=args.update_checksums)
    skipped = [e.key for e in manifest.values()
               if e.large and e not in entries and not args.keys]
    if skipped and not args.include_large:
        print(f"Skipped large files (use --include-large): {', '.join(skipped)}")


def run_samples(args):
    """Write the sample metadata table and print data-type coverage."""
    from . import mosaic

    meta = mosaic.build_sample_metadata()
    out = Path(args.output) if args.output else mosaic.default_data_dir() / "samples.csv"
    meta.to_csv(out, index=False)
    print(f"Wrote {len(meta)} samples to {out}")
    has = [c for c in meta.columns if c.startswith("has_")]
    if has:
        print(meta.groupby("species")[has].sum().to_string())


def run_build_tensor(args):
    """Build and save the expression tensor."""
    from . import mosaic
    from .tensor import build_expression_tensor

    t = build_expression_tensor(normalization=args.normalization, center=args.center,
                                min_count=args.min_count, min_fraction=args.min_fraction)
    name = f"tensor-{args.normalization}" + ("-centered" if args.center == "species" else "")
    out = Path(args.output) if args.output else mosaic.default_data_dir() / "processed" / name
    t.save(out)
    s = t.summary()
    print(f"Saved {out}")
    print(f"  shape (groups x colonies x timepoints): {tuple(s['shape'])}")
    print(f"  group filtering: {s['group_counts']}")
    print(f"  missing samples (NaN): {', '.join(s['missing_samples']) or 'none'}")
    print(f"  value range: {s['min']:.2f} to {s['max']:.2f}")


def _load_expression_tensor(spec):
    from .tensor import ExpressionTensor
    if spec == "talk":
        from .reproduce import talk_expression_tensor
        return talk_expression_tensor(Path("input-data/vst_counts_matrix.csv"))
    return ExpressionTensor.load(Path(spec))


def _write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def run_reproduce_talk(args):
    """Refit the talk model on the talk tensor and compare with published factors."""
    import time
    from . import mosaic
    from .reproduce import (TALK_PARAMS, compare_factors, factors_to_frames, fetch_published,
                            fit_talk_model, talk_tensor)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tensor, groups, colonies = talk_tensor(Path("input-data/vst_counts_matrix.csv"))
    print(f"Talk tensor {tensor.shape}; fitting rank {TALK_PARAMS['rank']} "
          f"with {TALK_PARAMS['n_initializations']} initialisations ...")
    t0 = time.time()
    model = fit_talk_model(tensor, verbose=1, n_jobs=args.n_jobs)
    ours = factors_to_frames(model.decomposition_.factors, groups, colonies)
    for name, df in ours.items():
        df.to_csv(out / f"{name}_factors.csv")
    published = fetch_published(mosaic.default_data_dir() / "raw" / "published-talk-rank35")
    cmp = compare_factors(ours, published)
    cmp.to_csv(out / "comparison.csv", index=False)
    _write_json(out / "run.json", {
        "params": TALK_PARAMS, "seconds": time.time() - t0,
        "final_losses": [l[-1] for l in model.candidate_losses_],
        "iterations": [len(l) for l in model.candidate_losses_],
        "mean_similarity": float(cmp.similarity.mean())})
    print(f"Saved {out}")
    print(f"  mean matched similarity to published: {cmp.similarity.mean():.3f}")
    print(cmp.round(3).to_string(index=False))


def run_select_rank(args):
    from .selection import select_rank
    t = _load_expression_tensor(args.tensor)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rank, df = select_rank(t.values, t.colonies.species.to_numpy(), args.ranks,
                           n_splits=args.splits, test_fraction=args.test_fraction,
                           n_iter_max=args.max_iter, tol=args.tol, seed=args.seed,
                           workers=args.workers)
    df.to_csv(out / "cv.csv", index=False)
    _write_json(out / "selection.json", {"selected_rank": rank, "args": vars(args)})
    print(df.groupby("rank")[["test_sse", "test_r2"]].agg(["mean", "std"]).round(4).to_string())
    print(f"Selected rank (1-SE rule): {rank}")


def run_select_lambda(args):
    from .selection import select_lambda
    t = _load_expression_tensor(args.tensor)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    lam, df = select_lambda(t.values, t.colonies.species.to_numpy(), t.colonies.site.to_numpy(),
                            rank=args.rank, lambda_grid=args.grid,
                            fixed_lambdas=(None, args.lambda_colony, args.lambda_time),
                            n_splits=args.splits, n_iter_max=args.max_iter, tol=args.tol,
                            seed=args.seed, workers=args.workers)
    df.to_csv(out / "fms.csv", index=False)
    _write_json(out / "selection.json", {"selected_lambda_gene": lam, "args": vars(args)})
    print(df.groupby("lam")[["fms", "gene_zero_fraction", "empty_components"]]
          .agg(["mean", "std"]).round(3).to_string())
    print(f"Selected gene lambda (1-SE rule): {lam}")


def run_sparse_cp(args):
    import time
    from .barnacle import SparseCP
    from .reproduce import factors_to_frames
    t = _load_expression_tensor(args.tensor)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model = SparseCP(args.rank, args.lambdas, nonneg_modes=[0], n_initializations=args.n_init,
                     random_state=args.seed, n_iter_max=args.max_iter, tol=args.tol)
    t0 = time.time()
    for old in out.glob("progress-init*.tsv"):
        old.unlink()
    print(f"Fitting rank {args.rank}, lambdas {args.lambdas}, {args.n_init} initialisations, "
          f"up to {args.max_iter} iterations; progress in {out}/progress-init*.tsv", flush=True)
    model.fit_transform(t.values, mask=t.mask, n_jobs=args.n_jobs, verbose=1, log_dir=out)
    frames = factors_to_frames(model.decomposition_.factors, list(t.groups.index),
                               list(t.colonies.index))
    for name, df in frames.items():
        df.index.name = {"gene": "group_id", "sample": "colony_id", "time": "timepoint"}[name]
        df.to_csv(out / f"{'colony' if name == 'sample' else name}_factors.csv")
    t.colonies.to_csv(out / "colonies.csv")
    _write_json(out / "run.json", {
        "args": vars(args), "tensor_params": t.params, "seconds": time.time() - t0,
        "final_losses": [l[-1] for l in model.candidate_losses_],
        "iterations": [len(l) for l in model.candidate_losses_],
        "converged": [len(l) < args.max_iter or
                      abs(l[-2] - l[-1]) / max(l[-1], 1) < args.tol
                      for l in model.candidate_losses_],
        "best_initialization": model._best + 1,
        "gene_zero_fraction": float((frames["gene"].to_numpy() == 0).mean())})
    np.savez(out / "losses.npz", *[np.asarray(l) for l in model.candidate_losses_])
    print(f"Saved {out}")
