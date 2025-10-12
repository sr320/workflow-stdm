# workflow-stdm

Sparse Tensor Decomposition for Multi-species gene expression data.

## Overview

This package provides a Python workflow for performing sparse tensor decomposition on gene expression data across multiple species and time points. It is optimized to handle:

- **~10,000 genes**
- **3 species**
- **4 time points**

The workflow uses efficient tensor decomposition methods (PARAFAC/CP and Tucker) to identify patterns and factors in multi-dimensional gene expression data.

## Features

- **Sparse tensor decomposition** using PARAFAC (CP) and Tucker methods
- **Optimized for gene expression data** with automatic preprocessing
- **Data loading utilities** for CSV-based gene expression datasets
- **Synthetic data generation** for testing and validation
- **Command-line interface** for easy workflow execution
- **UV package manager** for fast, reliable dependency management

## Installation

### Using UV (Recommended)

```bash
# Clone the repository
git clone https://github.com/sr320/workflow-stdm.git
cd workflow-stdm

# Install with uv
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

### Using pip

```bash
pip install -e .
```

## Quick Start

### Option 1: Use Provided Test Data (Recommended)

The repository includes test datasets in the `input-data/` directory. Simply run:

```bash
# Run workflow with default test data (10k genes, 3 species, 4 time points)
stdm run --rank 5 --method parafac

# Or use the smaller test dataset for quick testing
stdm run --input gene_expression_data_small.csv --rank 5 --method parafac
```

### Option 2: Generate and Use Synthetic Data

### 1. Generate Synthetic Data

```bash
stdm generate --output data.csv --genes 10000 --species 3 --timepoints 4
```

### 2. View Data Information

```bash
stdm info --input data.csv
```

### 3. Perform Tensor Decomposition

```bash
stdm decompose --input data.csv --output results/ --rank 5 --method parafac
```

## Usage

### Command-Line Interface

The package provides four main commands:

#### Run Complete Workflow (Recommended)

Use test data from `input-data/` directory:

```bash
stdm run [OPTIONS]

Options:
  -i, --input PATH              Input CSV file (default: gene_expression_data.csv from input-data/)
  -o, --output PATH             Output directory (default: results)
  -r, --rank INT               Decomposition rank (default: 5)
  -m, --method {parafac,tucker} Decomposition method (default: parafac)
  --sparsity-threshold FLOAT   Sparsity threshold (default: 0.01)
  --normalize                  Normalize tensor
  --log-transform              Apply log2(x+1) transformation (default: True)
  --no-log-transform           Skip log transformation
  --standardize                Standardize gene expression (default: True)
  --no-standardize             Skip standardization
  -v, --verbose                Verbose output

# Examples:
stdm run                                      # Use default test data
stdm run --input gene_expression_data_small.csv  # Use small test data
stdm run --method tucker --rank 10           # Tucker with rank 10
```

#### Generate Synthetic Data

```bash
stdm generate [OPTIONS]

Options:
  -o, --output PATH       Output CSV file path (required)
  -g, --genes INT        Number of genes (default: 10000)
  -s, --species INT      Number of species (default: 3)
  -t, --timepoints INT   Number of time points (default: 4)
  --sparsity FLOAT       Sparsity level 0-1 (default: 0.3)
  --noise FLOAT          Noise level (default: 0.1)
  --seed INT             Random seed (default: 42)
```

#### Decompose Tensor

```bash
stdm decompose [OPTIONS]

Options:
  -i, --input PATH              Input CSV file path (required)
  -o, --output PATH             Output directory (required)
  -r, --rank INT               Decomposition rank (default: 5)
  -m, --method {parafac,tucker} Decomposition method (default: parafac)
  --sparsity-threshold FLOAT   Sparsity threshold (default: 0.01)
  --normalize                  Normalize tensor
  --log-transform              Apply log2(x+1) transformation
  --standardize                Standardize gene expression
  -v, --verbose                Verbose output
```

#### Get Tensor Information

```bash
stdm info --input PATH
```

### Python API

```python
from stdm import SparseTensorDecomposer, GeneExpressionLoader

# Load or generate data
loader = GeneExpressionLoader(n_genes=10000, n_species=3, n_timepoints=4)
tensor = loader.generate_synthetic_data(sparsity=0.3, seed=42)

# Or load from CSV
# tensor = loader.load_from_csv("data.csv")

# Preprocess
tensor = loader.preprocess(tensor, log_transform=True, standardize=True)

# Decompose
decomposer = SparseTensorDecomposer(rank=5, method="parafac")
decomposer.fit(tensor, verbose=True)

# Get results
gene_factors = decomposer.get_gene_factors()
species_factors = decomposer.get_species_factors()
time_factors = decomposer.get_time_factors()
summary = decomposer.get_summary()

print(f"Reconstruction error: {summary['reconstruction_error']:.6f}")

# Reconstruct tensor
reconstructed = decomposer.reconstruct()
```

## Data Format

### Input CSV Format

The expected CSV format for gene expression data:

```csv
gene,species,timepoint,expression
gene1,species1,t0,0.5
gene1,species1,t1,0.7
gene1,species2,t0,0.3
...
```

### Output Files

The decomposition command generates:

- `gene_factors.npy` - Gene factor matrix (n_genes × rank)
- `species_factors.npy` - Species factor matrix (n_species × rank)
- `time_factors.npy` - Time factor matrix (n_timepoints × rank)
- `reconstructed_tensor.npy` - Reconstructed tensor
- `summary.json` - Decomposition summary with reconstruction error

## Methods

### PARAFAC (CP) Decomposition

Canonical Polyadic (CP) decomposition, also known as PARAFAC, decomposes the tensor into a sum of rank-1 tensors. It's efficient and provides interpretable factors.

### Tucker Decomposition

Tucker decomposition is a higher-order generalization of matrix SVD. It decomposes the tensor into a core tensor multiplied by factor matrices along each mode.

## Performance Considerations

- **Sparsity**: The package automatically handles sparse tensors, setting values below the threshold to zero
- **Normalization**: Tensor normalization improves numerical stability
- **Rank selection**: Lower ranks are faster but may lose information; higher ranks capture more detail
- **Memory**: With 10k genes, 3 species, and 4 timepoints, the tensor requires ~960 KB (double precision)

## Development

```bash
# Install development dependencies
uv sync

# Run tests (if available)
pytest

# Format code
ruff format src/

# Lint code
ruff check src/
```

## License

MIT License

## Citation

If you use this package in your research, please cite:

```
@software{workflow_stdm,
  title = {Sparse Tensor Decomposition for Multi-species Gene Expression},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/sr320/workflow-stdm}
}
```
