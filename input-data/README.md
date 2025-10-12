# Test Data for Sparse Tensor Decomposition

This directory contains test gene expression datasets for the sparse tensor decomposition workflow.

## Available Datasets

### 1. gene_expression_data.csv (Full Dataset)
- **Genes**: 10,000
- **Species**: 3
- **Time points**: 4
- **Sparsity**: ~32%
- **Total entries**: 120,000

This dataset matches the exact specifications for the workflow and can be used for full-scale testing and validation.

### 2. gene_expression_data_small.csv (Small Dataset)
- **Genes**: 1,000
- **Species**: 3
- **Time points**: 4
- **Sparsity**: ~32%
- **Total entries**: 12,000

A smaller dataset for quick testing and development.

## Data Format

All CSV files follow this format:

```csv
gene,species,timepoint,expression
gene_1,species_1,t0,0.5
gene_1,species_1,t1,0.7
gene_1,species_2,t0,0.3
...
```

## Usage

### Using with CLI

Run decomposition on the full dataset:
```bash
stdm run --rank 5 --method parafac
```

Run decomposition on the small dataset:
```bash
stdm run --input gene_expression_data_small.csv --rank 5 --method parafac
```

### Using with Python API

```python
from stdm import GeneExpressionLoader, SparseTensorDecomposer

# Load data
loader = GeneExpressionLoader()
tensor = loader.load_from_csv("input-data/gene_expression_data.csv")

# Preprocess
tensor = loader.preprocess(tensor, log_transform=True, standardize=True)

# Decompose
decomposer = SparseTensorDecomposer(rank=5, method="parafac")
decomposer.fit(tensor)
```

## Regenerating Data

To regenerate the test data:

```bash
# Full dataset
stdm generate --output input-data/gene_expression_data.csv --genes 10000 --species 3 --timepoints 4 --seed 42

# Small dataset
stdm generate --output input-data/gene_expression_data_small.csv --genes 1000 --species 3 --timepoints 4 --seed 123
```
