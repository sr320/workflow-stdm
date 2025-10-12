# Usage Guide for Sparse Tensor Decomposition Workflow

This guide provides detailed instructions on using the sparse tensor decomposition workflow for multi-species gene expression data.

## Installation

### Option 1: Using UV (Recommended)

```bash
# Clone the repository
git clone https://github.com/sr320/workflow-stdm.git
cd workflow-stdm

# Install with uv
uv sync

# Activate virtual environment
source .venv/bin/activate  # Unix/macOS
# or
.venv\Scripts\activate  # Windows
```

### Option 2: Using pip

```bash
pip install -e .
```

## Quick Start Example

### Option 1: Use Provided Test Data (Recommended)

The repository includes ready-to-use test datasets in the `input-data/` directory:

```bash
# Run workflow with default test data (10k genes, 3 species, 4 time points)
stdm run --rank 5 --method parafac --output results/

# Use the smaller test dataset for quick testing (1k genes)
stdm run --input gene_expression_data_small.csv --rank 5 --output results_small/

# Try Tucker decomposition
stdm run --method tucker --rank 5 --output results_tucker/
```

**Available test datasets:**
- `gene_expression_data.csv` - Full dataset (10,000 genes × 3 species × 4 time points)
- `gene_expression_data_small.csv` - Small dataset (1,000 genes × 3 species × 4 time points)
- `vst_counts_matrix.csv` - Real RNA-seq data (large dataset)

**New Feature: Auto-timestamping**

Each run automatically creates a timestamped subdirectory, so you can run the workflow multiple times without overwriting previous results:

```bash
# First run
stdm run --rank 5 --output results/
# Creates: results/20251012_143022/

# Second run (different parameters)
stdm run --rank 10 --output results/
# Creates: results/20251012_175729/

# All previous results are preserved!
```

**New Feature: Comprehensive README Report**

Every run generates a detailed `README.md` file in the output directory with:
- Run configuration and timestamp
- Input data summary and statistics
- Quality assessment and reconstruction error interpretation
- Parameter recommendations based on results
- Detailed descriptions of all output files
- Python code examples for loading and analyzing results
- Biological interpretation guidelines

### Option 2: Use Your Own Data

You can easily analyze your own gene expression data:

```bash
# Use data from input-data directory
stdm run --input my_data.csv --output my_results/

# Use data from any path
stdm run --input /path/to/my/data.csv --output /path/to/results/

# Use absolute path
stdm run --input ~/research/gene_expression.csv --output ~/results/analysis1/
```

**Required Data Format:**

Your CSV file must follow this structure:
```csv
gene,species,timepoint,expression
gene1,species1,t0,0.5
gene1,species1,t1,0.7
gene1,species2,t0,0.3
gene1,species2,t1,0.8
...
```

**Tips for Custom Data:**
- Column names must be: `gene`, `species`, `timepoint`, `expression`
- Expression values should be non-negative (counts or normalized values)
- The workflow will automatically determine the dimensions (# genes, species, timepoints)
- Missing combinations of gene/species/timepoint will be treated as zeros

### Option 3: Generate Your Own Synthetic Data

### 1. Generate Test Data

Generate synthetic gene expression data with the exact specifications (10k genes, 3 species, 4 time points):

```bash
stdm generate \
  --output gene_expression.csv \
  --genes 10000 \
  --species 3 \
  --timepoints 4 \
  --sparsity 0.3 \
  --noise 0.1 \
  --seed 42
```

### 2. Inspect the Data

```bash
stdm info --input gene_expression.csv
```

Expected output:
```
Tensor Information:
  Shape: (10000, 3, 4)
  Number of genes: 10000
  Number of species: 3
  Number of timepoints: 4
  Sparsity: 32.14%
  Value range: [0.0000, 12.2143]
  Mean: 0.7777
  Std: 1.0315
```

### 3. Run Tensor Decomposition

#### PARAFAC (CP) Decomposition

```bash
stdm decompose \
  --input gene_expression.csv \
  --output results_parafac \
  --rank 5 \
  --method parafac \
  --log-transform \
  --standardize
```

#### Tucker Decomposition

```bash
stdm decompose \
  --input gene_expression.csv \
  --output results_tucker \
  --rank 5 \
  --method tucker \
  --log-transform \
  --standardize
```

## Understanding the Output

### Output Directory Structure

After running decomposition, each run creates a timestamped directory with the following structure:

```
results/
└── 20251012_175729/  # Timestamped directory
    ├── README.md                  # Comprehensive analysis report (NEW!)
    ├── gene_factors.npy
    ├── species_factors.npy
    ├── time_factors.npy
    ├── reconstructed_tensor.npy
    └── summary.json
```

### Output Files

- **README.md**: **NEW!** Comprehensive analysis report that includes:
  - Run configuration and timestamp
  - Input data summary (shape, sparsity, statistics)
  - Quality assessment of the decomposition
  - **Optimal parameter recommendations** based on reconstruction error
  - Detailed descriptions of all output files
  - Python code examples for loading and analyzing results
  - Biological interpretation guidelines
  - **This is your first stop for understanding the results!**

- **gene_factors.npy**: Gene factor matrix (n_genes × rank)
  - Each row represents a gene's loading on each component
  - Higher absolute values indicate stronger association with that component

- **species_factors.npy**: Species factor matrix (n_species × rank)
  - Shows how each species contributes to each component

- **time_factors.npy**: Time point factor matrix (n_timepoints × rank)
  - Shows temporal patterns for each component

- **reconstructed_tensor.npy**: Reconstructed tensor from the decomposition
  - Can be compared with original for validation

- **summary.json**: Decomposition statistics
  - Includes reconstruction error (lower is better)
  - Factor shapes for verification

### Reading the Analysis Report

The automatically generated `README.md` in each results directory provides:

1. **Quality Assessment**: Interpretation of reconstruction error
   - Excellent (< 0.1): May be overfitting
   - Good (0.1 - 0.3): Most variation captured
   - Acceptable (0.3 - 0.5): Consider increasing rank
   - Poor (> 0.5): Increase rank or check data quality

2. **Parameter Recommendations**: Suggestions for:
   - Rank adjustment (increase/decrease)
   - Method selection (PARAFAC vs Tucker)

3. **Code Examples**: Ready-to-use Python code for:
   - Loading all result files
   - Finding top genes for each component
   - Visualizing temporal patterns
   - Comparing original vs reconstructed data

## Python API Usage

### Basic Usage

```python
from stdm import SparseTensorDecomposer, GeneExpressionLoader

# Load or generate data
loader = GeneExpressionLoader(n_genes=10000, n_species=3, n_timepoints=4)
tensor = loader.generate_synthetic_data(sparsity=0.3, seed=42)

# Preprocess
tensor = loader.preprocess(tensor, log_transform=True, standardize=True)

# Decompose
decomposer = SparseTensorDecomposer(rank=5, method="parafac")
decomposer.fit(tensor)

# Get results
gene_factors = decomposer.get_gene_factors()
species_factors = decomposer.get_species_factors()
time_factors = decomposer.get_time_factors()

# Check quality
summary = decomposer.get_summary()
print(f"Reconstruction error: {summary['reconstruction_error']:.6f}")
```

### Loading Your Own Data

Your CSV file should have this format:

```csv
gene,species,timepoint,expression
gene1,species1,t0,0.5
gene1,species1,t1,0.7
gene1,species2,t0,0.3
gene1,species2,t1,0.8
...
```

Then load it:

```python
loader = GeneExpressionLoader()
tensor = loader.load_from_csv("your_data.csv")
```

### Advanced Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

# Identify top contributing genes for each component
gene_factors = decomposer.get_gene_factors()

for component in range(5):
    # Get top 10 genes for this component
    top_indices = np.argsort(np.abs(gene_factors[:, component]))[-10:][::-1]
    
    print(f"\nComponent {component + 1} - Top 10 genes:")
    for idx in top_indices:
        print(f"  Gene {idx}: {gene_factors[idx, component]:.4f}")

# Visualize species patterns
species_factors = decomposer.get_species_factors()
plt.figure(figsize=(10, 6))
plt.imshow(species_factors.T, aspect='auto', cmap='RdBu_r')
plt.xlabel('Species')
plt.ylabel('Component')
plt.colorbar(label='Factor Value')
plt.title('Species Factor Matrix')
plt.show()
```

## Performance Tips

### Memory Optimization

For very large datasets (>10k genes), consider:

1. **Increase sparsity threshold** to reduce computational load:
   ```bash
   stdm decompose --sparsity-threshold 0.05 ...
   ```

2. **Reduce rank** for faster computation:
   ```bash
   stdm decompose --rank 3 ...
   ```

3. **Process genes in batches** (for custom analysis):
   ```python
   # Split into chunks
   chunk_size = 2000
   for i in range(0, 10000, chunk_size):
       tensor_chunk = tensor[i:i+chunk_size, :, :]
       # Process chunk...
   ```

### Speed vs. Accuracy

- **Fast but less accurate**: Fewer iterations
  ```python
  decomposer.fit(tensor, n_iter_max=10)
  ```

- **Slow but more accurate**: More iterations
  ```python
  decomposer.fit(tensor, n_iter_max=200, tol=1e-9)
  ```

### Choosing the Right Method

- **PARAFAC (CP)**: 
  - Faster
  - More interpretable
  - Better for discovering independent patterns
  - Good when you expect separable factors

- **Tucker**:
  - More flexible
  - Better reconstruction accuracy
  - Captures interactions between modes
  - Good when factors are correlated

## Interpreting Results

### Reconstruction Error

- **0.0 - 0.1**: Excellent fit (may be overfitting)
- **0.1 - 0.3**: Good fit
- **0.3 - 0.5**: Acceptable fit
- **> 0.5**: Poor fit (increase rank or check data quality)

### Factor Interpretation

1. **Gene Factors**: Identify gene modules
   - Genes with similar factor profiles are co-regulated
   - Look for biological pathways in top genes per component

2. **Species Factors**: Identify species-specific patterns
   - Components with high loading in one species indicate species-specific regulation

3. **Time Factors**: Identify temporal patterns
   - Positive/negative trends indicate up/down regulation over time
   - Oscillating patterns suggest cyclic behavior

## Troubleshooting

### Out of Memory Errors

- Reduce number of genes in test runs
- Increase sparsity threshold
- Reduce rank
- Close other applications

### Poor Reconstruction

- Increase rank
- Try different decomposition methods
- Check data preprocessing (log-transform, standardize)
- Verify data quality (no NaN, no extreme outliers)

### Slow Performance

- Reduce n_iter_max
- Increase tolerance (tol)
- Use PARAFAC instead of Tucker
- Pre-filter low-variance genes

## Citation

If you use this workflow in your research, please cite:

```bibtex
@software{workflow_stdm,
  title = {Sparse Tensor Decomposition for Multi-species Gene Expression},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/sr320/workflow-stdm}
}
```

## Support

For issues, questions, or contributions, please visit:
https://github.com/sr320/workflow-stdm/issues
