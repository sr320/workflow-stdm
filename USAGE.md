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

After running decomposition, you'll find the following files in the output directory:

- **gene_factors.npy**: Gene factor matrix (10000 × rank)
  - Each row represents a gene's loading on each component
  - Higher absolute values indicate stronger association with that component

- **species_factors.npy**: Species factor matrix (3 × rank)
  - Shows how each species contributes to each component

- **time_factors.npy**: Time point factor matrix (4 × rank)
  - Shows temporal patterns for each component

- **reconstructed_tensor.npy**: Reconstructed tensor from the decomposition
  - Can be compared with original for validation

- **summary.json**: Decomposition statistics
  - Includes reconstruction error (lower is better)
  - Factor shapes for verification

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
