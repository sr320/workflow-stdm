# Changelog: README Report and Auto-timestamping Features

## Summary of New Features

This update adds comprehensive reporting and auto-timestamping capabilities to the workflow-stdm package, making it easier for users to understand their results and manage multiple analyses.

## 1. Auto-Timestamping

### What it does
Every time you run the workflow (`stdm run` or `stdm decompose`), results are automatically saved to a timestamped subdirectory, preventing accidental overwrites of previous analyses.

### Example
```bash
# First run
stdm run --rank 5 --output results/
# Creates: results/20251012_143022/

# Second run with different parameters
stdm run --rank 10 --output results/
# Creates: results/20251012_175729/

# Both result sets are preserved!
```

### Directory Structure
```
results/
├── 20251012_143022/  # First run
│   ├── README.md
│   ├── gene_factors.npy
│   ├── species_factors.npy
│   ├── time_factors.npy
│   ├── reconstructed_tensor.npy
│   └── summary.json
└── 20251012_175729/  # Second run
    ├── README.md
    ├── gene_factors.npy
    ├── species_factors.npy
    ├── time_factors.npy
    ├── reconstructed_tensor.npy
    └── summary.json
```

## 2. Comprehensive README Report

### What it does
Each analysis run automatically generates a detailed `README.md` file in the output directory that serves as a comprehensive analysis report.

### What's included

#### Run Information
- Timestamp of the analysis
- Input file path
- Output directory
- Method and rank used

#### Input Data Summary
- Tensor shape (genes × species × timepoints)
- Data statistics (sparsity, value ranges, mean, std)
- Number of genes, species, and timepoints

#### Analysis Parameters
- Decomposition method (PARAFAC/Tucker)
- Rank
- Sparsity threshold
- Normalization settings
- Preprocessing steps (log transform, standardization)

#### Results Summary
- Reconstruction error
- **Quality assessment** (Excellent/Good/Acceptable/Poor)
- Factor matrix shapes

#### Quality Assessment & Recommendations

The report includes intelligent recommendations based on the reconstruction error:

| Reconstruction Error | Quality | Recommendation |
|---------------------|---------|----------------|
| < 0.1 | Excellent | May be overfitting - consider reducing rank |
| 0.1 - 0.3 | Good | Model captures most variation |
| 0.3 - 0.5 | Acceptable | Consider increasing rank |
| > 0.5 | Poor | Increase rank or check data quality |

**Rank Suggestions:**
- Poor fit: Suggests increasing rank by 5 or 10
- Excellent fit: Suggests decreasing rank to reduce overfitting
- Acceptable/Good fit: Indicates current rank is appropriate

#### Output Files Documentation

Detailed description of each output file:
- **Purpose**: What the file contains
- **Dimensions**: Matrix/tensor shapes
- **Interpretation**: How to understand the values
- **Usage Examples**: Python code to load the file

#### Python Code Examples

Ready-to-use code snippets for:
- Loading all result files
- Identifying top genes for each component
- Visualizing temporal patterns
- Comparing original vs reconstructed data
- Examining factor matrices

#### Biological Interpretation Guide

Guidance on:
- Identifying gene modules (co-regulated genes)
- Finding species-specific patterns
- Analyzing temporal dynamics
- Suggested follow-up analyses

## 3. Enhanced Documentation

### Updated README.md
- New features section describing auto-timestamping and README reports
- Expanded documentation on using custom data
- Example directory structures
- Data format requirements

### Updated USAGE.md
- Auto-timestamping examples
- Custom data format requirements and tips
- Comprehensive output file descriptions
- Quality assessment guidelines
- Step-by-step guide for interpreting results

## 4. Using Your Own Data

The workflow now has clearer documentation on how to use custom data:

```bash
# Use data from input-data directory
stdm run --input my_data.csv --output my_results/

# Use data from any path
stdm run --input /path/to/my/data.csv --output results/

# Use absolute path
stdm run --input ~/research/expression_data.csv --output ~/results/
```

### Required Data Format
```csv
gene,species,timepoint,expression
gene1,species1,t0,0.5
gene1,species1,t1,0.7
gene1,species2,t0,0.3
...
```

## Example Workflow

```bash
# Run analysis with default test data
stdm run --rank 5 --method parafac --output my_analysis/

# Output:
# ======================================================================
# Sparse Tensor Decomposition Workflow
# ======================================================================
# 
# Run Timestamp: 20251012_180231
# Input: input-data/gene_expression_data.csv
# Output: my_analysis/20251012_180231/
# Method: PARAFAC
# Rank: 5
# 
# [... analysis steps ...]
# 
# Results saved to: my_analysis/20251012_180231/
# Read my_analysis/20251012_180231/README.md for detailed analysis report

# View the analysis report
cat my_analysis/20251012_180231/README.md

# The README.md will contain:
# - Quality assessment (e.g., "Acceptable" for error 0.329)
# - Rank recommendations (e.g., "Current rank appears appropriate")
# - Detailed file descriptions
# - Python code examples
# - Interpretation guidelines
```

## Benefits

1. **Never lose results**: Auto-timestamping prevents accidental overwrites
2. **Self-documenting**: Each analysis includes a comprehensive report
3. **Parameter optimization**: Get intelligent recommendations based on results
4. **Easy to share**: README reports make it easy to share and understand results
5. **Reproducibility**: Complete run configuration is documented
6. **Learning tool**: Interpretation guidelines help users understand their results
7. **Quick start**: Python code examples get users analyzing results faster

## Implementation Details

### Files Modified
- `src/stdm/cli.py`: Added timestamp generation, README report generation, updated both `run_workflow()` and `run_decompose()` functions
- `README.md`: Added new features documentation and custom data usage
- `USAGE.md`: Enhanced with auto-timestamping examples and output documentation

### New Function
- `generate_readme_report()`: Generates comprehensive analysis reports with quality assessment and recommendations

### Backward Compatibility
- All existing functionality remains unchanged
- Auto-timestamping is always enabled (ensures results safety)
- Old scripts will work but will now create timestamped directories

## Testing

All features have been tested with:
- Default test data (10,000 genes)
- Small test data (1,000 genes)
- Multiple runs with different parameters
- Both PARAFAC and Tucker methods
- Different rank values (3, 4, 5, 7, 8)
- Both `stdm run` and `stdm decompose` commands

Results verified:
- ✅ Timestamped directories created correctly
- ✅ README reports generated with accurate information
- ✅ Quality assessments match reconstruction errors
- ✅ Rank recommendations adjust based on results
- ✅ All output files present and correct
- ✅ Python code examples are syntactically correct
