#!/usr/bin/env python
"""
Example: Running the workflow with test data from input-data directory.

This example demonstrates the simplest way to run the sparse tensor 
decomposition workflow using the provided test datasets.
"""

import subprocess

print("=" * 70)
print("Example: Using Test Data with the Workflow")
print("=" * 70)

# Example 1: Use default test data
print("\n1. Running with default test data (10k genes)...")
print("   Command: stdm run --rank 5 --method parafac")
print()

result = subprocess.run(
    ["stdm", "run", "--rank", "5", "--method", "parafac", "--output", "example_results/default"],
    capture_output=False,
    text=True
)

if result.returncode == 0:
    print("\n✓ Example 1 completed successfully!")
    print("  Results saved to: example_results/default/")

# Example 2: Use small test data
print("\n" + "=" * 70)
print("\n2. Running with small test data (1k genes - faster)...")
print("   Command: stdm run --input gene_expression_data_small.csv --rank 5")
print()

result = subprocess.run(
    ["stdm", "run", "--input", "gene_expression_data_small.csv", 
     "--rank", "5", "--output", "example_results/small"],
    capture_output=False,
    text=True
)

if result.returncode == 0:
    print("\n✓ Example 2 completed successfully!")
    print("  Results saved to: example_results/small/")

# Example 3: Tucker decomposition
print("\n" + "=" * 70)
print("\n3. Running with Tucker decomposition...")
print("   Command: stdm run --input gene_expression_data_small.csv --method tucker")
print()

result = subprocess.run(
    ["stdm", "run", "--input", "gene_expression_data_small.csv", 
     "--method", "tucker", "--rank", "5", "--output", "example_results/tucker"],
    capture_output=False,
    text=True
)

if result.returncode == 0:
    print("\n✓ Example 3 completed successfully!")
    print("  Results saved to: example_results/tucker/")

print("\n" + "=" * 70)
print("All examples completed!")
print("=" * 70)
print("\nNext steps:")
print("  - View results in the example_results/ directory")
print("  - Load factor matrices using numpy: np.load('example_results/default/gene_factors.npy')")
print("  - Check summary.json for reconstruction error and factor shapes")
