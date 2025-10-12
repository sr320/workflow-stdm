#!/usr/bin/env python
"""
Test script to validate the workflow with test data from input-data directory.
"""

import subprocess
import sys
from pathlib import Path
import json

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {description}")
    print(f"{'=' * 70}")
    print(f"Command: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
        print(f"✓ {description} - PASSED")
        return True
    else:
        print(f"✗ {description} - FAILED")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False


def main():
    print("=" * 70)
    print("Sparse Tensor Decomposition Workflow - Test Data Validation")
    print("=" * 70)
    
    # Check that input-data directory exists
    input_data_dir = Path("input-data")
    if not input_data_dir.exists():
        print(f"✗ Error: input-data directory not found")
        sys.exit(1)
    
    print(f"\n✓ input-data directory exists")
    
    # Check test data files
    test_files = ["gene_expression_data.csv", "gene_expression_data_small.csv"]
    for f in test_files:
        if (input_data_dir / f).exists():
            print(f"✓ Found: {f}")
        else:
            print(f"✗ Missing: {f}")
            sys.exit(1)
    
    # Test 1: Run with default data
    test1 = run_command(
        ".venv/bin/stdm run --output /tmp/test_default --rank 5 --method parafac",
        "Run workflow with default test data"
    )
    
    # Verify output
    if test1:
        summary_file = Path("/tmp/test_default/summary.json")
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
                print(f"  Reconstruction error: {summary['reconstruction_error']:.6f}")
                print(f"  Gene factors shape: {summary['gene_factors_shape']}")
        else:
            print("  ✗ Summary file not created")
            test1 = False
    
    # Test 2: Run with small dataset
    test2 = run_command(
        ".venv/bin/stdm run --input gene_expression_data_small.csv --output /tmp/test_small --rank 5 --method parafac",
        "Run workflow with small test data"
    )
    
    # Test 3: Run with Tucker method
    test3 = run_command(
        ".venv/bin/stdm run --input gene_expression_data_small.csv --output /tmp/test_tucker --rank 5 --method tucker",
        "Run workflow with Tucker decomposition"
    )
    
    # Test 4: Test info command
    test4 = run_command(
        ".venv/bin/stdm info --input input-data/gene_expression_data.csv",
        "Get info on full test dataset"
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    tests = [
        ("Default data with PARAFAC", test1),
        ("Small data with PARAFAC", test2),
        ("Small data with Tucker", test3),
        ("Info command", test4),
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for name, result in tests:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:.<50} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! The workflow works correctly with test data.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
