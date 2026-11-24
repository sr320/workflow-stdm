# Plan: moving workflow-stdm onto the MOSAiC data

Written 2026-11-23, after reviewing the codebase, the 2026 eScience talk
(text only; slide figures were not readable), and <https://urol-e5.github.io/MOSAiC/>.

Working assumption: this repo becomes the clean, reproducible version of the
workflow in the talk (sparse tensor decomposition, then elastic net prediction
of expression from epigenetic features). If the repo has a different role,
phases 2-5 change.

## Where the codebase stands

- **Method differs from the talk.** `src/stdm/decomposition.py` wraps plain
  tensorly PARAFAC/Tucker. No L1 sparsity, no non-negativity, no
  cross-validated rank or sparsity selection (the Barnacle approach).
  "Sparse" only means zeroing input values below 0.01.
- **Tensor shape loses the replication.** The code assumes
  genes x 3 species x 4 timepoints, so colonies have no place in it.
- **Every committed result is synthetic.** `input-data/vst_counts_matrix.csv`
  (9,800 ortholog groups x 117 samples: 40 ACR, 39 POC, 38 POR) is not read
  by any code.
- **The rank-35 runs are not meaningful.** A 1,000 x 3 x 4 tensor has at most
  12 independent columns, so rank 35 reconstructs it exactly (error ~1e-13).
- **Preprocessing would be wrong for real data.** `stdm run` log-transforms by
  default (double-transforming VST values), and missing colony-timepoints
  would be filled with zeros.
- **2 of 13 tests fail.** `test_csv_io`: names are sorted alphabetically, so
  `gene_10` lands before `gene_2` and rows are scrambled on reload.
  `test_reconstruction`: not yet diagnosed.
- **Step 2 of the talk is absent.** No elastic net, noise-injection
  validation, or stability selection code.
- **Housekeeping.** `.npy` outputs, `.DS_Store` and `.Rhistory` are tracked;
  four markdown docs describe synthetic benchmarks.

## MOSAiC data

Three species (*A. pulchra*, *P. evermanni*, *P. tuahiniensis*), 10 colonies
each, four timepoints (Jan, Mar, Sep, Nov 2020).

| Data type | Features (Apul / Peve / Ptua) | Samples (Apul / Peve / Ptua) |
|---|---|---|
| Gene counts | 44,371 / 40,389 / 31,840 | 40 / 38 / 39 |
| miRNA | 51 / 48 / 40 | 40 / 37 / 39 |
| lncRNA | 15,559 / 8,319 / 11,236 | 40 / 38 / 39 |
| CpG methylation | 6.6M / 4.8M / 6.0M | 39 / 37 / 32 |
| Gene-body methylation | 21,621 / 20,824 / 23,369 | 40 / 37 / 32 |

Also: exon-level plasticity (CV) matrices, ITS2 relative abundance,
physiology (`master_timeseries.csv`), metabolomics (141 compounds x 96
samples), lipidomics (371 lipids x 96 samples), annotated ortholog groups,
and calcification-related gene count subsets.

Wrinkles:

- miRNA columns are named like `1A10_ACR-145_TP4`; other matrices use
  `ACR-145-TP4`.
- *P. tuahiniensis* methylation has 32 samples against 39 for expression.
- The design is not fully balanced (117 of 120 colony-timepoints).
- CpG-level methylation files are about 1.2 GB each; gene-body methylation
  tables are about 10 MB.

All file URLs are listed on the site's `counts`, `plasticity`, `ortho`,
`its` and `physiology` pages; five were spot-checked and are live.

## Phases

### 0. Cleanup
Done (2026-11-23).

- Fix the loader ordering bug; diagnose and fix `test_reconstruction`.
- Add pytest as a dev dependency.
- Untrack result `.npy` files, `.DS_Store`, `.Rhistory`.
- Collapse README / USAGE / OPTIMIZATION / PROJECT_SUMMARY into one
  accurate README.

### 1. Data layer
Done (2026-11-23).

- Manifest of MOSAiC URLs plus a fetch script that caches to a gitignored
  `data/` directory and records checksums.
- One sample metadata table: sample, species, colony, timepoint, site.
- Harmonised sample IDs across data types.
- Ortholog map from `ortholog_groups_annotated.csv`.

### 2. Tensor on real data
- Build ortholog group x colony (30, labelled by species) x timepoint (4).
- Mask the 3 missing samples instead of filling zeros.
- Keeps colony replication, which rank selection needs.

Done (2026-11-23). Notes for phase 3:
- The default tensor is 8,200 groups x 30 colonies x 4 timepoints.
- Species-level offsets are about 52% of within-group variance unless
  `--center species` is used. Colony effects (12%) are larger than time
  effects (7-9%).
- The joint VST used in the talk is reproduced by `--normalization vst-joint`.
- Decide between fitting the centred tensor and the uncentred one. Barnacle's
  non-negativity constraints need uncentred values.

### 3. Decomposition as in the talk
- Replace the tensorly wrapper with Barnacle's sparse CP.
- Fit with the missing-sample mask.
- Choose rank by held-out reconstruction error across random splits.
- Choose sparsity by split-half agreement of component membership.

Done (2026-11-23); details in the README's Sparse CP section. Notes for
phase 4:
- The talk's rank-35 model reproduces exactly but its components are not
  stable: masked and converged, mean similarity to it is 0.32.
- Held-out-sample CV cannot choose a rank (sample-level noise dominates), so
  rank was chosen by split-half stability on the species-centred tensor.
- Use the rank-3 model on the top 2,000 variable genes,
  species-centred, with no penalties
  (`results/centered-top2000-r3`). Component 1 is a site difference shared
  by all species. Component 2 is an *A. pulchra* seasonal pattern. Component
  3 is driven by outlier samples.
- Check before interpreting component 1: whether site is confounded with
  sequencing batch or extraction date.
- Gene sparsity is not useful at this rank: l1 removes whole components.

### 4. Interpretation
- Component gene lists and GO annotation.
- Overlap with the calcification gene set.
- Component scores against physiology and temperature.

In progress (2026-11-23):
- Batch check done (README, Extraction batch). Component 1 is a real site
  difference in *P. evermanni* and *P. tuahiniensis*. In *A. pulchra*, site
  and component 2's timepoint pattern are confounded with RNA extraction
  batch.
- GO slim enrichment done for components 1 and 2.
- Next: add the upstream RNA metadata to the manifest and carry extraction
  batch into the model (e.g. regress it out of *A. pulchra* before
  decomposing, or drop the re-extracted samples). Then overlap with the
  calcification genes and test against physiology.

### 5. Elastic net
- Predict component genes from miRNA, lncRNA and gene-body methylation,
  per species.
- 50 replicates, noise-injection validation, stability selection.
- Split by colony, not by sample: with about 40 samples from 10 colonies per
  species, random 80/20 splits put the same colony in train and test and
  inflate accuracy.

### 6. Extensions (from the talk's conclusions)
- Genes not shared by all three species.
- lncRNA in the tensor.
- Metabolomics and lipidomics.

## Open questions

- Resolved: the talk used rank 35 with lambdas 0.2/0.1/0.05, from
  `urol-e5/timeseries_molecular` (`M-multi-species/scripts/14.1-barnacle`).
  Its tensor build is ported in `stdm.reproduce`; Barnacle is vendored.
- Resolved in part: site is not confounded with batch in *P. evermanni* or
  *P. tuahiniensis*, but is in *A. pulchra*. Whether the dated sample names
  really mark extraction batches should be confirmed with the lab.
