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
- Batch handling done. The RNA metadata is in the manifest
  (`multi_rna_metadata`), `data/samples.csv` has plate, RNA concentration
  and inferred extraction batch, and `--remove-batch` corrects the tensor.
  After correction rank 2 is used (rank 3 is degenerate):
  `results/centered-top2000-r2-nobatch`. The site component holds in
  *P. evermanni* and *P. tuahiniensis* and is absent in *A. pulchra*. The
  *A. pulchra* timepoint component is provisional.
- Physiology, ITS2 and calcification-gene tests done (`stdm interpret`,
  `results/phase4`). No association holds within site x timepoint. The
  *A. pulchra* timepoint component shares calcification's TP1/TP4 vs
  TP2/TP3 split, and calcification genes are not enriched in either
  component.

Done (2026-11-23) apart from two questions for the lab: whether the dated
RNA sample names mark extraction batches, and whether the near-zero TP1 and
TP4 calcification values are real. Notes for phase 5:
- The site component (*P. evermanni*, *P. tuahiniensis*) is the most solid
  response to predict.
- Colony-level variation beyond site and season is mostly sample-specific,
  so expect modest prediction accuracy.

### 5. Elastic net
- Predict component genes from miRNA, lncRNA and gene-body methylation,
  per species.
- 50 replicates, noise-injection validation, stability selection.
- Split by colony, not by sample: with about 40 samples from 10 colonies per
  species, random 80/20 splits put the same colony in train and test and
  inflate accuracy.

First run done (2026-11-23) on the site component (README, Elastic net):
`stdm elastic-net`, colony-pair outer CV, within-site permutation null,
noise-injection stability selection; `results/phase5-site`.
- lncRNA predicts the site component with held-out R² 0.93 (0.62–0.75 within
  site), but the component is lncRNA's own leading axis (PC1 r = −0.91 in
  *P. evermanni*). lncRNA shares libraries with the gene counts, so this
  mostly reflects shared biology and technical variation.
- Gene-body methylation, the only independent assay, predicts nothing.
- miRNA: weak within-site signal in *P. evermanni* only (p = 0.04,
  uncorrected).

- Shared-variation check (`--adjust`): removing technical covariates
  (library sizes, RNA concentration, plate, extraction batch) leaves the
  within-site prediction significant (R² 0.32 *P. evermanni*, 0.75
  *P. tuahiniensis*). Removing the leading gene-expression PCs removes all
  predictive signal, and unadjusted lncRNA predicts other gene PCs equally
  well. lncRNA tracks the transcriptome's main axes; cause cannot be
  separated with these data.

- Gene-level methylation test (`stdm cis-meth`, `results/phase5-cis`):
  each gene's body methylation against its own expression, with a colony
  permutation null. Between colonies, coupling is real in all three species
  (3-6x the expected number of strong correlations, mostly positive,
  in weakly methylated, variable genes). Over time within a colony, only
  *A. pulchra* shows any (12 genes at q < 0.1). Component genes are not more
  coupled than other genes. The between-colony signal may be genotype
  (CpG SNPs, cis-eQTL, mapping) rather than regulation.

Possible next steps:
- Check the between-colony methylation-expression genes for genotype
  effects (CpG coverage and SNP calls in the methylation data).
- Run the *A. pulchra* timepoint component (Component_1), keeping in mind it
  is provisional.

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
