# workflow-stdm

Sparse tensor decomposition and elastic net regression for multi-species,
multi-omic coral time-series data from the
[E5 MOSAiC project](https://urol-e5.github.io/MOSAiC/): *Acropora pulchra*,
*Porites evermanni* and *Pocillopora tuahiniensis*, 10 colonies each
(5 at Manava, 5 at Mahana), sampled at four timepoints in 2020.

The roadmap is in [PLAN.md](PLAN.md).

## Status

| Phase | State |
|---|---|
| 0. Cleanup | done |
| 1. Data layer (`stdm fetch`, `stdm samples`, `stdm.mosaic`) | done |
| 2. Tensor on real data (`stdm build-tensor`, `stdm.tensor`) | done |
| 3. Barnacle sparse CP with rank and sparsity selection | done; see [Sparse CP](#sparse-cp) |
| 4. Interpretation (`stdm interpret`) | done; see [Physiology, symbionts and calcification genes](#physiology-symbionts-and-calcification-genes) |
| 5. Elastic net (`stdm elastic-net`, `stdm.elasticnet`) | first run done; see [Elastic net](#elastic-net) |
| 6. Extensions | not started |

The `generate` / `decompose` / `run` / `info` commands and
`SparseTensorDecomposer` are the original synthetic-data scaffold: plain
tensorly PARAFAC/Tucker on a genes x species x timepoints tensor. They are not
the method from the talk and should not be pointed at real data; use
`build-tensor` and `sparse-cp` instead.

## Install

```bash
uv sync
```

Commands below are run with `uv run`, or activate `.venv` first.

## Data

```bash
uv run stdm fetch --list
```

```bash
uv run stdm fetch
```

```bash
uv run stdm samples
```

`fetch` downloads every file in [data/manifest.csv](data/manifest.csv) into
`data/raw/` (gitignored, about 200 MB) and skips files already cached. Each
file's SHA-256 is checked against [data/checksums.csv](data/checksums.csv); a
mismatch means the upstream file changed and is reported as a warning
(`--update-checksums` accepts it). Narrow a fetch with `--species`,
`--data-type` or `--key`. The three CpG-level methylation files are about
1 GB each and are only fetched with `--include-large`.

`samples` writes [data/samples.csv](data/samples.csv): one row per
colony x timepoint with species, colony, site, timepoint, month and a
`has_<type>` flag per data type, plus RNA extraction details and, when
`multi_genotypes` is cached, `genotype_id` and `identity_flagged` (see
[Sample identity](#sample-identity-genotypes-and-mix-ups)).

### Python API

```python
from stdm import mosaic

meta = mosaic.build_sample_metadata()            # or pd.read_csv("data/samples.csv")
counts, _ = mosaic.load_matrix("apul_gene")      # features x samples
lnc, coords = mosaic.load_matrix("peve_lncrna")  # annotation columns split out
mirna, _ = mosaic.load_matrix("ptua_mirna")      # well-prefixed labels harmonised
orthologs = mosaic.load_orthologs()              # one row per group, canonical gene IDs
physiology = mosaic.load_table("multi_physiology")
```

Matrix columns are canonical sample IDs (`ACR-145-TP4`) sorted by species,
colony and timepoint. Gene-keyed matrices use canonical gene IDs, which match
the ortholog table.

### What is in the data

| Data type | Apul | Peve | Ptua | Notes |
|---|---|---|---|---|
| Gene counts | 44,371 x 40 | 40,389 x 38 | 31,840 x 39 | raw counts |
| miRNA | 51 x 40 | 48 x 37 | 40 x 39 | raw counts |
| lncRNA | 15,559 x 40 | 8,319 x 38 | 11,236 x 39 | raw counts |
| Gene-body methylation | 21,621 x 40 | 20,824 x 37 | 23,369 x 32 | percent; 2-5% missing |
| Exon CV | 10,006 x 40 | 12,174 x 38 | 11,342 x 39 | 6-10% missing |
| Metabolomics / lipidomics | 29 | 38 | 29 | samples per species; 141 / 371 features |

Also cached: epigenetic-machinery and calcification-gene subsets, exon-level
counts, the annotated ortholog table, physiology and ITS2.

### Things to know before modelling

- **Unbalanced design.** 117 of 120 colony-timepoints have RNA-seq.
  *P. tuahiniensis* has gene-body methylation for only 32 samples.
- **Genes in more than one ortholog group.** 230 *A. pulchra* genes appear in
  two or more groups, because different transcripts of one gene were assigned
  to different groups. A gene must not be counted twice in a tensor.
- **Ortholog genes missing from counts.** 3.2% of *A. pulchra* ortholog genes
  (IDs from about `FUN_044375` on) are not in the gene count matrix.
- **Duplicate ortholog rows.** 46 ortholog groups are repeated upstream (one
  row per annotation hit); `load_orthologs` keeps the first.
- **CpG file filtering.** The CpG-level files are named `filtered_n20` and
  contain many missing values. They are not "covered in all samples", as the
  website states.
- **`input-data/vst_counts_matrix.csv`** (9,800 three-way ortholog groups x
  117 samples) is a blind DESeq2 VST computed jointly across the three
  species. `stdm build-tensor --normalization vst-joint` reproduces it
  (r = 0.9999). Its group filter could not be matched exactly and was probably
  applied to an older ortholog table.

## Expression tensor

```bash
uv run stdm build-tensor
```

This builds an ortholog group x colony x timepoint tensor from the raw gene
counts and saves it to `data/processed/tensor-<normalization>/` (gitignored).
The output holds `values.npy`, `groups.csv`, `colonies.csv` and `params.json`.

- **Groups.** Three-way ortholog groups whose genes are in all three count
  matrices, excluding groups that share a gene with another group. A group is
  kept when, in every species, at least `--min-fraction` (0.5) of samples have
  a normalised count of at least `--min-count` (10). With the defaults this
  keeps 8,200 of 10,346 groups.
- **Colonies.** All 30 colonies, ordered by species and then colony number,
  labelled with species and site. Keeping colonies as a mode preserves the
  replication that rank selection needs.
- **Missing samples.** `POR-236-TP3`, `POR-236-TP4` and `POC-201-TP4` are
  `NaN` and `False` in `.mask`; they are never filled with zeros.
- **`--normalization log2-mor`** (default): median-of-ratios size factors
  within each species, then log2(x + 1).
- **`--normalization vst-joint`**: blind DESeq2 VST across species, as in the
  2026 talk. It needs `uv sync --extra vst`.
- **`--center species`**: subtracts each group's mean within each species.

```python
from stdm.tensor import ExpressionTensor, build_expression_tensor

t = ExpressionTensor.load("data/processed/tensor-log2-mor")
t.values, t.mask, t.groups, t.colonies, t.timepoints
t.to_matrix()  # groups x observed samples
```

Share of within-group variance in each tensor (sums of squares):

| Tensor | Species | Site | Colony | Time | Residual |
|---|---|---|---|---|---|
| log2-mor | 0.53 | 0.03 | 0.12 | 0.07 | 0.25 |
| vst-joint | 0.51 | 0.04 | 0.12 | 0.09 | 0.24 |
| log2-mor, species-centred | 0 | 0.07 | 0.25 | 0.15 | 0.53 |

Site is measured within species, colony within site, and time within species.
Without centring, half the variance is species-level offsets, so the first
components of a decomposition would mostly describe species differences.
Colony (genotype) effects are larger than seasonal time effects.

## Sparse CP

`src/stdm/barnacle/` is a vendored copy of
[Barnacle](https://github.com/blasks/barnacle) (MIT), the l1-penalised CP
used in the 2026 talk. It is vendored because Barnacle pins Python < 3.12
and numpy < 2. One addition: a `mask` argument fits around missing samples
instead of treating them as values.

```bash
uv run stdm sparse-cp --tensor talk-centered --top-variable 2000 --nonneg-modes \
    --rank 3 --lambdas 0 0 0 --n-init 20 --output results/centered-top2000-r3
```

- **`--tensor`**: `talk` is `input-data/vst_counts_matrix.csv` with missing
  samples masked; `talk-centered` also removes each species' gene means; a
  directory from `build-tensor` also works.
- **`--top-variable N`**: keeps the N groups with the highest within-species
  variance.
- **`--nonneg-modes`**: modes held non-negative (default: gene). Pass the flag
  alone for none, which centred data needs.

Output: gene, colony and time factor CSVs, per-initialisation progress logs,
and `run.json` with losses, iterations and convergence.

Selection commands:

| Command | Chooses | How |
|---|---|---|
| `select-rank` | rank | held-out samples, 1-SE rule on test SSE |
| `select-rank-stability` | rank | components reproduced across site-balanced colony halves |
| `select-lambda` | gene sparsity | split-half factor match score, 1-SE rule |
| `reproduce-talk` | — | refits the talk's rank-35 model and compares with the published factors |

### Results on the talk's VST data

- **The talk model reproduces exactly.** With zero-filled missing samples and
  the talk's settings (rank 35, lambdas 0.2/0.1/0.05, seed 81), the factors
  match the published ones to 1.2e-7. All three initialisations stopped at
  the 2,000-iteration cap without converging.
- **Those components are not stable.** Masking the missing samples and running
  to convergence (about 6,000 iterations) gives a mean component similarity
  of 0.32 to the published model, with none above 0.9. The published model
  reconstructs the three missing samples as zeros.
- **Held-out-sample CV cannot choose a rank.** On uncentred data it picks
  rank 3, which only reproduces each species' mean profile (flat time
  factors). On species-centred data no rank from 1 to 20 predicts held-out
  samples better than the species means, and neither do colony or timepoint
  averages: sample-level noise outweighs the shared colony and time signal.
- **Gene sparsity removes components, not genes.** At rank 3 the l1 penalty
  leaves genes dense up to lambda 10, then switches whole components off.
  The selected gene lambda is 0.
- **Split-half stability** on the centred tensor finds no rank whose
  components all reproduce (similarity > 0.9) across colony halves, with all
  genes or the top 1,000 or 2,000 variable genes. Ranks 2-3 give the best
  agreement (about 0.6).

### Rank-3 model, species-centred, top 2,000 variable genes

Three components explain 23% of the within-species, gene-centred variance.
Similarity to refits on each of 10 colony halves is in brackets.

| Component | Variance | Colonies | Time (Jan, Mar, Sep, Nov) | Top genes |
|---|---|---|---|---|
| 1, site (0.87) | 8% | Mahana > Manava in all three species (5-vs-5 separation in apul and peve, p = 0.008; ptua p = 0.016) | positive throughout | + MCM7, E2F8, EIF2AK3; − hemicentin-1, MMP-25, FHL2, RUNX1 |
| 2, *A. pulchra* season (0.96) | 10% | *A. pulchra* only (Mahana 62, Manava 31; others near 0) | + Jan, Nov; − Mar, Sep | + eIF5B, DnaJC8, PHF14; − KLF5, malectin |
| 3, outliers (0.51) | 5% | dominated by POC-52 and ACR-186 | TP1 only | not interpreted |

**Extraction batch.** The upstream RNA metadata
(`M-multi-species/data/rna_metadata.csv` in `urol-e5/timeseries_molecular`)
gives a sequencing plate, RNA concentration and, for some samples, an
extraction date in the sample name. All samples went in one sequencing
order and plate is not related to site. Per-sample component scores
(projection onto the gene loadings), regressed on site with and without
extraction batch, log RNA concentration and timepoint:

| | Site effect, component 1 | Adjusted | Undated samples only |
|---|---|---|---|
| *P. evermanni* | −28 (p = 3e-6) | −29 (p = 9e-6) | Mahana +15, Manava −20 (p = 4e-4, n = 26) |
| *P. tuahiniensis* | −18 (p = 3e-4) | −19 (p = 4e-4) | Mahana +9, Manava −10 (p = 0.008, n = 32) |
| *A. pulchra* | −12 (p = 1e-4) | −3 (p = 0.34) | Mahana +1, Manava −5 (p = 0.21, n = 21) |

- In *P. evermanni* and *P. tuahiniensis* the site component holds after
  adjustment and within one batch.
- In *A. pulchra*, site and extraction batch are confounded: the 10 samples
  extracted in January 2023 are all Mahana, and 8 of 9 extracted in August
  2024 are Manava. The site effect disappears after adjustment, and RNA
  concentration is associated with the score within site (p = 0.007).
- Component 2 is also partly batch: the 2023 and 2024 extractions are mostly
  TP1 and TP4, which is where the component is positive. A timepoint effect
  remains after adjustment (p = 8e-5), mainly a TP2 dip, but it is about
  half the unadjusted size.

Batch here assumes dated sample names mark extraction batches and undated
(numbered) names are one earlier batch; this is inferred from the names,
not documented.

**Batch-corrected model.** `--remove-batch` subtracts per-gene extraction
batch effects within each species, keeping site and timepoint effects in
the regression so that they are not removed with batch (as in limma's
`removeBatchEffect` with a design). Batch labels come from
`data/samples.csv` (`extraction_batch`). After correction, rank 3 becomes
degenerate (two components with congruence −0.57, each explaining negative
variance alone), so rank 2 is used:

```bash
uv run stdm sparse-cp --tensor talk-centered --remove-batch --top-variable 2000 \
    --nonneg-modes --rank 2 --lambdas 0 0 0 --n-init 20 --output results/centered-top2000-r2-nobatch
```

| Component | Matches uncorrected | Variance | Pattern | Site p (apul, peve, ptua) | Half-refit similarity |
|---|---|---|---|---|---|
| Site | component 1 (gene 0.99, time 0.99) | 9% | Manava > Mahana in *P. evermanni* and *P. tuahiniensis*; none in *A. pulchra* | 0.13, 2e-5, 9e-4 | 0.78 |
| *A. pulchra* time | component 2 (time 0.97, gene 0.65) | 6% | *A. pulchra* colonies at both sites alike; Mar and Sep up, Jan and Nov down | timepoint p = 2e-6 in *A. pulchra* | 0.51 (min 0.01) |

- The site component is a real difference between sites in two species.
  The *A. pulchra* site difference was extraction batch.
- An *A. pulchra* timepoint pattern survives correction, but it is less
  reproducible and its gene content changed (cosine 0.65 to the uncorrected
  component). Because the 2023 extraction holds only TP1 and TP4 samples,
  batch and timepoint are hard to separate in *A. pulchra*; treat this
  component as provisional.

### Physiology, symbionts and calcification genes

```bash
uv run stdm interpret --model results/centered-top2000-r2-nobatch --output results/phase4
```

`interpret` rebuilds the model's tensor from its `run.json` and scores each
sample on every component (projection onto the gene loadings). It then tests
those scores within each species against 11 physiology traits and the ITS2
profiles (Spearman, BH q). Each test is run three ways: on all samples,
within site, and within site x timepoint. The last version rules out
associations created only by a shared site or seasonal pattern. It also
compares the top loadings with the 657 calcification-gene ortholog groups
from the `*_biomin` files.

- **Physiology:** 8 of 66 tests reach q < 0.1 on all samples or within site;
  **none hold within site x timepoint** (all q > 0.3).
  - The *A. pulchra* timepoint component tracks calcification rate in
    *A. pulchra* (rho = 0.54) and *P. evermanni* (0.49). Calcification is
    near zero at TP1 and TP4 and high at TP2 and TP3 in all three species,
    the same split as the component. Within a timepoint, the correlation is
    about 0.
  - The site component tracks host biomass (rho = 0.50) and calcification
    (−0.49) in *A. pulchra*, again only through shared timepoint structure.
  - Seasonal calcification is measured on separate fragments from the RNA, so
    its TP1/TP4 vs TP2/TP3 split is independent support that the timepoint
    pattern is not only extraction batch. The near-zero TP1 and TP4 values
    may also reflect a measurement difference; this needs checking with the
    physiology team.
- **ITS2:** no associations (0 of 12 tests at q < 0.1).
- **Calcification genes:** 127 of the 657 groups are in the top 2,000
  variable genes. They are not over-represented among the top 200 loadings
  of either component (Fisher p ≥ 0.6). Their loadings lean slightly toward
  the Manava side of the site component (rank-sum p = 0.04, uncorrected).

**GO slim enrichment** (`goslim_enrichment.csv`; top 200 loadings of the 737
annotated groups against the rest, Fisher test, BH q):

- Component 1, Mahana side: DNA binding (q = 0.02), DNA repair (q = 0.03).
- Component 1, Manava side: nothing at q < 0.1. The top terms are
  development, mitochondrion, plasma membrane and cell motility.
- Component 2, positive side: nucleolus, RNA binding, nucleoplasm and mRNA
  metabolic process (q < 0.001). This ribosome-biogenesis signature is also
  typical of RNA-quality differences, consistent with the batch
  confounding.

## Elastic net

```bash
uv run stdm elastic-net --component Component_2 --permutations 50 --workers 7 \
    --output results/phase5-site
```

Predicts each sample's score on a CP component (from `stdm interpret`'s
`sample_scores.csv`) from regulatory layers, per species. Component_2 of
`results/centered-top2000-r2-nobatch` is the site component.

- **Predictors:**
  - miRNA and lncRNA counts: median-of-ratios normalisation (DESeq2
    `poscounts` when no feature is positive in every sample), expression
    filter, log2(x + 1).
  - Gene-body methylation: genes with at most 20% missing values; gaps are
    filled with training-fold means.
  - The 1,000 most variable features per block (no target information used).
  - Libraries under 10% of the median size are dropped with a warning. This
    removes 9 *P. evermanni* and 7 *P. tuahiniensis* miRNA samples, so miRNA
    is fitted on its own rather than joined to the other blocks.
- **Validation:** each outer fold holds out one colony from each site (all
  25 pairs), so no colony is in training and test at once. An inner grouped
  `ElasticNetCV` chooses the penalty and l1 ratio.
- **Targets:** `raw` (the score, compared with a baseline that predicts each
  site's training mean) and `within_site` (the score minus its training site
  mean: what the predictors explain beyond the site difference).
- **Null:** 50 refits with colony labels shuffled within site. With 50
  permutations the smallest possible p is 0.02.
- **Stable features:** selected in more outer folds than the 95th percentile
  of injected noise features (column-permuted copies of real features).

Outputs per run: `{species}_{predictors}_{target}_predictions.csv`,
`_selection.csv`, `_folds.csv`, plus `summary.csv` and `run.json`.

**Site component, held-out R²** (Spearman in brackets; * p_perm = 0.02, the
minimum):

| Predictors | *P. evermanni* raw | within site | *P. tuahiniensis* raw | within site |
|---|---|---|---|---|
| Site-mean baseline | 0.37–0.43 | 0 | 0.07–0.18 | 0 |
| miRNA | −0.02 (0.12) | 0.17 (0.43), p = 0.04 | −0.36 | −0.16 |
| lncRNA | 0.93 (0.96)* | 0.62 (0.80)* | 0.93 (0.95)* | 0.75 (0.86)* |
| Gene-body methylation | −0.10 | −0.11 | −0.07 | −1.19 |
| lncRNA + methylation | 0.91 (0.95)* | 0.49 (0.74)* | 0.96 (0.97)* | 0.65 (0.79)* |

- **lncRNA predicts the site component almost perfectly,** including the
  variation among colonies and timepoints within a site. Adding methylation
  does not help.
- **But this is mostly shared structure, not regulation.** The site component
  is the leading axis of lncRNA variation itself: lncRNA PC1 correlates −0.91
  with the score in *P. evermanni* (−0.82 within site) and −0.68 in
  *P. tuahiniensis*, where PC2 adds −0.60. lncRNA and gene counts come from the
  same RNA-seq libraries, so they share biology and also library-level
  technical variation. Correspondingly, 40–130 lncRNAs per model pass the
  stability threshold, many at frequency 1.0: a broad axis, not a few
  regulators.
- **Gene-body methylation predicts nothing** in either species or target
  (p ≥ 0.16). It is the only layer measured on a separate assay, so this is
  the cleanest test of an epigenetic link, and it is negative at this sample
  size.
- **miRNA:** a weak within-site signal in *P. evermanni* (R² 0.17, p = 0.04,
  one of 16 tests, not corrected); nothing in *P. tuahiniensis*. It does
  not survive the genotype-corrected rerun (R² −0.02, p = 0.24).

**Is lncRNA's signal anything beyond what it shares with gene expression?**
`--adjust` regresses every predictor on covariates before fitting (no
target information, so it is done once on all samples):

```bash
uv run stdm elastic-net --component Component_2 --block-sets lncrna --adjust technical \
    --permutations 50 --workers 7 --output results/phase5-site-technical
uv run stdm elastic-net --component Component_2 --block-sets lncrna --adjust gene-pcs --n-pcs 3 ...
```

- `technical`: log library size of genes and lncRNA, log RNA concentration,
  plate and extraction batch.
- `gene-pcs`: the top K principal components of the species' gene
  expression (2,000 most variable genes, log normalised counts).

| lncRNA predictors | *P. evermanni* raw | within site | *P. tuahiniensis* raw | within site |
|---|---|---|---|---|
| Unadjusted | 0.93* | 0.62* | 0.93* | 0.75* |
| Technical covariates removed | 0.10 (p = 0.14) | 0.32* | 0.67* | 0.75* |
| Gene PCs 1–3 removed | −6.8 | −1.8 | −0.74 | −0.51 |
| Gene PCs 1–10 removed | −0.02 | 0.23 | −0.05 | 0.00 |

(* p_perm = 0.02. Gene-PC rows are single CV runs without permutations;
`results/phase5-site/lncrna_adjustment.csv`.)

- **Measured technical variation does not explain the within-site
  prediction.** It stays at R² 0.32 (*P. evermanni*) and 0.75
  (*P. tuahiniensis*). The *P. evermanni* raw R² falls because technical
  covariates, mainly extraction batch, partly line up with site in that
  species (they explain 19% of the score), so removing them also removes
  part of the site difference.
- **Removing gene-expression axes removes all of it.** The site component
  is gene PC1 in *P. evermanni* (r = −0.76) and gene PC2 in
  *P. tuahiniensis* (r = 0.93). Once those axes are taken out of lncRNA,
  nothing predicts the score; the large negative R² values are the model
  fitting leftover noise. This is close to circular, since the score is
  itself built from gene expression, but it shows there is no lncRNA
  signal for the component outside the shared axes.
- **lncRNA mirrors the transcriptome generally, not the site component in
  particular.** As a control, unadjusted lncRNA also predicts the other
  leading gene PCs: held-out R² 0.80, 0.51 and 0.31 for gene PCs 1, 2 and 5
  in *P. evermanni*, and 0.86, 0.81 and 0.70 for PCs 1, 2 and 5 in
  *P. tuahiniensis*.
- **Conclusion:** lncRNA and gene expression vary together along the same
  few axes, beyond what library size, RNA concentration, plate and batch
  explain. These data cannot tell whether that is lncRNA regulating genes,
  genes and lncRNAs responding to the same conditions, or unmeasured
  library effects. A predictor from an independent assay is needed to
  separate them, and the one available (gene-body methylation) predicts
  nothing.
- One *P. evermanni* sample, POR-73-TP2, has RNA concentration recorded as
  0.0 ng/µL in the RNA metadata; worth checking with the lab.

### Gene-level methylation and expression

```bash
uv run stdm cis-meth --permutations 200 --output results/phase5-cis
```

For every gene with both measurements, `cis-meth` (`stdm.cis`) correlates
the ranks of the gene's expression with the ranks of its own gene-body
methylation.

- **Genes kept:** expressed (normalised count ≥ 10 in half the samples),
  at most 20% missing methylation and methylation SD ≥ 2 percentage points.
  Missing methylation is filled with the site × timepoint mean.
- **Three levels:**
  - `all`: all samples.
  - `between_colony`: after removing site × timepoint means (colonies
    differing at the same place and time).
  - `within_colony`: after also removing colony means (a colony's methylation
    and expression changing together over time, beyond the shared season).
- **Null:** methylation profiles are reassigned between colonies within a site,
  matched by timepoint, 200 times. Null correlations from all genes are pooled
  to give per-gene p-values, followed by BH q. The global test counts genes
  whose |r| is above the null's 99th percentile; with 200 permutations the
  smallest p is 0.005.
- **Component genes:** the top 200 loadings on each side of each component
  (`results/centered-top2000-r2-nobatch`) are compared with the other genes
  in the tensor's 2,000 (Mann-Whitney on |r|).

| | *A. pulchra* | *P. evermanni* | *P. tuahiniensis* |
|---|---|---|---|
| Samples / genes | 40 / 8,479 | 37 / 6,828 | 32 / 4,575 |
| Between colony: genes above null 99th pct (null median) | 463 (80)** | 245 (66)** | 137 (44)** |
| Between colony: genes at q < 0.1 | 151 | 77 | 35 |
| Between colony: mean r (null) | 0.057 (0.000)** | 0.031 (0.002)** | 0.031 (0.006)* |
| Within colony: genes above null 99th pct (null median) | 291 (76)** | 59 (69) | 61 (45) |
| Within colony: genes at q < 0.1 | 12 | 1 | 0 |

(** p = 0.005, the minimum; * p = 0.015.)

- **Methylation and expression co-vary between colonies in all three
  species.** About 3–6 times more genes than expected by chance have strong
  correlations. Significant genes are mostly positive (70–94%), and they
  tend to be weakly methylated genes whose methylation varies more than
  average.
- **Over time within a colony, only *A. pulchra* shows coupling.** That
  species has 12 genes at q < 0.1, 9 of them negative. Nothing appears in
  *P. evermanni* or *P. tuahiniensis* beyond chance.
- **The component genes are not special.** Their |r| is no higher than for
  other genes, and none of the component gene sets is enriched for
  significant genes. In *P. evermanni*, the Manava-side site genes are
  slightly *less* coupled (p = 0.008, uncorrected).
- **The between-colony signal may be genetic rather than regulatory.** Each
  colony is a genet, so sequence differences between colonies can move both
  measurements. A SNP at a CpG removes the methylation call, and a cis-eQTL
  or a mapping difference changes counts. Over time within a colony, the
  genotype is fixed, and that is where coupling is weak or absent.
- **Overall:** gene-body methylation does relate to expression gene by gene,
  mostly as stable colony differences. It does not explain the site
  component, and there is little sign of methylation tracking expression
  over time, except in a few *A. pulchra* genes.

Outputs: `{species}_genes.csv` (r, p, q per level plus methylation mean and
SD), `{species}_null.csv`, `summary.csv`, `component_sets.csv`.

### Genotype check at the CpG level

```bash
uv run stdm fetch --key apul_cpg peve_cpg ptua_cpg apul_gff peve_gff ptua_gff --include-large
uv run stdm cpg-check --output results/phase5-cpg
```

`cpg-check` (`stdm.cpg`) asks whether the between-colony coupling could come
from sequence differences between colonies.

- **Inputs:**
  - **Genes:** every gene at q < 0.1 in `results/phase5-cis`, plus three
    control genes per hit. Controls are non-significant genes (q ≥ 0.5)
    matched on mean and SD of gene methylation.
  - **CpG data:** the merged CpG-level methylation files (`*_cpg`, about 1 GB
    each). CpGs are assigned to genes with the upstream gene models
    (`*_gff`, added to the manifest). CpGs must be present in at least 80% of
    samples.
- **Variant-like CpGs:** a C/T mutation that destroys a CpG makes bisulfite
  sequencing read it as unmethylated. A CpG is flagged when one colony is at
  or below 5% at every timepoint while at least two other colonies average
  at least 40%. This catches homozygous CpG loss only; a heterozygous loss
  halves the level and is not flagged.
- **Per gene:**
  - The methylation–expression correlation is recomputed from the gene's
    mean CpG methylation (`r_cpg`). It is then recomputed without the
    variant-like CpGs, and without the single CpG whose removal weakens it
    most.
  - The fraction of the gene's CpGs whose own correlation has the gene's
    sign is recorded.

| | *A. pulchra* | *P. evermanni* | *P. tuahiniensis* |
|---|---|---|---|
| Hits / controls with CpG data | 149 / 439 | 60 / 122 | 35 / 104 |
| Hits reproduced from CpG data (\|r\| ≥ 0.5, same sign) | 68 | 16 | 26 |
| ...still \|r\| ≥ 0.5 without variant-like CpGs | 47 | 13 | 20 |
| ...still \|r\| ≥ 0.5 without the most influential CpG | 57 | 14 | 22 |
| Variant-like CpGs, hits vs controls | 18.5% vs 9.3% | 9.9% vs 7.8% | 4.8% vs 4.6% |
| Genes with ≥ 1 variant-like CpG, hits vs controls | 62% vs 52% (p = 0.04) | 33% vs 21% (p = 0.10) | 66% vs 61% (p = 0.69) |
| CpGs agreeing in sign with the gene, hits vs controls (median) | 0.74 vs 0.62 (p < 0.001) | 0.85 vs 0.80 (p = 0.23) | 0.68 vs 0.62 (p = 0.03) |

- **The CpG file only partly reproduces the gene-level results.** Its
  filter (CpGs with data in at least 20 samples) differs from the per-sample
  10x bedgraphs behind the gene-methylation table. Rebuilt from CpGs, the
  hits' correlations agree with the gene-level ones (r = 0.67–0.83), but only
  15–43% of hits reach the q < 0.1 cut-off again. Part of that is regression
  to the mean: hits were chosen for extreme correlations. The robustness
  counts are taken on the hits that reproduce.
- **In most coupled genes the signal is spread across the gene.**
  - It survives removing variant-like CpGs in 47 of 68 *A. pulchra*, 13 of
    16 *P. evermanni* and 20 of 26 *P. tuahiniensis* genes.
  - It survives removing the most influential single CpG in 57 of 68, 14 of
    16 and 22 of 26 genes.
  - More of a hit gene's CpGs agree in direction than in matched controls.
  - So most coupled genes show a gene-wide shift rather than one site
    switched off.
- **In *A. pulchra*, genotype contributes.** Variant-like CpGs are twice as
  common in hit genes as in matched controls, and about 30% of reproduced
  hits fall below |r| = 0.5 without them. There is no clear enrichment in
  the other two species. **This did not hold up**: with one colony per
  genotype the enrichment disappears (see
  [Genotype-corrected rerun](#genotype-corrected-rerun)).
- **What this rules out, and what it doesn't.** It rules out homozygous
  CpG-destroying mutations as the main explanation. It cannot rule out
  heterozygous variants, or genetic variants elsewhere that shift
  methylation across a whole gene. Those need genotype calls from the WGBS
  reads (for example Biscuit or BS-SNPer), which these processed files
  don't contain. The within-colony test, where genotype is fixed, remains
  the genotype-free comparison, and it found coupling only in a few
  *A. pulchra* genes.
- **Data note:** ACR-225-TP1 is in the gene-methylation table but not in the
  CpG file, so *A. pulchra* uses 39 samples here.

Outputs: `{species}_genes.csv` (per gene: group, CpG counts, `r_cpg`,
`r_drop_variant`, `r_drop_top`, `frac_same_sign`, `r_gene_meth`) and
`summary.csv`.

## Sample identity: genotypes and mix-ups

```bash
uv run stdm fetch --key multi_genotypes
uv run stdm samples
```

`multi_genotypes` is the sample-to-genotype table from SNP calls on the
WGBS reads (timeseries_molecular script 58, pinned to commit `f6e99789`).
It finds two problems with the colony labels:

- **Clonemates (*A. pulchra*).** ACR-145 (Manava) and ACR-265 (Mahana) are
  one genotype; ACR-150 (Manava) and ACR-237 (Mahana) are another. Ten
  colonies are eight genotypes.
- **Likely sample mix-ups (5 samples).** ACR-145-TP2 and ACR-150-TP2 carry
  each other's genotype (a TP2 swap); POR-245-TP3 and TP4 match POR-236;
  POC-57-TP4 matches POC-53.

`samples.csv` gets `genotype_id` (each colony's genotype, from its unflagged
samples) and `identity_flagged`. Every command that fits a tensor or tests
colonies takes `--identity`:

| Mode | Samples | What it does |
|---|---|---|
| `as-given` (default) | 117 | colony labels as recorded |
| `drop-flagged` | 112 | drops the five mismatched samples |
| `one-per-genotype` | 106 | also keeps one colony per genotype (ACR-265 and ACR-237, which have all four timepoints) |

The genotype-corrected rerun below uses `one-per-genotype` throughout. In
*P. evermanni* and *P. tuahiniensis* it is the same as `drop-flagged`.

### Genotype-corrected rerun

```bash
uv run stdm sparse-cp --tensor talk-centered --remove-batch --identity one-per-genotype \
    --top-variable 2000 --nonneg-modes --rank 2 --lambdas 0 0 0 --n-init 20 \
    --max-iter 5000 --tol 1e-7 --output results/geno-r2-nobatch
uv run stdm interpret --model results/geno-r2-nobatch --output results/phase4-geno
uv run stdm elastic-net --scores results/phase4-geno/sample_scores.csv --component Component_2 \
    --identity one-per-genotype --permutations 50 --output results/phase5-site-geno
uv run stdm cis-meth --model results/geno-r2-nobatch --identity one-per-genotype \
    --output results/phase5-cis-geno
uv run stdm cpg-check --cis results/phase5-cis-geno --identity one-per-genotype \
    --output results/phase5-cpg-geno
```

**Phase 4: unchanged.**

- **Site component:** gene loadings have cosine 0.96 with the original model.
  Its site difference holds with colonies as the unit (t-test on colony mean
  scores, Manava vs Mahana): *P. evermanni* p = 2e-4 and *P. tuahiniensis*
  p = 0.016, against 4e-4 and 0.008 before.
- ***A. pulchra* timepoint component:** cosine 0.83, with the same pattern
  (March and September up).
- **Physiology:** 7 of 66 tests at q < 0.1, and none within site x
  timepoint.

**Gene-level methylation vs expression:**

| | Apul before | Apul after | Peve before | Peve after | Ptua before | Ptua after |
|---|---|---|---|---|---|---|
| Samples | 40 | 32 | 37 | 35 | 32 | 31 |
| Between colonies: genes above null 99th pct (null median) | 463 (80) | 307 (78) | 245 (66) | 229 (67) | 137 (44) | 120 (44) |
| Between colonies: genes at q < 0.1 | 151 | 70 | 77 | 62 | 35 | 26 |
| Within colony: genes above null 99th pct (null median) | 291 (76) | 194 (71) | 59 (69) | 61 (67) | 61 (45) | 60 (45) |
| Within colony: genes at q < 0.1 | 12 | 0 | 1 | 0 | 0 | 0 |

- **Between-colony coupling survives in all three species** (permutation
  p = 0.005 each).
  - Per-gene correlations agree with the original run: r = 0.92 (Apul),
    0.98 (Peve), 0.99 (Ptua).
  - Most of the new hits were hits before: 61 of 70, 55 of 62 and all 26.
  - Fewer *A. pulchra* hits is expected with 8 colonies instead of 10.
- **No single *A. pulchra* gene is coupled within colony any more.** The 12
  genes from the original run depended on the swapped or cloned samples.
  The excess over the null is smaller but still there (194 vs 71,
  p = 0.025).

**CpG genotype check:**

| | Apul before | Apul after | Peve before | Peve after | Ptua before | Ptua after |
|---|---|---|---|---|---|---|
| Hits reproduced from CpG data | 68 | 43 | 16 | 14 | 26 | 19 |
| ...still \|r\| ≥ 0.5 without variant-like CpGs | 47 | 36 | 13 | 12 | 20 | 14 |
| ...still \|r\| ≥ 0.5 without the top CpG | 57 | 35 | 14 | 12 | 22 | 19 |
| Variant-like CpGs, hits vs controls | 18.5% vs 9.3% | 9.6% vs 8.5% | 9.9% vs 7.8% | 8.5% vs 7.5% | 4.8% vs 4.6% | 6.1% vs 6.7% |
| CpGs agreeing in sign, hits vs controls (median) | 0.74 vs 0.62 | 0.80 vs 0.65 (p < 0.001) | 0.85 vs 0.80 | 0.83 vs 0.78 | 0.68 vs 0.62 | 0.68 vs 0.63 |

- **The *A. pulchra* enrichment of variant-like CpGs is gone** (9.6% vs
  8.5%; genes with any such CpG 56% vs 45%, p = 0.16). The earlier
  twofold excess came from the clone pairs. With each clone pair counted
  twice, a variant shared by two colonies looked like a between-colony
  effect twice over.
- **Coupled genes still show a gene-wide shift.** Most hits survive
  dropping variant-like CpGs or the single most influential CpG. More of
  their CpGs agree in direction than in matched controls, clearly so in
  *A. pulchra*.

**Elastic net** (held-out R², permutation p in brackets where < 0.05;
`results/phase5-site-geno`):

| Predictors | Peve before | Peve after | Peve within site before | Peve within site after | Ptua before | Ptua after | Ptua within site before | Ptua within site after |
|---|---|---|---|---|---|---|---|---|
| Samples | 38 | 36 | 38 | 36 | 39 | 38 | 39 | 38 |
| lncRNA | 0.93 (0.02) | 0.91 (0.02) | 0.62 (0.02) | 0.62 (0.02) | 0.93 (0.02) | 0.93 (0.02) | 0.75 (0.02) | 0.78 (0.02) |
| Gene-body methylation | −0.10 | −0.15 | −0.10 | −0.03 | −0.07 | −0.22 | −1.19 | −0.04 |
| miRNA | −0.02 | 0.01 | 0.17 (0.04) | −0.02 | −0.36 | −0.29 | −0.16 | −0.02 |
| lncRNA + methylation | 0.91 (0.02) | 0.92 (0.02) | 0.49 (0.02) | 0.57 (0.02) | 0.96 (0.02) | 0.95 (0.02) | 0.65 (0.02) | 0.44 (0.04) |

Sample counts are for lncRNA; methylation and miRNA have a few fewer.
p = 0.02 is the smallest possible with 50 permutations.

- **lncRNA still predicts the site component**, raw and within site, at the
  same accuracy.
- **Gene-body methylation still predicts nothing** (all p ≥ 0.14).
- **The weak *P. evermanni* miRNA signal is gone** (within site R² −0.02,
  p = 0.24), so it should not be counted as a finding.

## Tests

```bash
uv run pytest
```

Tests that read the real data are skipped until `stdm fetch` has run.

## Layout

```
data/manifest.csv       MOSAiC file list (tracked)
data/checksums.csv      SHA-256 of fetched files (tracked)
data/samples.csv        sample metadata (tracked, rebuilt by `stdm samples`)
data/raw/               downloaded files (ignored)
data/processed/         built tensors (ignored)
src/stdm/mosaic.py      data access, ID harmonising, sample table
src/stdm/tensor.py      ortholog x colony x timepoint tensor
src/stdm/barnacle/      vendored Barnacle sparse CP (masked fitting), factor match score
src/stdm/selection.py   rank and sparsity selection
src/stdm/reproduce.py   the talk's rank-35 model and published factors
src/stdm/interpret.py   component scores vs physiology, ITS2 and gene sets
src/stdm/elasticnet.py  colony-held-out elastic net, permutation null, stability selection
src/stdm/cis.py         gene-level methylation vs expression with a colony permutation null
src/stdm/cpg.py         CpG-level genotype check of coupled genes
src/stdm/cli.py         `stdm` command
results/                run outputs (ignored)
src/stdm/decomposition.py, data_loader.py   synthetic-data scaffold
input-data/             synthetic test data and the VST ortholog matrix
tests/
```
