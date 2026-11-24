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
| 4-6. Interpretation, elastic net, extensions | not started |

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
`has_<type>` flag per data type.

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

**GO slim enrichment** (`goslim_enrichment.csv`; top 200 loadings of the 737
annotated groups against the rest, Fisher test, BH q):

- Component 1, Mahana side: DNA binding (q = 0.02), DNA repair (q = 0.03).
- Component 1, Manava side: nothing at q < 0.1. The top terms are
  development, mitochondrion, plasma membrane and cell motility.
- Component 2, positive side: nucleolus, RNA binding, nucleoplasm and mRNA
  metabolic process (q < 0.001). This ribosome-biogenesis signature is also
  typical of RNA-quality differences, consistent with the batch
  confounding.

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
src/stdm/cli.py         `stdm` command
results/                run outputs (ignored)
src/stdm/decomposition.py, data_loader.py   synthetic-data scaffold
input-data/             synthetic test data and the VST ortholog matrix
tests/
```
