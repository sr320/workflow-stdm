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
| 3. Barnacle sparse CP with rank and sparsity selection | not started |
| 4-6. Interpretation, elastic net, extensions | not started |

The `generate` / `decompose` / `run` / `info` commands and
`SparseTensorDecomposer` are the original synthetic-data scaffold: plain
tensorly PARAFAC/Tucker on a genes x species x timepoints tensor. They are not
yet the method from the talk and should not be pointed at real data; phases 2
and 3 replace them.

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
src/stdm/cli.py         `stdm` command
src/stdm/decomposition.py, data_loader.py   synthetic-data scaffold
input-data/             synthetic test data and the VST ortholog matrix
tests/
```
