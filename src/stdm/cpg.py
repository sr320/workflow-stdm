"""Phase 5c: is the between-colony methylation-expression coupling genotype?

Each colony is a genet, so a sequence difference can move both measurements
of a gene. The clearest case at the CpG level is a C/T (or G/A) mutation
that destroys a CpG: bisulfite sequencing then reads that site as
unmethylated in the carrying colony at every timepoint. A regulatory
change is expected to shift methylation across many CpGs of a gene.

For each gene this module:

- flags *variant-like* CpGs: some colony is near 0% at every timepoint
  while at least two other colonies are clearly methylated (homozygous
  CpG loss; heterozygous loss halves the level and is not flagged);
- recomputes the gene's between-colony methylation-expression correlation
  from its CpGs, then without the variant-like CpGs, and without the single
  most influential CpG;
- records the fraction of its CpGs whose own correlation has the gene's
  sign (a gene-wide shift gives a high fraction).

Significant genes are compared with control genes matched on mean and SD of
gene methylation.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import mosaic
from .cis import _fill_by_cell, design, residual_maker, row_corr


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_genes(species: str, data_dir=None) -> pd.DataFrame:
    """Gene coordinates from the species' GFF: gene_id, chrom, start, end."""
    entry = mosaic.load_manifest(data_dir)[f"{species}_gff"]
    gff = pd.read_csv(mosaic.raw_path(entry, data_dir), sep="\t", comment="#", header=None,
                      usecols=[0, 2, 3, 4, 8], names=["chrom", "type", "start", "end", "attr"],
                      dtype={"chrom": str})
    gff = gff[gff.type == "gene"]
    ids = gff.attr.str.extract(r"ID=([^;]+)")[0].map(mosaic.canonical_gene_id)
    return pd.DataFrame({"gene_id": ids.to_numpy(), "chrom": gff.chrom.to_numpy(),
                         "start": gff.start.to_numpy(int), "end": gff.end.to_numpy(int)})


def _split_cpg(ids: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """``CpG_<chrom>_<pos>`` -> chrom, pos (chrom may contain underscores)."""
    body = ids.str.slice(4)
    parts = body.str.rsplit("_", n=1, expand=True)
    return parts[0].to_numpy(), parts[1].astype(int).to_numpy()


def assign_cpgs(chrom: np.ndarray, pos: np.ndarray, genes: pd.DataFrame) -> np.ndarray:
    """Index into ``genes`` of the gene containing each CpG, or -1.

    Uses the gene with the nearest start at or before the CpG; a CpG in two
    overlapping genes is given to the later-starting one.
    """
    out = np.full(len(pos), -1)
    for c, g in genes.reset_index(drop=True).groupby("chrom"):
        sel = np.flatnonzero(chrom == c)
        if not len(sel):
            continue
        g = g.sort_values("start")
        k = np.searchsorted(g.start.to_numpy(), pos[sel], side="right") - 1
        ok = k >= 0
        inside = np.zeros(len(sel), bool)
        inside[ok] = pos[sel][ok] <= g.end.to_numpy()[k[ok]]
        out[sel[inside]] = g.index.to_numpy()[k[inside]]
    return out


def load_gene_cpgs(species: str, genes: pd.DataFrame, samples: Sequence[str],
                   chunksize: int = 200_000, data_dir=None) -> tuple[pd.DataFrame, pd.Series]:
    """CpG methylation (fraction) for CpGs inside ``genes``, CpGs x ``samples``,
    and the gene of each CpG. Reads the ~1 GB file in chunks. Samples
    missing from the CpG file are left out."""
    entry = mosaic.load_manifest(data_dir)[f"{species}_cpg"]
    path = mosaic.raw_path(entry, data_dir)
    genes = genes.reset_index(drop=True)
    header = pd.read_csv(path, nrows=0).columns
    canon = {c: mosaic.canonical_sample_id(c) for c in header[1:]}
    want = [c for c in header[1:] if canon[c] in set(samples)]
    vals, gene_of = [], []
    for chunk in pd.read_csv(path, usecols=["CpG", *want], chunksize=chunksize,
                             dtype={c: "float32" for c in want}):
        chrom, pos = _split_cpg(chunk.CpG)
        k = assign_cpgs(chrom, pos, genes)
        keep = k >= 0
        if keep.any():
            v = chunk.loc[keep].set_index("CpG")[want]
            vals.append(v)
            gene_of.append(pd.Series(genes.gene_id.to_numpy()[k[keep]], index=v.index))
    have = {canon[c] for c in want}
    V = pd.concat(vals).rename(columns=canon)[[s for s in samples if s in have]] / 100.0
    return V, pd.concat(gene_of)


# ---------------------------------------------------------------------------
# Per-gene statistics
# ---------------------------------------------------------------------------

def variant_like(V: pd.DataFrame, colony: pd.Series, low: float = 0.05,
                 high: float = 0.4, min_samples: int = 2) -> pd.Series:
    """CpGs where a colony is at or below ``low`` in all its (>= ``min_samples``)
    samples while at least two other colonies average at least ``high``."""
    col = colony.loc[V.columns].to_numpy()
    groups = {c: np.flatnonzero(col == c) for c in pd.unique(col)}
    X = V.to_numpy(float)
    cmax, cmean, cn = [], [], []
    for idx in groups.values():
        sub = X[:, idx]
        n = np.sum(~np.isnan(sub), axis=1)
        with np.errstate(all="ignore"):
            cmax.append(np.where(n > 0, np.nanmax(np.where(np.isnan(sub), -np.inf, sub), axis=1), np.nan))
            cmean.append(np.nanmean(sub, axis=1))
        cn.append(n)
    cmax, cmean, cn = np.array(cmax), np.array(cmean), np.array(cn)  # colonies x CpGs
    is_low = (cn >= min_samples) & (cmax <= low)
    is_high = (cn >= min_samples) & (cmean >= high)
    flag = is_low.any(axis=0) & (is_high.sum(axis=0) >= 2)
    return pd.Series(flag, index=V.index)


def _corr_with(expr_res: np.ndarray, M: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Correlation of one residualised expression rank vector with each row of
    M (ranked, then residualised)."""
    Mr = stats.rankdata(M, axis=1) @ R
    return row_corr(np.broadcast_to(expr_res, Mr.shape), Mr)


def gene_stats(V: pd.DataFrame, gene_of: pd.Series, expr: pd.DataFrame, meta: pd.DataFrame,
               genes: Sequence[str], level: str = "between_colony",
               min_present: float = 0.8) -> pd.DataFrame:
    """Per-gene CpG-level checks of the methylation-expression correlation."""
    meta = meta.loc[V.columns]
    R = residual_maker(design(meta, level))
    flags = variant_like(V, meta.colony_id)
    present = V.notna().mean(axis=1) >= min_present
    rows = []
    for gene in genes:
        cp = gene_of.index[(gene_of == gene).to_numpy()]
        cp = [c for c in cp if present.get(c, False)]
        if gene not in expr.index or len(cp) == 0:
            rows.append(dict(gene=gene, n_cpg=len(cp)))
            continue
        e = stats.rankdata(expr.loc[gene, V.columns].to_numpy(float)) @ R
        sub = _fill_by_cell(V.loc[cp], meta)
        X = sub.to_numpy(float)
        fl = flags.loc[cp].to_numpy()

        def gene_r(mask):
            if not mask.any():
                return np.nan
            return float(_corr_with(e, X[mask].mean(axis=0, keepdims=True), R)[0])

        allm = np.ones(len(cp), bool)
        r_all = gene_r(allm)
        r_drop_var = gene_r(~fl) if fl.any() else r_all
        loo = np.array([gene_r(np.arange(len(cp)) != i) for i in range(len(cp))]) \
            if len(cp) > 1 else np.array([np.nan])
        # The CpG whose removal moves r furthest toward zero.
        r_drop_top = float(loo[np.nanargmin(np.abs(loo))]) if np.isfinite(loo).any() else np.nan
        varies = X.std(axis=1) > 0
        per = _corr_with(e, X[varies], R) if varies.any() else np.array([])
        per = per[np.isfinite(per)]
        same = float(np.mean(np.sign(per) == np.sign(r_all))) if len(per) and np.isfinite(r_all) else np.nan
        rows.append(dict(gene=gene, n_cpg=len(cp), n_cpg_varying=int(varies.sum()),
                         n_variant_like=int(fl.sum()), r_cpg=r_all, r_drop_variant=r_drop_var,
                         r_drop_top=r_drop_top, frac_same_sign=same))
    return pd.DataFrame(rows).set_index("gene")


def matched_controls(cis_genes: pd.DataFrame, hits: Sequence[str], level: str = "between_colony",
                     per_hit: int = 3, min_q: float = 0.5, seed: int = 0) -> list[str]:
    """Non-significant genes (q >= ``min_q``) nearest each hit in standardised
    (meth_mean, meth_sd), ``per_hit`` per hit, without replacement."""
    pool = cis_genes[(cis_genes[f"q_{level}"] >= min_q)].copy()
    feats = ["meth_mean", "meth_sd"]
    mu, sd = cis_genes[feats].mean(), cis_genes[feats].std()
    P = ((pool[feats] - mu) / sd).to_numpy()
    H = ((cis_genes.loc[list(hits), feats] - mu) / sd).to_numpy()
    rng = np.random.default_rng(seed)
    used = np.zeros(len(pool), bool)
    out = []
    for i in rng.permutation(len(H)):
        d = np.sum((P - H[i]) ** 2, axis=1)
        d[used] = np.inf
        pick = np.argsort(d)[:per_hit]
        used[pick] = True
        out.extend(pool.index[pick])
    return out
