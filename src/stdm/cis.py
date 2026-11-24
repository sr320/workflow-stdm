"""Phase 5b: does a gene's own body methylation track its expression?

For every gene with both measurements, ranks of expression and of gene-body
methylation are correlated across samples at three levels:

- ``all``: all samples (site, season and colony differences all count).
- ``between_colony``: after removing site x timepoint means, so the
  correlation comes from colonies differing at the same place and time.
- ``within_colony``: after also removing colony means, so it comes from a
  colony's methylation and expression changing together over time beyond
  the shared seasonal pattern.

Samples from one colony are not independent, so analytic p-values would be
too small. The null instead reassigns methylation profiles between colonies
within a site (matching timepoints) and recomputes every gene's correlation.
Null correlations are pooled across genes, which gives fine-grained p-values
from a modest number of permutations; BH q-values follow.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import mosaic
from .elasticnet import _log_counts
from .interpret import _bh

LEVELS = ("all", "between_colony", "within_colony")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_pair(species: str, min_count: float = 10, min_fraction: float = 0.5,
              max_missing: float = 0.2, min_meth_sd: float = 0.02, data_dir=None):
    """Expression (log2 normalised counts) and methylation (fraction), genes x
    samples, on shared genes and samples, plus sample metadata.

    Genes need expression of at least ``min_count`` in ``min_fraction`` of
    samples, at most ``max_missing`` missing methylation values and a
    methylation SD of at least ``min_meth_sd`` (fraction scale).
    """
    expr, _ = mosaic.load_matrix(f"{species}_gene", data_dir)
    meth, _ = mosaic.load_matrix(f"{species}_gene_meth", data_dir)
    samples = mosaic.sort_samples(set(expr.columns) & set(meth.columns))
    expr = _log_counts(expr[samples].astype(float), min_count, min_fraction)
    samples = [s for s in samples if s in expr.columns]  # failed libraries dropped
    meth = meth[samples].astype(float) / 100.0
    meth = meth[(meth.isna().mean(axis=1) <= max_missing) & (meth.std(axis=1) >= min_meth_sd)]
    genes = expr.index.intersection(meth.index)
    base = mosaic.default_data_dir() if data_dir is None else data_dir
    meta = pd.read_csv(f"{base}/samples.csv").set_index("sample_id").loc[samples,
                                                                         ["colony_id", "site", "timepoint"]]
    return expr.loc[genes, samples], meth.loc[genes, samples], meta


def _fill_by_cell(M: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Fill missing methylation with the gene's site x timepoint mean, then gene mean."""
    cell = meta.site + "-" + meta.timepoint.astype(str)
    filled = M.T.groupby(cell.to_numpy()).transform("mean").T
    M = M.fillna(filled)
    return M.T.fillna(M.mean(axis=1)).T


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def design(meta: pd.DataFrame, level: str) -> np.ndarray:
    """Nuisance design (samples x p) whose span is removed before correlating."""
    one = np.ones((len(meta), 1))
    if level == "all":
        return one
    cell = pd.get_dummies(meta.site + "-" + meta.timepoint.astype(str)).to_numpy(float)
    if level == "between_colony":
        return cell
    if level == "within_colony":
        col = pd.get_dummies(meta.colony_id).to_numpy(float)
        return np.hstack([cell, col])
    raise ValueError(level)


def residual_maker(D: np.ndarray) -> np.ndarray:
    """I - H for the column space of ``D`` (handles rank deficiency)."""
    return np.eye(len(D)) - D @ np.linalg.pinv(D)


def row_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson correlation of matching rows of A and B (rows already centred)."""
    num = np.sum(A * B, axis=1)
    den = np.sqrt(np.sum(A * A, axis=1) * np.sum(B * B, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def colony_permutation(meta: pd.DataFrame, rng) -> np.ndarray:
    """Sample index giving each sample the methylation of a sample from another
    colony in the same site at the same timepoint (colony labels shuffled
    within site; a sample keeps its own value where the timepoint is missing)."""
    pos = {s: i for i, s in enumerate(meta.index)}
    by = {(c, t): s for s, c, t in zip(meta.index, meta.colony_id, meta.timepoint)}
    col_site = meta.groupby("colony_id")["site"].first()
    idx = np.arange(len(meta))
    for site in col_site.unique():
        cols = list(col_site[col_site == site].index)
        for dst, src in zip(cols, rng.permutation(cols)):
            for s in meta.index[meta.colony_id == dst]:
                t = meta.loc[s, "timepoint"]
                if (src, t) in by:
                    idx[pos[s]] = pos[by[(src, t)]]
    return idx


def cis_correlations(E: pd.DataFrame, M: pd.DataFrame, meta: pd.DataFrame,
                     levels: Sequence[str] = LEVELS, permutations: int = 200,
                     seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-gene rank correlations with a pooled colony-permutation null.

    Returns ``(genes, null_summary)``. ``genes`` has, per level, ``r_<level>``,
    ``p_<level>`` and ``q_<level>``. ``null_summary`` has one row per level
    and permutation with the number of genes whose |r| exceeds the observed
    99th-percentile null cut-off, for a global test.
    """
    M = _fill_by_cell(M, meta)
    Er = stats.rankdata(E.to_numpy(float), axis=1)
    Mr = stats.rankdata(M.to_numpy(float), axis=1)
    rng = np.random.default_rng(seed)
    perms = [colony_permutation(meta, rng) for _ in range(permutations)]
    out = pd.DataFrame({"meth_mean": M.mean(axis=1), "meth_sd": M.std(axis=1),
                        "expr_mean": E.mean(axis=1)}, index=E.index)
    rows = []
    for level in levels:
        R = residual_maker(design(meta, level))
        Ea = Er @ R
        r = row_corr(Ea, Mr @ R)
        null = np.stack([row_corr(Ea, Mr[:, p] @ R) for p in perms])  # perms x genes
        flat = np.sort(np.abs(null[np.isfinite(null)]))
        exceed = flat.size - np.searchsorted(flat, np.abs(r), side="left")
        p = (1 + exceed) / (1 + flat.size)
        p = np.where(np.isfinite(r), p, np.nan)
        out[f"r_{level}"] = r
        out[f"p_{level}"] = p
        ok = np.isfinite(p)
        q = np.full(len(p), np.nan)
        q[ok] = _bh(p[ok])
        out[f"q_{level}"] = q
        cut = np.quantile(flat, 0.99) if flat.size else np.nan
        obs = int(np.sum(np.abs(r) > cut))
        for i, nr in enumerate(null):
            rows.append(dict(level=level, permutation=i, n_exceed=int(np.sum(np.abs(nr) > cut)),
                             mean_r=float(np.nanmean(nr))))
        rows.append(dict(level=level, permutation=-1, n_exceed=obs, mean_r=float(np.nanmean(r))))
    return out, pd.DataFrame(rows)


def summarise(genes: pd.DataFrame, null: pd.DataFrame, levels: Sequence[str] = LEVELS) -> list[dict]:
    """Per level: genes tested, mean r (observed and null), genes at q < 0.1,
    and a global p for the excess of |r| above the null 99th percentile."""
    rows = []
    for level in levels:
        n = null[null.level == level]
        obs = n[n.permutation == -1].iloc[0]
        perm = n[n.permutation >= 0]
        rows.append(dict(
            level=level, n_genes=int(genes[f"r_{level}"].notna().sum()),
            mean_r=float(obs.mean_r), null_mean_r=float(perm.mean_r.mean()),
            null_mean_r_95=float(perm.mean_r.quantile(0.95)),
            p_mean_r=float((1 + (perm.mean_r.abs() >= abs(obs.mean_r)).sum()) / (1 + len(perm))),
            n_exceed_99=int(obs.n_exceed), null_exceed_median=float(perm.n_exceed.median()),
            p_exceed=float((1 + (perm.n_exceed >= obs.n_exceed).sum()) / (1 + len(perm))),
            n_q10=int((genes[f"q_{level}"] < 0.1).sum()),
        ))
    return rows


# ---------------------------------------------------------------------------
# Component genes
# ---------------------------------------------------------------------------

def component_genes(gene_factors: pd.DataFrame, species: str, top: int = 200,
                    orthologs: Optional[pd.DataFrame] = None) -> dict[str, set[str]]:
    """Species gene IDs of the top ``top`` loadings on each side of each component."""
    o = mosaic.load_orthologs() if orthologs is None else orthologs
    gid = o.set_index("group_id")[species]
    out = {}
    for comp in gene_factors.columns:
        s = gene_factors[comp].sort_values()
        for side, groups in (("neg", s.index[:top]), ("pos", s.index[::-1][:top])):
            out[f"{comp}_{side}"] = set(gid.reindex(groups).dropna())
    return out


def compare_sets(genes: pd.DataFrame, sets: dict[str, set[str]], background: set[str],
                 levels: Sequence[str] = LEVELS) -> pd.DataFrame:
    """|r| of each gene set against the other genes in ``background``
    (Mann-Whitney), and the fraction at q < 0.1."""
    rows = []
    g = genes[genes.index.isin(background)]
    for name, members in sets.items():
        inset = g.index.isin(members)
        for level in levels:
            a, b = g.loc[inset, f"r_{level}"].dropna(), g.loc[~inset, f"r_{level}"].dropna()
            if len(a) < 5:
                continue
            p = stats.mannwhitneyu(a.abs(), b.abs(), alternative="two-sided").pvalue
            rows.append(dict(set=name, level=level, n=len(a), mean_abs_r=a.abs().mean(),
                             background_mean_abs_r=b.abs().mean(), mean_r=a.mean(),
                             p_abs=p,
                             frac_q10=float((g.loc[inset, f"q_{level}"] < 0.1).mean()),
                             background_frac_q10=float((g.loc[~inset, f"q_{level}"] < 0.1).mean())))
    return pd.DataFrame(rows)
