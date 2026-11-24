"""Relate CP component scores to physiology, symbionts and gene sets (phase 4)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import mosaic

# Physiology variables tested by default: one per measured trait, normalised
# to surface area where the table offers a choice.
PHYSIOLOGY_VARS = [
    "Host_AFDW.mg.cm2", "Sym_AFDW.mg.cm2", "prot_mg.cm2", "cre.umol.cm2",
    "cells.cm2", "chla.ug.cm2", "Total_Chl_cell", "Am", "AQY", "Rd",
    "calc.umol.cm2.hr",
]


def sample_scores(tensor, gene_factors: pd.DataFrame) -> pd.DataFrame:
    """Project each observed sample onto unit-norm gene loadings.

    Returns one row per sample with ``sample_id``, colony, species, site,
    timepoint and one column per component.
    """
    A = gene_factors.loc[tensor.groups.index].to_numpy(dtype=float)
    A = A / np.linalg.norm(A, axis=0)
    rows = []
    for j, colony in enumerate(tensor.colonies.index):
        for k, tp in enumerate(tensor.timepoints):
            x = tensor.values[:, j, k]
            if np.isnan(x).any():
                continue
            rows.append(dict(sample_id=f"{colony}-TP{tp}", colony_id=colony,
                             species=tensor.colonies["species"].iloc[j],
                             site=tensor.colonies["site"].iloc[j], timepoint=int(tp),
                             **dict(zip(gene_factors.columns, x @ A))))
    return pd.DataFrame(rows)


def load_physiology(data_dir=None) -> pd.DataFrame:
    """Physiology indexed by canonical sample ID (``ACR-139-TP1``)."""
    p = mosaic.load_table("multi_physiology", data_dir)
    p = p.assign(sample_id=(p["colony_id_corr"] + "-TP"
                            + p["timepoint"].str.extract(r"(\d)$")[0]).map(mosaic.canonical_sample_id))
    p = p.dropna(subset=["sample_id"]).drop_duplicates("sample_id")
    return p.set_index("sample_id")


def load_its2(data_dir=None, min_mean: float = 0.02) -> pd.DataFrame:
    """ITS2 relative abundance of profiles with mean abundance >= ``min_mean``."""
    t = mosaic.load_table("multi_its2", data_dir)
    t = t.assign(sample_id=t["sample_id"].map(mosaic.canonical_sample_id)).dropna(subset=["sample_id"])
    t = t.drop_duplicates("sample_id").set_index("sample_id")
    num = t.select_dtypes("number")
    num = num.div(num.sum(axis=1).replace(0, np.nan), axis=0)
    return num.loc[:, num.mean() >= min_mean]


def _resid_rank(y, site):
    r = pd.Series(stats.rankdata(y), index=y.index)
    return r - r.groupby(site).transform("mean")


def associate(scores: pd.DataFrame, traits: pd.DataFrame, components: Sequence[str],
              min_n: int = 12) -> pd.DataFrame:
    """Spearman correlations of component scores with traits, per species.

    ``rho`` uses all samples; ``rho_within_site`` correlates ranks after
    removing each site's mean rank, so a shared site difference alone does
    not produce an association; ``rho_within_site_tp`` removes each
    site x timepoint mean, so shared seasonal trends do not either. BH
    q-values (``q``, ``q_within_site``, ``q_within_site_tp``) are computed
    over all tests of each kind.
    """
    df = scores.join(traits, on="sample_id", how="inner")
    rows = []
    for sp, x in df.groupby("species"):
        for comp in components:
            for trait in traits.columns:
                ok = x[[comp, trait]].notna().all(axis=1) & np.isfinite(x[trait].astype(float))
                y = x.loc[ok]
                if len(y) < min_n or y[trait].nunique() < 3:
                    continue
                rho, p = stats.spearmanr(y[comp], y[trait])
                rw, pw = stats.pearsonr(_resid_rank(y[comp], y.site), _resid_rank(y[trait].astype(float), y.site))
                cell = y.site + "-" + y.timepoint.astype(str)
                rc, pc = stats.pearsonr(_resid_rank(y[comp], cell), _resid_rank(y[trait].astype(float), cell))
                rows.append(dict(species=sp, component=comp, trait=trait, n=len(y),
                                 rho=rho, p=p, rho_within_site=rw, p_within_site=pw,
                                 rho_within_site_tp=rc, p_within_site_tp=pc))
    out = pd.DataFrame(rows)
    for col in ("p", "p_within_site", "p_within_site_tp"):
        if len(out):
            out[col.replace("p", "q", 1)] = _bh(out[col].to_numpy())
    return out


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def calcification_groups(data_dir=None) -> set[str]:
    """Ortholog group IDs in any species' calcification (biomin) gene set."""
    manifest = mosaic.load_manifest(data_dir)
    groups: set[str] = set()
    for sp in mosaic.SPECIES["species"]:
        path = mosaic.raw_path(manifest[f"{sp}_biomin"], data_dir)
        groups |= set(pd.read_csv(path, usecols=["group_id"])["group_id"].dropna())
    return groups


def gene_set_test(gene_factors: pd.DataFrame, members: set[str], top: int = 200) -> pd.DataFrame:
    """For each component and side, overlap of the ``top`` loadings with a set
    (Fisher test) and a rank-sum test of set members' loadings against the rest.
    """
    inset = gene_factors.index.isin(list(members))
    rows = []
    for comp in gene_factors.columns:
        v = gene_factors[comp]
        for side, sel in [("+", v.rank(ascending=False) <= top), ("-", v.rank() <= top)]:
            a = int((sel & inset).sum()); b = int((sel & ~inset).sum())
            c = int((~sel & inset).sum()); d = int((~sel & ~inset).sum())
            odds, p = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
            rows.append(dict(component=comp, side=side, in_top=a, set_size=int(inset.sum()),
                             expected=round(top * inset.mean(), 1), odds=odds, p_fisher=p))
        u = stats.mannwhitneyu(v[inset], v[~inset]) if inset.any() else None
        rows.append(dict(component=comp, side="rank-sum", set_size=int(inset.sum()),
                         median_in_set=float(v[inset].median()), median_rest=float(v[~inset].median()),
                         p_ranksum=u.pvalue if u else np.nan))
    return pd.DataFrame(rows)
