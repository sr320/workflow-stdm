"""Tests for the CpG-level genotype check (stdm.cpg)."""

import numpy as np
import pandas as pd

from stdm.cpg import _split_cpg, assign_cpgs, gene_stats, matched_controls, variant_like


def _meta():
    rows = []
    for site in ("Mahana", "Manava"):
        for c in range(5):
            for tp in (1, 2, 3, 4):
                rows.append(dict(sample_id=f"{site[:3]}-{c}-TP{tp}", colony_id=f"{site[:3]}-{c}",
                                 site=site, timepoint=tp))
    return pd.DataFrame(rows).set_index("sample_id")


def test_split_and_assign_cpgs():
    chrom, pos = _split_cpg(pd.Series(["CpG_scaf_1_150", "CpG_scaf_1_900", "CpG_x_10"]))
    assert list(chrom) == ["scaf_1", "scaf_1", "x"] and list(pos) == [150, 900, 10]
    genes = pd.DataFrame({"gene_id": ["a", "b"], "chrom": ["scaf_1", "scaf_1"],
                          "start": [100, 500], "end": [200, 800]})
    assert list(assign_cpgs(chrom, pos, genes)) == [0, -1, -1]


def test_variant_like_flags_colony_specific_loss_only():
    meta = _meta()
    rng = np.random.default_rng(0)
    base = 0.7 + 0.05 * rng.normal(size=(3, len(meta)))
    V = pd.DataFrame(base, index=["snp", "graded", "flat"], columns=meta.index)
    V.loc["snp", meta.colony_id == "Mah-0"] = 0.0
    V.loc["graded"] = np.where(meta.colony_id.str.endswith("0"), 0.35, 0.7)
    flags = variant_like(V, meta.colony_id)
    assert flags["snp"] and not flags["graded"] and not flags["flat"]


def _gene_data(kind, seed=0):
    """One gene, 10 CpGs. 'variant': one CpG drives the colony coupling;
    'shift': all CpGs shift with the colony effect."""
    meta = _meta()
    rng = np.random.default_rng(seed)
    cols = meta.colony_id.unique()
    eff = dict(zip(cols, rng.normal(size=len(cols))))
    ce = meta.colony_id.map(eff).to_numpy()
    expr = pd.DataFrame([ce + 0.3 * rng.normal(size=len(meta))], index=["g"], columns=meta.index)
    V = 0.5 + 0.02 * rng.normal(size=(10, len(meta)))
    if kind == "shift":
        V += 0.08 * ce
    else:
        low = ce < np.quantile(list(eff.values()), 0.3)
        V[0] = np.where(low, 0.0, 0.8)
    V = pd.DataFrame(np.clip(V, 0, 1), index=[f"c{i}" for i in range(10)], columns=meta.index)
    gene_of = pd.Series("g", index=V.index)
    return V, gene_of, expr, meta


def test_gene_stats_separates_single_cpg_from_gene_wide_shift():
    V, gene_of, expr, meta = _gene_data("variant")
    s = gene_stats(V, gene_of, expr, meta, ["g"]).loc["g"]
    assert s.n_variant_like >= 1
    assert abs(s.r_cpg) > 0.5
    assert abs(s.r_drop_variant) < 0.3 and abs(s.r_drop_top) < 0.3

    V, gene_of, expr, meta = _gene_data("shift")
    s = gene_stats(V, gene_of, expr, meta, ["g"]).loc["g"]
    assert s.n_variant_like == 0
    assert abs(s.r_drop_top) > 0.5
    assert s.frac_same_sign > 0.8


def test_matched_controls_are_close_and_unique():
    rng = np.random.default_rng(0)
    g = pd.DataFrame({"meth_mean": rng.uniform(size=300), "meth_sd": rng.uniform(size=300),
                      "q_between_colony": np.r_[np.zeros(5), np.ones(295)]},
                     index=[f"g{i}" for i in range(300)])
    ctrl = matched_controls(g, [f"g{i}" for i in range(5)], per_hit=3)
    assert len(ctrl) == len(set(ctrl)) == 15
    assert not set(ctrl) & {f"g{i}" for i in range(5)}
