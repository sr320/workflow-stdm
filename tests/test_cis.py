"""Tests for gene-level methylation-expression correlation (stdm.cis)."""

import numpy as np
import pandas as pd

from stdm.cis import cis_correlations, colony_permutation, compare_sets, summarise


def _meta():
    rows = []
    for site in ("Mahana", "Manava"):
        for c in range(5):
            for tp in (1, 2, 3, 4):
                rows.append(dict(sample_id=f"{site[:3]}-{c}-TP{tp}", colony_id=f"{site[:3]}-{c}",
                                 site=site, timepoint=tp))
    return pd.DataFrame(rows).set_index("sample_id")


def _data(seed=0, n_genes=300, n_signal=30, effect=1.5):
    """Methylation with colony effects; the first ``n_signal`` genes' expression
    follows the colony part of methylation. All genes share site and season."""
    rng = np.random.default_rng(seed)
    meta = _meta()
    cols = meta.colony_id.unique()
    site = (meta.site == "Manava").to_numpy(float)
    season = meta.timepoint.isin([2, 3]).to_numpy(float)
    colony_m = {c: rng.normal(size=n_genes) for c in cols}
    Mc = np.stack([colony_m[c] for c in meta.colony_id], axis=1)  # genes x samples
    M = Mc + 0.5 * rng.normal(size=Mc.shape) + site + season
    E = rng.normal(size=Mc.shape) + 2 * site + 2 * season
    E[:n_signal] += effect * Mc[:n_signal]
    idx = [f"g{i}" for i in range(n_genes)]
    return (pd.DataFrame(E, index=idx, columns=meta.index),
            pd.DataFrame(0.5 + 0.05 * M, index=idx, columns=meta.index), meta)


def test_colony_permutation_keeps_site_and_timepoint():
    meta = _meta()
    idx = colony_permutation(meta, np.random.default_rng(3))
    src = meta.iloc[idx]
    assert (src.site.to_numpy() == meta.site.to_numpy()).all()
    assert (src.timepoint.to_numpy() == meta.timepoint.to_numpy()).all()
    assert sorted(idx) == list(range(len(meta)))


def test_detects_planted_colony_coupling_only_in_signal_genes():
    E, M, meta = _data()
    M.iloc[5, 3] = np.nan  # missing values are filled, not fatal
    genes, null = cis_correlations(E, M, meta, permutations=50, seed=1)
    hits = set(genes.index[genes.q_between_colony < 0.1])
    signal = {f"g{i}" for i in range(30)}
    assert len(hits & signal) >= 20
    assert len(hits - signal) <= 3
    s = {r["level"]: r for r in summarise(genes, null)}
    assert s["between_colony"]["p_exceed"] < 0.05
    # Colony effects are constant over time: nothing left within colony.
    assert s["within_colony"]["n_q10"] <= 3


def test_null_data_gives_no_hits():
    E, M, meta = _data(seed=2, n_signal=0)
    genes, null = cis_correlations(E, M, meta, permutations=50, seed=1)
    for level in ("between_colony", "within_colony"):
        assert (genes[f"q_{level}"] < 0.1).sum() <= 3


def test_compare_sets_flags_signal_set():
    E, M, meta = _data()
    genes, _ = cis_correlations(E, M, meta, levels=("between_colony",), permutations=20)
    sets = {"signal": {f"g{i}" for i in range(30)}}
    out = compare_sets(genes, sets, set(genes.index), levels=("between_colony",))
    row = out.iloc[0]
    assert row.mean_abs_r > row.background_mean_abs_r
    assert row.p_abs < 1e-4
