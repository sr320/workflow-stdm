"""Tests for the phase 5 elastic net (stdm.elasticnet)."""

import numpy as np
import pandas as pd
import pytest

from stdm.elasticnet import (ENetSpec, _poscounts_size_factors, colony_pair_folds,
                             cross_validate, residualize, stable_features, summarise)


def _design(seed=0, n_features=150, signal=2.0, site_effect=3.0):
    rng = np.random.default_rng(seed)
    rows = []
    for site in ("Mahana", "Manava"):
        for c in range(5):
            for tp in (1, 2, 3, 4):
                rows.append(dict(sample_id=f"{site[:3]}-{c}-TP{tp}", colony_id=f"{site[:3]}-{c}",
                                 site=site, timepoint=tp))
    meta = pd.DataFrame(rows).set_index("sample_id")
    X = pd.DataFrame(rng.normal(size=(len(meta), n_features)), index=meta.index,
                     columns=[f"f{i}" for i in range(n_features)])
    y = (signal * X["f0"] - signal * X["f1"] + site_effect * (meta.site == "Manava")
         + 0.5 * rng.normal(size=len(meta)))
    return X, y, meta


def test_colony_pair_folds_hold_out_one_colony_per_site():
    _, _, meta = _design()
    folds = colony_pair_folds(meta.colony_id, meta.site)
    assert len(folds) == 25
    for test in folds:
        held = meta[test]
        assert held.colony_id.nunique() == 2
        assert set(held.site) == {"Mahana", "Manava"}
        assert not set(held.colony_id) & set(meta[~test].colony_id)


def test_recovers_signal_and_stable_features():
    X, y, meta = _design()
    pred, sel, folds = cross_validate(X, y, meta, ENetSpec(n_alphas=15, seed=0))
    s = summarise(pred)
    assert s["r2"] > s["r2_site_baseline"] + 0.2
    stable = stable_features(sel)
    assert {"f0", "f1"} <= set(stable.feature)
    assert len(folds) == 25


def test_within_site_target_ignores_site_offset():
    X, y, meta = _design(signal=0.0, site_effect=5.0)
    pred, _, _ = cross_validate(X, y, meta, ENetSpec(n_alphas=10, target="within_site",
                                                     noise_fraction=0.0))
    # Pure site effect: nothing left to predict within site.
    assert summarise(pred)["r2"] < 0.1
    assert pred.observed.abs().mean() < 1.0


def test_residualize_removes_shared_axis_and_keeps_nans():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(30, 1))
    X = pd.DataFrame(2 * z + 0.1 * rng.normal(size=(30, 4)), columns=list("abcd"))
    X.iloc[0, 0] = np.nan
    R = residualize(X, z)
    assert np.isnan(R.iloc[0, 0])
    for c in "bcd":  # complete columns are exactly orthogonal to z
        assert abs(np.corrcoef(R[c], z[:, 0])[0, 1]) < 1e-6
    assert R.abs().stack().mean() < 0.2


def test_poscounts_handles_zeros():
    counts = pd.DataFrame({"a": [10, 0, 5], "b": [20, 4, 0], "c": [5, 2, 3]})
    sf = _poscounts_size_factors(counts)
    assert np.isfinite(sf).all()
    assert sf["b"] > sf["c"]
