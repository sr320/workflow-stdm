"""Tests for phase 4 association helpers (stdm.interpret)."""

import numpy as np
import pandas as pd
import pytest

from stdm.interpret import _bh, associate, gene_set_test


def test_bh_matches_reference():
    p = np.array([0.01, 0.04, 0.03, 0.2])
    # reference: statsmodels multipletests(p, method="fdr_bh")
    np.testing.assert_allclose(_bh(p), [0.04, 0.16 / 3, 0.16 / 3, 0.2])


def _scores(rng, n_per=6):
    rows = []
    for site in ("x", "y"):
        for tp in (1, 2, 3, 4):
            for i in range(n_per):
                rows.append(dict(sample_id=f"{site}{tp}{i}", species="a", site=site, timepoint=tp))
    return pd.DataFrame(rows)


def test_associate_separates_shared_season_from_within_timepoint_signal():
    rng = np.random.default_rng(0)
    s = _scores(rng)
    season = s.timepoint.isin([2, 3]).astype(float) * 5
    noise = rng.normal(size=len(s))
    s["comp"] = season + noise
    traits = pd.DataFrame(index=s.sample_id)
    traits["seasonal"] = (season + rng.normal(size=len(s))).to_numpy()   # shares season only
    traits["linked"] = (season + noise + 0.1 * rng.normal(size=len(s))).to_numpy()  # tracks within-tp too
    out = associate(s, traits, ["comp"]).set_index("trait")
    assert out.loc["seasonal", "p"] < 1e-4
    assert out.loc["seasonal", "p_within_site_tp"] > 0.01
    assert out.loc["linked", "rho_within_site_tp"] > 0.9


def test_gene_set_test_detects_enriched_side():
    idx = [f"g{i}" for i in range(400)]
    g = pd.DataFrame({"c": np.linspace(1, -1, 400)}, index=idx)
    members = set(idx[:30])  # all at the top of the + side
    out = gene_set_test(g, members, top=50).set_index("side")
    assert out.loc["+", "in_top"] == 30
    assert out.loc["+", "p_fisher"] < 1e-10
    assert out.loc["-", "in_top"] == 0
    assert out.loc["rank-sum", "p_ranksum"] < 1e-10
