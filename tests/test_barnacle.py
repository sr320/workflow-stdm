"""Tests for the vendored sparse CP solver, FMS, and rank/sparsity selection."""

import numpy as np
import pytest

from stdm.barnacle import SparseCP, als_lasso, factor_match_score
from stdm.selection import (colony_halves, one_se_lambda, one_se_rank, sample_holdout_masks,
                            select_lambda, select_rank)
import pandas as pd


def planted(I=120, J=12, K=4, R=2, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    A = np.zeros((I, R))
    for r in range(R):
        A[rng.choice(I, I // 4, replace=False), r] = rng.uniform(1, 2, I // 4)
    B = rng.normal(size=(J, R))
    C = rng.normal(size=(K, R)) + 2
    Y = np.einsum("ir,jr,kr->ijk", A, B, C)
    return Y + noise * rng.normal(size=Y.shape), (A, B, C)


def test_recovers_planted_components():
    X, truth = planted()
    cp = SparseCP(2, [0.05, 0, 0], nonneg_modes=[0], n_initializations=2,
                  random_state=0, tol=1e-6, n_iter_max=300).fit_transform(X)
    assert factor_match_score(cp.factors, truth, absolute=True)[0] > 0.95
    assert (cp.factors[0] >= 0).all()


def test_l1_penalty_creates_zeros():
    X, _ = planted()
    dense, _ = als_lasso(X, 2, [0, 0, 0], random_state=0, n_iter_max=300)
    sparse, _ = als_lasso(X, 2, [5.0, 0, 0], nonneg_modes=[0], random_state=0, n_iter_max=300)
    assert (sparse.factors[0] == 0).mean() > (dense.factors[0] == 0).mean() + 0.3


def test_masked_fit_ignores_missing_values():
    X, truth = planted(seed=1)
    rng = np.random.default_rng(1)
    mask = rng.random(X.shape) > 0.2
    garbage = X.copy()
    garbage[~mask] = 1e6  # must have no influence
    cp, losses = als_lasso(garbage, 2, [0.01, 0, 0], nonneg_modes=[0], random_state=0,
                           n_iter_max=2000, tol=1e-9, mask=mask)
    recon = np.einsum("ir,jr,kr->ijk", *cp.factors)
    rel = np.linalg.norm((recon - X)[~mask]) / np.linalg.norm(X[~mask])
    assert rel < 0.05
    assert losses[-1] <= losses[0]


def test_nan_without_mask_is_an_error():
    X, _ = planted()
    X[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="mask"):
        als_lasso(X, 2, [0, 0, 0], n_iter_max=5)


def test_unmasked_matches_full_mask():
    X, _ = planted()
    a, la = als_lasso(X, 2, [0.1, 0, 0], random_state=3, n_iter_max=50)
    b, lb = als_lasso(X, 2, [0.1, 0, 0], random_state=3, n_iter_max=50,
                      mask=np.ones(X.shape, bool))
    for fa, fb in zip(a.factors, b.factors):
        np.testing.assert_allclose(fa, fb)


def test_fms_identity_permutation_and_zero_components():
    rng = np.random.default_rng(0)
    f = [rng.random((10, 3)), rng.random((5, 3)), rng.random((4, 3))]
    perm = [x[:, [2, 0, 1]] for x in f]
    score, per, match = factor_match_score(f, perm)
    assert score == pytest.approx(1.0)
    assert list(match) == [1, 2, 0]

    zeroed = [x.copy() for x in f]
    zeroed[0][:, 1] = 0
    score, per, _ = factor_match_score(f, zeroed)
    assert per[1] == 0
    assert score == pytest.approx(2 / 3)


def test_fms_modes_allow_different_colony_counts():
    rng = np.random.default_rng(0)
    a = [rng.random((10, 2)), rng.random((6, 2)), rng.random((4, 2))]
    b = [a[0] * 2, rng.random((5, 2)), a[2]]
    score, _, _ = factor_match_score(a, b, modes=(0, 2), consider_weights=False)
    assert score == pytest.approx(1.0)


def test_sample_holdouts_keep_factors_estimable():
    observed = np.ones((6, 4), bool)
    observed[1, 3] = False
    species = ["a", "a", "a", "b", "b", "b"]
    tests = sample_holdout_masks(observed, species, n_splits=10, test_fraction=0.5, seed=0)
    assert len(tests) == 10
    for t in tests:
        assert not (t & ~observed).any()
        train = observed & ~t
        assert train.any(axis=1).all()
        for sp in "ab":
            rows = np.array(species) == sp
            assert train[rows].any(axis=0).all()
        assert t.sum() == round(0.5 * observed.sum())


def test_colony_halves_balance_sites():
    species = ["a"] * 10 + ["b"] * 10
    site = (["x"] * 5 + ["y"] * 5) * 2
    for a, b in colony_halves(species, site, n_splits=5, seed=0):
        assert (a ^ b).all()
        for sp in "ab":
            for st in "xy":
                rows = (np.array(species) == sp) & (np.array(site) == st)
                assert a[rows].sum() in (2, 3)
        assert a.sum() == 10


def test_one_se_rules():
    df = pd.DataFrame({"rank": [2, 2, 3, 3, 4, 4], "test_sse": [10, 12, 8, 8.2, 7.9, 8.1]})
    assert one_se_rank(df) == 3
    df = pd.DataFrame({"lam": [0, 0, 1, 1, 10, 10], "fms": [0.9, 0.92, 0.91, 0.9, 0.5, 0.6]})
    assert one_se_lambda(df) == 1


@pytest.mark.slow
def test_selection_end_to_end_small():
    X, _ = planted(I=80, J=12, K=4, R=2, seed=2, noise=0.1)
    X[:, 0, 3] = np.nan
    species = ["a"] * 6 + ["b"] * 6
    site = (["x"] * 3 + ["y"] * 3) * 2
    rank, df = select_rank(X, species, ranks=[1, 2, 4], n_splits=3, n_iter_max=300,
                           workers=1)
    assert rank == 2
    assert set(df["rank"]) == {1, 2, 4}
    lam, fdf = select_lambda(X, species, site, rank=2, lambda_grid=[0.0, 1.0, 1e4],
                             fixed_lambdas=(None, 0.0, 0.0), n_splits=3, n_iter_max=300,
                             workers=1)
    assert lam in (0.0, 1.0)
    assert fdf.groupby("lam").fms.mean()[1e4] < 0.5
