"""Rank and sparsity selection for sparse CP, following the 2026 talk.

Stage 1, rank
    Fit each candidate rank with no sparsity penalty to a random half of the
    samples and measure squared error on the held-out half. Repeat over
    splits; choose the smallest rank whose mean held-out error is within one
    standard error of the best rank (1-SE rule).

    A "sample" is one colony at one timepoint (a gene fibre of the tensor).
    Held-out samples are masked during fitting and predicted from the
    colony's other timepoints and the timepoint's other colonies, so every
    held-out cell is a genuine prediction. Each colony keeps at least one
    training timepoint and each species keeps at least one training colony
    per timepoint, so all factors remain estimable.

Stage 2, sparsity
    At the chosen rank, split each species' colonies into two halves
    (balanced across sites), fit each half separately, and compute the factor
    match score (FMS) between the two fits on the gene and time modes (colony
    factors cannot be compared because the halves hold different colonies).
    Choose the largest lambda whose mean FMS is within one standard error of
    the best mean FMS.

The talk's implementation compared fits on leave-one-group-out folds; this
version uses held-out samples for stage 1 and colony halves for stage 2 so
both stages work with the colony x timepoint layout and missing samples.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .barnacle import SparseCP, factor_match_score


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def sample_holdout_masks(observed: np.ndarray, species: Sequence[str], n_splits: int,
                         test_fraction: float = 0.5, seed: int = 0,
                         max_tries: int = 1000) -> list[np.ndarray]:
    """Random sample-level holdouts.

    ``observed`` is the colony x timepoint boolean array of sequenced samples.
    Returns ``n_splits`` boolean colony x timepoint arrays marking test samples.
    """
    rng = np.random.default_rng(seed)
    species = np.asarray(species)
    idx = np.argwhere(observed)
    n_test = int(round(test_fraction * len(idx)))
    out = []
    for _ in range(n_splits):
        for _ in range(max_tries):
            pick = idx[rng.choice(len(idx), n_test, replace=False)]
            test = np.zeros_like(observed)
            test[pick[:, 0], pick[:, 1]] = True
            train = observed & ~test
            ok = train.any(axis=1)[observed.any(axis=1)].all()
            for sp in np.unique(species):
                ok &= train[species == sp].any(axis=0).all()
            if ok:
                out.append(test)
                break
        else:
            raise RuntimeError("could not draw a valid holdout; lower test_fraction")
    return out


def colony_halves(species: Sequence[str], site: Sequence[str], n_splits: int,
                  seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Random halves of colonies within each species, balanced across sites."""
    rng = np.random.default_rng(seed)
    species = np.asarray(species)
    site = np.asarray(site)
    out = []
    for _ in range(n_splits):
        a = np.zeros(len(species), bool)
        flip = rng.integers(2)  # which half gets the extra colony of an odd site group
        for sp in np.unique(species):
            for st in np.unique(site[species == sp]):
                members = np.flatnonzero((species == sp) & (site == st))
                members = rng.permutation(members)
                n_a = len(members) // 2 + (len(members) % 2) * flip
                a[members[:n_a]] = True
                flip = 1 - flip
        out.append((a, ~a))
    return out


# ---------------------------------------------------------------------------
# Fitting jobs (top-level so they can run in worker processes)
# ---------------------------------------------------------------------------

@dataclass
class FitSpec:
    rank: int
    lambdas: tuple[float, float, float]
    nonneg_modes: tuple[int, ...] = (0,)
    n_initializations: int = 1
    n_iter_max: int = 1000
    tol: float = 1e-5
    seed: int = 0


def _fit(tensor, mask, spec: FitSpec):
    model = SparseCP(rank=spec.rank, lambdas=list(spec.lambdas),
                     nonneg_modes=list(spec.nonneg_modes),
                     n_initializations=spec.n_initializations, n_iter_max=spec.n_iter_max,
                     tol=spec.tol, random_state=spec.seed)
    cp = model.fit_transform(tensor, mask=mask, threads=1)
    return cp.factors, len(model.loss_)


def _rank_job(args):
    tensor, observed_cells, test_samples, spec, split = args
    test_cells = observed_cells & test_samples[None, :, :]
    train_cells = observed_cells & ~test_samples[None, :, :]
    factors, iters = _fit(np.where(train_cells, tensor, np.nan), train_cells, spec)
    recon = np.einsum("ir,jr,kr->ijk", *factors)
    resid = tensor[test_cells] - recon[test_cells]
    total = tensor[test_cells] - tensor[test_cells].mean()
    return dict(rank=spec.rank, split=split, test_sse=float(resid @ resid),
                test_r2=float(1 - (resid @ resid) / (total @ total)),
                n_test_cells=int(test_cells.sum()), iterations=iters)


def _half_job(args):
    tensor, observed_cells, cols, spec = args
    sub = tensor[:, cols, :]
    m = observed_cells[:, cols, :]
    factors, iters = _fit(np.where(m, sub, np.nan), m, spec)
    return factors, iters


def _run(jobs, fn, workers):
    if workers == 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, jobs))


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


# ---------------------------------------------------------------------------
# Stage 1: rank
# ---------------------------------------------------------------------------

def select_rank(tensor: np.ndarray, species: Sequence[str], ranks: Sequence[int],
                n_splits: int = 5, test_fraction: float = 0.5,
                nonneg_modes=(0,), n_iter_max: int = 1000, tol: float = 1e-5,
                seed: int = 0, workers: Optional[int] = None) -> tuple[int, pd.DataFrame]:
    """Held-out-sample CV over ``ranks`` with no sparsity penalty; 1-SE rule."""
    observed_cells = ~np.isnan(tensor)
    observed = observed_cells.any(axis=0)
    tests = sample_holdout_masks(observed, species, n_splits, test_fraction, seed)
    jobs = [(tensor, observed_cells, t, FitSpec(r, (0.0, 0.0, 0.0), tuple(nonneg_modes),
                                                1, n_iter_max, tol, seed + 1000 * s + r), s)
            for r in ranks for s, t in enumerate(tests)]
    df = pd.DataFrame(_run(jobs, _rank_job, workers or default_workers()))
    return one_se_rank(df), df


def one_se_rank(df: pd.DataFrame) -> int:
    """Smallest rank whose mean test SSE <= best mean + SE of the best rank."""
    g = df.groupby("rank")["test_sse"].agg(["mean", "std", "count"])
    g["se"] = g["std"] / np.sqrt(g["count"])
    best = g["mean"].idxmin()
    threshold = g.loc[best, "mean"] + g.loc[best, "se"]
    return int(g.index[g["mean"] <= threshold].min())


def select_rank_stability(tensor: np.ndarray, species: Sequence[str], site: Sequence[str],
                          ranks: Sequence[int], n_splits: int = 5,
                          lambdas=(0.0, 0.0, 0.0), nonneg_modes=(0,),
                          n_initializations: int = 3, threshold: float = 0.9,
                          n_iter_max: int = 2000, tol: float = 1e-6, seed: int = 0,
                          workers: Optional[int] = None) -> tuple[Optional[int], pd.DataFrame]:
    """Choose rank by split-half reproducibility of gene and time factors.

    For each rank and split, the model is fit separately to two site-balanced
    halves of the colonies and the components are matched on the gene and
    time modes. Use this when held-out-sample prediction cannot separate
    ranks (sample-level noise dominates the shared signal).

    Similarity is the absolute cosine product over gene and time modes,
    without the weight term: halves contain different colonies, and with no
    normalised mode the split of scale between modes is arbitrary, so
    weights are not comparable. Absolute values allow for sign flips, which
    unconstrained modes permit.

    Returns ``(rank, df)``: the rank with the most reproducible components
    (mean count of components with similarity above ``threshold``; ties go
    to the smaller rank; None if no rank averages at least one), and one row per rank and split with the mean FMS,
    the worst component, and the number of components above ``threshold``
    and above 0.5. A mean-FMS cutoff is not used because a too-large rank
    can still score well when its extra components happen to agree, while
    a too-small rank can score badly when it mixes similar-sized components.
    """
    observed_cells = ~np.isnan(tensor)
    halves = colony_halves(species, site, n_splits, seed)
    jobs, keys = [], []
    for r in ranks:
        for s, (a, b) in enumerate(halves):
            spec = FitSpec(r, tuple(float(x) for x in lambdas), tuple(nonneg_modes),
                           n_initializations, n_iter_max, tol, seed + 1000 * s + r)
            jobs += [(tensor, observed_cells, a, spec), (tensor, observed_cells, b, spec)]
            keys.append((r, s))
    fits = _run(jobs, _half_job, workers or default_workers())
    rows = []
    for i, (r, s) in enumerate(keys):
        (fa, ia), (fb, ib) = fits[2 * i], fits[2 * i + 1]
        fms, per, _ = factor_match_score(fa, fb, modes=(0, 2), consider_weights=False,
                                         absolute=True)
        rows.append(dict(rank=r, split=s, fms=fms, min_component=float(per.min()),
                         n_reproducible=int((per > threshold).sum()),
                         n_above_0_5=int((per > 0.5).sum()), iterations=ia + ib))
    df = pd.DataFrame(rows)
    return stability_rank(df), df


def stability_rank(df: pd.DataFrame, min_reproducible: float = 1.0) -> Optional[int]:
    """Rank with the highest mean ``n_reproducible``; ties go to the smaller rank.

    Returns None when no rank reproduces at least ``min_reproducible``
    components on average, since a pick among ranks that all fail is noise.
    """
    means = df.groupby("rank")["n_reproducible"].mean()
    if means.max() < min_reproducible:
        return None
    return int(means.index[np.isclose(means, means.max())].min())


# ---------------------------------------------------------------------------
# Stage 2: sparsity
# ---------------------------------------------------------------------------

def select_lambda(tensor: np.ndarray, species: Sequence[str], site: Sequence[str], rank: int,
                  lambda_grid: Sequence[float], fixed_lambdas=(None, 0.1, 0.05),
                  vary_mode: int = 0, n_splits: int = 5, nonneg_modes=(0,),
                  n_iter_max: int = 1000, tol: float = 1e-5, seed: int = 0,
                  workers: Optional[int] = None) -> tuple[float, pd.DataFrame]:
    """Split-half FMS over ``lambda_grid`` for mode ``vary_mode``; 1-SE rule.

    ``fixed_lambdas`` gives the penalties of the other modes (the entry for
    ``vary_mode`` is ignored). The talk fixed colony 0.1 and time 0.05 and
    varied the gene penalty.
    """
    observed_cells = ~np.isnan(tensor)
    halves = colony_halves(species, site, n_splits, seed)
    jobs, keys = [], []
    for lam in lambda_grid:
        lambdas = list(fixed_lambdas)
        lambdas[vary_mode] = lam
        lambdas = tuple(float(x) for x in lambdas)
        for s, (a, b) in enumerate(halves):
            spec = FitSpec(rank, lambdas, tuple(nonneg_modes), 1, n_iter_max, tol,
                           seed + 1000 * s)
            jobs += [(tensor, observed_cells, a, spec), (tensor, observed_cells, b, spec)]
            keys.append((lam, s))
    fits = _run(jobs, _half_job, workers or default_workers())
    rows = []
    for i, (lam, s) in enumerate(keys):
        (fa, ia), (fb, ib) = fits[2 * i], fits[2 * i + 1]
        fms, per, _ = factor_match_score(fa, fb, modes=(0, 2))
        rows.append(dict(lam=lam, split=s, fms=fms,
                         gene_zero_fraction=float(np.mean([(fa[0] == 0).mean(), (fb[0] == 0).mean()])),
                         empty_components=int(((fa[0] == 0).all(0)).sum() + ((fb[0] == 0).all(0)).sum()),
                         iterations=ia + ib))
    df = pd.DataFrame(rows)
    return one_se_lambda(df), df


def one_se_lambda(df: pd.DataFrame) -> float:
    """Largest lambda whose mean FMS >= best mean FMS - SE of the best lambda."""
    g = df.groupby("lam")["fms"].agg(["mean", "std", "count"])
    g["se"] = g["std"] / np.sqrt(g["count"])
    best = g["mean"].idxmax()
    threshold = g.loc[best, "mean"] - g.loc[best, "se"]
    return float(g.index[g["mean"] >= threshold].max())
