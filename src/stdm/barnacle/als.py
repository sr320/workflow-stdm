"""Sparse CP decomposition by l1-penalised ALS, adapted from Barnacle.

``als_lasso`` and ``SparseCP`` follow ``barnacle/decomposition.py``
(https://github.com/blasks/barnacle, MIT License, Copyright (c) 2021 Stephen
Blaskowski; see LICENSE in this directory). Barnacle is not installable here
because it pins Python < 3.12 and numpy < 2.

Changes from upstream:

- **Missing data.** ``mask`` (``True`` = observed) enables EM-style fitting:
  missing entries start at the observed mean of their mode-0 fibre and are
  replaced by the current reconstruction after every ALS sweep. The loss is
  then computed on observed entries only. Without a mask the algorithm and the
  loss formula are unchanged from upstream.
- Returns a tensorly ``CPTensor`` instead of Barnacle's ``SparseCPTensor``.
- Plotting helpers and simulated-data utilities are not included.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import opt_einsum as oe
import tensorly as tl
from tensorly import check_random_state
from tensorly.cp_tensor import CPTensor
from tensorly.decomposition._cp import initialize_cp
from threadpoolctl import threadpool_limits

from .fista import fista_solve


def _create_mttkrp_function(shape, rank):
    """MTTKRP for a 3-way tensor using cached opt_einsum contraction paths."""
    paths = [
        oe.contract_expression("jr,kr,ijk -> ir", (shape[1], rank), (shape[2], rank), shape),
        oe.contract_expression("ir,kr,ijk -> jr", (shape[0], rank), (shape[2], rank), shape),
        oe.contract_expression("ir,jr,ijk -> kr", (shape[0], rank), (shape[1], rank), shape),
    ]

    def mttkrp(X, factors, mode):
        summed = [fm for m, fm in enumerate(factors) if m != mode]
        return paths[mode](*summed, X)

    return mttkrp


def _initial_fill(tensor: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill missing entries with the observed mean of their mode-0 fibre."""
    filled = np.where(mask, tensor, 0.0)
    counts = mask.reshape(mask.shape[0], -1).sum(axis=1)
    sums = filled.reshape(filled.shape[0], -1).sum(axis=1)
    means = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts > 0)
    return np.where(mask, tensor, means[:, None, None])


def als_lasso(
    tensor: np.ndarray,
    rank: int,
    lambdas: Sequence[float],
    nonneg_modes: Optional[Sequence[int]] = None,
    norm_constraint: bool = True,
    init="random",
    tol: float = 1e-6,
    n_iter_max: int = 1000,
    random_state=None,
    threads: Optional[int] = None,
    verbose: int = 0,
    mask: Optional[np.ndarray] = None,
    log_path: Optional[str] = None,
    log_every: int = 25,
) -> tuple[CPTensor, list[float]]:
    r"""Rank-``rank`` CP decomposition minimising

    .. math:: \|M \odot (Y - \hat{Y})\|^2 + \sum_i \lambda_i \|F_i\|_1

    where ``M`` is the observation mask (all ones when ``mask`` is None).
    Modes in ``nonneg_modes`` are constrained non-negative. With
    ``norm_constraint``, modes with ``lambda == 0`` are constrained to unit
    l2 norm (unless every lambda is 0).

    Returns ``(cp_tensor, losses)``.
    """
    tensor = np.asarray(tensor, dtype=float)
    if tensor.ndim != 3:
        raise ValueError("als_lasso supports 3-way tensors only")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != tensor.shape:
            raise ValueError("mask must have the same shape as tensor")
        if mask.all():
            mask = None
    if mask is None and np.isnan(tensor).any():
        raise ValueError("tensor contains NaN; pass mask=~np.isnan(tensor)")

    with threadpool_limits(limits=threads, user_api="blas"):
        n_modes = tensor.ndim
        nonneg = [i in (nonneg_modes or []) for i in range(n_modes)]

        lambdas = np.array(lambdas, dtype=float)
        if len(lambdas) != n_modes:
            raise ValueError("The number of sparsity coefficients in `lambdas` "
                             "is not equal to the number of modes in `tensor`.")
        if np.any(lambdas < 0):
            raise ValueError("l1 sparsity penalty must be nonnegative.")

        if not norm_constraint or not np.any(lambdas):
            normalize = [False] * n_modes
        else:
            normalize = [lam == 0 for lam in lambdas]

        X = tensor if mask is None else _initial_fill(tensor, mask)

        _, factors = initialize_cp(X, rank, init=init, random_state=random_state)
        factors = [np.asarray(f, dtype=float) for f in factors]
        weights = None

        tensor_norm_sq = tl.norm(X) ** 2
        compute_mttkrp = _create_mttkrp_function(X.shape, rank)
        losses: list[float] = []

        for iteration in range(n_iter_max):
            for mode in range(n_modes):
                DtD = np.ones((rank, rank))
                for other in range(n_modes):
                    if other != mode:
                        DtD *= factors[other].T @ factors[other]
                DtX = compute_mttkrp(X, factors, mode)
                # fista_solve minimises 0.5||DY - X||^2 + reg||Y||_1, hence 0.5 * lambda
                # Contiguous copies are ~20% faster than transposed views, same result
                factors[mode] = np.ascontiguousarray(fista_solve(
                    lhs=DtD,
                    rhs=np.ascontiguousarray(DtX.T),
                    l1_reg=0.5 * lambdas[mode],
                    nonnegative=nonneg[mode],
                    normalize=normalize[mode],
                    init=np.ascontiguousarray(factors[mode].T),
                ).T)

            penalties = float(lambdas @ np.array([np.abs(f).sum() for f in factors]))
            if mask is None:
                # Upstream fast loss: ||X||^2 + ||X_hat||^2 - 2<X, X_hat>
                iprod = np.sum(DtX * factors[mode])
                factors_norm_sq = tl.cp_tensor.cp_norm((weights, factors)) ** 2
                sse = tensor_norm_sq + factors_norm_sq - 2 * iprod
            else:
                recon = tl.cp_to_tensor((None, factors))
                sse = float(np.sum((tensor[mask] - recon[mask]) ** 2))
                X = np.where(mask, tensor, recon)
            loss = float(sse + penalties)
            losses.append(loss)
            if verbose > 1:
                print(f"iteration {iteration + 1}: loss {loss}", flush=True)
            change = (abs(losses[-2] - losses[-1]) / max(losses[-1], 1)
                      if iteration != 0 else float("nan"))
            if log_path and ((iteration + 1) % log_every == 0 or iteration == 0
                             or (tol != 0 and change < tol)):
                with open(log_path, "a") as fh:
                    fh.write(f"{iteration + 1}\t{loss:.6f}\t{change:.3e}\n")

            if loss == 0.0:
                break
            if tol != 0 and iteration != 0:
                if change < tol:
                    if verbose > 0:
                        print(f"converged after {iteration + 1} iterations", flush=True)
                    break
                if iteration == n_iter_max - 1:
                    warnings.warn(f"Algorithm failed to converge after {iteration + 1} iterations")

        return CPTensor((None, factors)), losses


class SparseCP:
    """Sparse CP with multiple random initialisations (best final loss kept).

    Parameters mirror Barnacle's ``SparseCP``; ``fit_transform`` also accepts
    ``mask`` for missing data.
    """

    def __init__(self, rank, lambdas, nonneg_modes=None, norm_constraint=True, init="random",
                 tol=1e-6, n_iter_max=1000, random_state=None, n_initializations=1):
        self.rank = rank
        self.lambdas = lambdas
        self.nonneg_modes = nonneg_modes
        self.norm_constraint = norm_constraint
        self.init = init
        self.tol = tol
        self.n_iter_max = n_iter_max
        self.random_state = random_state
        self.n_initializations = n_initializations
        self._best = None

    @property
    def decomposition_(self) -> CPTensor:
        if self._best is None:
            raise AttributeError("The model has not yet been fit with data.")
        return self.candidates_[self._best]

    @property
    def loss_(self) -> list[float]:
        if self._best is None:
            raise AttributeError("The model has not yet been fit with data.")
        return self.candidate_losses_[self._best]

    def fit_transform(self, tensor, mask=None, threads=None, verbose=0, n_jobs=1,
                      log_dir=None) -> CPTensor:
        """Fit ``n_initializations`` models and keep the one with lowest final loss.

        With ``n_jobs > 1`` initialisations run in parallel processes. Random
        starting factors are drawn up front in the same order as the
        sequential loop, so results are identical either way. With
        ``log_dir``, each initialisation appends ``iteration, loss, relative
        change`` lines to ``progress-init<i>.tsv`` there.
        """
        rns = check_random_state(self.random_state)
        tensor = np.asarray(tensor, dtype=float)
        if isinstance(self.init, str) and self.init == "random":
            shape_only = np.zeros(tensor.shape)
            inits = [initialize_cp(shape_only, self.rank, init="random", random_state=rns)
                     for _ in range(self.n_initializations)]
        else:
            inits = [self.init] * self.n_initializations
        kwargs = dict(nonneg_modes=self.nonneg_modes, norm_constraint=self.norm_constraint,
                      tol=self.tol, n_iter_max=self.n_iter_max, threads=threads,
                      verbose=verbose - 1, mask=mask)
        jobs = []
        for i, init in enumerate(inits):
            kw = dict(kwargs)
            if log_dir is not None:
                kw["log_path"] = str(Path(log_dir) / f"progress-init{i + 1}.tsv")
            jobs.append((tensor, self.rank, self.lambdas, init, kw))
        if n_jobs > 1 and len(jobs) > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=min(n_jobs, len(jobs))) as ex:
                results = list(ex.map(_als_job, jobs))
        else:
            results = []
            for i, job in enumerate(jobs):
                if verbose > 0:
                    print(f"initialization {i + 1} of {len(jobs)}", flush=True)
                results.append(_als_job(job))
        self.candidates_ = [cp for cp, _ in results]
        self.candidate_losses_ = [losses for _, losses in results]
        self._best = int(np.argmin([losses[-1] for losses in self.candidate_losses_]))
        return self.decomposition_


def _als_job(job):
    tensor, rank, lambdas, init, kwargs = job
    return als_lasso(tensor, rank, lambdas, init=init, **kwargs)
