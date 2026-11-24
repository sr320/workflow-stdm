"""Factor match score (FMS) between two CP decompositions."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def _unit_columns(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(f, axis=0)
    safe = np.where(norms > 0, norms, 1.0)
    return f / safe, norms


def factor_match_score(
    factors_a: Sequence[np.ndarray],
    factors_b: Sequence[np.ndarray],
    modes: Optional[Sequence[int]] = None,
    consider_weights: bool = True,
    absolute: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Match components of two CP models and score their similarity.

    For components ``r`` of A and ``s`` of B, similarity is the product over
    ``modes`` of the cosine between their factor vectors, multiplied (when
    ``consider_weights``) by ``1 - |w_r - w_s| / max(w_r, w_s)``, where ``w``
    is the product of factor-vector norms over *all* modes. Components are
    paired by the Hungarian algorithm to maximise total similarity.

    Only ``modes`` enter the cosine term, so factors for modes not listed
    (e.g. colonies, when A and B were fit on different colonies) may have
    different lengths; weights then use only the listed modes.

    A component whose vector is all zero in any listed mode has similarity 0
    with everything, so components removed by the sparsity penalty count as
    not reproduced.

    Returns ``(mean_score, per_component_scores, matching)`` where
    ``matching[r]`` is the B component paired with A component ``r``.
    """
    n_modes = len(factors_a)
    modes = list(range(n_modes)) if modes is None else list(modes)
    rank_a = factors_a[0].shape[1]
    rank_b = factors_b[0].shape[1]

    sim = np.ones((rank_a, rank_b))
    zero_a = np.zeros(rank_a, bool)
    zero_b = np.zeros(rank_b, bool)
    for m in modes:
        ua, na = _unit_columns(np.asarray(factors_a[m], float))
        ub, nb = _unit_columns(np.asarray(factors_b[m], float))
        cos = ua.T @ ub
        sim *= np.abs(cos) if absolute else cos
        zero_a |= na == 0
        zero_b |= nb == 0

    if consider_weights:
        weight_modes = modes if any(
            np.asarray(factors_a[m]).shape[0] != np.asarray(factors_b[m]).shape[0]
            for m in range(n_modes)) else range(n_modes)
        wa = np.prod([np.linalg.norm(factors_a[m], axis=0) for m in weight_modes], axis=0)
        wb = np.prod([np.linalg.norm(factors_b[m], axis=0) for m in weight_modes], axis=0)
        denom = np.maximum.outer(wa, wb)
        penalty = 1 - np.divide(np.abs(np.subtract.outer(wa, wb)), denom,
                                out=np.ones_like(denom), where=denom > 0)
        sim *= penalty

    sim[zero_a, :] = 0
    sim[:, zero_b] = 0

    rows, cols = linear_sum_assignment(-sim)
    scores = np.zeros(rank_a)
    matching = np.full(rank_a, -1)
    scores[rows] = sim[rows, cols]
    matching[rows] = cols
    return float(scores.mean()), scores, matching
