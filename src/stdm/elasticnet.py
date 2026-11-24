"""Phase 5: predict component scores from regulatory layers with elastic net.

For each species, a sample's score on a CP component is predicted from
miRNA, lncRNA and gene-body methylation. Validation holds out whole
colonies (one per site per fold), never single samples: with about four
samples per colony, sample-level splits put the same colony in training and
test and inflate accuracy.

Two targets are supported:

- ``raw``: the component score itself. A site-mean baseline is reported
  alongside, because the site component is largely a site difference.
- ``within_site``: the score minus its training-set site mean, so the model
  has to explain variation among colonies and timepoints within a site.

Stability selection counts how often each feature has a non-zero
coefficient across outer folds. Noise injection adds column-permuted copies
of real features; a real feature is called stable when it is selected more
often than the 95th percentile of the noise features.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from . import mosaic
from .tensor import size_factors

BLOCKS = ("mirna", "lncrna", "gene_meth")

# scikit-learn 1.7 takes the alpha-grid size as ``alphas=int`` and deprecates ``n_alphas``.
import sklearn as _sklearn
_SKLEARN_INT_ALPHAS = tuple(int(x) for x in _sklearn.__version__.split(".")[:2]) >= (1, 7)


# ---------------------------------------------------------------------------
# Predictor matrices
# ---------------------------------------------------------------------------

def _poscounts_size_factors(counts: pd.DataFrame) -> pd.Series:
    """DESeq2 ``poscounts`` size factors: geometric means over positive counts only."""
    c = counts.to_numpy(dtype=float)
    logc = np.where(c > 0, np.log(np.where(c > 0, c, 1.0)), np.nan)
    log_geo = np.nanmean(logc, axis=1, keepdims=True)
    ratios = np.where(c > 0, logc - log_geo, np.nan)
    sf = np.exp(np.nanmedian(ratios, axis=0))
    return pd.Series(sf / np.exp(np.mean(np.log(sf))), index=counts.columns)


def _log_counts(counts: pd.DataFrame, min_count: float, min_fraction: float,
                min_library: float = 0.1) -> pd.DataFrame:
    lib = counts.sum()
    failed = lib < min_library * lib.median()
    if failed.any():
        import warnings
        warnings.warn(f"dropping {int(failed.sum())} samples with library size below "
                      f"{min_library:.0%} of the median: {sorted(lib[failed].index)}")
        counts = counts.loc[:, ~failed]
    try:
        sf = size_factors(counts)
    except ValueError:  # small panels (miRNA) often have no feature positive everywhere
        sf = _poscounts_size_factors(counts)
    norm = counts / sf
    keep = (norm >= min_count).mean(axis=1) >= min_fraction
    return np.log2(norm[keep] + 1)


def load_block(species: str, block: str, top: Optional[int] = 1000, data_dir=None,
               min_count: float = 10, min_fraction: float = 0.5,
               max_missing: float = 0.2) -> pd.DataFrame:
    """One species' predictor block as samples x features.

    Counts (miRNA, lncRNA) get median-of-ratios normalisation, an expression
    filter and log2(x + 1). Methylation (percent) keeps features with at
    most ``max_missing`` missing values; remaining gaps stay NaN and are
    filled with training means at fit time. ``top`` keeps the most variable
    features; it uses no target information.
    """
    values, _ = mosaic.load_matrix(f"{species}_{block}", data_dir)
    values = values.astype(float)
    if block == "gene_meth":
        values = values[values.isna().mean(axis=1) <= max_missing] / 100.0
    else:
        values = _log_counts(values, min_count, min_fraction)
    if top is not None and len(values) > top:
        values = values.loc[values.var(axis=1).sort_values(ascending=False).index[:top]]
    out = values.T
    out.columns = [f"{block}:{c}" for c in out.columns]
    out.index.name = "sample_id"
    return out


def predictors(species: str, blocks: Sequence[str] = BLOCKS, top: Optional[int] = 1000,
               data_dir=None) -> pd.DataFrame:
    """Predictor blocks joined on samples present in every block."""
    frames = [load_block(species, b, top, data_dir) for b in blocks]
    return pd.concat(frames, axis=1, join="inner")


# ---------------------------------------------------------------------------
# Removing variation shared with the gene-expression libraries
# ---------------------------------------------------------------------------

def residualize(X: pd.DataFrame, Z: np.ndarray) -> pd.DataFrame:
    """Residuals of every column of ``X`` on ``Z`` plus an intercept.

    Missing values in ``X`` are filled with column means first and restored
    afterwards. Uses no target information, so it can be applied before CV.
    """
    Xv = X.to_numpy(float)
    miss = np.isnan(Xv)
    Xv = np.where(miss, np.nanmean(Xv, axis=0), Xv)
    D = np.column_stack([np.ones(len(X)), np.asarray(Z, float)])
    beta, *_ = np.linalg.lstsq(D, Xv, rcond=None)
    R = Xv - D @ beta
    R[miss] = np.nan
    return pd.DataFrame(R, index=X.index, columns=X.columns)


def technical_covariates(species: str, samples: Sequence[str], data_dir=None) -> pd.DataFrame:
    """Per-sample technical covariates: log library sizes (genes, lncRNA),
    log RNA concentration, plate and extraction batch (dummies)."""
    samples = list(samples)
    base = Path(data_dir) if data_dir else mosaic.default_data_dir()
    meta = pd.read_csv(base / "samples.csv").set_index("sample_id").loc[samples]
    out = {}
    for block in ("gene", "lncrna"):
        raw, _ = mosaic.load_matrix(f"{species}_{block}", data_dir)
        out[f"log_lib_{block}"] = np.log1p(raw.reindex(columns=samples).sum().to_numpy(float))
    conc = meta.rna_conc_ng_ul.astype(float)
    out["log_rna_conc"] = np.log1p(conc.fillna(conc.median()).to_numpy())
    T = pd.DataFrame(out, index=samples)
    dummies = pd.get_dummies(meta[["plate", "extraction_batch"]].astype(str),
                             drop_first=True).astype(float)
    T = pd.concat([T, dummies], axis=1)
    return T.loc[:, T.std() > 0]


def gene_pcs(species: str, samples: Sequence[str], k: int, top: int = 2000,
             data_dir=None) -> np.ndarray:
    """Scores on the top ``k`` principal components of the species' gene
    expression (log normalised counts, ``top`` most variable genes, scaled)."""
    g, _ = mosaic.load_matrix(f"{species}_gene", data_dir)
    g = _log_counts(g.astype(float), 10, 0.5)
    g = g.loc[g.var(axis=1).sort_values(ascending=False).index[:top]].T.loc[list(samples)]
    Z = ((g - g.mean()) / g.std()).fillna(0).to_numpy()
    u, d, _ = np.linalg.svd(Z, full_matrices=False)
    return u[:, :k] * d[:k]


def adjust(X: pd.DataFrame, species: str, how: str = "none", n_pcs: int = 3,
           data_dir=None) -> pd.DataFrame:
    """Remove technical covariates (``technical``) or the top ``n_pcs`` gene
    expression PCs (``gene-pcs``) from every predictor column."""
    if how == "none":
        return X
    if how == "technical":
        return residualize(X, technical_covariates(species, X.index, data_dir).to_numpy())
    if how == "gene-pcs":
        return residualize(X, gene_pcs(species, X.index, n_pcs, data_dir=data_dir))
    raise ValueError(f"unknown adjustment {how!r}")


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def colony_pair_folds(colonies: pd.Series, sites: pd.Series) -> list[np.ndarray]:
    """Outer folds holding out one colony from each site (all pairs).

    ``colonies`` and ``sites`` are per sample. Returns boolean test masks.
    """
    col_site = pd.Series(sites.to_numpy(), index=colonies.to_numpy()).groupby(level=0).first()
    by_site = [sorted(col_site[col_site == s].index) for s in sorted(col_site.unique())]
    folds = []
    for pair in product(*by_site):
        folds.append(colonies.isin(pair).to_numpy())
    return folds


@dataclass
class ENetSpec:
    l1_ratio: tuple[float, ...] = (0.1, 0.5, 0.9, 1.0)
    n_alphas: int = 30
    inner_folds: int = 5
    max_iter: int = 5000
    target: str = "raw"            # or "within_site"
    noise_fraction: float = 0.1    # permuted copies, as a fraction of real features
    seed: int = 0


def _fit_predict(X_train, y_train, groups_train, X_test, spec: ENetSpec):
    mu = np.nanmean(X_train, axis=0)
    mu = np.where(np.isnan(mu), 0.0, mu)
    X_train = np.where(np.isnan(X_train), mu, X_train)
    X_test = np.where(np.isnan(X_test), mu, X_test)
    scaler = StandardScaler().fit(X_train)
    Xs, Xt = scaler.transform(X_train), scaler.transform(X_test)
    Xs = np.nan_to_num(Xs)
    Xt = np.nan_to_num(Xt)
    n_groups = len(np.unique(groups_train))
    inner = list(GroupKFold(n_splits=min(spec.inner_folds, n_groups)).split(Xs, y_train, groups_train))
    grid = ({"alphas": spec.n_alphas} if _SKLEARN_INT_ALPHAS else {"n_alphas": spec.n_alphas})
    model = ElasticNetCV(l1_ratio=list(spec.l1_ratio), cv=inner, max_iter=spec.max_iter,
                         n_jobs=1, **grid)
    model.fit(Xs, y_train)
    return model.predict(Xt), model.coef_ != 0, float(model.alpha_), float(model.l1_ratio_)


def _add_noise(X: np.ndarray, names: list[str], fraction: float, rng,
               min_noise: int = 20) -> tuple[np.ndarray, list[str]]:
    if fraction <= 0:
        return X, names
    k = max(int(round(fraction * X.shape[1])), min_noise)  # enough for a stable quantile
    cols = rng.choice(X.shape[1], k, replace=k > X.shape[1])
    noise = np.column_stack([rng.permutation(X[:, c]) for c in cols])
    return np.hstack([X, noise]), names + [f"noise:{i}" for i in range(k)]


def cross_validate(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, spec: ENetSpec):
    """Outer colony-pair CV with an inner grouped ElasticNetCV.

    ``meta`` is indexed like ``X`` with ``colony_id`` and ``site``. Returns
    ``(predictions, selection, folds)``: per-sample mean held-out
    prediction and site baseline, per-feature selection frequency (with
    noise features), and one row per fold.
    """
    rng = np.random.default_rng(spec.seed)
    Xv, names = _add_noise(X.to_numpy(float), list(X.columns), spec.noise_fraction, rng)
    yv = y.loc[X.index].to_numpy(float)
    colonies, sites = meta.loc[X.index, "colony_id"], meta.loc[X.index, "site"]
    folds = colony_pair_folds(colonies, sites)
    preds = [[] for _ in range(len(yv))]
    base = [[] for _ in range(len(yv))]
    truth = [[] for _ in range(len(yv))]
    selected = np.zeros(len(names))
    rows = []
    site_arr = sites.to_numpy()
    for f, test in enumerate(folds):
        train = ~test
        site_mean = pd.Series(yv[train]).groupby(site_arr[train]).mean()
        offset_train = site_mean.reindex(site_arr[train]).to_numpy()
        offset_test = site_mean.reindex(site_arr[test]).to_numpy()
        if spec.target == "within_site":
            y_train, y_test, b = yv[train] - offset_train, yv[test] - offset_test, np.zeros(test.sum())
        else:
            y_train, y_test, b = yv[train], yv[test], offset_test
        p, sel, alpha, l1 = _fit_predict(Xv[train], y_train, colonies.to_numpy()[train],
                                         Xv[test], spec)
        selected += sel
        for i, pi, bi, ti in zip(np.flatnonzero(test), p, b, y_test):
            preds[i].append(pi); base[i].append(bi); truth[i].append(ti)
        rows.append(dict(fold=f, held_out=",".join(sorted(colonies[test].unique())),
                         alpha=alpha, l1_ratio=l1, n_selected=int(sel[:X.shape[1]].sum()),
                         n_noise_selected=int(sel[X.shape[1]:].sum())))
    pred = pd.DataFrame({
        "observed": [np.mean(t) for t in truth],
        "predicted": [np.mean(p) for p in preds],
        "site_baseline": [np.mean(b) for b in base],
    }, index=X.index)
    selection = pd.DataFrame({"feature": names, "frequency": selected / len(folds)})
    selection["noise"] = selection.feature.str.startswith("noise:")
    return pred, selection, pd.DataFrame(rows)


def summarise(pred: pd.DataFrame) -> dict:
    """R^2 of model and site baseline against the overall mean, and Spearman."""
    y, p, b = pred.observed, pred.predicted, pred.site_baseline
    ss = ((y - y.mean()) ** 2).sum()
    return dict(n=len(y), r2=float(1 - ((y - p) ** 2).sum() / ss),
                r2_site_baseline=float(1 - ((y - b) ** 2).sum() / ss),
                spearman=float(stats.spearmanr(y, p)[0]))


def stable_features(selection: pd.DataFrame, quantile: float = 0.95) -> pd.DataFrame:
    """Real features selected more often than the noise features' quantile."""
    noise = selection[selection.noise].frequency
    threshold = float(noise.quantile(quantile)) if len(noise) else 0.0
    real = selection[~selection.noise]
    out = real[real.frequency > threshold].sort_values("frequency", ascending=False)
    return out.assign(threshold=threshold)


def _perm_job(args):
    X, y, meta, spec, seed = args
    rng = np.random.default_rng(seed)
    # Shuffle colony labels within site: keeps site structure and the
    # within-colony correlation of samples, breaks colony-feature links.
    col_site = meta.groupby("colony_id")["site"].first()
    mapping = {}
    for s in col_site.unique():
        cols = list(col_site[col_site == s].index)
        mapping.update(dict(zip(cols, rng.permutation(cols))))
    yp = y.copy()
    for col, new in mapping.items():
        src = meta.index[meta.colony_id == new]
        dst = meta.index[meta.colony_id == col]
        # Match by timepoint where both colonies have it; otherwise keep.
        s_tp = meta.loc[src].set_index("timepoint").index
        src_by_tp = pd.Series(y.loc[src].to_numpy(), index=s_tp)
        for d in dst:
            tp = meta.loc[d, "timepoint"]
            if tp in src_by_tp.index:
                yp.loc[d] = src_by_tp.loc[tp]
    pred, _, _ = cross_validate(X, yp, meta, ENetSpec(**{**spec.__dict__, "noise_fraction": 0.0}))
    return summarise(pred)["r2"]


def permutation_null(X, y, meta, spec: ENetSpec, n: int = 50, workers: int = 1,
                     seed: int = 1) -> np.ndarray:
    """Held-out R^2 with colony labels shuffled within site (``n`` times)."""
    jobs = [(X, y, meta, spec, seed + i) for i in range(n)]
    if workers == 1:
        return np.array([_perm_job(j) for j in jobs])
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return np.array(list(ex.map(_perm_job, jobs)))
