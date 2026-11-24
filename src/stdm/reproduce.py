"""Reproduce the rank-35 Barnacle model presented in the 2026 eScience talk.

Source: ``urol-e5/timeseries_molecular``,
``M-multi-species/scripts/14.1-barnacle/build_tensor_and_run.py`` run by
``M-multi-species/scripts/26-rank35-optimization.rmd`` with
``--rank 35 --lambda-gene 0.2 --lambda-sample 0.1 --lambda-time 0.05
--max-iter 2000 --tol 1e-5 --seed 81`` on ``vst_counts_matrix.csv``
(``input-data/vst_counts_matrix.csv`` here). Published factors:
``M-multi-species/output/26-rank35-optimization/lambda_gene_0.2/``.

The talk tensor is built exactly as that script did, including two choices
this repo does not otherwise make:

- colonies are ordered by first appearance in the file (not numerically);
- the three missing colony-timepoints are filled with 0. VST values are
  >= 3.5, so these zeros are extreme values that the model must fit.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .barnacle import SparseCP

TALK_PARAMS = dict(
    rank=35,
    lambdas=[0.2, 0.1, 0.05],  # gene, colony, time
    nonneg_modes=[0],
    n_initializations=3,
    random_state=81,
    n_iter_max=2000,
    tol=1e-5,
)

PUBLISHED_URL = ("https://gannet.fish.washington.edu/v1_web/owlshell/bu-github/"
                 "timeseries_molecular/M-multi-species/output/26-rank35-optimization/"
                 "lambda_gene_0.2/barnacle_factors/{name}_factors.csv")

_COL = re.compile(r"(ACR|POR|POC)-(\d+)-TP(\d)")
_SPECIES = {"ACR": "apul", "POR": "peve", "POC": "ptua"}


def talk_tensor(vst_path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Build the talk's 3-way tensor from the wide VST matrix.

    Returns ``(tensor, group_ids, colony_labels)`` with labels such as
    ``apul_ACR-139``, matching the published ``sample_factors.csv``.
    """
    v = pd.read_csv(vst_path, index_col=0)
    groups = sorted(v.index)
    cols: dict[tuple[str, str, int], str] = {}
    order: list[tuple[str, str]] = []
    for prefix in ["ACR", "POR", "POC"]:
        for c in v.columns:
            m = _COL.fullmatch(c)
            if not m or m.group(1) != prefix:
                continue
            cols[(prefix, m.group(2), int(m.group(3)))] = c
            if (prefix, m.group(2)) not in order:
                order.append((prefix, m.group(2)))
    tensor = np.zeros((len(groups), len(order), 4))
    for j, (prefix, colony) in enumerate(order):
        for k in range(4):
            c = cols.get((prefix, colony, k + 1))
            if c is not None:
                tensor[:, j, k] = v.loc[groups, c].to_numpy(dtype=float)
    labels = [f"{_SPECIES[p]}_{p}-{c}" for p, c in order]
    return tensor, groups, labels


def fit_talk_model(tensor: np.ndarray, verbose: int = 0, n_jobs: int = 1,
                   **overrides) -> SparseCP:
    """Fit SparseCP with the talk's parameters (``overrides`` replace any of them)."""
    params = {**TALK_PARAMS, **overrides}
    model = SparseCP(**params)
    model.fit_transform(tensor, verbose=verbose, n_jobs=n_jobs)
    return model


def factors_to_frames(factors, groups, colonies, timepoints=("TP1", "TP2", "TP3", "TP4")):
    names = [f"Component_{i + 1}" for i in range(factors[0].shape[1])]
    return {
        "gene": pd.DataFrame(factors[0], index=pd.Index(groups, name="OG_ID"), columns=names),
        "sample": pd.DataFrame(factors[1], index=pd.Index(colonies, name="OG_ID"), columns=names),
        "time": pd.DataFrame(factors[2], index=pd.Index(timepoints, name="OG_ID"), columns=names),
    }


def fetch_published(dest: Path) -> dict[str, pd.DataFrame]:
    """Download (once) and load the talk's published factor matrices."""
    import urllib.request

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    frames = {}
    for name in ("gene", "sample", "time"):
        path = dest / f"{name}_factors.csv"
        if not path.exists():
            req = urllib.request.Request(PUBLISHED_URL.format(name=name),
                                         headers={"User-Agent": "stdm-fetch"})
            with urllib.request.urlopen(req, timeout=120) as r:
                path.write_bytes(r.read())
        frames[name] = pd.read_csv(path, index_col=0)
    return frames


def compare_factors(ours: dict[str, pd.DataFrame], theirs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Match components between two rank-R models and report agreement.

    Returns one row per component of ``theirs`` with the matched component of
    ``ours``, the FMS-style similarity over all three modes, and the Pearson
    correlation of gene, colony and time loadings.
    """
    from .barnacle import factor_match_score

    modes = ("gene", "sample", "time")
    for m in modes:
        if not ours[m].index.equals(theirs[m].index):
            raise ValueError(f"{m} labels differ between the two models")
    fa = [theirs[m].to_numpy() for m in modes]
    fb = [ours[m].to_numpy() for m in modes]
    _, scores, match = factor_match_score(fa, fb)
    rows = []
    for r, s in enumerate(match):
        row = dict(published=theirs["gene"].columns[r],
                   reproduced=ours["gene"].columns[s] if s >= 0 else None,
                   similarity=scores[r])
        for m in modes:
            a, b = fa[modes.index(m)][:, r], fb[modes.index(m)][:, s]
            row[f"r_{m}"] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def talk_expression_tensor(vst_path: Path, data_dir: Optional[Path] = None):
    """The talk's VST data as an ExpressionTensor with missing samples as NaN.

    Same groups and values as :func:`talk_tensor`, but colonies are ordered
    numerically, labelled with site, and the three missing samples are masked
    instead of zero-filled.
    """
    from . import mosaic
    from .tensor import ExpressionTensor, SPECIES, TIMEPOINTS

    v = pd.read_csv(vst_path, index_col=0)
    v = v.loc[sorted(v.index)]
    v.columns = [mosaic.canonical_sample_id(c) for c in v.columns]
    meta = mosaic.build_sample_metadata(data_dir)
    colonies = meta.drop_duplicates("colony_id").set_index("colony_id")[["species", "site"]]
    colonies = colonies.loc[sorted(colonies.index, key=lambda c: (
        SPECIES.index(colonies.loc[c, "species"]), int(c.split("-")[1])))]
    values = np.full((len(v), len(colonies), len(TIMEPOINTS)), np.nan)
    for ci, colony in enumerate(colonies.index):
        for ti, tp in enumerate(TIMEPOINTS):
            s = f"{colony}-TP{tp}"
            if s in v.columns:
                values[:, ci, ti] = v[s].to_numpy()
    groups = pd.DataFrame(index=pd.Index(v.index, name="group_id"))
    o = mosaic.load_orthologs(data_dir).set_index("group_id")
    groups = groups.join(o[[*SPECIES, "protein_name", "goslim_names"]])
    colonies.index.name = "colony_id"
    return ExpressionTensor(values, groups, colonies, list(TIMEPOINTS),
                            {"normalization": "talk-vst", "source": str(vst_path)})
