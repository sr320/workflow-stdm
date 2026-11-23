"""Build an ortholog x colony x timepoint expression tensor from MOSAiC data.

Layout
------
``values[g, c, t]`` is the expression of ortholog group ``g`` in colony ``c``
at timepoint ``t``. Colonies are the 30 sampled colonies (10 per species),
ordered by species then colony number; each colony slice therefore belongs to
one species, and ``colonies`` carries species and site labels. Keeping
colonies as a mode (rather than averaging to species) preserves the biological
replication that rank selection needs.

Colony-timepoints without RNA-seq are ``NaN`` in ``values`` and ``False`` in
``mask``; they are never filled with zeros.

Normalisation
-------------
``log2-mor`` (default)
    DESeq2-style median-of-ratios size factors computed within each species on
    all of that species' genes, then ``log2(normalised + 1)``.
``vst-joint``
    DESeq2 blind variance-stabilising transform on the combined three-species
    ortholog count matrix. This reproduces ``input-data/vst_counts_matrix.csv``
    (r = 0.9999 on its 9,800 groups) and so matches the analysis in the 2026
    talk. Needs the optional ``pydeseq2`` dependency
    (``uv sync --extra vst``).

Group filtering is the same for both: a group is kept when, in every species,
at least ``min_fraction`` of samples have a median-of-ratios normalised count
of at least ``min_count``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import mosaic

SPECIES = list(mosaic.SPECIES.species)
TIMEPOINTS = [1, 2, 3, 4]
NORMALIZATIONS = ("log2-mor", "vst-joint")
CENTERINGS = ("none", "species")


@dataclass
class ExpressionTensor:
    """Expression tensor with labelled modes.

    Attributes
    ----------
    values : ndarray, shape (n_groups, n_colonies, n_timepoints)
        Expression values; ``NaN`` where the sample was not sequenced.
    groups : DataFrame
        Indexed by ``group_id``; per-species gene IDs and annotation.
    colonies : DataFrame
        Indexed by ``colony_id``; ``species`` and ``site``.
    timepoints : list of int
    params : dict
        How the tensor was built (normalisation, filters, counts).
    """

    values: np.ndarray
    groups: pd.DataFrame
    colonies: pd.DataFrame
    timepoints: list[int]
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        expected = (len(self.groups), len(self.colonies), len(self.timepoints))
        if self.values.shape != expected:
            raise ValueError(f"values shape {self.values.shape} != labels {expected}")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.values.shape

    @property
    def mask(self) -> np.ndarray:
        """``True`` where a value was observed."""
        return ~np.isnan(self.values)

    def missing_samples(self) -> list[str]:
        observed = self.mask.any(axis=0)
        return [f"{c}-TP{t}"
                for i, c in enumerate(self.colonies.index)
                for j, t in enumerate(self.timepoints) if not observed[i, j]]

    def to_matrix(self) -> pd.DataFrame:
        """Groups x samples (observed samples only), columns ``PREFIX-colony-TPn``."""
        g, c, t = self.shape
        flat = self.values.reshape(g, c * t)
        cols = [f"{col}-TP{tp}" for col in self.colonies.index for tp in self.timepoints]
        df = pd.DataFrame(flat, index=self.groups.index, columns=cols)
        return df.loc[:, df.notna().any(axis=0)]

    def summary(self) -> dict:
        return {
            "shape": list(self.shape),
            "groups": len(self.groups),
            "colonies_per_species": {k: int(v) for k, v in
                                     self.colonies.groupby("species").size().items()},
            "colonies_per_site": {f"{s}/{site}": int(n) for (s, site), n in
                                  self.colonies.groupby(["species", "site"]).size().items()},
            "missing_samples": self.missing_samples(),
            "observed_fraction": float(self.mask.mean()),
            "min": float(np.nanmin(self.values)),
            "max": float(np.nanmax(self.values)),
            **self.params,
        }

    # -- persistence -------------------------------------------------------

    def save(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "values.npy", self.values)
        self.groups.to_csv(out_dir / "groups.csv")
        self.colonies.to_csv(out_dir / "colonies.csv")
        with open(out_dir / "params.json", "w") as fh:
            json.dump({**self.params, "timepoints": self.timepoints,
                       "summary": self.summary()}, fh, indent=2, default=str)
        return out_dir

    @classmethod
    def load(cls, in_dir: Path) -> "ExpressionTensor":
        in_dir = Path(in_dir)
        with open(in_dir / "params.json") as fh:
            params = json.load(fh)
        timepoints = params.pop("timepoints")
        params.pop("summary", None)
        return cls(
            values=np.load(in_dir / "values.npy"),
            groups=pd.read_csv(in_dir / "groups.csv", index_col="group_id"),
            colonies=pd.read_csv(in_dir / "colonies.csv", index_col="colony_id"),
            timepoints=timepoints,
            params=params,
        )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def size_factors(counts: pd.DataFrame) -> pd.Series:
    """DESeq2 median-of-ratios size factors (genes with any zero are skipped)."""
    c = counts.to_numpy(dtype=float)
    positive = (c > 0).all(axis=1)
    if not positive.any():
        raise ValueError("no gene has a positive count in every sample")
    logc = np.log(c[positive])
    log_geo_mean = logc.mean(axis=1, keepdims=True)
    return pd.Series(np.exp(np.median(logc - log_geo_mean, axis=0)), index=counts.columns)


def select_ortholog_groups(orthologs: pd.DataFrame, counts: dict[str, pd.DataFrame],
                           types: tuple[str, ...] = ("three_way",)) -> tuple[pd.DataFrame, dict]:
    """Groups of the given types whose genes are in every species' count matrix.

    Groups sharing a gene with another selected group are dropped so that no
    gene is counted twice.
    """
    g = orthologs[orthologs["type"].isin(types)].set_index("group_id")
    report = {"ortholog_groups_of_type": int(len(g))}
    present = np.all([g[sp].isin(counts[sp].index) for sp in SPECIES], axis=0)
    g = g[present]
    report["with_genes_in_all_count_matrices"] = int(len(g))
    shared = np.any([g[sp].duplicated(keep=False) for sp in SPECIES], axis=0)
    g = g[~shared]
    report["after_dropping_shared_genes"] = int(len(g))
    return g, report


def expressed_groups(groups: pd.DataFrame, norm: dict[str, pd.DataFrame],
                     min_count: float, min_fraction: float) -> np.ndarray:
    """Boolean mask: expressed in every species by the min_count/min_fraction rule."""
    keep = np.ones(len(groups), dtype=bool)
    for sp in SPECIES:
        m = norm[sp].loc[groups[sp]].to_numpy()
        keep &= (m >= min_count).mean(axis=1) >= min_fraction
    return keep


def _vst_joint(combined_counts: pd.DataFrame) -> pd.DataFrame:
    try:
        from pydeseq2.dds import DeseqDataSet
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("normalization='vst-joint' needs pydeseq2: uv sync --extra vst") from exc
    meta = pd.DataFrame({"species": [c[:3] for c in combined_counts.columns]},
                        index=combined_counts.columns)
    dds = DeseqDataSet(counts=combined_counts.T.astype(int), metadata=meta,
                       design="~species", quiet=True)
    dds.vst(use_design=False)
    return pd.DataFrame(np.asarray(dds.layers["vst_counts"]).T,
                        index=combined_counts.index, columns=combined_counts.columns)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_expression_tensor(
    data_dir: Optional[Path] = None,
    normalization: str = "log2-mor",
    min_count: float = 10,
    min_fraction: float = 0.5,
    center: str = "none",
    types: tuple[str, ...] = ("three_way",),
) -> ExpressionTensor:
    """Build the ortholog x colony x timepoint tensor from cached MOSAiC files.

    Parameters
    ----------
    normalization : {"log2-mor", "vst-joint"}
    min_count, min_fraction : group filter applied in every species
    center : {"none", "species"}
        ``species`` subtracts each group's mean within each species (over that
        species' observed samples), removing between-species level differences
        that are partly technical (mapping to different genomes).
    types : ortholog group types to include
    """
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS}")
    if center not in CENTERINGS:
        raise ValueError(f"center must be one of {CENTERINGS}")
    if not 0 <= min_fraction <= 1:
        raise ValueError("min_fraction must be between 0 and 1")

    manifest = mosaic.load_manifest(data_dir)
    counts = {sp: mosaic.load_matrix(f"{sp}_gene", data_dir, manifest)[0] for sp in SPECIES}
    orthologs = mosaic.load_orthologs(data_dir, manifest)

    groups, report = select_ortholog_groups(orthologs, counts, types)
    norm = {sp: counts[sp] / size_factors(counts[sp]) for sp in SPECIES}
    keep = expressed_groups(groups, norm, min_count, min_fraction)

    if normalization == "log2-mor":
        expr = {sp: np.log2(norm[sp].loc[groups[sp]].set_axis(groups.index) + 1) for sp in SPECIES}
    else:
        combined = pd.concat([counts[sp].loc[groups[sp]].set_axis(groups.index) for sp in SPECIES],
                             axis=1)
        vst = _vst_joint(combined)
        expr = {sp: vst[counts[sp].columns] for sp in SPECIES}

    groups = groups[keep]
    report["after_expression_filter"] = int(len(groups))

    # Colony and site labels
    meta = mosaic.build_sample_metadata(data_dir, manifest)
    colonies = (meta.drop_duplicates("colony_id")
                .set_index("colony_id")[["species", "site"]])
    colonies = colonies.loc[sorted(colonies.index,
                                   key=lambda c: (SPECIES.index(colonies.loc[c, "species"]),
                                                  int(c.split("-")[1])))]

    values = np.full((len(groups), len(colonies), len(TIMEPOINTS)), np.nan)
    for ci, (colony, row) in enumerate(colonies.iterrows()):
        e = expr[row["species"]]
        for ti, tp in enumerate(TIMEPOINTS):
            sample = f"{colony}-TP{tp}"
            if sample in e.columns:
                values[:, ci, ti] = e.loc[groups.index, sample].to_numpy()

    if center == "species":
        for sp in SPECIES:
            idx = np.flatnonzero(colonies["species"].to_numpy() == sp)
            block = values[:, idx, :]
            values[:, idx, :] = block - np.nanmean(block.reshape(len(groups), -1), axis=1)[:, None, None]

    group_table = groups[[*SPECIES, "protein_name", "goslim_names"]].copy()
    group_table.index.name = "group_id"
    colonies.index.name = "colony_id"
    params = {
        "normalization": normalization,
        "center": center,
        "min_count": min_count,
        "min_fraction": min_fraction,
        "ortholog_types": list(types),
        "group_counts": report,
    }
    return ExpressionTensor(values, group_table, colonies, list(TIMEPOINTS), params)
