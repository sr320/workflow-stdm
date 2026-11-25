"""Access to the E5 MOSAiC coral time-series data.

Data are described at https://urol-e5.github.io/MOSAiC/ and listed, one file
per row, in ``data/manifest.csv``. This module downloads them into a local
cache, harmonises sample and gene identifiers, and builds the sample metadata
table used by downstream analyses.

Identifiers
-----------
Samples are written ``<PREFIX>-<colony>-TP<n>`` (e.g. ``ACR-145-TP4``) in most
files, but miRNA columns carry a well prefix (``1A10_ACR-145_TP4``) and the
physiology/ITS2 tables use ``ACR-145_TP4``. :func:`canonical_sample_id`
maps all of these to ``ACR-145-TP4``.

Gene IDs carry a ``gene-`` prefix in the P. evermanni and P. tuahiniensis
count matrices but not in the ortholog table, and A. pulchra ortholog IDs
carry a transcript suffix (``FUN_000185-T1``) that the count matrices do not.
:func:`canonical_gene_id` removes both.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
import tempfile
import urllib.request
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

SPECIES = pd.DataFrame(
    [
        ("apul", "ACR", "Acropora", "Acropora pulchra"),
        ("peve", "POR", "Porites", "Porites evermanni"),
        ("ptua", "POC", "Pocillopora", "Pocillopora tuahiniensis"),
    ],
    columns=["species", "prefix", "genus", "species_name"],
)
_PREFIX_TO_SPECIES = dict(zip(SPECIES.prefix, SPECIES.species))
_GENUS_TO_SPECIES = dict(zip(SPECIES.genus, SPECIES.species))

TIMEPOINT_MONTH = {1: "2020-01", 2: "2020-03", 3: "2020-09", 4: "2020-11"}

_SAMPLE_RE = re.compile(r"(ACR|POC|POR)-(\d+)[-_]TP([1-4])(?!\d)")

_SEP = {"comma": ",", "tab": "\t"}

# Data types whose rows are genes and should get canonical gene IDs.
GENE_KEYED = {"gene", "epimachine", "gene_meth", "exon_cv", "exon_counts", "biomin"}


def default_data_dir() -> Path:
    """Directory holding ``manifest.csv``; override with ``STDM_DATA_DIR``."""
    env = os.environ.get("STDM_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data"


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------

def parse_sample_id(label: str) -> Optional[tuple[str, str, int]]:
    """Return ``(prefix, colony_number, timepoint)`` or ``None`` if no match."""
    m = _SAMPLE_RE.search(str(label))
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def canonical_sample_id(label: str) -> Optional[str]:
    """Map any MOSAiC sample label to ``PREFIX-colony-TPn``; ``None`` if not a sample."""
    parsed = parse_sample_id(label)
    if parsed is None:
        return None
    prefix, colony, tp = parsed
    return f"{prefix}-{colony}-TP{tp}"


def canonical_gene_id(gene_id: str) -> str:
    """Strip the ``gene-`` prefix and an A. pulchra ``-T<n>`` transcript suffix."""
    gid = str(gene_id)
    if gid.startswith("gene-"):
        gid = gid[len("gene-"):]
    return re.sub(r"-T\d+$", "", gid)


# ---------------------------------------------------------------------------
# Manifest and download cache
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Entry:
    key: str
    species: str
    data_type: str
    kind: str
    sep: str
    feature_cols: tuple[str, ...]
    annotation_cols: tuple[str, ...]
    large: bool
    description: str
    url: str

    @property
    def filename(self) -> str:
        suffix = Path(self.url.split("?")[0]).suffix or ".txt"
        return f"{self.key}{suffix}"


def load_manifest(data_dir: Optional[Path] = None) -> dict[str, Entry]:
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    entries = {}
    with open(data_dir / "manifest.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            split = lambda s: tuple(x for x in s.split(";") if x)
            entries[row["key"]] = Entry(
                key=row["key"],
                species=row["species"],
                data_type=row["data_type"],
                kind=row["kind"],
                sep=_SEP[row["sep"]],
                feature_cols=split(row["feature_cols"]),
                annotation_cols=split(row["annotation_cols"]),
                large=row["large"].strip().lower() == "yes",
                description=row["description"],
                url=row["url"],
            )
    return entries


def select(
    manifest: dict[str, Entry],
    species: Optional[Iterable[str]] = None,
    data_types: Optional[Iterable[str]] = None,
    keys: Optional[Iterable[str]] = None,
    include_large: bool = False,
) -> list[Entry]:
    species = set(species) if species else None
    data_types = set(data_types) if data_types else None
    keys = set(keys) if keys else None
    if keys:
        unknown = keys - set(manifest)
        if unknown:
            raise KeyError(f"Unknown manifest keys: {sorted(unknown)}")
    out = []
    for e in manifest.values():
        if keys and e.key not in keys:
            continue
        if species and e.species not in species:
            continue
        if data_types and e.data_type not in data_types:
            continue
        if e.large and not include_large and not keys:
            continue
        out.append(e)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_checksums(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        return {r["key"]: r for r in csv.DictReader(fh)}


def _write_checksums(path: Path, rows: dict[str, dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["key", "sha256", "bytes", "fetched"], lineterminator="\n")
        w.writeheader()
        for key in sorted(rows):
            w.writerow({k: rows[key][k] for k in ["key", "sha256", "bytes", "fetched"]})


def raw_path(entry: Entry, data_dir: Optional[Path] = None) -> Path:
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    return data_dir / "raw" / entry.filename


def fetch(
    entries: Iterable[Entry],
    data_dir: Optional[Path] = None,
    force: bool = False,
    update_checksums: bool = False,
    log=print,
) -> dict[str, Path]:
    """Download entries into ``data/raw/`` unless already cached.

    Each file's SHA-256 is compared with ``data/checksums.csv``. A mismatch
    means the upstream file changed since the checksum was recorded; it is
    reported as a warning, and the recorded value is replaced only when
    ``update_checksums`` is true. Files with no recorded checksum are added.
    """
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    (data_dir / "raw").mkdir(parents=True, exist_ok=True)
    ck_path = data_dir / "checksums.csv"
    checksums = _read_checksums(ck_path)
    changed = False
    paths = {}
    for e in entries:
        dest = raw_path(e, data_dir)
        if dest.exists() and not force:
            log(f"cached   {e.key}")
        else:
            log(f"fetching {e.key} ...")
            fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".part-")
            os.close(fd)
            try:
                req = urllib.request.Request(e.url, headers={"User-Agent": "stdm-fetch"})
                with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as out:
                    shutil.copyfileobj(r, out, 1 << 20)
                os.replace(tmp, dest)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        digest = _sha256(dest)
        rec = checksums.get(e.key)
        if rec is None:
            checksums[e.key] = dict(key=e.key, sha256=digest, bytes=dest.stat().st_size,
                                    fetched=date.today().isoformat())
            changed = True
        elif rec["sha256"] != digest:
            msg = (f"{e.key}: checksum differs from data/checksums.csv "
                   f"(recorded {rec['fetched']}); the upstream file may have changed")
            if update_checksums:
                checksums[e.key] = dict(key=e.key, sha256=digest, bytes=dest.stat().st_size,
                                        fetched=date.today().isoformat())
                changed = True
                log(f"updated  {msg}")
            else:
                warnings.warn(msg)
        paths[e.key] = dest
    if changed:
        _write_checksums(ck_path, checksums)
    return paths


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read(entry: Entry, data_dir: Optional[Path]) -> pd.DataFrame:
    path = raw_path(entry, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `stdm fetch --key {entry.key}` first")
    return pd.read_csv(path, sep=entry.sep, low_memory=False)


def load_matrix(
    key: str,
    data_dir: Optional[Path] = None,
    manifest: Optional[dict[str, Entry]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a feature x sample matrix.

    Returns ``(values, annotation)``. ``values`` is indexed by feature with one
    column per sample, named by :func:`canonical_sample_id` and sorted by
    species, colony and timepoint. ``annotation`` holds the non-sample columns
    (e.g. lncRNA coordinates) on the same index. Gene-keyed matrices get
    :func:`canonical_gene_id` feature IDs.
    """
    manifest = manifest or load_manifest(data_dir)
    e = manifest[key]
    if e.kind != "matrix":
        raise ValueError(f"{key} is a {e.kind}, not a matrix; use load_table()")
    df = _read(e, data_dir)

    # Files whose first column has no header (metabolomics, lipidomics)
    feature_cols = list(e.feature_cols) or [df.columns[0]]

    sample_map = {}
    other = []
    for c in df.columns:
        if c in feature_cols:
            continue
        cid = canonical_sample_id(c)
        if cid is None:
            other.append(c)
        else:
            sample_map[c] = cid
    dupes = pd.Series(list(sample_map.values())).duplicated()
    if dupes.any():
        raise ValueError(f"{key}: duplicate samples after harmonising IDs: "
                         f"{sorted(set(pd.Series(list(sample_map.values()))[dupes]))}")
    unexpected = [c for c in other if c not in e.annotation_cols]
    if unexpected:
        warnings.warn(f"{key}: columns that are neither samples nor listed annotation: {unexpected}")

    ids = df[feature_cols].astype(str)
    if e.data_type in GENE_KEYED and "gene_id" in feature_cols:
        ids["gene_id"] = ids["gene_id"].map(canonical_gene_id)
    index = ids.agg(":".join, axis=1) if len(feature_cols) > 1 else ids.iloc[:, 0]
    index.name = "feature_id"

    values = df[list(sample_map)].rename(columns=sample_map)
    values.index = index
    values = values[sort_samples(values.columns)]
    values = values.apply(pd.to_numeric, errors="coerce")

    annotation = df[other].copy()
    annotation.index = index
    return values, annotation


def load_table(key: str, data_dir: Optional[Path] = None,
               manifest: Optional[dict[str, Entry]] = None) -> pd.DataFrame:
    """Load a non-matrix file (orthologs, physiology, its2) as-is."""
    manifest = manifest or load_manifest(data_dir)
    e = manifest[key]
    if e.kind != "table":
        raise ValueError(f"{key} is a {e.kind}, not a table; use load_matrix()")
    return _read(e, data_dir)


def sort_samples(samples: Iterable[str]) -> list[str]:
    """Order canonical sample IDs by species prefix, colony number, timepoint."""
    def k(s):
        prefix, colony, tp = parse_sample_id(s)
        return prefix, int(colony), tp
    return sorted(samples, key=k)


# ---------------------------------------------------------------------------
# Orthologs and sample metadata
# ---------------------------------------------------------------------------

def load_orthologs(data_dir: Optional[Path] = None,
                   manifest: Optional[dict[str, Entry]] = None) -> pd.DataFrame:
    """One row per ortholog group with canonical gene IDs per species.

    The upstream table repeats some groups (one row per annotation hit); the
    first row is kept. Columns: ``group_id, type, apul, peve, ptua,
    protein_name, goslim_names``.
    """
    o = load_table("multi_orthologs", data_dir, manifest)
    o = o.drop_duplicates("group_id", keep="first").copy()
    for sp in SPECIES.species:
        o[sp] = o[sp].where(o[sp].isna(), o[sp].astype(str).map(canonical_gene_id))
    cols = ["group_id", "type", *SPECIES.species, "protein_name", "goslim_names"]
    return o[cols].reset_index(drop=True)


#: Data types that contribute a presence flag to the sample table.
AVAILABILITY_TYPES = ["gene", "mirna", "lncrna", "gene_meth", "exon_cv", "metabolomics", "lipidomics"]


def extraction_batch(sample_name: str) -> tuple[Optional[str], str]:
    """Infer ``(extraction_date, batch)`` from an RNA submission sample name.

    Some sample names carry an extraction date (``20230123_ACR-225_TP1``,
    ``291-20220211``); the rest are plain numbers. Batches are inferred, not
    documented upstream: dates from 2023 on are named by year and month, the
    scattered 2021-2022 dates are pooled as ``2021-22``, and undated names
    form ``undated``.
    """
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", str(sample_name))
    if not m:
        return None, "undated"
    year, month, day = m.groups()
    date = f"{year}-{month}-{day}"
    return date, (f"{year}-{month}" if int(year) >= 2023 else "2021-22")


def load_rna_metadata(data_dir: Optional[Path] = None,
                      manifest: Optional[dict[str, Entry]] = None) -> pd.DataFrame:
    """RNA submission metadata, one row per canonical sample ID.

    Columns: ``rna_sample_name``, ``plate``, ``rna_conc_ng_ul``,
    ``extraction_date`` and ``extraction_batch`` (see :func:`extraction_batch`).
    """
    raw = load_table("multi_rna_metadata", data_dir, manifest)
    ids = (raw["ColonyID"].str.strip() + "-" + raw["Timepoint"].str.strip()).map(canonical_sample_id)
    dates, batches = zip(*raw["SampleName"].map(extraction_batch))
    out = pd.DataFrame({
        "sample_id": ids,
        "rna_sample_name": raw["SampleName"].astype(str),
        "plate": raw["Plate"],
        "rna_conc_ng_ul": pd.to_numeric(raw["Conc-ng.uL"], errors="coerce"),
        "extraction_date": dates,
        "extraction_batch": batches,
    })
    if out["sample_id"].isna().any() or out["sample_id"].duplicated().any():
        raise ValueError("RNA metadata has unparseable or duplicated colony/timepoint labels")
    return out.set_index("sample_id")


def build_sample_metadata(data_dir: Optional[Path] = None,
                          manifest: Optional[dict[str, Entry]] = None) -> pd.DataFrame:
    """One row per colony x timepoint sampled for any molecular data type.

    Site comes from the physiology table (``colony_id_corr``). Boolean
    ``has_<type>`` columns record which data types include each sample; data
    types whose files are not cached locally are left out.
    """
    manifest = manifest or load_manifest(data_dir)
    present: dict[str, set[str]] = {}
    for dt in AVAILABILITY_TYPES:
        cols: set[str] = set()
        found = False
        for e in manifest.values():
            if e.data_type != dt or not raw_path(e, data_dir).exists():
                continue
            found = True
            header = pd.read_csv(raw_path(e, data_dir), sep=e.sep, nrows=0).columns
            cols |= {s for c in header if (s := canonical_sample_id(c))}
        if found:
            present[dt] = cols

    samples = sort_samples(set().union(*present.values())) if present else []
    rows = []
    for s in samples:
        prefix, colony, tp = parse_sample_id(s)
        rows.append(dict(sample_id=s, species=_PREFIX_TO_SPECIES[prefix],
                         colony_id=f"{prefix}-{colony}", timepoint=tp,
                         month=TIMEPOINT_MONTH[tp]))
    meta = pd.DataFrame(rows, columns=["sample_id", "species", "colony_id", "timepoint", "month"])

    phys_entry = manifest.get("multi_physiology")
    if phys_entry and raw_path(phys_entry, data_dir).exists():
        phys = load_table("multi_physiology", data_dir, manifest)
        site = (phys.groupby("colony_id_corr")["site"]
                .agg(lambda s: ";".join(sorted(set(s.dropna())))))
        meta.insert(3, "site", meta["colony_id"].map(site))
        multi = meta["site"].str.contains(";", na=False)
        if multi.any():
            warnings.warn(f"colonies recorded at more than one site: "
                          f"{sorted(meta.loc[multi, 'colony_id'].unique())}")

    for dt, cols in present.items():
        meta[f"has_{dt}"] = meta["sample_id"].isin(cols)

    rna_entry = manifest.get("multi_rna_metadata")
    if rna_entry and raw_path(rna_entry, data_dir).exists():
        rna = load_rna_metadata(data_dir, manifest)
        meta = meta.join(rna[["plate", "rna_conc_ng_ul", "extraction_date", "extraction_batch"]],
                         on="sample_id")

    geno_entry = manifest.get("multi_genotypes")
    if geno_entry and raw_path(geno_entry, data_dir).exists():
        meta = add_genotypes(meta, load_table("multi_genotypes", data_dir, manifest))
    return meta


# ---------------------------------------------------------------------------
# Genotype identity
# ---------------------------------------------------------------------------

IDENTITY_MODES = ("as-given", "drop-flagged", "one-per-genotype")


def add_genotypes(meta: pd.DataFrame, geno: pd.DataFrame) -> pd.DataFrame:
    """Add ``genotype_id`` and ``identity_flagged`` to sample metadata.

    ``geno`` is the WGBS-based sample-to-genotype table (``sample``,
    ``colony``, ``genotype``, ``flagged``). A sample is flagged when its own
    genotype differs from its colony's (a likely sample mix-up). Each colony's
    genotype is the most common genotype among its unflagged samples, and is
    given to all its samples, including those without WGBS. Colonies with no
    genotyped sample keep their colony ID as genotype.
    """
    g = geno.assign(sample=geno["sample"].map(canonical_sample_id),
                    flagged=geno["flagged"].astype(str).str.upper() == "TRUE")
    ok = g[~g.flagged]
    colony_geno = ok.groupby("colony")["genotype"].agg(lambda s: s.mode().iloc[0])
    out = meta.copy()
    out["genotype_id"] = out["colony_id"].map(colony_geno).fillna(out["colony_id"])
    out["identity_flagged"] = out["sample_id"].isin(set(g.loc[g.flagged, "sample"]))
    return out


def identity_keep(meta: pd.DataFrame, mode: str = "as-given") -> pd.Series:
    """Boolean per sample: keep it under the identity ``mode``.

    - ``as-given``: keep everything (colony labels as recorded).
    - ``drop-flagged``: drop samples whose genotype differs from their colony.
    - ``one-per-genotype``: also keep only one colony per genotype (the one
      with the most remaining samples; ties go to the lower colony number), so
      clonemates are not counted as independent colonies.

    ``meta`` is indexed by sample ID with ``colony_id``, and, for the last
    two modes, ``identity_flagged`` and ``genotype_id`` (from
    :func:`add_genotypes`; run ``stdm fetch --key multi_genotypes`` then
    ``stdm samples``).
    """
    if mode not in IDENTITY_MODES:
        raise ValueError(f"identity mode must be one of {IDENTITY_MODES}")
    keep = pd.Series(True, index=meta.index)
    if mode == "as-given":
        return keep
    if "identity_flagged" not in meta:
        raise ValueError("sample metadata has no genotype columns; "
                         "run `stdm fetch --key multi_genotypes` and `stdm samples`")
    keep &= ~meta["identity_flagged"].astype(bool)
    if mode == "one-per-genotype":
        kept = meta[keep]
        n = kept.groupby("colony_id").size()
        order = pd.DataFrame({"n": n, "num": [int(c.rsplit("-", 1)[1]) for c in n.index],
                              "genotype": kept.groupby("colony_id")["genotype_id"].first()})
        order = order.sort_values(["n", "num"], ascending=[False, True])
        chosen = set(order.groupby("genotype").head(1).index)
        keep &= meta["colony_id"].isin(chosen)
    return keep
