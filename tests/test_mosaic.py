"""Tests for the MOSAiC data layer (stdm.mosaic)."""

import csv
import warnings
from pathlib import Path

import pandas as pd
import pytest

from stdm import mosaic

REPO_DATA = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label, expected", [
    ("ACR-145-TP4", "ACR-145-TP4"),
    ("1A10_ACR-145_TP4", "ACR-145-TP4"),   # miRNA columns
    ("ACR-139_TP1", "ACR-139-TP1"),        # physiology / ITS2 sample_id
    ("POR-69-TP2", "POR-69-TP2"),
    ("2G2_POC-57_TP4", "POC-57-TP4"),
    ("gene_id", None),
    ("Chr", None),
    ("ACR-145-TP5", None),
    ("ACR-145-TP12", None),
])
def test_canonical_sample_id(label, expected):
    assert mosaic.canonical_sample_id(label) == expected


@pytest.mark.parametrize("gene_id, expected", [
    ("FUN_000185-T1", "FUN_000185"),
    ("FUN_000185", "FUN_000185"),
    ("gene-Peve_00000032", "Peve_00000032"),
    ("Peve_00000032", "Peve_00000032"),
    ("gene-Pocillopora_meandrina_HIv1___RNAseq.g20905.t1",
     "Pocillopora_meandrina_HIv1___RNAseq.g20905.t1"),
    ("Pocillopora_meandrina_HIv1___TS.g25680.t1b",
     "Pocillopora_meandrina_HIv1___TS.g25680.t1b"),
])
def test_canonical_gene_id(gene_id, expected):
    assert mosaic.canonical_gene_id(gene_id) == expected


def test_sort_samples_orders_colonies_numerically():
    got = mosaic.sort_samples(["POR-216-TP1", "POR-69-TP2", "ACR-145-TP2", "ACR-145-TP1"])
    assert got == ["ACR-145-TP1", "ACR-145-TP2", "POR-69-TP2", "POR-216-TP1"]


# ---------------------------------------------------------------------------
# Fixture data directory served over file:// URLs
# ---------------------------------------------------------------------------

def _write(path, text):
    path.write_text(text)
    return path


@pytest.fixture
def data_dir(tmp_path):
    src = tmp_path / "upstream"
    src.mkdir()
    _write(src / "gene.csv",
           "gene_id,POR-69-TP1,POR-69-TP2,POR-216-TP1\n"
           "gene-Peve_001,1,2,3\n"
           "gene-Peve_002,4,5,6\n")
    _write(src / "mirna.txt",
           "Name\t1A1_POR-216_TP1\t1A2_POR-69_TP2\n"
           "Cluster_1\t10\t20\n")
    _write(src / "lnc.txt",
           "Geneid\tChr\tStart\tPOR-69-TP1\n"
           "lncRNA_1\tscaf1\t100\t7\n")
    _write(src / "metab.csv",
           ",POR-69-TP1,ACR-145-TP1\n"
           "Ethanolamine,1.5,2.5\n")
    _write(src / "orthologs.csv",
           "group_id,apul,peve,ptua,type,protein_name,goslim_names\n"
           "OG_1,FUN_001-T1,Peve_001,Poc.g1.t1,three_way,P1,a\n"
           "OG_1,FUN_001-T1,Peve_001,Poc.g1.t1,three_way,P1-dup,a\n"
           "OG_2,FUN_002-T2,Peve_002,,apul_peve,P2,b\n")
    _write(src / "phys.csv",
           "colony_id,colony_id_corr,site,sample_id\n"
           "POR-69,POR-69,Manava,POR-69_TP1\n"
           "POR-216,POR-216,Mahana,POR-216_TP1\n"
           "ACR-145,ACR-145,Manava,ACR-145_TP1\n")

    dd = tmp_path / "data"
    dd.mkdir()
    rows = [
        ("peve_gene", "peve", "gene", "matrix", "comma", "gene_id", "", "no", src / "gene.csv"),
        ("peve_mirna", "peve", "mirna", "matrix", "tab", "Name", "", "no", src / "mirna.txt"),
        ("peve_lncrna", "peve", "lncrna", "matrix", "tab", "Geneid", "Chr;Start", "no", src / "lnc.txt"),
        ("multi_metabolomics", "multi", "metabolomics", "matrix", "comma", "", "", "no", src / "metab.csv"),
        ("multi_orthologs", "multi", "orthologs", "table", "comma", "group_id", "", "no", src / "orthologs.csv"),
        ("multi_physiology", "multi", "physiology", "table", "comma", "", "", "no", src / "phys.csv"),
        ("peve_cpg", "peve", "cpg", "matrix", "comma", "CpG", "", "yes", src / "gene.csv"),
    ]
    with open(dd / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "species", "data_type", "kind", "sep", "feature_cols",
                    "annotation_cols", "large", "description", "url"])
        for *fields, path in rows:
            w.writerow([*fields, "test", path.as_uri()])
    return dd


@pytest.fixture
def fetched(data_dir):
    m = mosaic.load_manifest(data_dir)
    mosaic.fetch(mosaic.select(m), data_dir=data_dir, log=lambda *_: None)
    return data_dir, m


# ---------------------------------------------------------------------------
# Manifest and fetch
# ---------------------------------------------------------------------------

def test_select_skips_large_unless_asked(data_dir):
    m = mosaic.load_manifest(data_dir)
    assert "peve_cpg" not in {e.key for e in mosaic.select(m)}
    assert "peve_cpg" in {e.key for e in mosaic.select(m, include_large=True)}
    assert [e.key for e in mosaic.select(m, keys=["peve_cpg"])] == ["peve_cpg"]
    assert {e.key for e in mosaic.select(m, data_types=["mirna"])} == {"peve_mirna"}
    with pytest.raises(KeyError):
        mosaic.select(m, keys=["nope"])


def test_fetch_caches_and_records_checksums(fetched):
    data_dir, m = fetched
    assert (data_dir / "raw" / "peve_gene.csv").exists()
    ck = pd.read_csv(data_dir / "checksums.csv")
    assert set(ck.key) == {e.key for e in mosaic.select(m)}
    assert ck.sha256.str.len().eq(64).all()

    calls = []
    mosaic.fetch(mosaic.select(m, keys=["peve_gene"]), data_dir=data_dir, log=calls.append)
    assert calls == ["cached   peve_gene"]


def test_fetch_warns_when_upstream_changes(fetched):
    data_dir, m = fetched
    raw = data_dir / "raw" / "peve_gene.csv"
    raw.write_text(raw.read_text() + "gene-Peve_003,7,8,9\n")
    with pytest.warns(UserWarning, match="checksum differs"):
        mosaic.fetch(mosaic.select(m, keys=["peve_gene"]), data_dir=data_dir, log=lambda *_: None)
    before = pd.read_csv(data_dir / "checksums.csv").set_index("key").sha256["peve_gene"]

    mosaic.fetch(mosaic.select(m, keys=["peve_gene"]), data_dir=data_dir,
                 update_checksums=True, log=lambda *_: None)
    after = pd.read_csv(data_dir / "checksums.csv").set_index("key").sha256["peve_gene"]
    assert before != after


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_load_matrix_harmonises_samples_and_genes(fetched):
    data_dir, m = fetched
    v, a = mosaic.load_matrix("peve_gene", data_dir, m)
    assert list(v.index) == ["Peve_001", "Peve_002"]
    assert list(v.columns) == ["POR-69-TP1", "POR-69-TP2", "POR-216-TP1"]
    assert v.loc["Peve_002", "POR-216-TP1"] == 6
    assert a.shape == (2, 0)


def test_load_matrix_mirna_labels(fetched):
    data_dir, m = fetched
    v, _ = mosaic.load_matrix("peve_mirna", data_dir, m)
    assert list(v.columns) == ["POR-69-TP2", "POR-216-TP1"]
    assert v.loc["Cluster_1", "POR-216-TP1"] == 10


def test_load_matrix_keeps_annotation(fetched):
    data_dir, m = fetched
    v, a = mosaic.load_matrix("peve_lncrna", data_dir, m)
    assert list(a.columns) == ["Chr", "Start"]
    assert a.loc["lncRNA_1", "Chr"] == "scaf1"
    assert list(v.columns) == ["POR-69-TP1"]


def test_load_matrix_unnamed_feature_column(fetched):
    data_dir, m = fetched
    v, _ = mosaic.load_matrix("multi_metabolomics", data_dir, m)
    assert list(v.index) == ["Ethanolamine"]
    assert list(v.columns) == ["ACR-145-TP1", "POR-69-TP1"]


def test_load_matrix_rejects_duplicate_samples(fetched):
    data_dir, m = fetched
    raw = data_dir / "raw" / "peve_mirna.txt"
    raw.write_text("Name\t1A1_POR-216_TP1\t9Z9_POR-216_TP1\nCluster_1\t1\t2\n")
    with pytest.raises(ValueError, match="duplicate samples"):
        mosaic.load_matrix("peve_mirna", data_dir, m)


def test_load_orthologs_dedupes_and_canonicalises(fetched):
    data_dir, m = fetched
    o = mosaic.load_orthologs(data_dir, m)
    assert list(o.group_id) == ["OG_1", "OG_2"]
    assert o.loc[0, "apul"] == "FUN_001"
    assert o.loc[1, "apul"] == "FUN_002"
    assert pd.isna(o.loc[1, "ptua"])


def test_build_sample_metadata(fetched):
    data_dir, m = fetched
    meta = mosaic.build_sample_metadata(data_dir, m).set_index("sample_id")
    assert list(meta.index) == ["ACR-145-TP1", "POR-69-TP1", "POR-69-TP2", "POR-216-TP1"]
    assert meta.loc["POR-216-TP1", "site"] == "Mahana"
    assert meta.loc["POR-69-TP2", "month"] == "2020-03"
    assert meta.loc["POR-69-TP2", "has_mirna"]
    assert not meta.loc["POR-69-TP2", "has_lncrna"]
    assert not meta.loc["ACR-145-TP1", "has_gene"]
    assert meta.loc["ACR-145-TP1", "has_metabolomics"]


# ---------------------------------------------------------------------------
# Checks against the real cached data (skipped until `stdm fetch` has run)
# ---------------------------------------------------------------------------

real = pytest.mark.skipif(not (REPO_DATA / "raw" / "apul_gene.csv").exists(),
                          reason="MOSAiC data not fetched; run `stdm fetch`")


@real
def test_real_sample_table():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        meta = mosaic.build_sample_metadata(REPO_DATA)
    assert len(meta) == 117
    assert meta.groupby("species").size().to_dict() == {"apul": 40, "peve": 38, "ptua": 39}
    per_site = meta.drop_duplicates("colony_id").groupby(["species", "site"]).size()
    assert (per_site == 5).all() and len(per_site) == 6


@real
def test_real_orthologs_match_gene_matrices():
    o = mosaic.load_orthologs(REPO_DATA)
    for sp, min_frac in [("apul", 0.95), ("peve", 0.999), ("ptua", 0.999)]:
        genes, _ = mosaic.load_matrix(f"{sp}_gene", REPO_DATA)
        assert o[sp].dropna().isin(genes.index).mean() >= min_frac


def test_genotypes_and_identity_modes():
    meta = pd.DataFrame({"sample_id": [f"ACR-{c}-TP{t}" for c in (1, 2, 3) for t in (1, 2)],
                         "colony_id": [f"ACR-{c}" for c in (1, 2, 3) for _ in (1, 2)]})
    geno = pd.DataFrame({
        "sample": ["ACR-1-TP1", "ACR-1-TP2", "ACR-2-TP1", "ACR-2-TP2", "ACR-3-TP1"],
        "colony": ["ACR-1", "ACR-1", "ACR-2", "ACR-2", "ACR-3"],
        # ACR-1 and ACR-2 are clonemates; ACR-2-TP2 is a mix-up.
        "genotype": ["G1", "G1", "G1", "G9", "G2"],
        "flagged": ["FALSE", "FALSE", "FALSE", "TRUE", "FALSE"]})
    m = mosaic.add_genotypes(meta, geno).set_index("sample_id")
    assert list(m.genotype_id) == ["G1", "G1", "G1", "G1", "G2", "G2"]  # ACR-3-TP2 has no WGBS
    assert m.identity_flagged.sum() == 1 and m.identity_flagged["ACR-2-TP2"]
    assert mosaic.identity_keep(m, "as-given").all()
    assert list(m.index[~mosaic.identity_keep(m, "drop-flagged")]) == ["ACR-2-TP2"]
    keep = mosaic.identity_keep(m, "one-per-genotype")
    # ACR-1 keeps more samples than ACR-2 after the flag, so it represents G1.
    assert sorted(m[keep].colony_id.unique()) == ["ACR-1", "ACR-3"]
    with pytest.raises(ValueError):
        mosaic.identity_keep(m, "bogus")
