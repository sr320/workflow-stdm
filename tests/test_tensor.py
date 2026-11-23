"""Tests for building the ortholog x colony x timepoint tensor (stdm.tensor)."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stdm import mosaic
from stdm.tensor import ExpressionTensor, build_expression_tensor, size_factors

REPO_DATA = Path(__file__).resolve().parents[1] / "data"

# Two colonies per species; POR-2 is missing TP4.
COLONIES = {"apul": ["ACR-1", "ACR-2"], "peve": ["POR-1", "POR-2"], "ptua": ["POC-1", "POC-2"]}
GENE_PREFIX = {"apul": "FUN_", "peve": "gene-Peve_", "ptua": "gene-Poc_"}
ORTHO_PREFIX = {"apul": "FUN_", "peve": "Peve_", "ptua": "Poc_"}


def _samples(sp):
    out = [f"{c}-TP{t}" for c in COLONIES[sp] for t in range(1, 5)]
    if sp == "peve":
        out.remove("POR-2-TP4")
    return out


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    dd = tmp_path / "data"
    (dd / "raw").mkdir(parents=True)

    rows = []
    for sp in ["apul", "peve", "ptua"]:
        samples = _samples(sp)
        # 8 genes: g0-g4 well expressed, g5 low in this species only for peve,
        # g6-g7 extra genes not in any ortholog group
        base = np.array([200, 400, 800, 1600, 300, 300, 50, 70], dtype=float)
        if sp == "peve":
            base[5] = 1
        scale = rng.uniform(0.5, 2.0, size=len(samples))  # library size differences
        counts = np.round(base[:, None] * scale[None, :] * rng.uniform(0.8, 1.2, (8, len(samples))))
        df = pd.DataFrame(counts.astype(int), columns=samples)
        df.insert(0, "gene_id", [f"{GENE_PREFIX[sp]}{i:03d}" for i in range(8)])
        df.to_csv(dd / "raw" / f"{sp}_gene.csv", index=False)
        rows.append((f"{sp}_gene", sp, "gene", "matrix", "comma", "gene_id", "", f"{sp}_gene.csv"))

    # Orthologs: OG_0-OG_5 three-way; OG_6 reuses apul gene 004 (shared -> both dropped);
    # OG_7 is apul_peve only; OG_8 points at a ptua gene not in the counts.
    with open(dd / "raw" / "multi_orthologs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group_id", "apul", "peve", "ptua", "type", "protein_name", "goslim_names"])
        for i in range(6):
            w.writerow([f"OG_{i}", f"FUN_{i:03d}-T1", f"Peve_{i:03d}", f"Poc_{i:03d}",
                        "three_way", f"P{i}", ""])
        w.writerow(["OG_6", "FUN_004-T2", "Peve_006", "Poc_006", "three_way", "P6", ""])
        w.writerow(["OG_7", "FUN_007-T1", "Peve_007", "", "apul_peve", "P7", ""])
        w.writerow(["OG_8", "FUN_007-T1", "Peve_007", "Poc_999", "three_way", "P8", ""])
    rows.append(("multi_orthologs", "multi", "orthologs", "table", "comma", "group_id", "",
                 "multi_orthologs.csv"))

    with open(dd / "raw" / "multi_physiology.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["colony_id", "colony_id_corr", "site"])
        for sp, cols in COLONIES.items():
            w.writerow([cols[0], cols[0], "Manava"])
            w.writerow([cols[1], cols[1], "Mahana"])
    rows.append(("multi_physiology", "multi", "physiology", "table", "comma", "", "",
                 "multi_physiology.csv"))

    with open(dd / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "species", "data_type", "kind", "sep", "feature_cols",
                    "annotation_cols", "large", "description", "url"])
        for key, sp, dt, kind, sep, f, a, fname in rows:
            w.writerow([key, sp, dt, kind, sep, f, a, "no", "test",
                        f"https://example.invalid/{fname}"])
    return dd


def test_size_factors_recover_library_scaling():
    base = np.array([10, 50, 100, 400.0])
    counts = pd.DataFrame(np.outer(base, [1, 2, 4]), columns=["a", "b", "c"])
    sf = size_factors(counts)
    np.testing.assert_allclose(sf / sf.iloc[0], [1, 2, 4])
    np.testing.assert_allclose(np.prod(sf), 1.0)


def test_build_shape_filtering_and_mask(data_dir):
    t = build_expression_tensor(data_dir, min_count=10, min_fraction=0.5)
    # OG_4 and OG_6 share FUN_004 -> dropped; OG_5 is low in peve -> filtered;
    # OG_7 is not three-way; OG_8 has a missing ptua gene.
    assert list(t.groups.index) == ["OG_0", "OG_1", "OG_2", "OG_3"]
    assert t.params["group_counts"] == {
        "ortholog_groups_of_type": 8,
        "with_genes_in_all_count_matrices": 7,
        "after_dropping_shared_genes": 5,
        "after_expression_filter": 4,
    }
    assert t.shape == (4, 6, 4)
    assert list(t.colonies.index) == ["ACR-1", "ACR-2", "POR-1", "POR-2", "POC-1", "POC-2"]
    assert t.colonies.loc["POR-2", "site"] == "Mahana"
    assert t.missing_samples() == ["POR-2-TP4"]
    assert np.isnan(t.values[:, 3, 3]).all()
    assert t.mask.sum() == 4 * (6 * 4 - 1)


def test_values_are_log2_median_of_ratios(data_dir):
    t = build_expression_tensor(data_dir)
    raw, _ = mosaic.load_matrix("ptua_gene", data_dir)
    expected = np.log2(raw / size_factors(raw) + 1).loc["Poc_002", "POC-2-TP3"]
    got = t.values[t.groups.index.get_loc("OG_2"), t.colonies.index.get_loc("POC-2"), 2]
    assert got == pytest.approx(expected)


def test_species_centering(data_dir):
    t = build_expression_tensor(data_dir, center="species")
    for sp in ["apul", "peve", "ptua"]:
        idx = np.flatnonzero(t.colonies.species.to_numpy() == sp)
        block = t.values[:, idx, :].reshape(len(t.groups), -1)
        np.testing.assert_allclose(np.nanmean(block, axis=1), 0, atol=1e-12)
    assert np.isnan(t.values[:, 3, 3]).all()


def test_to_matrix_and_round_trip(data_dir, tmp_path):
    t = build_expression_tensor(data_dir)
    m = t.to_matrix()
    assert m.shape == (4, 23)
    assert "POR-2-TP4" not in m.columns

    t.save(tmp_path / "out")
    t2 = ExpressionTensor.load(tmp_path / "out")
    np.testing.assert_array_equal(t.values, t2.values)
    assert list(t2.groups.index) == list(t.groups.index)
    assert t2.colonies.equals(t.colonies)
    assert t2.params["normalization"] == "log2-mor"


def test_invalid_options(data_dir):
    with pytest.raises(ValueError):
        build_expression_tensor(data_dir, normalization="tpm")
    with pytest.raises(ValueError):
        build_expression_tensor(data_dir, center="gene")
    with pytest.raises(ValueError):
        build_expression_tensor(data_dir, min_fraction=2)


real = pytest.mark.skipif(not (REPO_DATA / "raw" / "apul_gene.csv").exists(),
                          reason="MOSAiC data not fetched; run `stdm fetch`")


@real
def test_real_tensor():
    t = build_expression_tensor(REPO_DATA)
    assert t.shape[1:] == (30, 4)
    assert t.shape[0] > 5000
    assert sorted(t.missing_samples()) == ["POC-201-TP4", "POR-236-TP3", "POR-236-TP4"]
    assert t.colonies.groupby(["species", "site"]).size().eq(5).all()
    assert not t.groups[["apul", "peve", "ptua"]].apply(lambda s: s.duplicated()).any().any()
