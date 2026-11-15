"""Quick checks of the princals comparison (optional: skipped if pyGifi is not installed).

Run from the repository root:  python -m pytest comparison/princals/tests
"""
import hashlib

import numpy as np
import pytest

pygifi = pytest.importorskip("pygifi")
import compare as c  # noqa: E402

pytestmark = pytest.mark.comparison
MAN = c.load_manifest()
LOCAL = ["cylinder_0", "cylinder_10", "cylinder_25", "st_gender", "neumann"]   # no download needed


@pytest.mark.parametrize("name", LOCAL)
def test_inputs_match_manifest(name):
    df = c.load_dataset(name, MAN)
    assert list(df.columns) == MAN["inputs"][name]["columns"]
    assert hashlib.md5(c._csv_bytes(df)).hexdigest() == MAN["inputs"][name]["md5"]


@pytest.mark.parametrize("name", LOCAL)
def test_manifest_knots_are_qlpca_quantile_knots(name):
    df = c.load_dataset(name, MAN)
    for r in ("1", "2"):
        for col in df.columns:
            k = np.asarray(MAN["knots"][name][r][col])
            np.testing.assert_array_equal(k, c.quantile_knots(df[col].to_numpy(), int(r)))
            assert np.all((k > df[col].min()) & (k < df[col].max()))


@pytest.mark.parametrize("name,run", [("st_gender", "knots1"), ("neumann", "knots2"), ("cylinder_0", "knots1")])
def test_same_knots_same_solution(name, run):
    df = c.load_dataset(name, MAN)
    q, p = c.run_qlpca(df, run, MAN, name), c.run_pygifi(df, run, MAN, name)
    assert c.within(c.compare(q, p, MAN["ndim"]), MAN["tolerances"]["unrestricted"])


def test_monotone_degree_1_same_solution():
    df = c.load_dataset("neumann", MAN)
    q, p = c.run_qlpca(df, "monotone_deg1", MAN, "neumann"), c.run_pygifi(df, "monotone_deg1", MAN, "neumann")
    assert c.within(c.compare(q, p, MAN["ndim"]), MAN["tolerances"]["monotone"])


def test_princals_does_not_discretise_numeric_columns():
    df = c.load_dataset("st_gender", MAN).reset_index(drop=True)
    np.testing.assert_array_equal(np.asarray(pygifi.make_numeric(df), float), df.to_numpy())
    p = c.run_pygifi(df, "knots1", MAN, "st_gender")
    for j in range(df.shape[1]):          # as many distinct transformed values as data values
        assert len(np.unique(p["transform"][:, j].round(12))) == len(np.unique(df.iloc[:, j]))


@pytest.mark.parametrize("name", list(MAN["datasets"]))
def test_r_ground_truth_shapes(name):
    rgt = c.load_r_ground_truth(name)
    if rgt is None:
        pytest.skip(f"no R ground truth for {name} yet")
    n, m = MAN["inputs"][name]["n"], len(MAN["inputs"][name]["columns"])
    for run in c.runs_for(MAN, name):
        r = rgt[run]
        assert r["loadings"].shape == (m, MAN["ndim"]), run
        assert r["objectscores"].shape == (n, MAN["ndim"]) and r["transform"].shape == (n, m)
        assert r["evals"].shape == (m,)


@pytest.mark.parametrize("name", LOCAL)
def test_r_ground_truth_if_available(name):
    rgt = c.load_r_ground_truth(name)
    if rgt is None:
        pytest.skip(f"no R ground truth for {name} yet")
    assert rgt["meta"]["input_md5"] == MAN["inputs"][name]["md5"]
    df = c.load_dataset(name, MAN)
    for run in ("knots1", "knots2"):
        q = c.run_qlpca(df, run, MAN, name)
        assert c.within(c.compare(q, rgt[run], MAN["ndim"]), MAN["tolerances"]["unrestricted"]), run
