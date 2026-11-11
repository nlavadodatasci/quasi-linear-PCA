"""Tests of the qlpca package against numbers printed in the thesis and the prototype.

Targets come from tests/reference_values.json and are never edited to make tests pass.
Loadings are compared in absolute value (sign is convention-dependent).
"""
import json
import pathlib
import sys

import numpy as np
import pytest

from qlpca import QLPCA, ispline_basis, pseudo_indicator
from qlpca.datasets import load_st_gender
from qlpca.splines import interior_knots

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))
import qlpca_prototype as proto  # noqa: E402

REF = json.load(open(ROOT / "tests" / "reference_values.json", encoding="utf8"))
X, VARIABLES, _ = load_st_gender()
TOL = 0.005
CASES = [("qlpca1", 1), ("qlpca2", 2)]


@pytest.fixture(scope="module")
def fits():
    return {key: QLPCA(2, degree=1, n_knots=r).fit(X) for key, r in CASES}


def test_data_matches_variable_order():
    assert VARIABLES == REF["variables"]


def test_table_4_1_summary_statistics():
    t = REF["table_4_1_summary"]
    np.testing.assert_allclose(X.min(0), t["min"], atol=0.005)
    np.testing.assert_allclose(X.max(0), t["max"], atol=0.005)
    # the thesis prints m_top's mean as 19.4 (data: 19.42), i.e. rounded to the column's precision
    np.testing.assert_allclose(X.mean(0), t["mean"], atol=0.025)
    np.testing.assert_allclose(X.std(0, ddof=1), t["std"], atol=0.01)


# ------------------------------------------------------------------ splines
@pytest.mark.parametrize("degree,r", [(1, 0), (1, 1), (1, 2), (2, 2), (3, 3)])
def test_vectorised_basis_equals_prototype(degree, r):
    h = proto.zscore(X)[:, 3]
    G, knots = pseudo_indicator(h, degree, r)
    G_ref, knots_ref = proto.isplines(h, degree, r)
    np.testing.assert_allclose(knots, knots_ref, atol=1e-12)
    np.testing.assert_allclose(G, G_ref, atol=1e-12)


def test_ispline_properties():
    h = np.random.default_rng(0).normal(size=40)
    G, _ = pseudo_indicator(h, 1, 2)
    assert G.shape == (40, 3)
    assert np.all((G >= -1e-12) & (G <= 1 + 1e-12))
    np.testing.assert_allclose(G[h.argmax()], 1.0)
    np.testing.assert_allclose(G[h.argmin()], 0.0, atol=1e-12)


def test_degree1_one_knot_basis_explicit():
    """WCE paper: I_1 rises linearly m1 -> m2 then stays 1; I_2 is 0 up to m2, then rises to 1 at m3."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0])
    G, knots = pseudo_indicator(x, 1, 1)
    m1, m2, m3 = knots[0], knots[2], knots[-1]
    np.testing.assert_allclose(G[:, 0], np.clip((x - m1) / (m2 - m1), 0, 1), atol=1e-12)
    np.testing.assert_allclose(G[:, 1], np.clip((x - m2) / (m3 - m2), 0, 1), atol=1e-12)


def test_interior_knots_follow_matlab_prctile():
    x = np.arange(1.0, 12.0) ** 2
    assert len(interior_knots(x, 3)) == 3
    np.testing.assert_allclose(interior_knots(x, 1), np.percentile(x, 50, method="hazen"))
    # r = 2 -> percentiles 33 and 67 (MATLAB round)
    np.testing.assert_allclose(interior_knots(x, 2), np.percentile(x, [33, 67], method="hazen"))
    # r = 7 -> 12.5 rounds up to 13 in MATLAB (numpy's round would give 12)
    np.testing.assert_allclose(interior_knots(np.arange(100.0), 7)[0], np.percentile(np.arange(100.0), 13, method="hazen"))


# ------------------------------------------------------------ thesis tables
def test_table_4_3_pca_eigenvalues():
    ev = np.linalg.eigvalsh(np.corrcoef(X.T))[::-1]
    np.testing.assert_allclose(ev, REF["table_4_3_pca"]["eigenvalues"], atol=TOL)
    np.testing.assert_allclose(np.cumsum(ev) / ev.sum() * 100, REF["table_4_3_pca"]["cumulative_pct"], atol=0.01)


def test_table_4_4_pca_loadings_via_qlpca_zero_knots():
    m = QLPCA(4, degree=1, n_knots=0).fit(X)
    ref = np.abs(np.array(REF["table_4_4_pca_loadings"]))
    np.testing.assert_allclose(np.abs(m.loadings_), ref, atol=0.01)


def test_pca_communality_of_mf_funds():
    m = QLPCA(2, degree=1, n_knots=0).fit(X)
    j = VARIABLES.index("mf_funds")
    assert m.communalities_[j] == pytest.approx(REF["pca_communality_mf_funds_2pc"], abs=0.002)


@pytest.mark.parametrize("key,r", CASES)
def test_table_4_5_eigenvalues(fits, key, r):
    t = REF["table_4_5_qlpca"][key]
    np.testing.assert_allclose(fits[key].eigenvalues_, t["eigenvalues"], atol=TOL)
    assert fits[key].vaf_.sum() == pytest.approx(t["total_pct"], abs=0.01)
    np.testing.assert_allclose(fits[key].vaf_, t["pct_variance"], atol=0.05)


@pytest.mark.parametrize("key,r", CASES)
def test_table_4_6_absolute_loadings(fits, key, r):
    ref = np.abs(np.array(REF["table_4_6_loadings"][key]))
    np.testing.assert_allclose(np.abs(fits[key].loadings_), ref, atol=TOL)


@pytest.mark.parametrize("key,r", CASES)
def test_tables_4_7_4_8_communalities(fits, key, r):
    np.testing.assert_allclose(fits[key].communalities_, REF["communalities"][key], atol=TOL)


@pytest.mark.parametrize("key,r", CASES)
def test_matches_prototype(fits, key, r):
    o = proto.qlpca(X, p=2, degree=1, r=r)
    np.testing.assert_allclose(fits[key].eigenvalues_, o["eig"], atol=1e-6)
    np.testing.assert_allclose(np.abs(fits[key].loadings_), np.abs(o["A"]), atol=1e-4)


# --------------------------------------------------------------- properties
def test_degree1_zero_knots_equals_pca():
    m = QLPCA(3, degree=1, n_knots=0).fit(X)
    ev = np.linalg.eigvalsh(np.corrcoef(X.T))[::-1]
    np.testing.assert_allclose(m.all_eigenvalues_, ev, atol=1e-8)
    np.testing.assert_allclose(m.eigenvalues_, ev[:3], atol=1e-8)
    Z = (X - X.mean(0)) / X.std(0, ddof=1)
    # same component subspace as PCA: projector of the scores
    U, _, _ = np.linalg.svd(Z, full_matrices=False)
    P_pca = U[:, :3] @ U[:, :3].T
    S = m.scores_ / np.sqrt(len(X))
    np.testing.assert_allclose(S @ S.T, P_pca, atol=1e-6)


@pytest.mark.parametrize("init", ["matlab", "pca"])
@pytest.mark.parametrize("key,r", CASES)
def test_constraints(init, key, r):
    m = QLPCA(2, degree=1, n_knots=r, init=init).fit(X)
    n = len(X)
    np.testing.assert_allclose(m.scores_.T @ m.scores_, n * np.eye(2), atol=1e-8)
    np.testing.assert_allclose(m.transformed_.mean(0), 0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(m.transformed_, axis=0), np.sqrt(n), atol=1e-8)
    # VAF per component equals the eigenvalue of F'F/n; communalities sum to the same total
    np.testing.assert_allclose((m.loadings_ ** 2).sum(0), m.eigenvalues_, atol=1e-6)
    assert m.communalities_.sum() == pytest.approx(m.eigenvalues_.sum(), abs=1e-6)


@pytest.mark.parametrize("init", ["matlab", "pca"])
@pytest.mark.parametrize("r", [1, 2])
def test_loss_non_increasing(init, r):
    m = QLPCA(2, degree=1, n_knots=r, init=init).fit(X)
    h = m.loss_history_
    assert len(h) > 2
    assert np.all(np.diff(h) <= 1e-8), np.diff(h).max()


def test_sign_convention():
    m = QLPCA(2, degree=1, n_knots=2).fit(X)
    H = (X - X.mean(0)) / X.std(0, ddof=1)
    for j in range(X.shape[1]):
        assert np.corrcoef(m.transformed_[:, j], H[:, j])[0, 1] >= 0
    top = np.abs(m.loadings_).argmax(0)
    assert np.all(m.loadings_[top, [0, 1]] > 0)


def test_start_values_agree_for_two_components():
    a = QLPCA(2, 1, 1, init="matlab").fit(X)
    b = QLPCA(2, 1, 1, init="pca").fit(X)
    np.testing.assert_allclose(a.eigenvalues_, b.eigenvalues_, atol=0.005)


# ---------------------------------------------------- projection / reconstruction
@pytest.mark.parametrize("r", [0, 1, 2])
def test_projection_of_training_rows_returns_scores(r):
    m = QLPCA(2, degree=1, n_knots=r).fit(X)
    np.testing.assert_allclose(m.transform_variables(X), m.transformed_, atol=1e-8)
    np.testing.assert_allclose(m.transform(X), m.scores_, atol=1e-8)


def test_projection_extrapolates_linearly():
    m = QLPCA(2, degree=1, n_knots=1).fit(X)
    x = X[:1].copy()
    j = 0
    lo = x.copy(); lo[0, j] = X[:, j].min() - 1.0
    lo2 = x.copy(); lo2[0, j] = X[:, j].min() - 2.0
    hi = x.copy(); hi[0, j] = X[:, j].max() + 1.0
    hi2 = x.copy(); hi2[0, j] = X[:, j].max() + 2.0
    f = lambda z: m.transform_variables(z)[0, j]
    assert f(lo2) - f(lo) == pytest.approx(f(lo) - m.transform_variables(
        np.where(np.arange(X.shape[1]) == j, X[:, j].min(), X[:1]))[0, j], abs=1e-9)
    assert f(hi2) - f(hi) == pytest.approx(f(hi) - m.transform_variables(
        np.where(np.arange(X.shape[1]) == j, X[:, j].max(), X[:1]))[0, j], abs=1e-9)


def test_reconstruction_with_all_components_returns_F_and_H():
    m = QLPCA(X.shape[1], degree=1, n_knots=1).fit(X)
    Fh = m.scores_ @ m.loadings_.T
    np.testing.assert_allclose(Fh, m.transformed_, atol=1e-8)
    Hh = m.inverse_transform(m.scores_)
    # f_j is invertible where monotone; check those variables against H, and that
    # the reconstruction re-transforms to F for every variable
    monotone = []
    for j in range(X.shape[1]):
        _, f = m._curve(j)
        monotone.append(bool(np.all(np.diff(f) > 0) or np.all(np.diff(f) < 0)))
    assert any(monotone)
    np.testing.assert_allclose(Hh[:, monotone], X[:, monotone], atol=1e-7)
    np.testing.assert_allclose(m.transform_variables(Hh), m.transformed_, atol=1e-7)


def test_not_fitted_and_degree_guard():
    with pytest.raises(AttributeError):
        QLPCA().transform(X)
    m = QLPCA(2, degree=2, n_knots=1).fit(X)
    with pytest.raises(NotImplementedError):
        m.transform(X)
