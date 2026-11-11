"""Tests of the modernised default behaviour (docs/DESIGN.md D1-D10), chapter 4 data."""
import warnings

import numpy as np
import pytest

from qlpca import QLPCA
from qlpca.core import ConvergenceWarning, _als, orthonormal_basis
from qlpca.datasets import load_st_gender
from qlpca.splines import bspline_basis, ispline_basis

X, VARIABLES, _ = load_st_gender()
M = X.shape[1]


def total_vaf(m):
    return 100 * m.eigenvalues_.sum() / X.shape[1]


# --------------------------------------------------- D1: reaches the optimum
@pytest.mark.parametrize("r,p,target", [(1, 2, 68.90), (2, 2, 76.33), (1, 3, 84.02)])
def test_default_reaches_legacy_optimum(r, p, target):
    m = QLPCA(p, n_knots=r, random_state=0).fit(X)
    assert total_vaf(m) == pytest.approx(target, abs=0.005)


def test_default_reaches_at_least_90_30_for_two_knots_three_components():
    m = QLPCA(3, n_knots=2, random_state=0).fit(X)
    assert total_vaf(m) >= 90.30 - 0.005
    # legacy start gets stuck at 90.00 or 90.30 depending on column order: never better
    leg = QLPCA(3, n_knots=2, legacy=True).fit(X)
    assert total_vaf(m) >= total_vaf(leg) - 1e-9


def test_pca_start_alone_agrees_with_legacy_for_two_components():
    a = QLPCA(2, n_knots=1, legacy=True).fit(X)
    b = QLPCA(2, n_knots=1, n_init=0).fit(X)
    np.testing.assert_allclose(a.eigenvalues_, b.eigenvalues_, atol=0.005)
    np.testing.assert_allclose(np.abs(a.loadings_), np.abs(b.loadings_), atol=0.005)


def test_default_zero_knots_is_pca():
    m = QLPCA(3, n_knots=0, random_state=0).fit(X)
    ev = np.linalg.eigvalsh(np.corrcoef(X.T))[::-1]
    np.testing.assert_allclose(m.all_eigenvalues_, ev, atol=1e-8)


@pytest.mark.parametrize("r", [1, 2])
def test_default_constraints_and_loss(r):
    m = QLPCA(2, n_knots=r, random_state=0).fit(X)
    n = len(X)
    np.testing.assert_allclose(m.scores_.T @ m.scores_, n * np.eye(2), atol=1e-8)
    np.testing.assert_allclose(m.transformed_.mean(0), 0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(m.transformed_, axis=0), np.sqrt(n), atol=1e-8)
    np.testing.assert_allclose((m.loadings_ ** 2).sum(0), m.eigenvalues_, atol=1e-8)
    assert np.all(np.diff(m.loss_history_) <= 1e-10)
    # loss = m - sum of the first p eigenvalues of F'F/n (D3)
    assert m.loss_ == pytest.approx(M - m.eigenvalues_.sum(), abs=1e-9)
    assert m.loss_ == pytest.approx(((m.scores_ @ m.loadings_.T - m.transformed_) ** 2).sum() / n, abs=1e-9)


def test_sign_convention():
    m = QLPCA(2, n_knots=2, random_state=0).fit(X)
    H = (X - X.mean(0)) / X.std(0, ddof=1)
    for j in range(M):
        assert np.corrcoef(m.transformed_[:, j], H[:, j])[0, 1] >= 0
    assert np.all(m.loadings_[np.abs(m.loadings_).argmax(0), [0, 1]] > 0)


# ------------------------------------------------------------- invariance
def test_invariant_to_column_order():
    perm = np.random.default_rng(1).permutation(M)
    a = QLPCA(3, n_knots=2, random_state=5).fit(X)
    b = QLPCA(3, n_knots=2, random_state=5).fit(X[:, perm])
    np.testing.assert_allclose(a.eigenvalues_, b.eigenvalues_, atol=1e-6)
    np.testing.assert_allclose(np.abs(a.loadings_[perm]), np.abs(b.loadings_), atol=1e-4)


def test_invariant_to_affine_rescaling():
    rng = np.random.default_rng(2)
    a_ = rng.uniform(0.1, 50, M)
    b_ = rng.uniform(-100, 100, M)
    a = QLPCA(3, n_knots=2, random_state=5).fit(X)
    b = QLPCA(3, n_knots=2, random_state=5).fit(X * a_ + b_)
    np.testing.assert_allclose(a.eigenvalues_, b.eigenvalues_, atol=1e-8)
    np.testing.assert_allclose(a.transformed_, b.transformed_, atol=1e-6)


# --------------------------------------------- D2: three bases, same fit
def test_b_spline_i_spline_intercept_and_standardised_i_splines_agree():
    n, p = X.shape[0], 2
    H = (X - X.mean(0)) / X.std(0, ddof=1)
    est = QLPCA(p, n_knots=2).fit(X)
    kn = est.knots_

    def run(make):
        Qs = [orthonormal_basis(make(j)) for j in range(M)]

        def project(j, y):
            f = Qs[j] @ (Qs[j].T @ y)
            f = f - f.mean()
            return f * np.sqrt(n) / np.linalg.norm(f)

        U, _, _ = np.linalg.svd(H, full_matrices=False)
        X0 = np.sqrt(n) * U[:, :p]
        F0 = H * (np.sqrt(n) / np.linalg.norm(H, axis=0))
        return _als(project, F0, X0, H.T @ X0 / n, p, 1e-12, 500)

    r_b = run(lambda j: bspline_basis(H[:, j], 1, kn[j]))
    r_i = run(lambda j: ispline_basis(H[:, j], 1, kn[j]))
    r_ii = run(lambda j: np.c_[np.ones(n), ispline_basis(H[:, j], 1, kn[j])])
    r_z = run(lambda j: (lambda G: (G - G.mean(0)) / G.std(0, ddof=1))(ispline_basis(H[:, j], 1, kn[j])))
    for r in (r_i, r_ii, r_z):
        np.testing.assert_allclose(r["F"], r_b["F"], atol=1e-8)
        np.testing.assert_allclose(r["ev"], r_b["ev"], atol=1e-8)


# ----------------------------------------------------------------- D7: ties
def test_ties_fit_without_error():
    Xr = np.round(X, 2)
    assert any(len(np.unique(Xr[:, j])) < len(Xr) for j in range(M)), "data set has no ties"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = QLPCA(2, n_knots=2, random_state=0).fit(Xr)
    assert np.all(np.isfinite(m.loadings_))
    # rounding to two decimals barely changes the solution
    ref = QLPCA(2, n_knots=2, random_state=0).fit(X)
    np.testing.assert_allclose(m.eigenvalues_, ref.eigenvalues_, atol=0.02)
    with pytest.raises(ValueError):
        QLPCA(2, n_knots=2, legacy=True).fit(Xr)


def test_coinciding_knots_are_merged_with_warning():
    Xt = X.copy()
    Xt[:, 0] = np.r_[np.zeros(10), np.arange(1.0, 6.0)]  # percentiles 33 and 67 coincide at 0
    with pytest.warns(UserWarning, match="merged"):
        m = QLPCA(2, n_knots=2, random_state=0).fit(Xt)
    assert np.isfinite(m.loss_)


# ------------------------------------------------------------- D5: monotone
@pytest.mark.parametrize("r", [1, 2])
def test_monotone(r):
    free = QLPCA(2, n_knots=r, random_state=0).fit(X)
    mono = QLPCA(2, n_knots=r, monotone=True, random_state=0).fit(X)
    assert all(np.all(c >= -1e-12) for c in mono.coef_)
    assert mono.loss_ >= free.loss_ - 1e-9
    assert np.all(np.diff(mono.loss_history_) <= 1e-9)
    for j in range(M):
        _, f = mono._curve(j)
        assert np.all(np.diff(f) >= -1e-9)
        o = np.argsort(X[:, j])
        assert np.all(np.diff(mono.transformed_[o, j]) >= -1e-9)
    np.testing.assert_allclose(mono.scores_.T @ mono.scores_, len(X) * np.eye(2), atol=1e-8)


def test_monotone_inverse_is_unique_for_increasing_f():
    m = QLPCA(M, n_knots=1, monotone=True, n_init=0).fit(X)
    Hh = m.inverse_transform(m.scores_)
    strict = [bool(np.all(np.diff(m._curve(j)[1]) > 1e-9)) for j in range(M)]
    assert any(strict)
    np.testing.assert_allclose(Hh[:, strict], X[:, strict], atol=1e-6)


# --------------------------------------------------------- reproducibility
def test_random_state_gives_identical_results():
    a = QLPCA(3, n_knots=2, random_state=42).fit(X)
    b = QLPCA(3, n_knots=2, random_state=42).fit(X)
    np.testing.assert_array_equal(a.scores_, b.scores_)
    np.testing.assert_array_equal(a.losses_, b.losses_)
    assert len(a.losses_) == 11  # PCA start + n_init random starts


# -------------------------------------------------------------- D8, D9, D10
def test_per_variable_degree_and_knots_and_explicit_knots():
    nk = [1, 2] * 5 + [1]
    m = QLPCA(2, n_knots=nk, random_state=0).fit(X)
    assert [len(k) - 4 for k in m.knots_] == nk
    explicit = [np.percentile(X[:, j], [25, 75]) for j in range(M)]
    m2 = QLPCA(2, knots=explicit, random_state=0).fit(X)
    assert all(len(k) == 6 for k in m2.knots_)
    m3 = QLPCA(2, degree=[1] * 10 + [2], n_knots=1, random_state=0).fit(X)
    with pytest.raises(NotImplementedError):
        m3.transform(X)
    with pytest.raises(ValueError):
        QLPCA(2, n_knots=[1, 2]).fit(X)


@pytest.mark.parametrize("kw", [dict(), dict(monotone=True)])
def test_projection_of_training_rows_returns_scores(kw):
    m = QLPCA(2, n_knots=2, random_state=0, **kw).fit(X)
    np.testing.assert_allclose(m.transform_variables(X), m.transformed_, atol=1e-8)
    np.testing.assert_allclose(m.transform(X), m.scores_, atol=1e-8)


def test_reconstruction_with_all_components():
    m = QLPCA(M, n_knots=1, random_state=0).fit(X)
    np.testing.assert_allclose(m.scores_ @ m.loadings_.T, m.transformed_, atol=1e-8)
    Hh = m.inverse_transform(m.scores_)
    np.testing.assert_allclose(m.transform_variables(Hh), m.transformed_, atol=1e-7)


def test_convergence_warning():
    with pytest.warns(ConvergenceWarning):
        QLPCA(2, n_knots=2, n_init=0, max_iter=2).fit(X)
