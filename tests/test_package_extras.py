"""Tests of diagnostics.py, make_cylinder and plots.py."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from qlpca import QLPCA  # noqa: E402
from qlpca.datasets import load_st_gender, make_cylinder  # noqa: E402
from qlpca.diagnostics import communalities, piecewise_loadings, vaf, vaf_percent  # noqa: E402
from qlpca.plots import biplot, scree_plot, transformation_plots  # noqa: E402

X, VARIABLES, COUNTRIES = load_st_gender()


@pytest.fixture(scope="module")
def model():
    return QLPCA(2, n_knots=2, random_state=0).fit(X)


# -------------------------------------------------------------- diagnostics
def test_vaf_equals_eigenvalues_and_communalities_sum(model):
    np.testing.assert_allclose(vaf(model.loadings_), model.eigenvalues_, atol=1e-8)
    np.testing.assert_allclose(vaf_percent(model.loadings_), 100 * model.eigenvalues_ / X.shape[1], atol=1e-8)
    np.testing.assert_allclose(communalities(model.loadings_), model.communalities_)
    assert communalities(model.loadings_).sum() == pytest.approx(vaf(model.loadings_).sum())


def test_thesis_vaf_percent_table_4_5():
    m = QLPCA(2, n_knots=1, legacy=True).fit(X)
    np.testing.assert_allclose(vaf_percent(m.loadings_), [41.65, 27.25], atol=0.05)


def test_piecewise_loadings_sign_rule_and_shape(model):
    pl = piecewise_loadings(model, X)
    assert len(pl) == X.shape[1]
    for d in pl:
        assert d["loadings"].shape == (len(d["edges"]) - 1, 2)
        assert len(d["slope_sign"]) == len(d["edges"]) - 1
        # eqs. 3.7-3.8: corr with h = sign(slope) * corr with f, on the truncated vectors
        np.testing.assert_allclose(d["loadings"], d["loadings_via_f"], atol=1e-9, equal_nan=True)
        assert np.all(np.abs(d["loadings"][~np.isnan(d["loadings"])]) <= 1 + 1e-12)


def test_piecewise_loadings_linear_case_matches_pearson():
    """With no interior knot there is one segment: the loading is the correlation of the
    scores with the whole variable."""
    m = QLPCA(2, n_knots=0, random_state=0).fit(X)
    pl = piecewise_loadings(m, X)
    S = m.scores_
    for j, d in enumerate(pl):
        assert d["loadings"].shape == (1, 2)
        np.testing.assert_allclose(d["loadings"][0], [np.corrcoef(S[:, s], X[:, j])[0, 1] for s in range(2)],
                                   atol=1e-9)
        # for p components of PCA this correlation is the loading itself (up to sign)
        np.testing.assert_allclose(np.abs(d["loadings"][0]), np.abs(m.loadings_[j]), atol=1e-6)


def test_piecewise_loadings_sign_flips_with_decreasing_segment():
    """A V-shaped transformation has slopes of opposite sign on the two segments, and the
    piecewise loadings of that variable change sign accordingly."""
    rng = np.random.default_rng(3)
    t = rng.normal(size=80)
    Xv = np.c_[t, np.abs(t) + 0.1 * rng.normal(size=80), np.abs(t) + 0.1 * rng.normal(size=80)]
    m = QLPCA(1, n_knots=1, random_state=0).fit(Xv)
    d = piecewise_loadings(m, Xv)[0]
    assert d["slope_sign"][0] * d["slope_sign"][1] < 0
    assert d["loadings"][0, 0] * d["loadings"][1, 0] < 0
    assert d["n"].sum() >= len(Xv)  # the interior knot belongs to both closed segments


def test_piecewise_loadings_requires_degree_one():
    m = QLPCA(2, degree=2, n_knots=1, random_state=0).fit(X)
    with pytest.raises(NotImplementedError):
        piecewise_loadings(m, X)


# ----------------------------------------------------------------- cylinder
def test_make_cylinder_shape_and_formulas():
    Xc, names = make_cylinder(51, 0, random_state=0)
    assert Xc.shape == (51, 12) and len(names) == 12
    a, b = Xc[:, 0], Xc[:, 1]
    assert np.all((a > 0) & (a < 1) & (b > 0) & (b < 1))
    np.testing.assert_allclose(Xc[:, 2], 2 * np.sqrt(b * np.pi))
    np.testing.assert_allclose(Xc[:, 5], a * b ** 2 / (2 * np.pi))
    np.testing.assert_allclose(np.tan(Xc[:, 7]), a * np.sqrt(np.pi) / (2 * np.sqrt(b)))
    np.testing.assert_allclose(1 / np.tan(Xc[:, 8]), a * np.sqrt(np.pi) / (2 * np.sqrt(b)))
    np.testing.assert_allclose(Xc[:, 11], 2 * a * np.pi / b ** 2)
    assert Xc[:, 5].max() <= 1 / (2 * np.pi)  # thesis figures: at most about 0.14


def test_make_cylinder_noise_on_linearised_scale():
    X0, _ = make_cylinder(51, 0, random_state=7)
    for noise in (0.10, 0.25):
        X1, _ = make_cylinder(51, noise, random_state=7)
        assert np.all(np.isfinite(X1))
        # relative noise sd on the log scale of variable 1 (a): about `noise` x sd(log a)
        d = np.log(X1[:, 0]) - np.log(X0[:, 0])
        assert d.std(ddof=1) == pytest.approx(noise * np.log(X0[:, 0]).std(ddof=1), rel=0.5)
    # angles stay in (0, pi/2)
    assert np.all((X1[:, 7:9] > 0) & (X1[:, 7:9] < np.pi / 2))


def test_make_cylinder_reproducible():
    a, _ = make_cylinder(51, 0.1, random_state=5)
    b, _ = make_cylinder(51, 0.1, random_state=5)
    np.testing.assert_array_equal(a, b)


def test_qlpca_beats_pca_on_cylinder_without_noise():
    """Qualitative target (thesis Table 3.3: PCA 82%, qlPCA 97-98%); n and draws differ."""
    gains = []
    for seed in range(4):
        Xc, _ = make_cylinder(51, 0, random_state=seed)
        pca = QLPCA(2, n_knots=0).fit(Xc)
        ql = QLPCA(2, n_knots=2, random_state=0).fit(Xc)
        gains.append(ql.eigenvalues_.sum() - pca.eigenvalues_.sum())
        assert 100 * ql.eigenvalues_.sum() / 12 > 90
    assert min(gains) > 0


# -------------------------------------------------------------------- plots
def test_plots_run(model):
    f1, ax = plt.subplots()
    scree_plot(model, ax=ax, label="qlPCA", reference=np.linalg.eigvalsh(np.corrcoef(X.T))[::-1],
               reference_label="PCA")
    f2 = transformation_plots(model, X, VARIABLES, ncols=4)
    f3, ax3 = plt.subplots()
    biplot(model, names=VARIABLES, labels=COUNTRIES, ax=ax3)
    assert len(f2.axes) >= X.shape[1]
    assert len(ax.lines) >= 2
    plt.close("all")
