"""Input validation and DataFrame support (docs/PACKAGING.md, P5)."""
import warnings

import numpy as np
import pandas as pd
import pytest

from qlpca import QLPCA, SmallSampleWarning
from qlpca.datasets import load_st_gender, make_cylinder


@pytest.fixture
def cyl():
    X, names = make_cylinder(n=40, random_state=0)
    return X, names


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nan_inf_rejected_with_variable_name(cyl, bad):
    X, names = cyl
    df = pd.DataFrame(X, columns=names)
    df.iloc[3, 4] = bad
    with pytest.raises(ValueError, match=r"NaN or inf .*a\*b.*missing"):
        QLPCA(n_init=1).fit(df)


def test_nan_in_transform(cyl):
    X, _ = cyl
    model = QLPCA(n_init=1, random_state=0).fit(X)
    Xn = X[:2].copy()
    Xn[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        model.transform(Xn)


def test_constant_column_named(cyl):
    X, names = cyl
    df = pd.DataFrame(X, columns=names).assign(const=1.0)
    with pytest.raises(ValueError, match="constant variable.*const"):
        QLPCA(n_init=1).fit(df)


def test_fewer_than_two_observations(cyl):
    with pytest.raises(ValueError, match="minimum of 2"):
        QLPCA(n_components=1).fit(cyl[0][:1])


def test_legacy_ties_message_names_variable():
    X, names, _ = load_st_gender()
    with pytest.raises(ValueError, match="ties in .*default mode"):
        QLPCA(legacy=True).fit(pd.DataFrame(X.round(0), columns=names))


def test_small_sample_warning(cyl):
    X, _ = cyl
    with pytest.warns(SmallSampleWarning, match="fewer than 5"):
        QLPCA(n_init=1, n_knots=3).fit(X[:15])      # w = 4, 15 < 20
    with warnings.catch_warnings():
        warnings.simplefilter("error", SmallSampleWarning)
        QLPCA(n_init=1, n_knots=2).fit(X[:15])      # w = 3, 15 = 15: no warning


def test_chapter4_fits_raise_no_small_sample_warning():
    X, _, _ = load_st_gender()                      # n = 15: thesis fits with 1 and 2 knots
    with warnings.catch_warnings():
        warnings.simplefilter("error", SmallSampleWarning)
        for r in (1, 2):
            QLPCA(legacy=True, n_knots=r).fit(X)


def test_dataframe_names_kept(cyl):
    X, names = cyl
    df = pd.DataFrame(X, columns=names)
    model = QLPCA(n_init=1, random_state=0).fit(df)
    assert list(model.feature_names_in_) == names
    assert model.n_features_in_ == 12
    ref = QLPCA(n_init=1, random_state=0).fit(X)
    assert np.allclose(model.scores_, ref.scores_)
    assert np.allclose(model.transform(df.iloc[:5]), ref.transform(X[:5]))
    out = QLPCA(n_init=1, random_state=0).set_output(transform="pandas").fit(df).transform(df)
    assert list(out.columns) == ["qlpca0", "qlpca1"]
    with pytest.raises(ValueError, match="feature names"):
        model.transform(df[names[::-1]])
