"""ExtrapolationWarning (DESIGN D11): new rows outside the training range are flagged,
and the values returned are unchanged by the warning."""
import warnings

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from qlpca import ExtrapolationWarning, QLPCA
from qlpca.datasets import CYLINDER_VARIABLES, make_cylinder

# transform(X_test) computed with qlpca before ExtrapolationWarning was added (same split, same fit)
BEFORE_ROWS_0_1 = np.array([[-1.1691754043892166, 0.28673783735772995],
                            [0.8159529230803593, -1.0025895710355348]])
BEFORE_ROW_38 = np.array([96.78960445609204, -43.61795368150922])
BEFORE_COLUMN_SUMS = np.array([103.03404620344232, -35.46851264830452])


@pytest.fixture(scope="module")
def split():
    X, names = make_cylinder(n=200, random_state=0)
    X_train, X_test = train_test_split(X, test_size=0.25, random_state=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = QLPCA(n_components=2, random_state=0).fit(X_train)
    return model, X_train, X_test


def test_no_warning_for_training_rows_or_fit(split):
    model, X_train, _ = split
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExtrapolationWarning)
        QLPCA(n_components=2, n_init=1, random_state=0).fit(X_train)
        QLPCA(n_components=2, n_init=1, random_state=0).fit_transform(X_train)
        S = model.transform(X_train)
        F = model.transform_variables(X_train)
    assert np.allclose(S, model.scores_) and np.allclose(F, model.transformed_)


def test_warning_on_test_rows_names_count_and_variables(split):
    model, X_train, X_test = split
    outside = (X_test < X_train.min(0)) | (X_test > X_train.max(0))
    n_out = int(outside.any(1).sum())
    assert n_out == 4
    with pytest.warns(ExtrapolationWarning, match=rf"^{n_out} of {len(X_test)} observation") as rec:
        model.transform(X_test)
    assert len(rec) == 1                               # transform warns once, not twice
    msg = str(rec[0].message)
    listed = msg.split("variable(s) ")[1].split(";")[0].split(", ")
    assert listed == [f"x{j}" for j in np.where(outside.any(0))[0]]   # array input: x0, x1, ...
    with pytest.warns(ExtrapolationWarning):
        model.transform_variables(X_test)


def test_warning_names_dataframe_columns(split):
    pd = pytest.importorskip("pandas")
    _, X_train, X_test = split
    model = QLPCA(n_components=2, n_init=1, random_state=0).fit(pd.DataFrame(X_train, columns=CYLINDER_VARIABLES))
    with pytest.warns(ExtrapolationWarning, match=r"b/a.*2\*a\*pi/b\^2"):
        model.transform(pd.DataFrame(X_test, columns=CYLINDER_VARIABLES))


def test_values_unchanged_by_the_warning(split):
    model, _, X_test = split
    with pytest.warns(ExtrapolationWarning):
        S = model.transform(X_test)
    np.testing.assert_allclose(S[[0, 1]], BEFORE_ROWS_0_1, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(S[38], BEFORE_ROW_38, rtol=1e-9)
    np.testing.assert_allclose(S.sum(0), BEFORE_COLUMN_SUMS, rtol=1e-9)
    assert np.abs(S).max() > 30 * np.abs(model.scores_).max()   # the heavy-tail problem itself
