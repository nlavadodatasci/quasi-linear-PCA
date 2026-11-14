"""scikit-learn compatibility (docs/PACKAGING.md, P2-B).

Each excluded estimator check is listed with its reason in docs/PACKAGING.md,
section "Estimator checks excluded".
"""
from sklearn.utils.estimator_checks import parametrize_with_checks

from qlpca import QLPCA

EXPECTED_FAILED_DEFAULT = {}
_TIES = ("legacy=True requires unique values within each variable (the 2012 MATLAB code); "
         "this check fits integer-valued data with ties")
EXPECTED_FAILED_LEGACY = {
    "check_estimators_dtypes": _TIES,
    "check_positive_only_tag_during_fit": _TIES,
}


def _expected_failed(estimator):
    return EXPECTED_FAILED_LEGACY if estimator.legacy else EXPECTED_FAILED_DEFAULT


@parametrize_with_checks([QLPCA(n_init=2), QLPCA(legacy=True)], expected_failed_checks=_expected_failed)
def test_sklearn_compatible(estimator, check):
    check(estimator)


def test_feature_names_out_clone_and_pipeline():
    import numpy as np
    from sklearn.base import clone
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from qlpca.datasets import make_cylinder

    X, _ = make_cylinder(n=40, random_state=1)
    model = QLPCA(n_components=3, n_init=1, random_state=0)
    assert list(model.fit(X).get_feature_names_out()) == ["qlpca0", "qlpca1", "qlpca2"]
    assert clone(model).get_params() == model.get_params()
    assert not hasattr(clone(model), "scores_")
    pipe = make_pipeline(StandardScaler(), QLPCA(n_components=2, n_init=1, random_state=0))
    Z = pipe.fit_transform(X)
    assert Z.shape == (40, 2) and np.allclose(Z, pipe.transform(X))
    assert repr(QLPCA(n_knots=2)) == "QLPCA(n_knots=2)"
