"""Qualitative replication of thesis Table 3.3 (cylinder problem).

The original random draws are lost (a, b ~ U(0,1) is inferred), so the targets are
averages over simulated draws with a tolerance of 3 percentage points, not exact values.
"""
import numpy as np
import pytest

from qlpca import QLPCA
from qlpca.datasets import make_cylinder

TABLE_3_3 = {  # noise -> (PCA, qlPCA2, qlPCA3), % variance explained by two components
    0.0: (82.15, 97.77, 98.43),
    0.10: (80.18, 96.90, 97.68),
    0.25: (72.62, 92.59, 94.41),
}


@pytest.fixture(scope="module")
def simulated():
    out = {}
    for noise in TABLE_3_3:
        rows = []
        for seed in range(10):
            X, _ = make_cylinder(51, noise, random_state=seed)
            rows.append([100 * QLPCA(2, n_knots=r, n_init=2 if r else 0, random_state=0).fit(X).eigenvalues_.sum() / 12
                         for r in (0, 2, 3)])
        out[noise] = np.array(rows)
    return out


@pytest.mark.parametrize("noise", list(TABLE_3_3))
def test_mean_variance_explained_is_close_to_table_3_3(simulated, noise):
    np.testing.assert_allclose(simulated[noise].mean(0), TABLE_3_3[noise], atol=3.0)


def test_orderings_of_table_3_3_hold_in_every_draw(simulated):
    for noise, v in simulated.items():
        assert np.all(v[:, 1] > v[:, 0] + 5)           # qlPCA far better than PCA
        assert np.all(v[:, 2] >= v[:, 1] - 0.2)        # three knots at least as good as two
    means = np.array([simulated[k].mean(0) for k in TABLE_3_3])
    assert np.all(np.diff(means, axis=0) < 0)           # more noise, less variance explained


def test_linearised_cylinder_matrix_has_rank_two():
    X, _ = make_cylinder(51, 0, random_state=0)
    L = np.log(X)
    L[:, 7], L[:, 8] = np.log(np.tan(X[:, 7])), -np.log(np.tan(X[:, 8]))
    s = np.linalg.svd(L - L.mean(0), compute_uv=False)
    assert s[2] < 1e-8 * s[0]
