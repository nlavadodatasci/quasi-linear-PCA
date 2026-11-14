"""Spline bases for qlPCA (thesis section 3.1.1 and the MATLAB listings of 3.2).

The pseudo-indicator matrix of a variable is its I-spline basis. With B-splines
B_1..B_{w+1} of degree v on the knot sequence, I_i = sum_{q>i} B_q (i = 1..w,
w = v + r); the first B-spline is dropped, so the constant is not in the span
(it is recovered by centring in the estimator).
"""
import warnings

import numpy as np
from scipy.interpolate import BSpline


def matlab_prctile(x, q):
    """Percentiles as MATLAB ``prctile`` (Hazen definition)."""
    return np.percentile(x, q, method="hazen")


def interior_knots(x, n_knots):
    """Interior knots at the percentiles round(100*i/(r+1)), i = 1..r.

    Rounding is MATLAB's (half away from zero), not numpy's banker's rounding.
    """
    if n_knots == 0:
        return np.array([])
    q = np.floor(100.0 * np.arange(1, n_knots + 1) / (n_knots + 1) + 0.5)
    return np.sort(matlab_prctile(np.asarray(x, float), q))


def knot_sequence(x, degree, n_knots, interior=None):
    """min repeated degree+1 times, interior knots, max repeated degree+1 times.

    ``interior`` gives explicit interior knots instead of percentiles. Interior knots
    that coincide (ties in the data) or fall on/outside the boundary are merged or
    dropped, with a warning, which reduces the basis dimension (DESIGN D7).
    """
    x = np.asarray(x, float)
    lo, hi = x.min(), x.max()
    inner = interior_knots(x, n_knots) if interior is None else np.sort(np.asarray(interior, float))
    keep = np.unique(inner[(inner > lo) & (inner < hi)])
    if len(keep) < len(inner):
        warnings.warn("coinciding or boundary interior knots were merged; the basis dimension is reduced",
                      stacklevel=3)
    return np.r_[[lo] * (degree + 1), keep, [hi] * (degree + 1)]


def bspline_basis(x, degree, knots):
    """Dense (n, len(knots) - degree - 1) B-spline design matrix.

    ``x`` must lie in [knots[degree], knots[-degree-1]]; the right endpoint is included.
    """
    x = np.asarray(x, float)
    return BSpline.design_matrix(x, np.asarray(knots, float), degree).toarray()


def ispline_basis(x, degree, knots):
    """I-spline basis (n, degree + r): reverse cumulative sum of the B-splines, first dropped."""
    B = bspline_basis(x, degree, knots)
    return np.cumsum(B[:, ::-1], axis=1)[:, ::-1][:, 1:]


def pseudo_indicator(x, degree=1, n_knots=1, knots=None):
    """Pseudo-indicator matrix G (n, degree + n_knots) of a continuous variable.

    Returns ``(G, knots)``; pass ``knots`` to evaluate on other points.
    """
    if knots is None:
        knots = knot_sequence(x, degree, n_knots)
    return ispline_basis(x, degree, knots), knots
