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
    """Percentiles as MATLAB ``prctile`` (Hazen definition).

    Parameters
    ----------
    x : array-like
        Data.
    q : float or array-like
        Percentiles, between 0 and 100.

    Returns
    -------
    float or ndarray
        The percentiles of ``x``.
    """
    return np.percentile(x, q, method="hazen")


def interior_knots(x, n_knots):
    """Interior knots at the percentiles round(100 i / (r + 1)), i = 1..r.

    Rounding is MATLAB's (half away from zero), not numpy's banker's rounding.

    Parameters
    ----------
    x : array-like
        Data of one variable.
    n_knots : int
        Number r of interior knots.

    Returns
    -------
    ndarray of shape (n_knots,)
        Sorted interior knots, in the units of ``x``.
    """
    if n_knots == 0:
        return np.array([])
    q = np.floor(100.0 * np.arange(1, n_knots + 1) / (n_knots + 1) + 0.5)
    return np.sort(matlab_prctile(np.asarray(x, float), q))


def knot_sequence(x, degree, n_knots, interior=None):
    """Full knot sequence: min repeated degree+1 times, interior knots, max repeated degree+1 times.

    Interior knots that coincide (ties in the data) or fall on or outside the boundary are merged or
    dropped, with a warning, which reduces the basis dimension (DESIGN D7).

    Parameters
    ----------
    x : array-like
        Data of one variable.
    degree : int
        Spline degree.
    n_knots : int
        Number of interior knots at the percentiles (ignored if ``interior`` is given).
    interior : array-like, optional
        Explicit interior knots.

    Returns
    -------
    ndarray
        The knot sequence.
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
    """Dense B-spline design matrix.

    Parameters
    ----------
    x : array-like of shape (n,)
        Points, within [knots[degree], knots[-degree-1]]; the right endpoint is included.
    degree : int
        Spline degree.
    knots : array-like
        Full knot sequence.

    Returns
    -------
    ndarray of shape (n, len(knots) - degree - 1)
        B-spline basis functions evaluated at ``x``.
    """
    x = np.asarray(x, float)
    return BSpline.design_matrix(x, np.asarray(knots, float), degree).toarray()


def ispline_basis(x, degree, knots):
    """I-spline basis: reverse cumulative sum of the B-splines, the first column dropped.

    Parameters
    ----------
    x : array-like of shape (n,)
        Points.
    degree : int
        Spline degree.
    knots : array-like
        Full knot sequence.

    Returns
    -------
    ndarray of shape (n, degree + r)
        Monotone I-spline columns, from 0 to 1.
    """
    B = bspline_basis(x, degree, knots)
    return np.cumsum(B[:, ::-1], axis=1)[:, ::-1][:, 1:]


def pseudo_indicator(x, degree=1, n_knots=1, knots=None):
    """Pseudo-indicator matrix G of a continuous variable (its I-spline basis).

    Parameters
    ----------
    x : array-like of shape (n,)
        Data of one variable.
    degree : int, default=1
        Spline degree.
    n_knots : int, default=1
        Number of interior knots at the percentiles.
    knots : array-like, optional
        Full knot sequence; pass it to evaluate the basis on other points.

    Returns
    -------
    G : ndarray of shape (n, degree + n_knots)
        Pseudo-indicator matrix.
    knots : ndarray
        The knot sequence used.
    """
    if knots is None:
        knots = knot_sequence(x, degree, n_knots)
    return ispline_basis(x, degree, knots), knots
