"""Diagnostics for a fitted qlPCA (thesis section 3.1.4 and 3.1.5; docs/SPEC.md section 5)."""
import numpy as np


def vaf(loadings):
    """Variance accounted for by each component: sum_j a_js^2.

    Equals the s-th eigenvalue of the correlation matrix F'F/n of the transformed variables.
    """
    return (np.asarray(loadings, float) ** 2).sum(0)


def vaf_percent(loadings):
    """VAF per component as a percentage of the number of variables (VAF_s * 100 / m).

    Refers to the *transformed* matrix; comparing with PCA percentages needs caution
    (thesis pp. 60-61).
    """
    L = np.asarray(loadings, float)
    return 100.0 * vaf(L) / L.shape[0]


def communalities(loadings):
    """Communality of each variable: sum_s a_js^2. Rule of thumb: keep variables >= 0.25."""
    return (np.asarray(loadings, float) ** 2).sum(1)


def _corr(a, b):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def piecewise_loadings(model, X, min_points=3):
    """Piecewise loadings of a degree-1 qlPCA (thesis eqs. 3.7-3.8).

    For every variable j and every segment between consecutive distinct knots, the
    vectors are truncated to the observations whose h_j lies in the (closed) segment, and
    the loading on component s is the correlation of the scores x_s with the *original*
    variable on that segment. Because f_j is linear on the segment,

        corr(x_s, h_j) = sign(slope) * corr(x_s, f_j)    (truncated vectors),

    so the sign of the transformation's slope tells whether the original variable is
    positively or negatively related to the component on that segment.

    Returns a list with one entry per variable: a dict with ``edges`` (segment limits in the
    units of the data), ``slope_sign``, ``loadings`` (n_segments, p; NaN where a segment has
    fewer than ``min_points`` observations or a constant vector), ``n`` (observations per
    segment) and ``loadings_via_f`` (the same computed through f_j and the slope sign).
    """
    model._check()
    X = np.asarray(X, float)
    S = model.transform(X)
    F = model.transform_variables(X)
    Hs = (X - model.mean_) / model.std_
    out = []
    for j in range(X.shape[1]):
        t, f = model._curve(j)
        edges = t * model.std_[j] + model.mean_[j]
        sgn = np.sign(np.diff(f))
        L, L2, cnt = [], [], []
        for i in range(len(edges) - 1):
            sel = (Hs[:, j] >= t[i] - 1e-9) & (Hs[:, j] <= t[i + 1] + 1e-9)  # standardised scale: no rounding loss
            cnt.append(int(sel.sum()))
            if sel.sum() < min_points:
                L.append([np.nan] * S.shape[1])
                L2.append([np.nan] * S.shape[1])
                continue
            L.append([_corr(S[sel, s], X[sel, j]) for s in range(S.shape[1])])
            L2.append([sgn[i] * _corr(S[sel, s], F[sel, j]) for s in range(S.shape[1])])
        out.append(dict(edges=edges, slope_sign=sgn, loadings=np.array(L), loadings_via_f=np.array(L2),
                        n=np.array(cnt)))
    return out
