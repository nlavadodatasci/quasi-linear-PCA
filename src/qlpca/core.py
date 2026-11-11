"""The qlPCA estimator (thesis chapter 3; design decisions in docs/DESIGN.md).

Minimises sigma(X, F) = (1/n) sum_j |X a_j - f_j|^2 subject to X'X = nI, where each
f_j is a spline of degree ``degree`` with ``n_knots`` interior knots in the
standardised variable h_j, by alternating least squares.

Two modes:

default       modernised algorithm: full B-spline basis centred (D2), PCA start plus
              random orthonormal starts keeping the lowest loss (D1), truncated-SVD
              update of (X, A) (D3), projector update of f_j (D4), optional monotone
              transformations (D5), ties allowed (D7).
``legacy``    the 2012 MATLAB behaviour: standardised I-spline columns, start from the
              first p columns of G, eigen-decomposition update, single start, unique
              values required. Reproduces thesis Tables 4.3-4.8.

Sign convention (D6): each transformed variable is oriented so that
corr(f_j, h_j) >= 0 and each component so that its loading of largest absolute
value is positive. For V-shaped transformations corr(f_j, h_j) is close to zero and
the orientation of that variable is fragile: read it through the transformation
plot, not through the sign of the loading.
"""
import warnings

import numpy as np
from scipy.optimize import nnls

from .splines import bspline_basis, ispline_basis, knot_sequence


class ConvergenceWarning(UserWarning):
    """The loss change did not fall below ``tol`` within ``max_iter`` iterations."""


def _zscore_cols(M):
    mean = M.mean(0)
    std = M.std(0, ddof=1)
    std = np.where(std == 0, 1.0, std)
    return (M - mean) / std, mean, std


def _inv_sqrt_sym(S):
    lam, U = np.linalg.eigh(S)
    return (U / np.sqrt(lam)) @ U.T


def _per_variable(value, m, name):
    if np.ndim(value) == 0:
        return [int(value)] * m
    value = list(value)
    if len(value) != m:
        raise ValueError(f"{name} must be a scalar or have one value per variable")
    return [int(v) for v in value]


def orthonormal_basis(B, rtol=1e-10):
    """Orthonormal basis (rank-revealing) of the column span of B after centring the columns."""
    Bc = B - B.mean(0)
    U, s, _ = np.linalg.svd(Bc, full_matrices=False)
    return U[:, : int((s > rtol * s[0]).sum())]


def _unit(f, n):
    """Centre f and scale to norm sqrt(n); None if f is (numerically) constant."""
    f = f - f.mean()
    nrm = np.linalg.norm(f)
    if nrm < 1e-12:
        return None
    return f * (np.sqrt(n) / nrm)


def _als(project, F0, X, A, p, tol, max_iter):
    """Two-block ALS (D3, D4): f_j = project(j, X a_j), then (X, A) from the truncated SVD of F.

    ``project(j, y)`` returns the best admissible f_j for target y (centred, norm sqrt(n)),
    or None when the projection is degenerate; f_j is then kept.
    """
    n, m = F0.shape
    F = F0.copy()
    history, prev, kept, converged = [], np.inf, False, False
    for _ in range(max_iter):
        for j in range(m):
            f = project(j, X @ A[j])
            if f is None:
                kept = True
            else:
                F[:, j] = f
        U, S, Vt = np.linalg.svd(F, full_matrices=False)
        ev = S ** 2 / n
        X = np.sqrt(n) * U[:, :p]
        A = Vt[:p].T * (S[:p] / np.sqrt(n))
        loss = m - ev[:p].sum()
        history.append(loss)
        if abs(prev - loss) < tol:
            converged = True
            break
        prev = loss
    if kept:
        warnings.warn("a projection onto the admissible cone was constant; the previous "
                      "transformation of that variable was kept", UserWarning, stacklevel=3)
    return dict(F=F, X=X, A=A, ev=ev, history=np.array(history), converged=converged)


class QLPCA:
    """Quasi-linear PCA.

    Parameters
    ----------
    n_components : number of retained components p.
    degree : spline degree v, scalar or one per variable (qlPCA proper uses 1).
    n_knots : number of interior knots r, scalar or one per variable. Knots are placed at
        the percentiles round(100 i/(r+1)) (MATLAB ``prctile``). ``degree=1, n_knots=0``
        is ordinary PCA.
    knots : optional explicit interior knots, a sequence with one array per variable, in
        the units of the data. Overrides ``n_knots``. Ignored when ``legacy``.
    monotone : restrict every f_j to be nondecreasing (I-splines with nonnegative
        coefficients and a free intercept). Decreasing relations appear as negative
        loadings. Ignored when ``legacy``.
    init : ``"pca"``: start from the PCA scores (plus ``n_init`` random orthonormal
        starts; the fit with the lowest loss is kept). Ignored when ``legacy``.
    n_init : number of random starts in addition to the PCA start.
    random_state : seed or ``numpy.random.Generator`` for the random starts.
    legacy : reproduce the 2012 MATLAB behaviour (start from the first p columns of the
        standardised pseudo-indicator matrix, single start, eigen-decomposition update,
        unique values required).
    tol, max_iter : stop when |loss_k - loss_{k-1}| < tol or after max_iter iterations
        (a ConvergenceWarning is issued in the latter case).

    Attributes (after ``fit``)
    --------------------------
    scores_ (n, p) object scores X;  transformed_ (n, m) matrix F;
    loadings_ (m, p) matrix A;  eigenvalues_ (p,) and all_eigenvalues_ (m,) of F'F/n;
    vaf_ (p,) percentage of variance accounted for; communalities_ (m,);
    loss_, loss_history_, n_iter_ (of the retained fit); losses_ (final loss of each start);
    knots_ (knot sequences, standardised scale); coef_ (per variable: coefficients on the
    centred basis -- B-splines by default, I-splines if ``monotone`` -- or on the
    standardised I-spline columns if ``legacy``).
    """

    def __init__(self, n_components=2, degree=1, n_knots=1, knots=None, monotone=False,
                 init="pca", n_init=10, random_state=None, legacy=False, tol=1e-6, max_iter=1000):
        self.n_components = n_components
        self.degree = degree
        self.n_knots = n_knots
        self.knots = knots
        self.monotone = monotone
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.legacy = legacy
        self.tol = tol
        self.max_iter = max_iter

    # ------------------------------------------------------------------ fit
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n, m = X.shape
        p = self.n_components
        if not 1 <= p <= m:
            raise ValueError("n_components must be between 1 and the number of variables")
        if self.init != "pca" and not self.legacy:
            raise ValueError("init must be 'pca'")
        degs = _per_variable(self.degree, m, "degree")
        nks = _per_variable(self.n_knots, m, "n_knots")
        std = X.std(0, ddof=1)
        if np.any(std == 0):
            raise ValueError("constant variables cannot be fitted")
        if self.legacy:
            for j in range(m):
                if len(np.unique(X[:, j])) < n:
                    raise ValueError("legacy=True requires unique values within each variable")
        self.mean_, self.std_ = X.mean(0), std
        H = (X - self.mean_) / self.std_

        explicit = [None] * m
        if self.knots is not None and not self.legacy:
            if len(self.knots) != m:
                raise ValueError("knots must have one array per variable")
            explicit = [(np.asarray(k, float) - self.mean_[j]) / self.std_[j] for j, k in enumerate(self.knots)]
        self.knots_ = [knot_sequence(H[:, j], degs[j], nks[j], explicit[j]) for j in range(m)]
        self._degrees = degs
        self._kinds = ["I" if (self.legacy or self.monotone) else "B"] * m

        if self.legacy:
            F, A, coef, shift, scale = self._fit_legacy(H, degs)
        else:
            F, A, coef, shift, scale = self._fit_modern(H, degs)
        self._shift, self._scale = shift, scale

        # sign convention (D6)
        s_var = np.sign([np.corrcoef(F[:, j], H[:, j])[0, 1] for j in range(m)])
        s_var[s_var == 0] = 1.0
        F = F * s_var
        A = A * s_var[:, None]
        coef = [c * s for c, s in zip(coef, s_var)]
        s_comp = np.sign(A[np.abs(A).argmax(0), np.arange(p)])
        s_comp[s_comp == 0] = 1.0
        A = A * s_comp
        Zs = F @ A
        T = _inv_sqrt_sym(Zs.T @ Zs)

        self.n_samples_ = n
        self.scores_, self.transformed_, self.loadings_ = np.sqrt(n) * Zs @ T, F, A
        self._T = T
        self.coef_ = coef
        self.vaf_ = 100.0 * (A ** 2).sum(0) / m
        self.communalities_ = (A ** 2).sum(1)
        return self

    def _fit_legacy(self, H, degs):
        """2012 MATLAB behaviour (SPEC section 4.2)."""
        n, m = H.shape
        p = self.n_components
        Gz, shift, scale = [], [], []
        for j in range(m):
            Z, mu, sd = _zscore_cols(ispline_basis(H[:, j], degs[j], self.knots_[j]))
            Gz.append(Z)
            shift.append(mu)
            scale.append(sd)
        Xs = np.hstack(Gz)[:, :p].copy()
        A = np.array([[np.corrcoef(H[:, j], Xs[:, s])[0, 1] for s in range(p)] for j in range(m)])
        F = np.zeros((n, m))
        prev, history = np.inf, []
        for _ in range(self.max_iter):
            for j in range(m):
                c = np.linalg.lstsq(Gz[j], Xs @ A[j], rcond=None)[0]
                f = Gz[j] @ c
                F[:, j] = f - f.mean()
            F /= np.sqrt((F ** 2).sum(0))
            F *= np.sqrt(n)
            ev, V = np.linalg.eigh(F.T @ F)
            o = np.argsort(ev)[::-1]
            ev, V = ev[o] / n, V[:, o]
            A = V[:, :p] * np.sqrt(ev[:p])
            Zs = F @ A
            Xs = np.sqrt(n) * Zs @ _inv_sqrt_sym(Zs.T @ Zs)
            loss = ((Xs @ A.T - F) ** 2).sum() / n
            history.append(loss)
            if abs(prev - loss) < self.tol:
                break
            prev = loss
        else:
            warnings.warn("maximum number of iterations reached", ConvergenceWarning, stacklevel=3)
        # coefficients against the final F (the 1/n scale in the listing is arbitrary)
        coef = [np.linalg.lstsq(Gz[j], F[:, j], rcond=None)[0] for j in range(m)]
        self.all_eigenvalues_, self.eigenvalues_ = ev, ev[:p]
        self.loss_, self.loss_history_, self.n_iter_ = history[-1], np.array(history), len(history)
        self.losses_ = np.array([history[-1]])
        return F, A, coef, shift, scale

    def _fit_modern(self, H, degs):
        n, m = H.shape
        p = self.n_components
        if self.monotone:
            Bs = [ispline_basis(H[:, j], degs[j], self.knots_[j]) for j in range(m)]
        else:
            Bs = [bspline_basis(H[:, j], degs[j], self.knots_[j]) for j in range(m)]
        shift = [B.mean(0) for B in Bs]
        scale = [np.ones(B.shape[1]) for B in Bs]
        Cs = [B - mu for B, mu in zip(Bs, shift)]

        if self.monotone:
            def project(j, y):
                c = nnls(Cs[j], y, maxiter=50 * Cs[j].shape[1] + 100)[0]
                return _unit(Cs[j] @ c, n)
        else:
            Qs = [orthonormal_basis(B) for B in Bs]

            def project(j, y):
                return _unit(Qs[j] @ (Qs[j].T @ y), n)

        F0 = H * (np.sqrt(n) / np.linalg.norm(H, axis=0))
        rng = np.random.default_rng(self.random_state)
        U, _, _ = np.linalg.svd(H, full_matrices=False)
        starts = [np.sqrt(n) * U[:, :p]]
        for _ in range(int(self.n_init)):
            R = rng.standard_normal((n, p))
            R -= R.mean(0)
            starts.append(np.sqrt(n) * np.linalg.qr(R)[0])
        runs = [_als(project, F0, X0, H.T @ X0 / n, p, self.tol, self.max_iter) for X0 in starts]
        losses = np.array([r["history"][-1] for r in runs])
        best = runs[int(np.argmin(losses))]  # first of equals: the PCA start wins ties
        if not best["converged"]:
            warnings.warn("maximum number of iterations reached", ConvergenceWarning, stacklevel=3)

        F = best["F"]
        if self.monotone:
            coef = [nnls(Cs[j], F[:, j], maxiter=50 * Cs[j].shape[1] + 100)[0] for j in range(m)]
        else:
            coef = [np.linalg.lstsq(Cs[j], F[:, j], rcond=None)[0] for j in range(m)]
        self.all_eigenvalues_, self.eigenvalues_ = best["ev"], best["ev"][:p]
        self.loss_, self.loss_history_, self.n_iter_ = best["history"][-1], best["history"], len(best["history"])
        self.losses_ = losses
        return F, best["A"], coef, shift, scale

    def fit_transform(self, X, y=None):
        return self.fit(X).scores_

    # --------------------------------------------------- new data (degree 1)
    def _check(self):
        if not hasattr(self, "scores_"):
            raise AttributeError("estimator is not fitted")
        if any(d != 1 for d in self._degrees):
            raise NotImplementedError("projection and reconstruction are defined for degree 1 only")

    def _curve(self, j):
        """Knot abscissae (standardised scale) and values of f_j at them."""
        t = np.unique(self.knots_[j])
        basis = ispline_basis if self._kinds[j] == "I" else bspline_basis
        G = basis(t, self._degrees[j], self.knots_[j])
        return t, ((G - self._shift[j]) / self._scale[j]) @ self.coef_[j]

    def transform_variables(self, X):
        """Optimal transformations f_j of new rows: linear interpolation of (h_j, f_j),
        linear extrapolation outside the training range."""
        self._check()
        H = (np.asarray(X, float) - self.mean_) / self.std_
        out = np.empty_like(H)
        for j in range(H.shape[1]):
            t, f = self._curve(j)
            x = H[:, j]
            y = np.interp(x, t, f)
            lo, hi = x < t[0], x > t[-1]
            y[lo] = f[0] + (f[1] - f[0]) / (t[1] - t[0]) * (x[lo] - t[0])
            y[hi] = f[-1] + (f[-1] - f[-2]) / (t[-1] - t[-2]) * (x[hi] - t[-1])
            out[:, j] = y
        return out

    def transform(self, X):
        """Object scores of (new) rows, X = sqrt(n) F A (A'F'FA)^(-1/2) (D10; equivalent to
        thesis eq. 3.5 and returns the training scores exactly for the training rows)."""
        Z = self.transform_variables(X) @ self.loadings_
        return np.sqrt(self.n_samples_) * Z @ self._T

    def inverse_transform(self, scores):
        """Reconstruct the data from object scores: F_hat = X A', then invert each f_j.

        Where f_j is not monotone the inverse is not unique: the first segment whose
        range contains the value is used, and values outside the range of f_j are clipped.
        """
        self._check()
        Fh = np.asarray(scores, float) @ self.loadings_.T
        H = np.empty_like(Fh)
        for j in range(Fh.shape[1]):
            t, f = self._curve(j)
            y = np.clip(Fh[:, j], f.min(), f.max())
            h = np.full_like(y, np.nan)
            for i in range(len(t) - 1):
                lo, hi = sorted((f[i], f[i + 1]))
                ok = np.isnan(h) & (y >= lo) & (y <= hi) & (f[i + 1] != f[i])
                h[ok] = t[i] + (y[ok] - f[i]) / (f[i + 1] - f[i]) * (t[i + 1] - t[i])
            H[:, j] = h
        return H * self.std_ + self.mean_
