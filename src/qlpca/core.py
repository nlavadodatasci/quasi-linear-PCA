"""The qlPCA estimator (thesis chapter 3).

Minimises sigma(X, F) = (1/n) sum_j |X a_j - f_j|^2 subject to X'X = nI, where each
f_j is a spline of degree ``degree`` with ``n_knots`` interior knots in the
standardised variable h_j, by alternating least squares.

Two start values are available (``init``):

``"matlab"``  X = first p columns of the standardised pseudo-indicator matrix, as in the
              MATLAB listings that produced the thesis results (default);
``"pca"``     X = scores of ordinary PCA of the standardised data, as in the printed
              pseudo-code.

Both variants centre and standardise the pseudo-indicator columns (this acts as the
intercept; without it the fit is far worse, see docs/SPEC.md section 6).

Sign convention (loadings are defined up to sign): each transformed variable is signed
so that corr(f_j, h_j) >= 0 and each component so that its loading of largest
absolute value is positive.
"""
import numpy as np

from .splines import ispline_basis, knot_sequence


def _zscore_cols(M):
    mean = M.mean(0)
    std = M.std(0, ddof=1)
    std = np.where(std == 0, 1.0, std)
    return (M - mean) / std, mean, std


def _inv_sqrt_sym(S):
    lam, U = np.linalg.eigh(S)
    return (U / np.sqrt(lam)) @ U.T


class QLPCA:
    """Quasi-linear PCA.

    Parameters
    ----------
    n_components : number of retained components p.
    degree : spline degree v (qlPCA proper uses 1).
    n_knots : number of interior knots r. ``degree=1, n_knots=0`` is ordinary PCA.
    init : ``"matlab"`` or ``"pca"`` start value.
    tol, max_iter : stop when |loss_k - loss_{k-1}| < tol or after max_iter iterations.

    Attributes (after ``fit``)
    --------------------------
    scores_ (n, p) object scores X;  transformed_ (n, m) matrix F;
    loadings_ (m, p) matrix A;  eigenvalues_ (p,) and all_eigenvalues_ (m,) of F'F/n;
    vaf_ (p,) percentage of variance accounted for; communalities_ (m,);
    loss_, loss_history_, n_iter_; knots_ (list of knot sequences, standardised scale);
    coef_ (list of spline coefficients on the standardised pseudo-indicator columns).
    """

    def __init__(self, n_components=2, degree=1, n_knots=1, init="matlab", tol=1e-6, max_iter=1000):
        self.n_components = n_components
        self.degree = degree
        self.n_knots = n_knots
        self.init = init
        self.tol = tol
        self.max_iter = max_iter

    # ------------------------------------------------------------------ fit
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n, m = X.shape
        p, v, r = self.n_components, self.degree, self.n_knots
        if self.init not in ("matlab", "pca"):
            raise ValueError("init must be 'matlab' or 'pca'")
        if not 1 <= p <= m:
            raise ValueError("n_components must be between 1 and the number of variables")

        self.mean_ = X.mean(0)
        self.std_ = X.std(0, ddof=1)
        H = (X - self.mean_) / self.std_

        self.knots_ = [knot_sequence(H[:, j], v, r) for j in range(m)]
        Gz, self._gmean, self._gstd = [], [], []
        for j in range(m):
            Z, mu, sd = _zscore_cols(ispline_basis(H[:, j], v, self.knots_[j]))
            Gz.append(Z)
            self._gmean.append(mu)
            self._gstd.append(sd)

        if self.init == "matlab":
            Xs = np.hstack(Gz)[:, :p].copy()
        else:
            U, _, _ = np.linalg.svd(H, full_matrices=False)
            Xs = np.sqrt(n) * U[:, :p]
        A = np.array([[np.corrcoef(H[:, j], Xs[:, s])[0, 1] for s in range(p)] for j in range(m)])

        F = np.zeros((n, m))
        prev, history = np.inf, []
        for _ in range(self.max_iter):
            for j in range(m):
                coef = np.linalg.lstsq(Gz[j], Xs @ A[j], rcond=None)[0]
                f = Gz[j] @ coef
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

        # coefficients against the final F (the 1/n scale in the listing is arbitrary)
        coef = [np.linalg.lstsq(Gz[j], F[:, j], rcond=None)[0] for j in range(m)]

        # sign convention: corr(f_j, h_j) >= 0, then largest |loading| positive per component
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
        Xs = np.sqrt(n) * Zs @ T

        self.n_samples_ = n
        self.scores_, self.transformed_, self.loadings_ = Xs, F, A
        self._T = T
        self.coef_ = coef
        self.all_eigenvalues_ = ev
        self.eigenvalues_ = ev[:p]
        self.vaf_ = 100.0 * (A ** 2).sum(0) / m
        self.communalities_ = (A ** 2).sum(1)
        self.loss_ = history[-1]
        self.loss_history_ = np.array(history)
        self.n_iter_ = len(history)
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X).scores_

    # --------------------------------------------------- new data (degree 1)
    def _check(self):
        if not hasattr(self, "scores_"):
            raise AttributeError("estimator is not fitted")
        if self.degree != 1:
            raise NotImplementedError("projection and reconstruction are defined for degree 1 only")

    def _curve(self, j):
        """Knot abscissae (standardised scale) and values of f_j at them."""
        t = np.unique(self.knots_[j])
        G = ispline_basis(t, self.degree, self.knots_[j])
        return t, ((G - self._gmean[j]) / self._gstd[j]) @ self.coef_[j]

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
        """Object scores of (new) rows, X = sqrt(n) F A (A'F'FA)^(-1/2) (thesis eq. 3.5)."""
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
