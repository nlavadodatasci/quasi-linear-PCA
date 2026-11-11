"""Faithful numpy port of the MATLAB listings in thesis section 3.2 (prototype)."""
import numpy as np

def prctile(x, p):                      # MATLAB prctile == numpy 'hazen'
    return np.percentile(x, p, method="hazen")

def zscore(M):
    return (M - M.mean(0)) / M.std(0, ddof=1)

def mbasis(x, order, i, knots):         # i is 1-based, as in MATLAB
    if order == 1:
        return 1.0/(knots[i]-knots[i-1]) if (knots[i-1] <= x < knots[i]) else 0.0
    d = knots[i+order-1]-knots[i-1]
    if d == 0: return 0.0
    return order*((x-knots[i-1])*mbasis(x,order-1,i,knots)
                  + (knots[i+order-1]-x)*mbasis(x,order-1,i+1,knots))/((order-1)*d)

def knot_seq(h, order, r):
    p = np.round(100/(r+1)*np.arange(1, r+1))
    inner = np.sort(prctile(h, p)) if r else np.array([])
    return np.r_[[h.min()]*order, inner, [h.max()]*order]

def create_mspline(h, order, r):
    kn = knot_seq(h, order, r)
    G = np.array([[mbasis(x, order, i, kn) for i in range(1, order+r+1)] for x in h])
    return G, kn

def isplines(x, degree, r):
    dim = degree + r; k = degree
    G2, kn = create_mspline(x, degree+1, r)
    n = len(x); I = np.zeros((n, dim))
    for i in range(1, dim+1):
        for t in range(n):
            if x[t] == kn[-1]: I[t, i-1] = 1; continue
            j = int((kn <= x[t]).sum())
            if i > j: I[t, i-1] = 0
            else:
                s = 0.0
                for m in range(i+1, j+1):
                    s += (kn[m+k]-kn[m-1])/(k+1)*G2[t, m-1]
                I[t, i-1] = s
    return I, kn

def qlpca(H, p, degree=1, r=1, tol=1e-6, max_it=1000, init="matlab"):
    n, m = H.shape
    H = zscore(H)
    w = degree + r
    Gs = [isplines(H[:, j], degree, r)[0] for j in range(m)]
    G = zscore(np.hstack(Gs))
    X = G[:, :p].copy()
    if init == "pca":
        U, s, Vt = np.linalg.svd(H, full_matrices=False); X = np.sqrt(n)*U[:, :p]
    A = np.array([[np.corrcoef(H[:, j], X[:, s])[0, 1] for s in range(p)] for j in range(m)])
    F = np.zeros((n, m)); prev = np.inf; coef = np.zeros((m, w))
    for it in range(1, max_it+1):
        for j in range(m):
            Gj = G[:, j*w:(j+1)*w]
            y = X @ A[j]
            coef[j] = np.linalg.lstsq(Gj, y, rcond=None)[0] / n
            f = Gj @ coef[j]; F[:, j] = f - f.mean()
        F = F / np.sqrt((F**2).sum(0)) * np.sqrt(n)
        ev, V = np.linalg.eigh(F.T @ F); o = np.argsort(ev)[::-1]; ev = ev[o]/n; V = V[:, o]
        eig = ev[:p]; A = V[:, :p] * np.sqrt(eig)
        Z = F @ A
        K, lam, Wt = np.linalg.svd(Z, full_matrices=False)
        X = np.sqrt(n) * K @ Wt
        loss = ((X @ A.T - F)**2).sum()/n
        if abs(prev-loss) < tol: break
        prev = loss
    return dict(F=F, X=X, A=A, eig=eig, all_eig=ev, it=it, loss=loss, H=H)
