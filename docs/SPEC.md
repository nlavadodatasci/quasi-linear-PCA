# qlPCA — algorithm specification

Extracted from N. Lavado (2012), *Extensões via splines da Análise em Componentes
Principais*, PhD thesis, ISCTE-IUL (chapters 1, 3, 4), and Lavado & Calapez (2011a,
2011b; the IJAM paper `IJAM_41_4_14.pdf` is the English account of thesis chapter 3
and of the cylinder study). Page numbers are thesis page numbers (PDF page = thesis page + 17).

There are two descriptions of the algorithm in the thesis and they differ in
details: the **pseudo-code** (section 3.1.3) and the **MATLAB listings**
(section 3.2, transcribed in `matlab_original/`). The numbers in chapter 4 come
from the MATLAB code. `prototype/qlpca_prototype.py` is a direct numpy port of
the listings and reproduces Tables 4.5-4.8 to the printed precision.

## 1. Notation

| symbol | meaning |
|---|---|
| H (n x m) | data matrix, columns standardised |
| p | number of retained components |
| v | spline degree (qlPCA proper: v = 1) |
| r | number of interior knots |
| w = v + r | number of I-spline basis functions per variable |
| G_j (n x w) | pseudo-indicator matrix of variable j (I-spline basis evaluated at h_j) |
| y_j (w) | spline coefficients of variable j |
| f_j = G_j y_j | optimally transformed variable, centred, norm sqrt(n) |
| X (n x p) | object scores (nonlinear principal components), X'X = nI |
| a_j (p) | nonlinear loadings of variable j; A collects them |

## 2. Objective

Minimise, subject to X'X = nI (eq. 3.1, p. 54):

    sigma(X, F) = (1/n) * sum_j (X a_j - f_j)' (X a_j - f_j)

with f_j restricted to splines of degree v with r interior knots (eqs. 3.2-3.3).
All variables get the "Single" treatment (rank-one quantification). Degree and
number of knots are model parameters, not optimised. With v = 1 and r = 0 the
solution is ordinary PCA (stated in the WCE paper) — use this as a unit test.

## 3. Spline basis (pseudo-indicator matrix)

As implemented in `create_mspline.m` / `create_Ispline.m`:

1. Interior knots at percentiles `round(100*i/(r+1))`, i = 1..r, of the
   **standardised** variable: r=1 -> 50; r=2 -> 33, 67; r=3 -> 25, 50, 75.
   Percentiles follow MATLAB `prctile` (numpy: `np.percentile(..., method="hazen")`).
   The default numpy method gives different knots.
2. Knot sequence t: min repeated (v+1) times, interior knots, max repeated (v+1) times.
3. M-splines of order v+1 by the recursion in `basis` (order 1 uses the half-open
   interval [t_i, t_{i+1})). There are v+1+r of them.
4. I-splines of degree v: I_i(x) = sum_{q=i+1}^{j} (t_{q+v+1} - t_q)/(v+1) * M_q(x),
   i = 1..w, where j = #{knots <= x}; I_i(x) = 0 if i > j; I_i(max) = 1.
   Equivalently: with B-splines of degree v on t, I_i = sum_{q>i} B_q. The first
   B-spline is dropped, so the constant is not in the span; it is recovered by
   centring (next point).
5. The MATLAB code standardises every column of G (`G = zscore(G)`) before the
   iterations. This is **not** in the pseudo-code and it matters: regression on
   centred columns is equivalent to including an intercept.

Quirk: line 33 of `create_Ispline.m` reads `if j-k<=i<=j`, which MATLAB evaluates
as `((j-k)<=i)<=j`, always true here. The result is still correct because the
terms summed for q < j-k are zero and the remaining ones sum to one.

For v = 1 and one knot at the median m2 (min m1, max m3) the basis is explicit
(WCE paper): I_1 rises linearly from 0 at m1 to 1 at m2 and stays 1; I_2 is 0 up
to m2 and rises linearly to 1 at m3. (The formula printed in the WCE paper, I_1 = x/m2, is
written for m1 = 0; the general form is (x - m1)/(m2 - m1).)

## 4. Algorithm

### 4.1 As printed in the pseudo-code (pp. 56-58)

    I.  Z <- linearPCA(H) (p components); [K, L, W] <- svd(Z); X <- sqrt(n) K W'
        a_j <- X' h_j / n;   G <- createIspline(H, v+1, r)
    II. for each j:  ytilde <- X a_j
                     y_j <- (G_j' G_j)^-1 G_j' ytilde        (thesis prints an extra 1/n)
                     f_j <- G_j y_j;  centre f_j;  scale to norm sqrt(n)
                     a_j <- X' f_j / n
    III. Z <- linearPCA(F) (p components); [K, L, W] <- svd(Z); X <- sqrt(n) K W'
         (IJAM paper, step III.1: Z <- sum_j f_j a_j' = F A', which is the precise form)
    IV. stop when |loss_k - loss_{k-1}| < 0.1e-5 or after max_it iterations

### 4.2 As implemented in MATLAB (pp. 68-73) — this is what produced the results

    H <- zscore(H)                      (sample std, n-1)
    G <- zscore(createIspline(H, v, r))
    X <- first p columns of G           (not a PCA start; a random start is commented out)
    A <- corr(H, X)                     (m x p)
    repeat (max_it = 1000):
        for j: y <- X A[j,:]';  coef_j <- (1/n) (G_j'G_j)^-1 G_j' y
               F[:,j] <- G_j coef_j, centred
        F <- columns scaled to norm sqrt(n)
        [V, lam] <- eig(F'F), descending;  eigval <- lam[1:p] / n
        A <- V[:,1:p] * diag(sqrt(eigval))
        X <- F A;  [K, ~, W] <- svd(X);  X <- sqrt(n) K W'
    until convergence (test not printed; 3.2.4 says successive loss values, tol 1e-6)

Differences to keep in mind:
- start values (PCA scores vs first columns of G);
- the standardisation of G;
- A is updated from the eigen-decomposition of F'F, not as X'F/n;
- the 1/n factor only rescales coefficients (F is renormalised), so `coef` is
  defined up to scale; recompute coefficients against the final F if they are
  to be reported;
- the convergence test code is missing (the listing keeps `evprev`).

Outputs: X, F, A, eigenvalues, knots, coefficients.

## 5. Properties (section 3.1.4)

- **VAF per dimension**: VAF_s = sum_j a_sj^2; equals the s-th eigenvalue of the
  correlation matrix R = F'F/n of the transformed variables. Percentage = VAF_s*100/m.
  These percentages refer to the transformed matrix; comparison with PCA
  percentages needs caution (thesis pp. 60-61).
- **Communality** of variable j: sum_s a_sj^2. Rule of thumb >= 0.25.
- **Scree plots** use all eigenvalues of R. Solutions are **not nested**: compare
  p-1, p, p+1.
- **Projection of new observations** (v = 1 only): f_new,j by linear interpolation
  of (h_j, f_j), linear extrapolation outside the range; then
  X = sqrt(n) F A' W L^-1/2 W' (eq. 3.5).
- **Reconstruction** (v = 1): Fhat = X A; Hhat by inverse linear interpolation.
- **Piecewise loadings** (v = 1): on segment i between consecutive knots,
  corr(x_i, h_i) = sign(slope_i) * corr(x_i, f_i) (eqs. 3.7-3.8). The paragraph on
  p. 67 of the thesis ends mid-sentence. IJAM section V.E gives the full text:
  piecewise loadings are computed from the truncated vectors x_i, f_i, h_i (the
  observations falling between two consecutive knots), not from analytic
  expressions; the closed form for one interior knot is in the WCE paper.
- The 2011 implementation does not enforce monotone (nondecreasing) splines
  (IJAM section IV.C); monotone shapes appear only if they are optimal.
- Over-parametrisation guard: at least ~5 (better 10) observations per basis
  function, i.e. n >= 5w.

## 6. Verified against the thesis (prototype, 2026-11)

| target | thesis | prototype |
|---|---|---|
| PCA eigenvalues, Table 4.3 | 4.276, 2.531, 1.878, ... | match (max diff 0.0006) |
| qlPCA v=1, r=1, p=2, Table 4.5 | 4.582, 2.998 (68.90%) | 4.582, 2.998 |
| qlPCA v=1, r=2, p=2, Table 4.5 | 4.760, 3.637 (76.33%) | 4.759, 3.637 |
| loadings Table 4.6 (absolute values) | | max diff 0.005 |
| communalities Tables 4.7, 4.8 | | max diff 0.005 |

Start values and centring, on the chapter 4 data (v = 1), % VAF of the retained components:

| r | p | MATLAB start | PCA start | PCA start, G not centred |
|---|---|---|---|---|
| 1 | 2 | 68.90 | 68.90 | 50.50 |
| 1 | 3 | 84.02 | 84.02 | 68.21 |
| 2 | 2 | 76.33 | 76.33 | 65.35 |
| 2 | 3 | 90.00 | 90.30 | 79.51 |

- For p = 2 both starts agree. For r = 2, p = 3 they reach different local optima
  (eigenvalues 4.417, 3.373, 2.110 vs 4.407, 3.137, 2.389); the MATLAB start also
  gives 90.00 or 90.30 depending on the column order of the data. Thesis Fig. 4.5
  (r = 2, three components) is therefore start-dependent. Use several starts and
  keep the lowest loss.
- Without centring G (the pseudo-code read literally, no intercept) the fit is far
  worse and falls below ordinary PCA (61.89%). Centring G, or an intercept in step
  II.2, is part of the method and must be stated in any published pseudo-code.

## 7. Cylinder problem (section 3.3)

n = 51 cylinders, height a and base area b drawn at random; ten derived variables:

    1 a   2 b   3 2*sqrt(b*pi)   4 2*a*sqrt(b*pi)   5 a*b   6 a*b^2/(2*pi)
    7 a/sqrt(2*b*pi)   8 arctan(a*sqrt(pi)/(2*sqrt(b)))   9 arccot(same)
    10 a/b   11 b/a   12 2*a*pi/b^2

Noise: transform each variable by its linearising transformation (log; log(tan)
and log(cot) for the angles), add Normal noise with sd = 10% or 25% of the sd of
the variable, transform back. Fits: v = 1 with r = 2 and r = 3, p = 2.

The IJAM paper (section VI.A) states that a and b are uniformly distributed
pseudorandom numbers; the interval is not given. Ranges printed in the figures
(variable 6 at most about 0.14; variable 12 up to 6498.7) are consistent with
U(0,1). The noise sd is "a proportion of the standard deviation of each
variable". IJAM Table II prints PCA without noise as 82.16 (thesis: 82.15). With U(0,1) and eight
different seeds the prototype gave, without noise, PCA 78-87%, r=2 96.8-99.1%,
r=3 98.2-99.5% (thesis: 82.15, 97.77, 98.43). "sd of the variable" for the
noise is ambiguous (original or linearised scale). Decided: linearised scale (on the natural
scale the noise for variable 12 would be hundreds of log-units). The MSc thesis (chapter 6) states
that a and b are uniform on [0, 1] for its 31-cylinder version (8 derived variables, CATPCA
spline ordinal (2,2) on the discretised matrix: 98.670 % in two dimensions; the data matrix is
in its Annex 1, an image in the PDF).

## 8. Sign indeterminacy (found while porting)

With unrestricted (non-monotone) splines, -f_j is as admissible as f_j, so the
sign of a transformed variable and of its loadings is arbitrary. On the chapter 4
data with two interior knots, the two start values give the same eigenvalues but
opposite signs for the loadings of "H-M Fundos". The thesis (p. 92) reads the sign
change of this variable between the one-knot and two-knot solutions as a
substantive difference; part of it may be this indeterminacy. The package should
fix a convention (for example corr(f_j, h_j) >= 0, or a positive slope on the
first segment) and the notebook should discuss it.

## 9. Open questions for the author

1. Convergence criterion actually used: author does not recall; use the loss-based
   test described in thesis 3.2.4 / IJAM IV.B.
2. Cylinder example: seed lost (accepted: new simulation with fixed seed, a, b ~ U(0,1));
   noise sd measured on the linearised scale (decided).
3. The illustrative pseudo-indicator matrices on pp. 52-53 are not reproduced by
   the printed code (different boundary knots?).
4. Reference year of the Eurostat indicators in the WCE 2011 example.
5. Permission to redistribute the chapter 4 data (provided by Oliveira & Carvalho).
6. CATPCA comparisons (WCE Table II; thesis section 2.2.2) were run in SPSS.

## 10. Design decisions for the Python package (accepted 2026-11-11)

Summary only; the comparison of options and the evidence are in `docs/DESIGN.md`.

The package is meant for new studies, so it does not have to mirror the 2012 code.
It keeps a `legacy` mode that does, for the replication notebooks.

| topic | 2012 code | package default | why |
|---|---|---|---|
| start values | first p columns of G | PCA scores, plus `n_init` random starts, keep lowest loss | order-independent; local optima exist |
| constant term | columns of G standardised | B-spline basis of full dimension v+r+1 (contains the constant), f centred afterwards | same fitted space as centred I-splines, standard and better conditioned |
| (X, A) update | eigen-decomposition of F'F | truncated SVD of F | identical result; exact joint minimiser given F |
| regression | explicit inverse every iteration | projector for each variable computed once (QR/pinv) | speed, stability |
| convergence | absolute change in loss < 1e-6 | same criterion, tolerance as parameter | |
| monotonicity | not available | option `monotone=True`: I-spline basis, non-negative coefficients (NNLS) | removes the sign indeterminacy (section 8), keeps loadings interpretable on the original variables |
| ties in the data | not allowed (third decimals added by hand) | allowed | |
| knots | same r for all variables, equally spaced percentiles | per-variable degree/knots, optional user-supplied knots | thesis conclusion lists this as future work |
| signs | none | f_j oriented so that corr(f_j, h_j) >= 0; components oriented by largest loading | reproducible tables |

Evidence for multiple starts (chapter 4 data, v = 1, 40 random orthonormal starts,
% VAF): r=1, p=2: 68.90 in 40/40. r=2, p=2: 76.33 in 35/40 and 70.55 in 5/40.
r=2, p=3: 90.30 in 28/40 and 90.00 in 12/40. The thesis values for p = 2 are the
best optima found.
