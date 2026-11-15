# Design decisions

Status: **accepted by the author on 2026-11-11.** Each section lists the options
that were compared, the evidence, and the decision. `docs/SPEC.md` describes what
the 2012 thesis and code did; this file describes what the package does and why
it differs.

The package has two modes:

- **default** — a modernised algorithm meant for new studies;
- **`legacy=True`** — the behaviour of the 2012 MATLAB code, kept so that the
  replication notebooks can be compared with the thesis tables number by number.

Evidence below was obtained with `prototype/qlpca_prototype.py` on the chapter 4
data (15 countries, 11 variables, linear splines). Percentages are the variance
accounted for by the retained components.

---

## D1. Start values

The 2012 pseudo-code (thesis 3.1.3, IJAM IV.B) starts from PCA scores; the 2012
MATLAB code starts from the first p columns of the standardised pseudo-indicator
matrix, i.e. basis functions of the first variable.

| option | pros | cons |
|---|---|---|
| A. first p columns of G (2012 code) | reproduces the thesis | depends on the column order of the data; not orthonormal |
| B. PCA scores (2012 pseudo-code) | deterministic; starts from the linear solution | can stop in a local optimum |
| C. PCA scores plus `n_init` random orthonormal starts, keep the lowest loss | robust to local optima; order-independent | more computing time; needs `random_state` for reproducibility |

Evidence:

| interior knots | components | A | B | 40 random starts |
|---|---|---|---|---|
| 1 | 2 | 68.90 | 68.90 | 68.90 in 40/40 |
| 2 | 2 | 76.33 | 76.33 | 76.33 in 35/40, 70.55 in 5/40 |
| 2 | 3 | 90.00 | 90.30 | 90.30 in 28/40, 90.00 in 12/40 |

With option A the result for two knots and three components is 90.00 or 90.30
depending on the order of the columns.

**Decision: C** by default (`init="pca"`, `n_init=10`, `random_state=None`).
`legacy=True` uses A. Consequence for the replication: thesis Figure 4.5 (two
knots, three components) shows a local optimum; the notebook reports both.

---

## D2. Constant term (centring)

I-splines of degree v with r interior knots span a space of dimension v+r that
does not contain the constant. The 2012 code standardises the columns of G before
iterating, which is equivalent to adding an intercept to the regression of step
II.2. The published pseudo-code omits this.

| option | pros | cons |
|---|---|---|
| A. standardise the columns of the I-spline matrix (2012 code) | reproduces the thesis | ad hoc step, absent from the published pseudo-code |
| B. I-splines plus an explicit intercept | faithful to the original formulation | same fit as A; one more parameter to describe |
| C. full B-spline basis (dimension v+r+1, contains the constant), then centre f | standard, well conditioned, available in scipy | the basis is no longer the one printed in the thesis |
| D. no constant (pseudo-code read literally) | none | wrong: fit falls below ordinary PCA |

Evidence (PCA start): A gives 68.90 / 84.02 / 76.33 / 90.30 for (r, p) = (1,2),
(1,3), (2,2), (2,3); D gives 50.50 / 68.21 / 65.35 / 79.51; ordinary PCA explains
61.89 with two components. A, B and C span the same function space, so they must
give the same fitted values; only A has been run numerically so far — the test
suite must confirm B = C = A.

**Decision: C** by default; **B** (I-splines, non-negative coefficients, free
intercept) when `monotone=True`; **A** in `legacy=True`. Any published
pseudo-code must state the centring.

---

## D3. Update of scores and loadings

| option | pros | cons |
|---|---|---|
| A. eigen-decomposition of F'F, then X = orthonormalised F A (2012 code) | reproduces the thesis | squares the condition number; two decompositions per iteration |
| B. a_j = X'f_j/n, then X from svd(F A') (2012 pseudo-code) | textbook ALS | a partial update: slower convergence |
| C. truncated SVD of F: X = sqrt(n) U_p, A = V_p S_p / sqrt(n) | exact joint minimiser of the loss over (X, A) given F; one decomposition | none known |

A and C give the same X and A up to signs. **Decision: C.** The algorithm is then
a two-block ALS: F given (X, A), then (X, A) given F, and the loss is
m - sum of the first p eigenvalues of F'F/n.

## D4. Regression step

The 2012 code inverts G_j'G_j at every iteration. **Decision:** compute an
orthonormal basis Q_j of the centred basis once (QR, rank-revealing so that
duplicated knots do not break it); the update is f_j = Q_j Q_j' (X a_j).
Coefficients are recovered once, after convergence, against the final F.

## D5. Monotone transformations

| option | pros | cons |
|---|---|---|
| A. unrestricted splines only (2012) | maximal fit; can show V-shaped relations | sign of f_j arbitrary; loadings cannot be read on the original variable |
| B. add `monotone=True`: I-spline basis, coefficients >= 0 (NNLS), free intercept | no sign indeterminacy; loadings interpretable; inverse transformation unique where f is strictly increasing | lower fit when the true relation is not monotone |

**Decision: B as an option, default `monotone=False`** (the unrestricted model is
the qlPCA of the thesis). The cone-constrained, normalised update is: project on
the cone, centre, rescale to norm sqrt(n). If the projection is constant (all
coefficients zero) keep the previous f_j and warn. Decreasing relations are
handled by the sign of the loading, not by the spline.

## D6. Sign convention

Unrestricted mode: f_j is oriented so that corr(f_j, h_j) >= 0; each component is
oriented so that its loading of largest absolute value is positive. For V-shaped
transformations corr(f_j, h_j) is close to zero and the orientation is fragile:
the documentation must say so, and such variables should be read through the
transformation plot and piecewise loadings, not through the sign of the loading.

## D7. Ties

The 2012 code needed unique values within each variable (third decimals were
added by hand in Appendix A.1). **Decision:** ties are allowed. Knots that
coincide because of ties are merged, with a warning, and the basis dimension of
that variable is reduced.

## D8. Knots and degree

2012: same degree and number of knots for all variables, knots at equally spaced
percentiles (MATLAB `prctile`, MATLAB rounding). **Decision:** `degree` and
`n_knots` accept a scalar or one value per variable; `knots` optionally gives the
interior knots explicitly. Default placement is unchanged. Projection,
reconstruction and piecewise loadings remain defined for degree 1 only.

## D9. Convergence

Absolute change of the loss below `tol` (default 1e-6, as in the thesis) or
`max_iter`. The loss history is stored; a warning is issued when `max_iter` is hit.

## D10. Projection and reconstruction

Projection of new rows uses X_new = sqrt(n) F_new A (A'F'F A)^(-1/2), the form
that returns the training scores exactly (equivalent to thesis eq. 3.5).
Reconstruction inverts each piecewise-linear f_j; where f_j is not monotone the
inverse is not unique and the first matching segment is used. Both points are
documented in the docstrings.

## D11. New observations outside the training range (added 2026-11-14, version 0.2.0)

qlPCA is descriptive: it fits and interprets one sample. Projecting new rows (D10)
needs f_j outside the training range of each variable, where the data say nothing.
The 2012 code and `transform_variables` extend f_j linearly from its first or last
segment. With heavy-tailed variables this can dominate the result: in
`make_cylinder(n=200, random_state=0)` split with
`train_test_split(test_size=0.25, random_state=1)`, the default fit gives training
scores within ±2.5 and test scores up to 96.8. Four test rows fall outside the
training range; the damage comes mainly from `2*a*pi/b^2` (one test value about 3,650
training standard deviations above the training maximum, where the extrapolated f_j
reaches 744 against at most 3.1 in training) and from `a/b` (f_j up to 18.5).

| option | pros | cons |
|---|---|---|
| A. linear extrapolation, silent (until 0.2.0) | continuous; returns the training scores exactly; keeps the slope of the outer segments | unbounded; no sign that anything is wrong |
| B. linear extrapolation with a warning | same values as A; the user is told how many rows and which variables are outside | still unbounded; a warning can be ignored |
| C. clip to the training range (f_j constant outside, `extrapolate="clip"`) | bounded scores; no invented slope | distinct new values get equal transformations; changes the values returned |
| D. return NaN or raise outside the range | never invents values | unusable in pipelines and cross-validation |
| E. extrapolate on a log scale or another transformed scale | can tame some tails | a modelling choice the method does not make; depends on the variable |

**Decision: B in 0.2.0.** The values returned do not change. `transform` and
`transform_variables` issue `ExtrapolationWarning` (exported in `qlpca`) with the number
of observations outside the training range and the names of the variables; nothing is
issued for the training rows, nor by `fit` or `fit_transform`. The training range is
the minimum and maximum of each variable in the training data, in its original units,
with a slack of 1e-9 standard deviations so that round-off does not trigger the
warning (rows rebuilt by `inverse_transform` can exceed the range by about 1e-16
standard deviations). The message lists at most five variables ("and N more").
The docstrings and the README ("Use with scikit-learn") say that the method is
descriptive and recommend transforming heavy-tailed variables beforehand (e.g. log), or
not projecting new data.

**Deferred to 0.3.0:** option C as a parameter `extrapolate={"linear","clip"}`, with
`"linear"` as the default so that results do not change.

## D12. Choice of the number of knots (status: proposed for 0.3.0)

Nothing in `src/qlpca/` implements this yet.

**Problem.** The VAF on the fitting sample always grows with the number of knots: every extra knot
enlarges the spline space of each variable, so the in-sample criterion cannot choose it. Today the user
picks `n_knots` by hand, guided by the transformation plots and the warning n < 5 (degree + knots).

| option | pros | cons |
|---|---|---|
| A. manual choice (today) | simple; the analyst keeps control; matches the descriptive use of the method and the thesis | no guard against over-fitting beyond `SmallSampleWarning`; results depend on the analyst |
| B. a `score(X)` method: VAF of the p components on new data, with new rows limited to the training range; usable with `GridSearchCV` | follows the scikit-learn convention (estimators that can be scored); any splitting scheme and grid; no new class | needs `extrapolate="clip"` (D11) to be well defined; the user has to set up the search; plain maximisation of the mean favours many knots when the curve is flat |
| C. a `QLPCACV` class: grid `n_knots` in {1, 2, 3, 4} at the percentiles, `cv=5`, one-standard-error rule, grid limited by n >= 5 (degree + knots), results in `cv_results_` | one call; parsimonious choice (one-SE rule); the limit avoids fits the thesis advises against; same pattern as `LassoCV`, `RidgeCV` | one more public class to document and test; cost of 5 x 4 fits (each with `n_init` starts); a fixed grid and knot placement |
| D. automatic choice inside `QLPCA` by default | nothing to do for the user | cost of every fit multiplied; results change with the random folds unless seeded (reproducibility); unstable on small samples (n = 15 in thesis chapter 4); against the scikit-learn convention that an estimator fits the model it is given and model selection is done outside it |

**Recommendation: B and C in 0.3.0; D rejected** (cost, reproducibility, small samples, scikit-learn
convention). Both B and C depend on the option `extrapolate="clip"` deferred in D11: held-out rows outside the
training range must not be extrapolated linearly.

**Preliminary evidence** (`comparison/princals/knot_selection.py`, written to
`comparison/princals/results/knot_selection.csv`). Out-of-sample VAF (%), p = 2, degree 1, default mode,
5-fold cross-validation repeated 4 times (`RepeatedKFold(5, 4, random_state=0)`, 20 splits); test rows
clipped to the training range; VAF = share of the test rows' transformed variance captured by the training
loading directions, ||F_test V||^2 / ||F_test||^2; mean over the 20 splits:

| data | 0 knots | 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|---|
| cylinders, n = 51, no noise | 83.4 | 94.5 | 96.5 | 97.3 | 97.7 | 98.3 |
| cylinders, n = 51, noise 0.25 | 72.8 | 87.6 | 89.5 | 90.6 | 91.1 | 91.4 |
| gas turbine, n = 400 (2015 sample) | 79.6 | 81.7 | 82.0 | 82.3 | 82.6 | 82.3 |

Standard errors (sd / sqrt(20)) between 0.2 and 1.1. On the cylinders the out-of-sample VAF still rises with
6 knots, by less with each knot; on the gas turbine data it stops rising after 4 knots. The one-SE rule on
the grid {1, 2, 3, 4} of option C would choose 4 knots (cylinders, no noise: the maximum), 3 (cylinders, noise
0.25) and 3 (gas turbine); maximising the mean would choose 4 in all three.

**Open:** the criterion on the original scale of the variables (reconstruction error) instead of the
transformed scale; knots per variable (a greedy search over variables); the location of the knots, not only
their number.

---

## Public interface

    QLPCA(n_components=2, degree=1, n_knots=1, knots=None, monotone=False,
          init="pca", n_init=10, random_state=None, legacy=False,
          tol=1e-6, max_iter=1000)

`legacy=True` forces: start A (D1), centring A (D2), update A (D3), no
multi-start, unique values required, and ignores `monotone` and `knots`.
