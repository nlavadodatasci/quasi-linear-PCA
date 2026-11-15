# User guide

## The model and the loss

The data are n observations of m continuous variables, standardised to h_j (mean 0, standard deviation 1). Each
variable is transformed by a spline f_j of degree v with r interior knots, centred and scaled to norm √n. qlPCA
looks for p object scores X (n × p, with X'X = nI), loadings A (m × p) and transformations F = [f_1 … f_m] that
minimise

$$
\sigma(X, A, F) = \frac{1}{n} \sum_{j=1}^{m} \lVert X a_j - f_j \rVert^2 ,
$$

which is the same as maximising the sum of the p largest eigenvalues of the correlation matrix F'F/n of the
transformed variables. With `degree=1, n_knots=0` every f_j is linear and qlPCA is ordinary PCA. The fit
alternates two steps until the loss changes by less than `tol`: each f_j is the projection of X a_j on the space
of its spline basis, and (X, A) come from the truncated SVD of F.

The variance accounted for (`vaf_`, `eigenvalues_`) refers to the transformed variables.

## Default mode and `legacy=True`

* **Default** — a modernised algorithm for new analyses (docs `DESIGN.md`, D1–D11): centred B-spline basis, PCA
  start plus random starts (the lowest loss is kept), truncated-SVD updates, ties allowed, optional monotone
  transformations, degree and knots per variable.
* **`legacy=True`** — the behaviour of the 2012 MATLAB code: standardised I-spline columns, a start from the first
  p columns of the basis, eigen-decomposition updates, a single start, unique values required within each
  variable. It reproduces the thesis tables number by number and is what the replication notebooks use.

## Knots

* `n_knots` — number of interior knots, one value for all variables or one per variable. Knots are placed at the
  percentiles round(100 i / (r + 1)), computed as MATLAB's `prctile`: one knot at the median, two at the 33rd and
  67th percentiles.
* `knots=` — explicit interior knots, one array per variable, in the units of the data; it overrides `n_knots`:

```python
QLPCA(n_components=2, knots=[[3.0, 12.5], [2050.0], [7.5, 15.0], ...])   # one array per variable
```

* Coinciding knots (ties) or knots on the boundary are merged, with a warning, and the basis of that variable
  becomes smaller. `knots_` holds the knot sequences actually used, on the standardised scale.
* `degree` — 1 by default; projection, reconstruction and piecewise loadings are defined for degree 1 only.

How many knots to use is discussed in [Choosing the knots](knot-selection.md).

## Monotone transformations

`monotone=True` restricts every f_j to be nondecreasing: the basis is the I-spline basis with nonnegative
coefficients (fitted by NNLS) and a free intercept. Decreasing relations then appear as negative loadings. The
constrained fit often has flat segments (a coefficient exactly zero); `inverse_transform` is not unique there.

## Starts and local optima

The loss has local optima. The default start is the PCA solution, plus `n_init=10` random orthonormal starts; the
fit with the lowest loss is kept and the PCA start wins ties. `losses_` gives the final loss of every start (the
first is the PCA start), `loss_history_` and `n_iter_` describe the retained fit, and `random_state` makes the
random starts reproducible. A `ConvergenceWarning` is issued when `max_iter` is reached.

## Diagnostics and plots

* `vaf_`, `eigenvalues_`, `all_eigenvalues_`, `communalities_`, `loadings_`, `scores_`, `transformed_`.
* `qlpca.diagnostics.piecewise_loadings(model, X)` — for degree 1, the loading of each variable on each component,
  segment by segment between its knots, on the original scale (thesis eqs. 3.7–3.8).
* `qlpca.plots.scree_plot`, `transformation_plots` (f_j against the original variable, with the knots), `biplot`.
* `model.variable_transformation(j)` — the curve of f_j on a grid over the training range.

## Projection and reconstruction (degree 1)

* `transform(X_new)` — object scores of new rows, X_new = √n F_new A (A'F'FA)^(-1/2); it returns `scores_` exactly
  for the training rows.
* `transform_variables(X_new)` — the transformed variables F_new.
* `inverse_transform(scores)` — the data reconstructed from scores: F̂ = X A', then each f_j is inverted (exactly
  where f_j is monotone; where it is not, the first segment that contains the value is used).

## Use with scikit-learn

`QLPCA` is a scikit-learn transformer, tested with scikit-learn's estimator checks:

* `clone`, `get_params` / `set_params`, `Pipeline`, `GridSearchCV` (inside a pipeline with a supervised step, or
  with your own scorer: `QLPCA` has no `score` method in 0.2.0);
* `set_output(transform="pandas")`, `get_feature_names_out()` (`qlpca0`, `qlpca1`, …);
* pandas DataFrames: the column names are kept in `feature_names_in_`, used in warnings and plots, and checked again
  by `transform`.

```python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

pipe = make_pipeline(QLPCA(n_components=2, n_knots=2, random_state=0), Ridge())
```

## Limits

* **qlPCA is descriptive.** It fits and interprets one sample; it is not a predictive model.
* **New data only with degree 1.**
* **Linear extrapolation.** Outside the training range of a variable, f_j is extended linearly from its first or
  last segment; `transform` and `transform_variables` then issue an `ExtrapolationWarning` with the number of rows
  and the variables concerned. Validation folds in cross-validation are new data, so expect the warning there.
* **Heavy tails.** With heavy-tailed variables the scores of new observations can be dominated by single extreme
  values (on the simulated cylinder data, test scores near 100 against training scores within ±2.5). Transform
  such variables beforehand (e.g. log), or do not project new data.
* **Sample size.** The thesis recommends n ≥ 5 (degree + knots) for every variable; a `SmallSampleWarning` is issued
  below that.
* **Complete data.** Missing or infinite values raise an error; constant variables are rejected.
