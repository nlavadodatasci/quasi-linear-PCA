# Choosing the knots

The VAF on the fitting sample always grows with the number of knots: every extra knot enlarges the spline space of
each variable, so the in-sample criterion cannot choose it. In 0.2.0 the number of knots is chosen by hand, guided
by the transformation plots and the `SmallSampleWarning` (n < 5 (degree + knots)).

Design decision D12 (status: **proposed for 0.3.0**, see [DESIGN.md](design/design.md)) compares four options:
manual choice (today); a `score(X)` method (VAF of the p components on new data, with new rows clipped to the
training range) for use with `GridSearchCV`; a `QLPCACV` class (grid of 1–4 knots at the percentiles, 5-fold
cross-validation, one-standard-error rule, grid limited by n ≥ 5 (degree + knots), results in `cv_results_`); and an
automatic choice inside `QLPCA`. The recommendation is **`score()` and `QLPCACV` in 0.3.0** — neither exists in
0.2.0 — and the automatic choice is rejected (cost, reproducibility, small samples, scikit-learn convention). Both
depend on the option `extrapolate="clip"` deferred in D11.

## Preliminary evidence

Out-of-sample VAF (%), p = 2, degree 1, default mode; 5-fold cross-validation repeated 4 times
(`RepeatedKFold(5, 4, random_state=0)`); test rows clipped to the training range; VAF = share of the test rows'
transformed variance captured by the training loading directions. Mean over the 20 splits, standard error in
brackets. Computed by
[`comparison/princals/knot_selection.py`](https://github.com/nlavadodatasci/quasi-linear-PCA/blob/main/comparison/princals/knot_selection.py)
(file `results/knot_selection.csv`).

--8<-- "knot_selection_table.md"

Open questions recorded in D12: a criterion on the original scale of the variables; knots per variable (greedy
search); the location of the knots, not only their number. The Kelmarsh [application](kelmarsh.md) compares knots
at the percentiles with knots placed at observed control regions.
