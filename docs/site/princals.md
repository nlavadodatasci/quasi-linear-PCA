# Comparison with princals

`qlpca` is compared with `princals` from the Gifi system in its two implementations: the R package **Gifi 1.0-0**
(Mair, De Leeuw & Groenen; CRAN, 2025) and its Python port **pyGifi** (commit `7aa81ed`). The full comparison —
pipelines with source lines, same-knot results, bases, defaults, local optima, monotone fits and the functions only
`qlpca` has — is the notebook
[`comparison/princals/qlpca_vs_princals.ipynb`](https://github.com/nlavadodatasci/quasi-linear-PCA/blob/main/comparison/princals/qlpca_vs_princals.ipynb);
this page shows only numbers already computed there.

## Independent implementation and licences

`qlpca` is an independent implementation, written from the author's PhD thesis (Lavado, 2012) and the method's
original paper (Lavado & Calapez, 2011). It contains no code from the Gifi R package or from pyGifi. **Gifi and
pyGifi are licensed under GPL-3**; in the comparison folder they are only called (R) or imported (Python), and the
notebook quotes a few of their source lines with file and line number. No data of theirs is stored in the
repository (`neumann` is loaded from pyGifi at run time). `qlpca` is MIT.

## Data and settings

Cylinders (`make_cylinder`, n = 51, noise 0, 0.10, 0.25); thesis chapter 4 (n = 15); `neumann` (3 variables); UCI
Gas Turbine CO and NOx Emission (doi:10.24432/C5WC95, CC BY 4.0; 2015, n = 400). Same knots: the qlpca percentile
knots written explicitly in a manifest and passed to all three programs, degree 1, `eps = 1e-12`. Monotone: 2
knots, degree 1 and 2. Defaults: `princals(data, ndim = 2)` and `QLPCA(n_components=2)`.

## VAF of two components (%)

From `comparison/princals/results/summary.csv`. Runs `knots1`, `knots2`: same knots, degree 1, no monotonicity;
`monotone_deg1/2`: `monotone=True` vs `ordinal=TRUE`; `defaults`: each program's own defaults (different models).

--8<-- "princals_tables.md"

## Conclusions of the notebook

As printed by the notebook (computed numbers):

--8<-- "princals_conclusions.md"

## Figures

Same knots (2), degree 1, no monotonicity — transformations of the chapter 4 data:

![Same-knot transformations, chapter 4 data](generated/princals/nb_b_same_knots_transformations_st_gender.png)

Condition numbers of B-splines and of [1, I-splines]:

![Condition numbers](generated/princals/nb_c_condition_numbers.png)

Each program's defaults (princals: a knot at every distinct value, ordinal):

![Defaults, chapter 4 data](generated/princals/nb_d_defaults_transformations_st_gender.png)

![Defaults, cylinders](generated/princals/nb_d_defaults_transformations_cylinder_0.png)

Local optima: 50 random starts, chapter 4 data, 2 knots:

![Local optima](generated/princals/nb_e_local_optima_st_gender_knots2.png)

Monotone, degree 2 (qlpca `monotone=True`, princals `ordinal=TRUE`):

![Monotone degree 2](generated/princals/nb_f_monotone_deg2_transformations_cylinder_0.png)

VAF summary over the data sets:

![VAF summary](generated/princals/compare_vaf_summary.png)
