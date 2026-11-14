# qlPCA — quasi-linear principal components analysis

[![CI](https://github.com/nlavadodatasci/quasi-linear-PCA/actions/workflows/ci.yml/badge.svg)](https://github.com/nlavadodatasci/quasi-linear-PCA/actions/workflows/ci.yml)

This repository has two purposes:

1. **A Python quant package, `qlpca`, for direct use in data analysis** —
   `from qlpca import QLPCA`, then `fit` / `transform` — implementing **quasi-linear PCA (qlPCA)**.
   `QLPCA` is a **scikit-learn transformer**: it works with `Pipeline`, `clone`, `GridSearchCV`,
   `set_output` and pandas DataFrames (see [Use with scikit-learn](#use-with-scikit-learn)).
2. **A chapter-by-chapter replication** of the PhD thesis in which the method was proposed
   (Lavado, 2012), in notebooks that use the package.

qlPCA extends principal components analysis to nonlinear relations by transforming each continuous
variable with a **low-order spline** — usually piecewise linear, with one to three interior knots —
chosen jointly with the components by alternating least squares. Continuous variables enter directly,
with no discretisation step.

The listings printed in the thesis are transcribed in
[`matlab_original/`](matlab_original/), and a literal numpy port of them is in
[`prototype/`](prototype/); the package in [`src/qlpca`](src/qlpca) is tested against both.

## Installation

Requires Python 3.10 or later.

```bash
pip install "git+https://github.com/nlavadodatasci/quasi-linear-PCA"
# or, to run the tests and notebooks from a clone:
git clone https://github.com/nlavadodatasci/quasi-linear-PCA
cd quasi-linear-PCA
pip install -e ".[dev,notebooks]"
```

## Quick start

```python
from qlpca import QLPCA
from qlpca.datasets import make_cylinder

X, names = make_cylinder(n=51, noise=0.10, random_state=0)   # simulated "cylinder problem": 12 variables
model = QLPCA(n_components=2, degree=1, n_knots=2, random_state=0).fit(X)

print(model.eigenvalues_)         # variance accounted for by each component
print(model.loadings_.shape)      # nonlinear loadings, one row per variable
print(model.communalities_)       # per-variable fit
print(model.transform(X[:3]))     # project (new) observations on the components (degree 1)
print(model.inverse_transform(model.scores_).shape)   # reconstruct data from object scores (degree 1)
```

The thesis chapter 4 data are shipped with the package: `X, variables, countries = load_st_gender()`
(see [Data](#data)). `QLPCA` is a scikit-learn transformer: it accepts DataFrames (the column names are kept
in `feature_names_in_` and used by the plots) and works in pipelines.

Diagnostics (`qlpca.diagnostics`: VAF, communalities, piecewise loadings) and plots
(`qlpca.plots`: scree plots, transformation plots, biplots) work on a fitted model.

### Two modes

* **Default** — a modernised algorithm for new analyses: centred B-spline basis, PCA start plus random
  starts (keeps the lowest loss), truncated-SVD updates, ties allowed, optional `monotone=True`
  (non-decreasing transformations), knots per variable.
* **`legacy=True`** — the behaviour of the 2012 MATLAB code, used to compare with the thesis number
  by number.

The choices and the evidence behind them are in [`docs/DESIGN.md`](docs/DESIGN.md); the algorithm as
printed in the thesis, and how it differs from the MATLAB listings, in [`docs/SPEC.md`](docs/SPEC.md).
The thesis pseudo-code omits one essential step — centring the spline basis (an intercept) — which the
MATLAB code performs; notebook 04 gives the corrected pseudo-code.

## Use with scikit-learn

`QLPCA` is a scikit-learn transformer (`BaseEstimator`, `TransformerMixin`), tested with scikit-learn's
estimator checks. What works:

* `clone`, `get_params` / `set_params`, `Pipeline`, `GridSearchCV` (inside a pipeline with a supervised
  step, or with your own scorer: `QLPCA` has no `score` method);
* `set_output(transform="pandas")`, `get_feature_names_out()` (`qlpca0`, `qlpca1`, ...);
* pandas DataFrames as input: the column names are kept in `feature_names_in_`, used in warnings and
  plots, and checked again by `transform`.

```python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

pipe = make_pipeline(QLPCA(n_components=2, n_knots=2, random_state=0), Ridge())
```

Keep in mind what the method is for:

* **qlPCA is descriptive.** It fits and interprets one sample: the transformations, loadings and scores
  describe the data they were fitted on. It is not a predictive model and makes no claim about new data.
* **New data only with degree 1.** `transform`, `transform_variables` and `inverse_transform` are defined
  for piecewise-linear transformations (`degree=1`) only.
* **Linear extrapolation.** For a new observation outside the training range of a variable, the
  transformation is extended linearly from its first or last segment. Nothing in the data supports it
  there, and a single extreme value can dominate the scores. `transform` and `transform_variables` then
  issue an `ExtrapolationWarning` with the number of observations and the variables concerned. Validation
  folds in cross-validation are new data, so expect the warning there.
* **Heavy tails.** With heavy-tailed variables (for example `a/b`, `b/a` and `2*a*pi/b^2` in `make_cylinder`) the
  scores of new observations can reach values around 100 when the training scores stay within ±2.5.
  Transform such variables before fitting (e.g. `log`), or do not project new data.

See [`docs/DESIGN.md`](docs/DESIGN.md), D10 and D11.

## Notebooks

| notebook | source | content |
|---|---|---|
| [01 Spline transformations](notebooks/01_splines.ipynb) | thesis §1.1 | B-, M- and I-splines, knots, pseudo-indicator matrix; thesis Fig. 1.1 |
| [02 PCA as loss minimisation](notebooks/02_pca_loss_function.ipynb) | thesis §1.2–1.3 | the common loss function; PCA, MCA and qlPCA as spline classes |
| [03 From CATPCA to qlPCA](notebooks/03_from_catpca_to_qlpca.ipynb) | thesis ch. 2 | HOMALS and CATPCA in brief; discretisation of continuous variables (Table 2.2) |
| [04 The qlPCA algorithm](notebooks/04_qlpca_algorithm.ipynb) | thesis §3.1–3.2 | corrected pseudo-code, legacy vs default, properties; Figs 3.1, 3.2 |
| [05 The cylinder problem](notebooks/05_cylinder_problem.ipynb) | thesis §3.3 | simulation study; Table 3.3, Figs 3.4–3.7 (**qualitative**) |
| [06 Science & technology: gender differences](notebooks/06_science_technology_gender.ipynb) | thesis ch. 4 | the application; Tables 4.1–4.8, Figs 4.1–4.10 (**exact**) |
| [07 Eurostat economic indicators](notebooks/07_wce2011_eurostat.ipynb) | WCE 2011 | introduction only (see below) |

The notebooks are stored with their outputs. They import everything from `qlpca`; no algorithm code is
duplicated in them.

## Results compared with the thesis

Chapter 4 is an **exact** replication (`legacy=True`). The tolerance for the qlPCA tables is 0.005,
the rounding of the printed two-decimal values.

| thesis | quantity | thesis | this repository | |
|---|---|---|---|---|
| Table 4.1 | summary statistics of the 11 variables | — | max. difference 0.02 | the thesis prints the mean of `m_top` as 19.4 (data: 19.42) |
| Table 4.3 | PCA eigenvalues | 4.276, 2.531, 1.878, … | 4.276, 2.532, 1.878, … | max. difference 0.0006 |
| Table 4.4 | PCA loadings (absolute values) | — | max. difference 0.005 | |
| Table 4.5 | qlPCA, 1 knot: eigenvalues, % VAF | 4.582, 2.998; 68.90 % | 4.582, 2.998; 68.90 % | |
| Table 4.5 | qlPCA, 2 knots: eigenvalues, % VAF | 4.760, 3.637; 76.33 % | 4.759, 3.637; 76.33 % | |
| Table 4.6 | qlPCA loadings (absolute values) | — | max. difference 0.005 | signs are a convention (see below) |
| Tables 4.7, 4.8 | communalities | — | max. difference 0.005, 0.003 | |
| p. 100 | PCA communality of `mf_funds` | 0.0244 | 0.0252 | the thesis value comes from the rounded loadings of Table 4.4 |
| Fig. 4.5 | 2 knots, 3 components | scree plot only | 90.00 % (legacy, thesis column order) or 90.30 % | local optima; the result depends on the start, both are reported |
| Fig. 1.1 | $I_1(0.58)$, $I_2(0.58)$, $f(0.58)$ | 1, 0.29, 1.37 | 1, 0.29, 1.37 | |
| Table 2.2 | discretisation by *multiplying* | — | identical | with the population standard deviation |

**Qualitative** comparisons, where the thesis data cannot be recovered:

| thesis | quantity | thesis | this repository |
|---|---|---|---|
| Table 3.3 | cylinder problem, % variance in 2 dimensions, PCA / qlPCA 2 knots / qlPCA 3 knots | no noise 82.15 / 97.77 / 98.43; noise 10 % 80.18 / 96.90 / 97.68; noise 25 % 72.62 / 92.59 / 94.41 | means of 50 simulated data sets: 82.67 / 98.14 / 98.80; 80.95 / 96.95 / 97.79; 73.47 / 92.57 / 93.92 (thesis values inside the simulated range in every cell) |
| Fig. 3.4 | PCA scree plot, noise 25 % | two breaks (3rd and 6th component) | one clear break in the simulated draw: not reproduced |

**Not reproduced**

| source | what | why |
|---|---|---|
| thesis pp. 52–53 | illustrative pseudo-indicator matrices | the printed code gives different values; probably other boundary knots (open question) |
| thesis §2.2.2, ch. 4; WCE Table II | CATPCA results | computed with SPSS |
| WCE 2011, §V | Eurostat example (58 % / 64 % / 54 %) | data values and reference year not printed in the paper; notebook 07 is an introduction only |

**Signs.** A loading is defined up to sign per component and — for a non-monotone transformation — also
per variable. The package orients each transformed variable so that $\mathrm{corr}(f_j, h_j) \ge 0$ and each
component so that its largest loading is positive (the convention of MATLAB's `biplot`). Comparisons with the
thesis are made in absolute value or after aligning signs. In the two-knot solution of chapter 4 the
V-shaped variable `mf_funds` is oriented in opposite ways by the package and by the thesis; the thesis reads
its sign change between one and two knots as substantive (p. 92), but part of it is this indeterminacy.

## Tests

```bash
python -m pytest            # package, thesis tables (legacy), modern default, cylinder study
python tools/run_notebooks.py   # execute every notebook
```

The numbers printed in the thesis and papers that serve as test targets are in
[`tests/reference_values.json`](tests/reference_values.json) and are never edited to make a test pass.

## Data

* `src/qlpca/data/st_gender_eu15.csv` — chapter 4 data, shipped with the package (`qlpca.datasets.load_st_gender()`) (thesis Appendix A.1): 11 indicators of gender differences in
  science and technology for 15 EU countries, 2003–2004, compiled by Oliveira & Carvalho (2009) from Eurostat
  and *She Figures 2006*, and provided by them for the thesis. See [`data/README.md`](data/README.md).
* The cylinder data are simulated (`qlpca.datasets.make_cylinder`).

## How to cite

Please cite the article that describes the method (see also [`CITATION.cff`](CITATION.cff)):

> Lavado, N. & Calapez, T. (2011). Principal Components Analysis with Spline Optimal Transformations for
> Continuous Data. *IAENG International Journal of Applied Mathematics*, 41(4), 367–375.
> <https://www.iaeng.org/IJAM/issues_v41/issue_4/IJAM_41_4_14.pdf>

Related work:

* Lavado, N. (2012). *Extensões via splines da Análise em Componentes Principais*. PhD thesis, ISCTE –
  Instituto Universitário de Lisboa. <http://hdl.handle.net/10071/6202>
* Lavado, N. & Calapez, T. (2011). Quasi-Linear PCA: Low Order Spline's Approach to Non-Linear Principal
  Components. *Proceedings of the World Congress on Engineering 2011*, vol. I, 360–364.
  <http://hdl.handle.net/10071/14371>

Full bibliography: [`references.bib`](references.bib).

## Related work and software

qlPCA builds on the Gifi framework of nonlinear multivariate analysis (Gifi, 1990), in which variables
are optimally transformed by alternating least squares; CATPCA (SPSS) belongs to the same family.
The closest software is `princals` in the R package
[Gifi](https://CRAN.R-project.org/package=Gifi) (Mair, De Leeuw & Groenen; GPL-3) and its Python port
[pyGifi](https://github.com/vam-math/pyGifi) (GPL-3).

`qlpca` is an independent implementation, written from the author's PhD thesis (Lavado, 2012) and the
method's original paper (Lavado & Calapez, 2011). It contains no code from the Gifi R package or from
pyGifi.

* Gifi, A. (1990). *Nonlinear Multivariate Analysis*. Chichester: Wiley.

## Acknowledgements

The method was developed under the supervision of
[Teresa Calapez](https://orcid.org/0000-0002-8584-2906) (ISCTE-IUL), who supervised the PhD thesis
(Lavado, 2012) and is co-author of the paper that introduced qlPCA (Lavado & Calapez, 2011).

## Licence

Code, including the code cells of the notebooks: [MIT](LICENSE).
Text and figures (notebook prose and outputs, `docs/`, this README): [CC BY 4.0](LICENSE-docs.md).
The chapter 4 data are not covered by these licences; see [`data/README.md`](data/README.md).

Author: Nuno Lavado, Coimbra Institute of Engineering (ISEC), Polytechnic University of Coimbra.
