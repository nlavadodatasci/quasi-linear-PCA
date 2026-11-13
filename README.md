# qlPCA — quasi-linear principal components analysis

[![CI](https://github.com/nlavadodatasci/quasi-linear-PCA/actions/workflows/ci.yml/badge.svg)](https://github.com/nlavadodatasci/quasi-linear-PCA/actions/workflows/ci.yml)

A Python re-implementation of **quasi-linear PCA (qlPCA)**, and a chapter-by-chapter replication of the
PhD thesis in which it was proposed (Lavado, 2012). qlPCA extends principal components analysis to
nonlinear relations by transforming each continuous variable with a **low-order spline** — usually
piecewise linear, with one to three interior knots — chosen jointly with the components by
alternating least squares. Continuous variables enter directly, with no discretisation step.

The original MATLAB files were lost. The listings printed in the thesis are transcribed in
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
from qlpca.datasets import load_st_gender

X, variables, countries = load_st_gender()        # thesis chapter 4: 15 countries, 11 indicators
model = QLPCA(n_components=2, degree=1, n_knots=1, random_state=0).fit(X)

model.eigenvalues_          # variance accounted for by each component (4.582, 2.998)
model.loadings_             # nonlinear loadings, one row per variable
model.communalities_        # per-variable fit
model.transform(X[:3])      # project (new) observations on the components (degree 1)
model.inverse_transform(model.scores_)   # reconstruct data from object scores (degree 1)
```

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

* `data/st_gender_eu15.csv` — chapter 4 data (thesis Appendix A.1): 11 indicators of gender differences in
  science and technology for 15 EU countries, 2003–2004, compiled by Oliveira & Carvalho (2009) from Eurostat
  and *She Figures 2006*, and provided by them for the thesis. See [`data/README.md`](data/README.md).
* The cylinder data are simulated (`qlpca.datasets.make_cylinder`).

## How to cite

Please cite the article that describes the method (see also [`CITATION.cff`](CITATION.cff)):

> Lavado, N. & Calapez, T. (2011). Principal Components Analysis with Spline Optimal Transformations for
> Continuous Data. *IAENG International Journal of Applied Mathematics*, 41(4), 367–375.

Related work:

* Lavado, N. (2012). *Extensões via splines da Análise em Componentes Principais*. PhD thesis, ISCTE –
  Instituto Universitário de Lisboa. <http://hdl.handle.net/10071/6202>
* Lavado, N. & Calapez, T. (2011). Quasi-Linear PCA: Low Order Spline's Approach to Non-Linear Principal
  Components. *Proceedings of the World Congress on Engineering 2011*, vol. I, 360–364.
  <http://hdl.handle.net/10071/14371>

Full bibliography: [`references.bib`](references.bib).

## Licence

Code: [MIT](LICENSE). Text and notebooks: [CC BY 4.0](LICENSE-docs.md).
The chapter 4 data are not covered by these licences; see [`data/README.md`](data/README.md).

Author: Nuno Lavado, Coimbra Institute of Engineering (ISEC), Polytechnic University of Coimbra.
