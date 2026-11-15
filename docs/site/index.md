# qlpca — quasi-linear principal components analysis

The repository [nlavadodatasci/quasi-linear-PCA](https://github.com/nlavadodatasci/quasi-linear-PCA) has two
purposes:

1. **A Python package, `qlpca`, for direct use in data analysis.** `QLPCA` implements quasi-linear PCA (qlPCA)
   and is a scikit-learn transformer (`fit`, `transform`, `Pipeline`, `clone`, `GridSearchCV`, `set_output`,
   pandas DataFrames). See the [user guide](user-guide.md) and the [API reference](api/qlpca.md).
2. **A chapter-by-chapter replication of the PhD thesis** in which the method was proposed (Lavado, 2012), in
   notebooks that use the package. See [Thesis replication](replication/index.md).

qlPCA extends principal components analysis to nonlinear relations by transforming each continuous variable with a
**low-order spline** — usually piecewise linear, with one to three interior knots — chosen jointly with the
components by alternating least squares. Continuous variables enter directly, with no discretisation step.

**What it gives the user.** qlPCA reads as ordinary PCA with flexibility added knot by knot: with no interior
knots it *is* PCA, and each knot lets a variable's contribution change from one range of its values to the next.
That change is reported directly as **piecewise loadings** — the loading of each variable on each component,
segment by segment, on the original scale. The package starts from the PCA solution and adds random starts, keeping the
best and reporting the loss of each, which guards against local optima; it projects new observations and reconstructs data (degree 1); knots can
be placed at percentiles or at values that mean something in the application; and `QLPCA` works as a
scikit-learn transformer. On wind turbine SCADA data, knots placed at the turbine's control regions add 11 to 13
points of out-of-sample variance over linear PCA, in all six turbines of a wind farm — see the
[Kelmarsh application](kelmarsh.md). How qlPCA relates to `princals` (Gifi) is in
[Comparison with princals](princals.md).

## Installation

Requires Python 3.10 or later.

```bash
pip install "git+https://github.com/nlavadodatasci/quasi-linear-PCA"
# or, to run the tests and notebooks from a clone:
git clone https://github.com/nlavadodatasci/quasi-linear-PCA
cd quasi-linear-PCA
pip install -e ".[dev,notebooks]"
```

The package is not yet on PyPI; `pip install qlpca` will work after the first release.

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

The thesis chapter 4 data ship with the package: `X, variables, countries = qlpca.datasets.load_st_gender()`.

## How to cite

**Preferred citation** — the article that describes the method:

> Lavado, N. & Calapez, T. (2011). Principal Components Analysis with Spline Optimal Transformations for
> Continuous Data. *IAENG International Journal of Applied Mathematics*, 41(4), 367–375.
> <https://www.iaeng.org/IJAM/issues_v41/issue_4/IJAM_41_4_14.pdf>

**Software** — the Zenodo DOI of the repository is **pending** (it will be created with the first release).
Until then cite the repository URL and version, as in
[`CITATION.cff`](https://github.com/nlavadodatasci/quasi-linear-PCA/blob/main/CITATION.cff).

Related work:

* Lavado, N. (2012). *Extensões via splines da Análise em Componentes Principais*. PhD thesis, ISCTE –
  Instituto Universitário de Lisboa. <http://hdl.handle.net/10071/6202>
* Lavado, N. & Calapez, T. (2011). Quasi-Linear PCA: Low Order Spline's Approach to Non-Linear Principal
  Components. *Proceedings of the World Congress on Engineering 2011*, vol. I, 360–364.
  <http://hdl.handle.net/10071/14371>

Author: Nuno Lavado, Coimbra Institute of Engineering (ISEC), Polytechnic University of Coimbra.
