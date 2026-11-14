# Packaging: `qlpca` as an installable Python library

Status: **P1-P7 accepted by the author on 2026-11-14 (P4: option B).** Same format as
`docs/DESIGN.md`: options compared, evidence, recommendation.

The repository has two purposes: (1) a Python statistical package for direct use
(`from qlpca import QLPCA`) and (2) the replication of the PhD thesis. Later
repositories with applications of qlPCA will depend on the package and cite it.
This file is about purpose (1).

## Findings on the current state (0.1.0, checked 2026-11-14)

- 74 tests pass. A wheel builds and contains only the library (6 modules).
- **Bug:** installed from a wheel (not `pip install -e`), `load_st_gender()` raises
  `FileNotFoundError`: it looks for `data/` two levels above the module, which only
  exists in a clone. The README quick start therefore fails for library users.
- `QLPCA` is not scikit-learn compatible: no `get_params`/`set_params`, no `clone`,
  no informative `repr`.
- Missing values give an opaque error from scipy ("Array must not contain infs or nans").
- No warning for over-parametrised fits: 5 observations with 3 interior knots fit
  "100%" silently (the thesis recommends n >= 5w, w = degree + knots).
- The version string lives in three places (`pyproject.toml`, `__init__.py`, `CITATION.cff`).
- `CITATION.cff` title still reads "reproducible notebooks".
- No tag, GitHub release or DOI yet.

## P1. Distribution channel

| option | pros | cons |
|---|---|---|
| A. install from GitHub only (today) | nothing to maintain | long command; no versions to pin; application repos cannot declare `qlpca>=0.2` cleanly |
| B. PyPI, published from GitHub Actions with Trusted Publishing | `pip install qlpca`; versions; no stored tokens; the standard for Python | a name is permanent; needs a PyPI account and a one-time setup |
| C. B plus conda-forge | reaches conda users | a second recipe to maintain; only worth it with users |

The names `qlpca`, `quasi-linear-pca` and `pyqlpca` were free on PyPI on 2026-11-14.
**Recommendation: B now, C on demand.** Rehearse on TestPyPI first.

## P2. Programming interface

| option | pros | cons |
|---|---|---|
| A. plain class on numpy/scipy (today) | two light dependencies | does not work in `Pipeline`, `clone`, `GridSearchCV`; users must learn a new interface |
| B. scikit-learn estimator (`TransformerMixin`, `BaseEstimator`), tested with scikit-learn's estimator checks | drop-in next to `PCA`/`KernelPCA`; pipelines, cross-validation, pandas output via `set_output` | adds scikit-learn as a dependency; a few naming conventions to follow |
| C. imitate the interface without the dependency | light | re-implements what scikit-learn gives for free and drifts from it |

**Recommendation: B.** The intended users already have scikit-learn installed.
Keep the current attribute names and add the conventional ones (`n_features_in_`,
`feature_names_in_`, `get_feature_names_out`). Some estimator checks will not
apply (`legacy=True` needs unique values); document each exclusion.

## P3. Repository layout

| option | pros | cons |
|---|---|---|
| A. one repository: library + replication notebooks (today) | one DOI, one citation; the thesis tables are permanent regression tests | the README must serve two audiences |
| B. two repositories (library; thesis replication) | each one is simpler | two DOIs to cite; the replication can silently break when the library changes |

**Recommendation: A.** The wheel already contains only the library. The README
opens with the two purposes and sends each reader to their section.

## P4. Example data inside the package

| option | pros | cons |
|---|---|---|
| A. data outside the package (today) | nothing | loader fails after `pip install` |
| B. ship the chapter 4 table as package data (`importlib.resources`) | loader works everywhere | redistributes the data in every wheel: needs the permission that is still pending |
| C. ship only the simulated cylinder data (`make_cylinder`); the chapter 4 loader works from a clone and otherwise raises a clear error | no licensing question; quick start works everywhere | the real-data example is not available to `pip` users |

**Recommendation: C now, B as soon as the permission is confirmed.** The README
quick start should use `make_cylinder`.

**Decision (author, 2026-11-14): B.** The chapter 4 data go into the package in the
first release archived on Zenodo; `load_st_gender()` must then work after `pip install`
(package data read with `importlib.resources`), and the CI wheel check must stop using
`--expect-no-data`. Until that release, C stays in place.

## P5. Robustness expected from a library

- Input validation with clear messages (NaN/inf, constant columns, too few rows).
- A warning when n < 5 (degree + knots) for some variable.
- Accept pandas DataFrames; keep variable names for loadings and plots.
- One source for the version (read it from the package metadata).
- `CHANGELOG.md`; semantic versioning; `0.x` while the interface may still change.
- Missing data: not supported in 0.2; say so. (Open design question for later.)

## P6. Documentation

| option | pros | cons |
|---|---|---|
| A. README + docstrings + notebooks (today) | already there | no searchable API reference |
| B. small documentation site built from the docstrings and notebooks (GitHub Pages) | API reference, user guide, gallery; expected by users and reviewers | one more thing to keep building in CI |

**Recommendation: A for 0.2, B for 0.3.**

## P7. Citation

- Zenodo: link the GitHub repository, then every GitHub release gets a DOI and the
  project gets a concept DOI that always points to the latest version. Application
  repositories cite the concept DOI plus the paper.
- `CITATION.cff`: change the title to name the package; add the DOI after the first release.
- Software paper: JOSS is free of charge but, since 2026, asks for at least six
  months of public development history before submission (the repository became
  public on 2026-11-13, so not before mid-May 2027) and looks at open-development
  signals (releases, issues). The methodological paper is a separate matter.

## P8. Related software (searched 2026-11-14)

| software | language | what it is | relation to qlPCA |
|---|---|---|---|
| `Gifi::princals` (Mair, De Leeuw, Groenen; CRAN, GPL-3) | R | nonlinear PCA by optimal scaling; spline degree and knots per variable, monotone or not | **closest relative.** Same family (Gifi system). To be compared numerically with `qlpca` on the same data; not yet done |
| `pyGifi` (GitHub, GPL-3) | Python | port of R Gifi (Homals, Princals, Morals); not on PyPI, early stage; its README says `transform` does not project new observations | closest in Python, but a port of the categorical framework |
| SPSS CATPCA | proprietary | reference implementation used in the thesis | not reproducible without a licence |
| `prince` (PyPI) | Python | PCA, CA, MCA, MFA, FAMD | linear/factorial methods, no spline transformations |
| scikit-learn `KernelPCA`; `SplineTransformer` + `PCA` | Python | other nonlinear routes | no joint optimisation of transformations and components |

No package named qlPCA, and no Python package implementing it, was found.
What `qlpca` offers beyond the above, to be stated carefully in the README and
the paper: continuous variables entered directly, projection of new observations,
reconstruction, piecewise loadings, multiple starts, a scikit-learn interface.
**A notebook comparing `qlpca` with `Gifi::princals` (linear splines, same knots)
is needed before any claim of difference in results**; it also replaces the SPSS
comparisons that cannot be reproduced.

## Estimator checks excluded (0.2.0)

`tests/test_sklearn.py` runs scikit-learn's `parametrize_with_checks` on
`QLPCA(n_init=2)` (default mode) and `QLPCA(legacy=True)`. The default mode passes
every check. Two checks are marked as expected failures, for `legacy=True` only:

| check | mode | reason |
|---|---|---|
| `check_estimators_dtypes` | `legacy=True` | fits integer-valued data, which has ties; `legacy=True` requires unique values within each variable, as the 2012 MATLAB code did (DESIGN D7) |
| `check_positive_only_tag_during_fit` | `legacy=True` | same reason: the data of this check have ties, so the error raised is the ties error, not a refusal of negative values |

The legacy mode exists to reproduce the thesis number by number, so its restriction
is kept rather than relaxed.

## Proposed sequence

- **0.1.0** — tag the current state as the thesis-replication release (GitHub release + Zenodo DOI).
- **0.2.0** — library release: P2-B, P4-C, P5, `CITATION.cff`, README quick start, PyPI via Trusted Publishing.
- **0.3.0** — documentation site; comparison notebook with `Gifi::princals`; per-variable `monotone`.
