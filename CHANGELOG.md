# Changelog

All notable changes to `qlpca` are recorded here. The project follows
[semantic versioning](https://semver.org/); while the version is `0.x` the
interface may still change between minor versions.

Version 0.1.0 was never published: it existed only as the version number of the
thesis-replication state of the repository. 0.2.0 is the first release.

## Unreleased

- README, documentation site and `CITATION.cff` now give the Zenodo DOI (concept DOI
  10.5281/zenodo.24260718; version 0.2.0: 10.5281/zenodo.24260719) and installation by `pip install qlpca`.

## 0.2.0 — 2026-11-15

First public release: the Python package and the replication of the PhD thesis
(Lavado, 2012) in one repository.

### Package
- `QLPCA` estimator with two modes: a modernised default (docs/DESIGN.md, D1-D10)
  and `legacy=True`, which reproduces the 2012 MATLAB code and thesis Tables 4.3-4.8.
- scikit-learn transformer (`BaseEstimator`, `TransformerMixin`): `get_params` /
  `set_params`, `clone`, pipelines, `set_output`, `get_feature_names_out`; tested with
  scikit-learn's estimator checks (exclusions in docs/PACKAGING.md).
- Input validation with clear messages: NaN/inf, constant variables, fewer than two
  observations; `SmallSampleWarning` when n < 5 (degree + knots). Missing data are not
  supported.
- pandas DataFrames accepted; column names kept in `feature_names_in_` and used by the plots.
- `ExtrapolationWarning` (exported in `qlpca`): `transform` and `transform_variables` warn when
  observations fall outside the training range, with their number and the variables (at most five
  named); the warning points at the caller's line; the values
  returned (linear extrapolation) are unchanged (docs/DESIGN.md, D11). README section "Use with
  scikit-learn": what works, the descriptive nature of the method, extrapolation and heavy tails.
- Splines (`qlpca.splines`), diagnostics (`qlpca.diagnostics`), plots (`qlpca.plots`).
- Data: the chapter 4 data set ships with the package (`qlpca.datasets.load_st_gender`,
  not covered by the code licence); `make_cylinder` simulates the cylinder problem.
- New dependency: scikit-learn >= 1.6.

### Replication
- Notebooks 01-07 (07 is an introduction only); results compared with the thesis in the README.

### Documentation
- Documentation site (https://nlavadodatasci.github.io/quasi-linear-PCA/): MkDocs Material, API reference from the numpydoc docstrings
  (mkdocstrings), user guide, choice of knots (D12), thesis notebooks (mkdocs-jupyter, not re-executed),
  comparison with princals, Kelmarsh application, DESIGN.md and PACKAGING.md. Built with
  `mkdocs build --strict` in CI and deployed to GitHub Pages from `main`; extra `docs` in pyproject.toml.
- Docstrings of all public functions in numpydoc format (no change of signatures or defaults).

### Authors and metadata
- Software authors: Nuno Lavado and Teresa Calapez (`CITATION.cff`, `pyproject.toml`, README "Authors");
  `.zenodo.json` with the creators, licence, keywords and related identifiers for the Zenodo archive.

### Infrastructure
- Version read from the package metadata (single source in `pyproject.toml`).
- CI: tests on Python 3.10-3.13, notebook execution, wheel install with the README quick start.
- Publishing workflow to TestPyPI and PyPI through Trusted Publishing.
