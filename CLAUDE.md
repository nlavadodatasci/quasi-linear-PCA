# qlPCA — reproducible notebooks for quasi-linear PCA

## Goal
Two purposes, both explicit at the top of the README: (1) a Python statistical
package for direct use (`from qlpca import QLPCA`), on which later application
repositories will depend; (2) the thesis replication described below.

A citable GitHub repository (Zenodo DOI) that re-implements in Python and
replicates, chapter by chapter, the results of Nuno Lavado's PhD thesis on
**qlPCA**: an extension of PCA to nonlinear relations through low-order spline
transformations of continuous variables, fitted by alternating least squares.
The original MATLAB files were lost; the listings printed in the thesis were
transcribed to `matlab_original/`.

Focus: qlPCA for continuous variables. HOMALS/CATPCA (categorical variables)
appear only as background and motivation, kept short.

Author: Nuno Lavado (ISEC, Polytechnic University of Coimbra). Repository
language: English. Proposed licences: MIT (code), CC BY 4.0 (text/notebooks).

## What is already here (prepared 2026-11-11)
- `docs/SPEC.md` — algorithm specification, pseudo-code vs MATLAB differences,
  verified targets, open questions. **Read this first.**
- `matlab_original/` — transcription of the thesis listings (not run in MATLAB).
- `prototype/qlpca_prototype.py` — literal numpy port of the listings;
  `prototype/check_against_thesis.py` shows it reproduces thesis Tables 4.3, 4.5-4.8.
  It is slow, loop-based reference code, not the package.
- `data/` — chapter 4 data set (Appendix A.1/A.2), Eurostat indicator codes for
  the WCE paper; see `data/README.md`.
- `tests/reference_values.json` — numbers printed in the thesis/paper, to be used
  as test targets.
- `references.bib` — own work, thesis reference list, and a few additions marked VERIFY.
- `IJAM_41_4_14.pdf` (Lavado & Calapez 2011, IJAM) is the English account of the
  algorithm, its properties and the cylinder study; cite it as the main reference.
- The four source PDFs sit in this folder but are git-ignored (link to them
  instead: thesis http://hdl.handle.net/10071/6202, WCE paper
  http://hdl.handle.net/10071/14371, IJAM paper on iaeng.org).

## Target layout
    src/qlpca/        splines.py (B/M/I-spline bases, pseudo-indicator matrix)
                      core.py (QLPCA estimator: fit, transform, inverse_transform)
                      diagnostics.py (VAF, communalities, piecewise loadings)
                      plots.py (scree, transformation plots, biplots)
                      datasets.py (S&T data loader, cylinder generator)
    notebooks/        01_splines.ipynb            thesis 1.1; MSc thesis ch. 4
                      02_pca_loss_function.ipynb  thesis 1.2-1.3; MSc ch. 2
                      03_from_catpca_to_qlpca.ipynb  thesis ch. 2 (brief; 2.2.2 discretisation)
                      04_qlpca_algorithm.ipynb    thesis 3.1-3.2
                      05_cylinder_problem.ipynb   thesis 3.3
                      06_science_technology_gender.ipynb  thesis ch. 4
                      07_wce2011_eurostat.ipynb   WCE 2011 paper (introduction only)
    tests/  data/  docs/  matlab_original/
    README.md  CITATION.cff  LICENSE  pyproject.toml  references.bib

## Work plan
1. **Package core.** `pyproject.toml`, `src/qlpca`. Vectorised spline basis
   (scipy `BSpline.design_matrix` + reverse cumulative sum) tested against the
   prototype basis. `QLPCA(n_components, degree=1, n_knots=1)` with a
   scikit-learn-like API. Options for `init={"matlab","pca"}` so both the printed
   pseudo-code and the MATLAB behaviour are available.
1b. **Modernised default (docs/DESIGN.md).** Commit the current state first.
   Then: `legacy=True` keeps today's behaviour and today's tests; the new default
   implements D1-D10. New tests: default reaches the legacy optimum for p = 2
   (68.90, 76.33) and at least 90.30 for r = 2, p = 3; result invariant to column
   order and to affine rescaling of the variables; B-spline basis, I-spline plus
   intercept and standardised I-splines give the same F from the same start; data
   with ties (chapter 4 data rounded to two decimals) fit without error;
   `monotone=True` gives non-negative coefficients, nondecreasing f_j, a loss not
   lower than the unrestricted one, and non-increasing loss history; fixed
   `random_state` gives identical results.
2. **Tests (pytest)** from `tests/reference_values.json`:
   PCA eigenvalues (Table 4.3); qlPCA eigenvalues, |loadings|, communalities
   (Tables 4.5-4.8) with tolerance 0.005; degree 1 with zero knots equals PCA;
   X'X = nI; F columns centred with norm sqrt(n); loss non-increasing;
   projection of training rows returns X; reconstruction with p = m returns H.
3. **Notebooks 01-04** (theory with figures: thesis Figs 1.1, 3.1-3.2).
4. **Notebooks 05-07** (replications). State clearly what is exact (chapter 4)
   and what is only qualitative (cylinder simulation, WCE Eurostat example).
5. **Release.** README with a results-vs-thesis table, `CITATION.cff` (with
   `preferred-citation` pointing to the IJAM 2011 paper and `references` to the
   thesis and WCE paper), LICENSE, GitHub Actions running pytest and executing the
   notebooks, GitHub release archived on Zenodo for the DOI.

6. **Library release (docs/PACKAGING.md, proposed 2026-11-14 — check that the author
   accepted it before starting).** The package must also work as an installed
   library. 0.1.0: tag + GitHub release of the current state. 0.2.0: scikit-learn
   compatible estimator, data loader fixed for wheel installs, input validation and
   over-parametrisation warning, single version source, README quick start on
   `make_cylinder`, `CITATION.cff` title, CHANGELOG, publish workflow (PyPI Trusted
   Publishing, TestPyPI first). 0.3.0: documentation site, comparison notebook with
   R `Gifi::princals`.

## Design: two modes (accepted by the author, 2026-11-11)
**Read `docs/DESIGN.md`.** It compares the options for each decision (start
values, centring, updates, monotone option, signs, ties, knots) with the evidence.
The package default is a modernised algorithm; `legacy=True` reproduces the 2012
MATLAB behaviour and is what the replication notebooks compare with the thesis.

## Status
- Done (2026-11-13): package complete; step 1b; notebooks 01-06 written and executed end
  to end (outputs saved in `notebooks/`); tests: `test_legacy.py`, `test_default.py`,
  `test_package_extras.py`, `test_cylinder.py` (qualitative Table 3.3).
- Notebook 05 (cylinder): qualitative only; means over 50 draws are within ~1 point of
  thesis Table 3.3 in all nine cells. Notebook 06 (chapter 4): exact for Tables 4.1,
  4.3-4.8 (<= 0.005, legacy=True); not unique: Fig. 4.5 (local optimum, start-dependent).
- Findings to keep in mind: (a) `monotone=True` reconstruction errors come only from flat
  segments of f_j; (b) `piecewise_loadings` selects segments on the standardised scale;
  (c) thesis pp. 52-53 matrices not reproduced (open question 3); (d) CATPCA/SPSS figures
  not reproduced; (e) PCA communality of mf_funds is 0.0252 exactly, 0.0244 only from the
  rounded Table 4.4 loadings; (f) in the two-knot biplot the package's component 2 has the
  opposite sign to the thesis's because of the arbitrary sign of mf_funds (SPEC 8);
  (g) the thesis describes the cylinder noise sd as "of each original variable": the
  linearised scale was chosen (the natural scale would give absurd noise for variable 12).
- Decided: `make_cylinder` measures the noise sd on the linearised scale; a, b ~ U(0,1)
  (also stated in the MSc thesis ch. 6).
- Blocked: the analysis of notebook 07 (WCE 2011 Eurostat): the data values and the reference
  year are not in the repository (open question 4).
- Notebook 07 exists as an introduction only (no analysis); the author will decide later.
- Notebook 08 dropped by the author (2026-11-13).
- Step 5 in progress (2026-11-13): README (results-vs-thesis table), LICENSE (MIT),
  LICENSE-docs.md (CC BY 4.0 for text/notebooks; data excluded), CITATION.cff (validated
  with cffconvert), `.github/workflows/ci.yml` (pytest on 3.10-3.13; notebooks executed
  with `tools/run_notebooks.py` on 3.12). Pushed 2026-11-13; CI green (pytest 3.10-3.13, notebooks).
- Step 6, 0.2.0 (2026-11-14; PACKAGING P1-P7 accepted, P4 = B; 0.1.0 will never be
  published, 0.2.0 is the first release archived on Zenodo): done and pushed, CI green.
  `QLPCA` is a scikit-learn transformer (`tests/test_sklearn.py`, estimator checks; default
  passes all, two legacy checks xfail for ties, listed in PACKAGING.md); input validation and
  `SmallSampleWarning` (`tests/test_validation.py`); DataFrames (`feature_names_in_`, used by
  the plots); chapter 4 data moved to `src/qlpca/data/` as package data, `QLPCA_DATA` kept;
  version 0.2.0 from pyproject via importlib.metadata; CHANGELOG.md; CI wheel job checks the
  README quick start and `load_st_gender()`; `.github/workflows/publish.yml` (manual run ->
  TestPyPI, published release -> PyPI, environments "testpypi"/"pypi"). Not yet published:
  no tag, no release; the PyPI/TestPyPI trusted publishers and GitHub environments must be
  configured by the author.
- Before the 0.2.0 GitHub release / Zenodo DOI: (1) `data/README.md` still says "check with
  Oliveira & Carvalho before redistributing" - update it once the permission is confirmed;
  (2) notebook 07 ships as an introduction only unless the author decides otherwise;
  (3) Zenodo linked to the GitHub account; (4) README installation still uses the GitHub URL
  (switch to `pip install qlpca` after the PyPI upload); (5) CHANGELOG date for 0.2.0.

- Comparison with princals (2026-11-14): `comparison/princals/` (README, manifest.json with explicit
  knots and input md5, compare.py, generate_r_ground_truth.R, install_pygifi.py, notebook
  qlpca_vs_princals.ipynb, results/, tests/). Gifi/pyGifi are GPL-3: only called/imported there, never
  copied into src/qlpca. pyGifi pinned to 7aa81ed; its fits cached in results/pygifi/. Gas turbine
  (UCI, CC BY 4.0) downloaded at run time to .cache/ (year 2015, n = 400, seed 2015, 9 variables).
  Workflows: `.github/workflows/r-ground-truth.yml` (manual; run once, JSON committed) and an optional
  CI job (`continue-on-error`). Local notes `comparison/princals/notes/` are git-ignored: never add them.
  Findings: same knots, degree 1, no monotonicity -> qlpca = pyGifi within tolerance in 12/12 cases;
  monotone degree 1 equal, degree 2 differ (different feasible sets; princals' Dykstra result not exactly
  isotone); pyGifi's default start (SVD) differs from R's (set.seed(123)); cylinder inputs rounded to
  10 significant digits because make_cylinder differs in the last bit across platforms.
  R ground truth (R 4.6.2, Gifi 1.0-0) committed 2026-11-15 in results/r_ground_truth/: qlpca = Gifi (R)
  within tolerance in 12/12 same-knot cases and 3/3 monotone degree 1; with defaults pyGifi and R differ on
  the cylinder data (and slightly on neumann). R JSON loadings are (ndim, m); compare.py transposes them.

- D12 (choice of the number of knots) recorded in docs/DESIGN.md as "proposed for 0.3.0" (2026-11-15):
  recommendation B (`score(X)`, out-of-sample VAF with new rows clipped to the training range, for
  GridSearchCV) and C (`QLPCACV`, grid {1,2,3,4}, cv=5, one-SE rule, n >= 5(degree+knots), cv_results_);
  D (automatic choice inside QLPCA) rejected; depends on `extrapolate="clip"` (D11). Not implemented:
  src/qlpca/ unchanged. Evidence: `comparison/princals/knot_selection.py` -> results/knot_selection.csv.

- Application screening (2026-11-15): `comparison/applications/` (README with the approval criterion committed
  before the run, screening.py, results/). Kelmarsh (Zenodo 16807551, CC-BY-4.0), turbine 1, 2019, 9 SCADA
  variables, n = 2000 (seed 2026), 5 contiguous time blocks: p = 2 out-of-sample VAF PCA 72.69, qlPCA 2 knots
  80.50 (gain 7.81 >= 5): passed; Hill of Towie not screened. Data downloaded to .cache/ (git-ignored).
  Case study `comparison/applications/kelmarsh_case_study.ipynb` (2026-11-15): 8 variables (generator_speed
  dropped), Senvion MM92 data from the archived product page; paired out-of-sample gain over PCA: percentile
  knots 8.47 (se 1.77), physical knots 2.96 (0.83). March 2019 empty after filtering (two long "Oil filter gear
  choked" warnings). No pitch value for a physical knot in the sources (none estimated).

## Conventions
- Percentiles must follow MATLAB `prctile`: `np.percentile(x, q, method="hazen")`.
- Standardise H with the sample sd (ddof=1); F has norm sqrt(n) (ddof=0).
- Loadings are defined up to sign per component, and with non-monotone splines
  also per variable (SPEC section 8). Fix and document a sign convention; compare
  with the thesis in absolute value or after aligning signs.
- Never change numbers in `tests/reference_values.json` to make tests pass; if a
  thesis number cannot be reproduced, record it in the README table.
- Notebooks import from `qlpca`; no algorithm code duplicated in notebooks.
- Cite sources in each notebook (BibTeX keys from `references.bib`).

## Open questions (ask the author, do not guess)
See `docs/SPEC.md` section 9: convergence criterion, cylinder simulation
settings, illustrative matrices on thesis pp. 52-53, Eurostat reference year,
permission to redistribute the chapter 4 data, CATPCA comparisons (SPSS).
