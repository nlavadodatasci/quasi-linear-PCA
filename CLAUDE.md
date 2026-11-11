# qlPCA — reproducible notebooks for quasi-linear PCA

## Goal
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
                      07_wce2011_eurostat.ipynb   WCE 2011 paper
                      08_msc_spline_simulation.ipynb  optional, MSc ch. 6
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

## Design: two modes (accepted by the author, 2026-11-11)
**Read `docs/DESIGN.md`.** It compares the options for each decision (start
values, centring, updates, monotone option, signs, ties, knots) with the evidence.
The package default is a modernised algorithm; `legacy=True` reproduces the 2012
MATLAB behaviour and is what the replication notebooks compare with the thesis.

## Status
- Done: package core and 22 tests reproducing the thesis tables (this is the
  future `legacy=True` path).
- Next: step 1b below.

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
