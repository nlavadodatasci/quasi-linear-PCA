# qlpca compared with princals (Gifi)

This folder compares `qlpca` with `princals` from the Gifi system, in its two implementations:

* **Gifi** 1.0-0, the R package (Mair, De Leeuw & Groenen; CRAN 2025-08-28; GPL-3);
* **pyGifi**, its Python port, pinned to commit
  [`7aa81ed`](https://github.com/vam-math/pyGifi/tree/7aa81ed) (GPL-3).

It covers the pipelines step by step, results with the same knots, each package's defaults, local optima and
starts, monotone transformations, and what only `qlpca` does. The notebook
[`qlpca_vs_princals.ipynb`](qlpca_vs_princals.ipynb) presents it; the numbers are in
[`results/summary.csv`](results/summary.csv) and the figures in [`results/figures/`](results/figures/).

The folder is not part of the `qlpca` package and is not needed to use it.

## Licences

Gifi and pyGifi are GPL-3. In this folder they are only **called** (R) or **imported** (Python); the notebook
quotes a few of their source lines with file and line number. No code of either package is copied or translated
into `src/qlpca/`, and the package was not changed after reading their code. No data of theirs is stored here:
`neumann` is loaded from pyGifi at run time. The code in this folder is MIT, like the rest of the repository.

## Files

| file | content |
|---|---|
| `manifest.json` | data sets, explicit knots per variable (`knots`), degree, `ndim`, `ordinal`, `eps`, tolerances; md5 of each input (`inputs`) |
| `compare.py` | loads the data, runs qlpca and pyGifi, compares with the R results (insensitive to sign changes), writes `results/summary.csv` and `results/figures/` |
| `generate_r_ground_truth.R` | runs `Gifi::princals` with the manifest's knots and with the package defaults; writes `results/r_ground_truth/<data set>.json` |
| `install_pygifi.py` | installs pyGifi at `7aa81ed` and compiles its C extension `pygifi_rng` |
| `requirements.txt` | Python requirements of this folder |
| `qlpca_vs_princals.ipynb` | the comparison, sections a)-h) |
| `tests/` | quick checks (skipped when pyGifi is not installed) |
| `results/pygifi/` | cached pyGifi fits (the monotone degree-2 fits take minutes) |

## Data

| name | source | n × m |
|---|---|---|
| `cylinder_0`, `cylinder_10`, `cylinder_25` | `qlpca.datasets.make_cylinder(n=51, noise=0 / 0.10 / 0.25, random_state=0)`, rounded to 10 significant digits (log, exp and arctan can differ in the last bit between platforms; rounding makes the input, and its md5, the same everywhere) | 51 × 12 |
| `st_gender` | `qlpca.datasets.load_st_gender()` (thesis chapter 4) | 15 × 11 |
| `neumann` | `pygifi.get_dataset("neumann")`, loaded at run time (also `data(neumann)` in Gifi) | 65 × 3 |
| `gas_turbine` | UCI *Gas Turbine CO and NOx Emission Data Set*, doi:[10.24432/C5WC95](https://doi.org/10.24432/C5WC95), CC BY 4.0 (Kaya, Tüfekci & Uzun 2019) | 400 × 9 |

**Gas turbine.** The zip file is downloaded at run time into `.cache/` (git-ignored). Sample: year **2015**
(`gt_2015.csv`, 7384 hourly records), 400 rows drawn with `pandas.DataFrame.sample(random_state=2015)`.
Variables: the nine process and ambient measurements AT (ambient temperature), AP (ambient pressure),
AH (ambient humidity), AFDP (air filter difference pressure), GTEP (gas turbine exhaust pressure),
TIT (turbine inlet temperature), TAT (turbine after temperature), TEY (turbine energy yield) and
CDP (compressor discharge pressure). All are continuous sensor readings. TIT and TAT pile up just below their
maxima (2015: 75th percentile 1099.9 vs maximum 1100.4 for TIT; 550.03 vs 550.59 for TAT), so their relations
with the load variables (TEY, CDP, GTEP) bend: a case for nonlinear transformations. CO and NOX are left out: they are the
emissions that the original study predicts from the other nine, not descriptors of the operating state, and CO
is heavy-tailed. NOX is used in section g) as the response of a pipeline example; the 2014 file provides new
observations for `transform`.

## How to run

From the repository root:

```bash
pip install -e .
pip install -r comparison/princals/requirements.txt     # pyGifi without its C extension
python comparison/princals/install_pygifi.py            # optional: pyGifi with pygifi_rng (needs a C compiler)

cd comparison/princals
python compare.py manifest     # only when data or settings change: knots and input md5 -> manifest.json
python compare.py              # qlpca + pyGifi (+ R if present) -> results/summary.csv, results/figures/
python compare.py --refresh    # same, recomputing the cached pyGifi fits
python -m pytest tests         # quick checks
```

`pygifi_rng` (R's random number generator) is needed only to reproduce R's random start in Python
(`r_seed=123`); everything else works without it.

### R ground truth

R is not needed locally. The manual workflow **R ground truth (princals)**
(`.github/workflows/r-ground-truth.yml`, Actions → *Run workflow*) installs R, Gifi 1.0-0 and jsonlite,
exports the inputs with `python compare.py inputs`, runs `generate_r_ground_truth.R` and publishes the JSON
files as the artefact `r-ground-truth`. Download them into `results/r_ground_truth/`, commit them, and rerun
`python compare.py` and the notebook: the R columns of `summary.csv` and of the notebook tables are then
filled in, and the R checks in `tests/` run. Each JSON records the R and Gifi versions and the md5 of the input
it was computed on; `compare.py` refuses a JSON whose input does not match the manifest.

Until then every comparison with R is marked **pending**; all numbers come from qlpca and pyGifi.

## Settings

* Same knots: the qlpca percentile knots (1 and 2 interior knots), written explicitly in the manifest;
  degree 1; `ordinal=FALSE`; `eps = 1e-12`; `itmax = 100000`.
* Monotone: 2 knots, degree 1 and 2, `ordinal=TRUE` vs `monotone=True`; `eps = 1e-8`; data sets
  `cylinder_0`, `st_gender`, `neumann` only (the inner Dykstra loop of princals stops at 1e-6, and one
  pyGifi fit of degree 2 takes several minutes).
* Defaults: `princals(data, ndim = 2)` and `QLPCA(n_components=2)`.
* Tolerances (manifest): both programs stop when the loss changes by less than `eps` in one iteration, which
  does not bound the distance to the optimum; with `eps = 1e-12` the two implementations still differ by about
  1e-6 in the eigenvalues. Transformations are compared through 1 − |correlation|; object scores and loadings
  after aligning signs.

## References

Gifi (1990); Lavado & Calapez (2011); Lavado (2012); Mair, De Leeuw & Groenen (2025), Gifi R package;
Perumal et al., pyGifi; Kaya, Tüfekci & Uzun (2019). BibTeX keys in [`references.bib`](../../references.bib):
`gifi1991nonlinear`, `lavado2011ijam`, `lavado2012phd`, `mair2025gifi`, `perumal2025pygifi`,
`kaya2019gasturbine`, `uci2019gasturbine`.
