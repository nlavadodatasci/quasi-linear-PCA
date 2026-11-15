# Screening data sets for a real application of qlPCA

Purpose: find a public, openly licensed data set of continuous variables on which qlPCA is worth applying in a
paper, using a criterion fixed **before** the analysis is run. This folder is not part of the `qlpca` package;
`src/qlpca/` is not changed by it.

Run from the repository root:

```bash
pip install -e .
python comparison/applications/screening.py              # Kelmarsh; Hill of Towie only if Kelmarsh fails
```

Data are downloaded at run time into `comparison/applications/.cache/` (git-ignored); results are written to
`comparison/applications/results/`.

## Approval criterion (fixed before running)

A candidate **passes** if, with **p = 2** components, the mean **out-of-sample** VAF of qlPCA with **1 or with
2 interior knots** is **at least 5 percentage points** higher than the mean out-of-sample VAF of linear PCA
(`n_knots=0`) on the same data and folds. The results for p = 3 and the in-sample VAF are reported but do not
enter the decision. The numbers are reported without interpretation.

If Kelmarsh does not pass, the same procedure is repeated with Hill of Towie.

## Procedure (same for every candidate)

* Sample: 2000 ten-minute records drawn at random from the filtered year with a fixed seed
  (`numpy.random.default_rng(2026)`), then sorted in time.
* Models: linear PCA (`QLPCA(n_knots=0)`), qlPCA with 1 and with 2 interior knots at the percentiles
  (`QLPCA(n_knots=1)`, `QLPCA(n_knots=2)`), degree 1, default mode (`random_state=0`), p = 2 and p = 3.
* In-sample VAF: fit on the 2000 records, `100 * sum(eigenvalues_[:p]) / m`.
* Out-of-sample VAF: the 2000 time-sorted records are split into **5 contiguous time blocks**; each block is the
  test set once, the other four the training set. Test rows are clipped to the training range of each variable
  (no linear extrapolation, DESIGN D11), transformed with the fitted f_j, and the VAF is
  `100 * ||F_test V||^2 / ||F_test||^2`, with V the training loading directions: the same criterion as
  `comparison/princals/knot_selection.py`. Mean and standard error over the 5 blocks.
* Also saved for qlPCA with 2 knots, p = 2, fitted on the whole sample: a figure of the transformations, a table
  of piecewise loadings, and the number of non-monotone transformations (for 1 and 2 knots).

## Candidate 1: Kelmarsh wind farm

* Source: Plumley, C. & Takeuchi, R., *Kelmarsh wind farm data*, Zenodo, CC-BY-4.0. Latest version checked on
  2026-11-15: record [16807551](https://zenodo.org/records/16807551), published 2025-08-12,
  doi:[10.5281/zenodo.16807551](https://doi.org/10.5281/zenodo.16807551); concept DOI (all versions)
  [10.5281/zenodo.5841833](https://doi.org/10.5281/zenodo.5841833). Data released by Cubico Sustainable
  Investments Ltd.
* Turbine **Kelmarsh 1** (Senvion MM92, 2050 kW rated, 78.5 m hub height, 92 m rotor), year **2019**
  (`Kelmarsh_SCADA_2019_3085.zip`, 311 MB, 52 560 ten-minute records, files
  `Turbine_Data_Kelmarsh_1_2019-...csv` and `Status_Kelmarsh_1_2019-...csv`). 2019 is a complete year in
  which the nine signals below are recorded (at most 4.6 % missing).

**Variables (9, ten-minute means).** They describe the operating state of the turbine through its main
physical relations, several of them known to be nonlinear (power curve, pitch regulation above rated wind
speed, rotor-speed limits, heating with load):

| variable | SCADA column |
|---|---|
| wind speed | `Wind speed (m/s)` |
| power | `Power (kW)` |
| rotor speed | `Rotor speed (RPM)` |
| generator speed | `Generator RPM (RPM)` |
| pitch angle | `Blade angle (pitch position) A (°)` |
| nacelle temperature | `Nacelle temperature (°C)` |
| generator temperature | `Stator temperature 1 (°C)` |
| gearbox temperature | `Gear oil temperature (°C)` |
| ambient temperature | `Nacelle ambient temperature (°C)` |

**Excluded**, and why:

* standard deviations, minima and maxima within each ten-minute interval: statistics of the same signals;
* `Long Term Wind`, the two wind sensors, density-adjusted wind speed: other versions of wind speed;
* `Gearbox speed`: a fixed multiple of rotor/generator speed;
* pitch angles B and C: duplicates of blade A;
* wind direction, nacelle position, vane and yaw angles: circular variables;
* energy counters, lost production, potential/theoretical power, availability and capacity indicators:
  derived or model quantities, not measurements;
* voltages, currents, apparent and reactive power, power factor, grid frequency: electrical quantities
  that follow power or the grid;
* bearing, transformer, converter, hub, CPU, motor-axis and other temperatures: further temperatures of the
  same kind (one each for nacelle, generator, gearbox and ambient is kept);
* pressures, accelerations, cable windings, metal particle counts: incomplete or condition-monitoring signals
  outside the operating-state description.

**Filtering** (in this order):

1. remove every ten-minute record whose interval `[t, t + 10 min)` overlaps an event of the status file with
   `Status` equal to **Stop** (stoppages), **Warning** (alarms) or **Communication** (data unavailable);
2. remove records with `Power (kW) <= 0` (turbine not producing: idling or stopped);
3. remove records with a missing value in any of the nine variables.

## Candidate 2: Hill of Towie wind farm (only if candidate 1 fails)

Clerc, A., Lingkan, E. et al., *Hill of Towie wind farm open dataset*, Zenodo, CC-BY-4.0; latest version
checked on 2026-11-15: 2.1.0, record [22662930](https://zenodo.org/records/22662930),
doi:[10.5281/zenodo.22662930](https://doi.org/10.5281/zenodo.22662930); concept DOI
[10.5281/zenodo.14870021](https://doi.org/10.5281/zenodo.14870021).

## Results (run on 2026-11-15)

### Kelmarsh 1, 2019

Filtering: 52 560 records; 12 387 removed for overlapping a Stop (118 events), Warning (87) or Communication
(10) event; 3 283 removed for `Power <= 0`; 0 for missing values; 36 890 remain. Sample of 2000 records from
2019-01-01 08:20 to 2019-12-31 19:30.

VAF (%), in sample and out of sample (5 contiguous time blocks; mean and standard error over the blocks):

| model | p | in sample | out of sample | se | out-of-sample gain over PCA |
|---|---|---|---|---|---|
| PCA | 2 | 72.32 | 72.69 | 2.77 | — |
| qlPCA, 1 knot | 2 | 76.42 | 75.73 | 3.30 | 3.04 |
| qlPCA, 2 knots | 2 | 80.63 | 80.50 | 1.69 | 7.81 |
| PCA | 3 | 86.26 | 86.07 | 0.77 | — |
| qlPCA, 1 knot | 3 | 86.83 | 86.42 | 1.07 | 0.35 |
| qlPCA, 2 knots | 3 | 89.55 | 89.18 | 0.91 | 3.12 |

**Criterion (p = 2, gain >= 5 points with 1 or 2 knots): passed** (2 knots: 7.81). Hill of Towie was therefore
not screened.

Non-monotone transformations (p = 2, whole sample): 4 of 9 with 1 knot (wind speed, power, rotor speed,
generator speed), 5 of 9 with 2 knots (the same and generator temperature).

Notes on the data, from the numbers: rotor speed and generator speed have correlation 1.0000 (to four
decimals) in the sample. The pitch angle equals its minimum (0°) in 58 % of the records, so its two percentile
knots coincide and are merged into one.

Files: `screening_summary.csv`, `kelmarsh_vaf.csv` (every block), `kelmarsh_transformations.pdf/.png`,
`kelmarsh_piecewise_loadings.csv`, `kelmarsh_monotonicity.csv`, `kelmarsh_sample_info.json` (filter counts,
sample, correlations).

## Case study: Kelmarsh 1, 2019 (`kelmarsh_case_study.ipynb`)

Same data, filtering and sample as the screening; `generator_speed` removed (correlation 1.0000 with
`rotor_speed`), 8 variables. Turbine data from the Senvion MM92 product page (archived copy of 2020-04-02 in the
Internet Archive): cut-in 3 m/s, nominal wind speed 12.5 m/s, nominal power 2,050 kW, rotor speed 7.5 – 15 rpm.
No source gives a pitch angle at which pitching starts, so the physical-knot model has no interior knot for the
pitch angle.

Models (p = 2): PCA; qlPCA with 2 percentile knots; qlPCA with physical knots (wind 3 and 12.5 m/s, power
2050 kW, rotor 7.5 and 15 rpm, temperatures at percentiles); both qlPCA models also with `monotone=True`.
Out-of-sample: 5 contiguous time blocks, criterion of `screening.py`, gain over PCA paired by block.

| model | VAF in sample | VAF out of sample (se) | paired gain over PCA (se) |
|---|---|---|---|
| PCA | 70.75 | 71.04 (3.51) | — |
| qlPCA, percentile knots | 79.60 | 79.52 (1.90) | 8.47 (1.77) |
| qlPCA, percentile knots, monotone | 77.32 | 76.74 (2.99) | 5.69 (0.70) |
| qlPCA, physical knots | 74.30 | 74.01 (3.30) | 2.96 (0.83) |
| qlPCA, physical knots, monotone | 73.89 | 73.61 (3.42) | 2.57 (0.73) |

Other facts from the notebook:

* March 2019 has no records after filtering: two "Oil filter gear choked" Warning events of 38 and 42 days
  (2019-02-24 to 2019-04-04, 2019-04-09 to 2019-05-21) are removed by the filtering rule fixed before the
  analysis; April and May keep 701 and 1153 of their records.
* With percentile knots the descending segments of wind speed, power and rotor speed hold 660, 660 and 659 of the
  2000 records, at low wind (median 4.35 m/s against 7.26 m/s elsewhere).
* With physical knots the decreasing segments are wind speed 2 – 3 m/s (63 records) and rotor speed 2 – 7.5 rpm
  (36 records).
* Projection of October – December from a January – September fit: no projected record is outside the training
  range, so no `ExtrapolationWarning` is issued; VAF in the projected period 75.88 (PCA), 81.13 (percentile
  knots), 79.14 (physical knots).

Results: `case_vaf.csv`, `case_vaf_blocks.csv`, `case_piecewise_loadings.csv`, `case_projection.csv`, figures
`case_operating_regimes`, `case_v_shapes`, `case_v_shapes_power_curve`, `case_piecewise_loadings`,
`case_biplot_regimes` (PDF and PNG).
