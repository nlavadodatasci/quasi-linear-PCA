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

**Filtering** (in this order; revised on 2026-11-15, see below):

1. remove every ten-minute record whose interval `[t, t + 10 min)` overlaps an event of the status file with
   `Status` equal to **Stop** (stoppages) or **Communication** (data unavailable);
2. remove records with `Power (kW) <= 0` (turbine not producing: idling or stopped);
3. remove records with a missing value in any of the nine variables.

Records overlapping a **Warning** event are kept and flagged in a column `warning_active`, which is not a model
variable; it is used in the figures and in a sensitivity analysis with and without those records.

**Missing values.** 2 428 records of 2019 (4.6 %) have missing values: 84 with all nine variables missing and
2 344 with some missing, from 2019-05-05 03:20 to 2019-05-21 09:50, during a "Data communication unavailable"
event. Step 1 removes 2 422 of them and step 2 the remaining 6 (their power is missing); step 3 finds none left.
They are removed because the gaps come from a technical failure (data not recorded or not transmitted),
independent of the values of the variables, and `qlpca` requires complete data. The counts are in
`results/kelmarsh_sample_info.json` (`missing_values`).

**Why Warning events are no longer removed.** The first run (rule above with Warning included in step 1) reported
"0 removed for missing values", which hid the missing values above (all were already removed by step 1), and
removed about 80 days in which the turbine produced power: two "Oil filter gear choked" warnings (code 1860) of
38.8 and 42.0 days (2019-02-24 to 2019-04-04 and 2019-04-09 to 2019-05-21), which emptied March and most of April
and May. A warning does not stop the turbine; removing its whole duration removes normal operation. The screening
was run again with the revised rule; the approval criterion did not change.

## Candidate 2: Hill of Towie wind farm (only if candidate 1 fails)

Clerc, A., Lingkan, E. et al., *Hill of Towie wind farm open dataset*, Zenodo, CC-BY-4.0; latest version
checked on 2026-11-15: 2.1.0, record [22662930](https://zenodo.org/records/22662930),
doi:[10.5281/zenodo.22662930](https://doi.org/10.5281/zenodo.22662930); concept DOI
[10.5281/zenodo.14870021](https://doi.org/10.5281/zenodo.14870021).

## Results (revised filter, run on 2026-11-15)

### Kelmarsh 1, 2019

Filtering: 52 560 records; 2 917 removed for overlapping a Stop (118 events) or Communication (10) event; 4 076
removed for `Power <= 0` or missing power; 0 for other missing values; 45 567 remain, of which 8 677 overlap a
Warning event (kept). Sample of 2000 records from 2019-01-01 10:20 to 2019-12-31 23:00 (365 `warning_active`).

VAF (%), in sample and out of sample (5 contiguous time blocks; mean and standard error over the blocks):

| model | p | in sample | out of sample | se | out-of-sample gain over PCA |
|---|---|---|---|---|---|
| PCA | 2 | 73.62 | 72.89 | 3.29 | — |
| qlPCA, 1 knot | 2 | 78.01 | 76.88 | 3.30 | 3.99 |
| qlPCA, 2 knots | 2 | 81.48 | 81.31 | 1.81 | 8.43 |
| PCA | 3 | 86.67 | 86.16 | 0.80 | — |
| qlPCA, 1 knot | 3 | 87.61 | 85.65 | 1.86 | −0.51 |
| qlPCA, 2 knots | 3 | 89.88 | 89.25 | 1.12 | 3.09 |

**Criterion (p = 2, gain >= 5 points with 1 or 2 knots): passed** (2 knots: 8.43). With the first filter it was
also passed (7.81). Hill of Towie was not screened.

Non-monotone transformations (p = 2, whole sample): 4 of 9 with 1 knot (wind speed, power, rotor speed,
generator speed), 5 of 9 with 2 knots (the same and generator temperature). Rotor speed and generator speed have
correlation 1.0000 to four decimals.

Files: `screening_summary.csv`, `kelmarsh_vaf.csv` (every block), `kelmarsh_transformations.pdf/.png`,
`kelmarsh_piecewise_loadings.csv`, `kelmarsh_monotonicity.csv`, `kelmarsh_sample_info.json` (filter counts, missing
values, Warning events, sample, correlations).

## Case study: Kelmarsh 1, 2019 (`kelmarsh_case_study.ipynb`)

Same data, filtering and sample as the screening; `generator_speed` removed (correlation 1.0000 with
`rotor_speed`); 8 variables.

**Knots from the observed control regions, estimated on 2018** (another year, same filter, 45 331 records). Wind
speed in classes of 0.5 m/s (classes with at least 30 records); medians of rotor speed, pitch angle and power per
class; each boundary is the wind speed at which a median crosses a threshold at 5 % of its range (linear
interpolation between class centres):

| boundary | threshold | wind (m/s) | power median (kW) | rotor median (rpm) | pitch median (deg) |
|---|---|---|---|---|---|
| B1 end of minimum rotor speed | rotor 8.492 rpm | 4.351 | 149.1 | 8.492 | 0.079 |
| B2 rated rotor speed reached | rotor 14.813 rpm | 8.625 | 1293.2 | 14.813 | 0.050 |
| B3 start of pitching | pitch 0.713° (above the low-wind pitch region) | 9.553 | 1645.9 | 15.100 | 0.713 |
| B4 rated power reached | power 1949.0 kW | 10.771 | 1949.0 | 15.155 | 3.332 |

Knots: wind speed at B1–B4; power at the power medians of B1–B4; rotor speed at B1 and B2 (where its regime
changes); pitch at B3 and at the pitch median of B4; temperatures at the percentiles. The Senvion datasheet knots
(wind 3 and 12.5 m/s, power 2050 kW, rotor 7.5 and 15 rpm; no pitch value given) are kept as a reference.

VAF, p = 2 (5 contiguous time blocks; gain over PCA paired by block):

| model | VAF in sample | VAF out of sample (se) | paired gain over PCA (se) |
|---|---|---|---|
| PCA | 72.57 | 71.95 (4.01) | — |
| percentile knots | 80.63 | 80.48 (2.10) | 8.53 (1.97) |
| percentile knots, monotone | 78.87 | 78.34 (3.16) | 6.39 (1.01) |
| control regions (2018) | 83.45 | 82.91 (2.16) | 10.97 (2.15) |
| control regions (2018), monotone | 82.64 | 81.97 (2.55) | 10.02 (1.80) |
| datasheet (reference) | 76.15 | 75.89 (4.15) | 3.94 (0.32) |

Without the 365 `warning_active` records the paired gains are 7.46 (percentile), 5.56 (percentile, monotone),
9.82 (control regions), 8.98 (control regions, monotone) and 3.43 (datasheet).

**Number or location of the knots.** A further model places at the percentiles the same number of interior knots
per variable as the control-region model uses (wind 4, power 4, rotor 2, pitch 2, temperatures 2; none merged).
Parameters = sum over variables of (degree + interior knots).

| model | parameters | VAF in sample | VAF out of sample (se) | paired gain over PCA (se) |
|---|---|---|---|---|
| PCA | 8 | 72.57 | 71.95 (4.01) | — |
| percentile knots (2 per variable) | 23 | 80.63 | 80.48 (2.10) | 8.53 (1.97) |
| percentiles, same number of knots | 27 | 81.02 | 80.93 (2.03) | 8.98 (2.06) |
| control regions (2018) | 28 | 83.45 | 82.91 (2.16) | 10.97 (2.15) |
| percentiles, same number of knots, monotone | 27 | 79.20 | 78.82 (3.08) | 6.87 (1.10) |
| control regions (2018), monotone | 28 | 82.64 | 81.97 (2.55) | 10.02 (1.80) |

The two "same number" models have 27 parameters, not 28, because one percentile knot of the pitch angle coincides
with its minimum (0°) and is merged. Paired difference, control regions minus percentiles with the same number of
knots: **1.98 (se 0.43)**; monotone versions: **3.15 (se 0.72)**.

**The decreasing branch below B1.** Records of the 2019 sample with wind speed below B1 = 4.351 m/s, split into a
start-up / idle state (pitch > 5° or rotor speed < 8.0 rpm) and the others; counts of records in the decreasing
segments of the control-region model; mean scores on the two components:

| group | records | decreasing wind / power / rotor segment | mean score 1 | mean score 2 | median pitch | median rotor | median power |
|---|---|---|---|---|---|---|---|
| wind < B1, start-up/idle | 26 | 26 / 26 / 26 | 0.37 | 0.91 | 16.44° | 5.60 rpm | 17.1 kW |
| wind < B1, other | 322 | 322 / 300 / 292 | −0.27 | 0.13 | 0.52° | 8.16 rpm | 77.1 kW |
| wind ≥ B1 | 1652 | 0 / 30 / 39 | 0.05 | −0.04 | 0.00° | 12.30 rpm | 614.4 kW |

Control-region model refitted without the 26 start-up/idle records (1974 records): VAF 84.62 in sample, 84.12 out of
sample, paired gain over PCA 7.88 (se 0.75); PCA on the same records 76.87 / 76.24. The lowest segments of wind
speed (322 records), power (330) and rotor speed (331) still have a negative slope. Figure:
`case_decreasing_branch` (pitch and rotor speed against wind speed, coloured by membership of the decreasing
segments; start-up/idle records marked with x).

Other facts from the notebook:

* Distance from each percentile knot to the nearest control-region knot, in standard deviations of 2019: wind
  0.38 and 0.58; power 0.23 and 0.86; rotor 0.49 and 0.61; pitch 0.21 (the percentile model has one pitch knot:
  the 33rd percentile is 0°, the minimum).
* Segments with fewer than 50 records: none for percentile and control-region knots; 3 for the datasheet knots
  (wind above 12.5 m/s, 49; power above 2050 kW, 25; rotor below 7.5 rpm, 22).
* Decreasing segments of wind speed, power and rotor speed: with percentile knots the lowest segment of each
  (660 records, about 150 of them `warning_active`); with control-region knots the segment below B1 (348, 356,
  357 records, 95–97 `warning_active`). In the control-region model the pitch segment above 3.33° and the
  generator-temperature segment below 60.6 °C also decrease.
* Projection of October – December from a January – September fit: 1 of 551 projected records is outside the
  training range (rotor speed and pitch); the `ExtrapolationWarning` is issued. VAF in the projected period,
  clipped / linearly extrapolated: PCA 75.03 / 74.71; percentile knots 81.93 / 81.88; control regions 83.49 /
  83.46.

Results: `case_vaf.csv`, `case_vaf_blocks.csv`, `case_vaf_sensitivity.csv`, `case_control_regions_2018_medians.csv`,
`case_control_region_boundaries.csv`, `case_knot_distances.csv`, `case_segment_counts.csv`,
`case_piecewise_loadings.csv`, `case_projection.csv`, `case_low_wind_groups.csv`, `case_vaf_without_idle.csv`;
figures `case_control_regions_2018`, `case_operating_regimes`, `case_v_shapes`, `case_piecewise_loadings`,
`case_biplot_regimes`, `case_decreasing_branch` (PDF and PNG).


## All turbines: protocol

Fixed on 2026-11-15, **before** any computation on turbines 2 to 6. Implemented in `kelmarsh_all_turbines.py`
(notebook `kelmarsh_all_turbines.ipynb`); results in `results/` with the prefix `all_`.

1. **Turbines and years.** Kelmarsh 1 to 6; analysis year 2019; boundary year 2018. The pipeline of the Kelmarsh 1
   case study is applied without changes: the same 8 variables (wind speed, power, rotor speed, pitch angle,
   nacelle, generator, gearbox and ambient temperatures); the same filter (remove records overlapping Stop or
   Communication events, `Power <= 0` and missing values; keep records overlapping Warning events, flagged
   `warning_active`); a sample of 2000 records with `numpy.random.default_rng(2026)`; 5 contiguous time blocks;
   the same out-of-sample VAF criterion (test rows clipped to the training range); gains paired by block.
2. **Boundaries.** B1 to B4 are estimated on 2018 for each turbine with the same method (medians by wind classes
   of 0.5 m/s with at least 30 records; thresholds at 5 % of the range of each median; B3 searched above the wind
   of the minimum pitch median). They are reported in a table per turbine.
3. **Exclusion rule.** A turbine is excluded only if, in 2019 or in 2018, it has fewer than 20 000 records after
   filtering, or if any of the boundaries cannot be estimated. Exclusions and their reason are reported.
4. **Models per turbine** (p = 2): PCA; percentiles with 2 knots per variable; percentiles with the same number of
   interior knots per variable as the control-region model; control regions (2018); the last two also with
   `monotone=True`.
5. **Main results.**
   (a) per turbine, the paired difference "control regions minus percentiles with the same number of knots" and its
   standard error over the 5 blocks, in the unrestricted and the monotone versions;
   (b) the number of turbines in which that difference is positive;
   (c) the mean of the differences over the turbines and its standard error between turbines
   (sd / sqrt(number of turbines)).
   Also reported: the gain of each model over PCA; the correlation between rotor speed and generator speed per
   turbine (to confirm that dropping generator speed applies to all); the number of start-up / idle records below B1
   (pitch > 5° or rotor speed < 8.0 rpm) and the gain of the control-region model without them; whether the segments
   below B1 of wind speed, power and rotor speed are decreasing in each turbine.
6. **Sensitivity.** The results of (a) without the `warning_active` records.
7. **Check.** Kelmarsh 1 must reproduce the case study (paired gain of the control-region model over PCA 10.97;
   difference control regions minus percentiles with the same number of knots 1.98). If it does not, the run stops.

Figures: the difference (a) per turbine with error bars, unrestricted and monotone; the 2018 operating curves of the
six turbines with B1 to B4. Numbers are reported without interpretation.
