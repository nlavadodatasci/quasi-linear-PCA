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

## Results

See `results/` and the section added below after the run.
