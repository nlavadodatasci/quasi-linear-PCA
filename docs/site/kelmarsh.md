# Application: Kelmarsh wind farm

Ten-minute SCADA data of the six Senvion MM92 turbines of Kelmarsh wind farm (UK): Plumley, C. & Takeuchi, R.,
*Kelmarsh wind farm data*, Zenodo, doi:[10.5281/zenodo.16807551](https://doi.org/10.5281/zenodo.16807551),
CC-BY-4.0, downloaded at run time. The data set was chosen by a screening with an approval criterion fixed before
the analysis; the study was then repeated on all six turbines under a protocol also fixed before the run. The full
material is in
[`comparison/applications/`](https://github.com/nlavadodatasci/quasi-linear-PCA/tree/main/comparison/applications)
(README, `screening.py`, `kelmarsh_case_study.ipynb`, `kelmarsh_all_turbines.py`). Numbers are reported without
interpretation.

## Screening criterion

--8<-- "kelmarsh_screening_criterion.md"

## All turbines: protocol

--8<-- "kelmarsh_protocol.md"

## All turbines: results

--8<-- "kelmarsh_results.md"

## Figures

Control regions in 2018 (medians in 0.5 m/s wind classes and the boundaries B1–B4), six turbines:

![Operating curves 2018](generated/kelmarsh/all_operating_curves_2018.png)

Control regions minus percentiles with the same number of knots, per turbine (± standard error over the 5 blocks):

![Differences per turbine](generated/kelmarsh/all_differences.png)

Kelmarsh 1, 2019 sample by control region:

![Operating regimes, Kelmarsh 1](generated/kelmarsh/case_operating_regimes.png)

Kelmarsh 1, piecewise loadings of the control-region model:

![Piecewise loadings](generated/kelmarsh/case_piecewise_loadings.png)

Kelmarsh 1, records below B1 by membership of the decreasing segments (start-up / idle records marked ×):

![Decreasing branch](generated/kelmarsh/case_decreasing_branch.png)
