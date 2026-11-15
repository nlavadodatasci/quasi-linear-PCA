"""Preliminary evidence for DESIGN D12 (choice of the number of knots): out-of-sample VAF.

    python comparison/princals/knot_selection.py     # writes results/knot_selection.csv

For each data set and number of interior knots r in {0, 1, 2, 3, 4, 6} (degree 1, p = 2):

* folds: sklearn RepeatedKFold(n_splits=5, n_repeats=4, random_state=0), i.e. 20 train/test splits;
* fit: QLPCA(n_components=2, n_knots=r, random_state=0) on the training rows (default mode: PCA start
  plus 10 random starts);
* test rows are clipped to the training range of each variable (no linear extrapolation, DESIGN D11),
  transformed with the fitted f_j (transform_variables), giving F_test;
* out-of-sample VAF = 100 * ||F_test V||^2 / ||F_test||^2, where V (m x p) holds the training loading
  directions (columns of loadings_ scaled to unit length): the share of the test rows' transformed
  variance captured by the training principal axes;
* reported: mean over the 20 splits and its standard error (sd / sqrt(20)).

Data: make_cylinder(n=51, noise=0 and 0.25, random_state=0) (not rounded: no R involved here) and the
gas turbine sample of manifest.json (2015, n = 400, 9 variables; downloaded at run time).
Only qlpca is used; this file does not need pyGifi.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare as c                                   # noqa: E402  (data loaders only)
from qlpca import QLPCA                               # noqa: E402
from qlpca.datasets import make_cylinder              # noqa: E402

KNOTS = (0, 1, 2, 3, 4, 6)
P = 2


def out_of_sample_vaf(X, n_knots, n_splits=5, n_repeats=4, random_state=0):
    """Out-of-sample VAF (%) of each of the n_splits * n_repeats test folds."""
    vaf = []
    for tr, te in RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state).split(X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # SmallSampleWarning for many knots on n = 40
            model = QLPCA(n_components=P, n_knots=n_knots, random_state=0).fit(X[tr])
        X_test = np.clip(X[te], X[tr].min(0), X[tr].max(0))
        F = model.transform_variables(X_test)
        V = model.loadings_ / np.linalg.norm(model.loadings_, axis=0)
        vaf.append(100 * np.linalg.norm(F @ V) ** 2 / np.linalg.norm(F) ** 2)
    return np.array(vaf)


def main():
    data = {
        "cylinder_0 (n=51)": make_cylinder(n=51, noise=0.0, random_state=0)[0],
        "cylinder_25 (n=51)": make_cylinder(n=51, noise=0.25, random_state=0)[0],
        "gas_turbine (n=400)": c.load_dataset("gas_turbine").to_numpy(),
    }
    rows = []
    for name, X in data.items():
        for r in KNOTS:
            v = out_of_sample_vaf(X, r)
            rows.append({"dataset": name, "n_knots": r, "vaf_oos_mean": v.mean(),
                         "vaf_oos_se": v.std(ddof=1) / np.sqrt(len(v)), "n_folds": len(v)})
            print(f"{name:20s} knots {r}: {v.mean():6.2f}  (se {rows[-1]['vaf_oos_se']:.2f})", flush=True)
    out = c.RESULTS / "knot_selection.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.4f", lineterminator="\n")
    table = pd.DataFrame(rows).pivot(index="dataset", columns="n_knots", values="vaf_oos_mean").round(1)
    print(table.to_string())
    print("wrote", out)


if __name__ == "__main__":
    main()
