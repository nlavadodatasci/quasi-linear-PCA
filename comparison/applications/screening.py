"""Screening of public data sets for a real application of qlPCA (see README.md for the criterion,
the data and the filtering, all fixed before the run).

    python comparison/applications/screening.py            # Kelmarsh; Hill of Towie only if Kelmarsh fails
    python comparison/applications/screening.py hill_of_towie   # force a candidate

Writes to comparison/applications/results/:
    screening_summary.csv                  one row per candidate x model x p (in- and out-of-sample VAF)
    <candidate>_vaf.csv                    out-of-sample VAF of every block
    <candidate>_transformations.pdf/.png   qlPCA, 2 knots, p = 2, whole sample
    <candidate>_piecewise_loadings.csv     same model
    <candidate>_monotonicity.csv           non-monotone transformations, 1 and 2 knots
    <candidate>_sample_info.json           records before/after each filter, sample size, seed
"""
import json
import sys
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from qlpca import QLPCA
from qlpca.diagnostics import piecewise_loadings

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
RESULTS = HERE / "results"
N_SAMPLE = 2000
SEED = 2026
N_BLOCKS = 5
MODELS = {"PCA": 0, "qlPCA 1 knot": 1, "qlPCA 2 knots": 2}
PS = (2, 3)
CRITERION_P, CRITERION_GAIN = 2, 5.0


# ----------------------------------------------------------------------------- data
def _download(url, path):
    CACHE.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        print("downloading", url, flush=True)
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
        tmp.rename(path)
    return path


KELMARSH = {
    "record": "16807551",
    "file": "Kelmarsh_SCADA_2019_3085.zip",
    "turbine": "Kelmarsh_1",
    "variables": {           # short name -> SCADA column prefix (the units contain a degree sign)
        "wind_speed": "Wind speed (",
        "power": "Power (kW)",
        "rotor_speed": "Rotor speed (",
        "generator_speed": "Generator RPM (",
        "pitch_angle": "Blade angle (pitch position) A (",
        "nacelle_temp": "Nacelle temperature (",
        "generator_temp": "Stator temperature 1 (",
        "gearbox_temp": "Gear oil temperature (",
        "ambient_temp": "Nacelle ambient temperature (",
    },
}


def load_kelmarsh():
    """Filtered ten-minute records of Kelmarsh 1, 2019 (README: filtering), and the counts per step."""
    spec = KELMARSH
    url = f"https://zenodo.org/api/records/{spec['record']}/files/{spec['file']}/content"
    z = zipfile.ZipFile(_download(url, CACHE / spec["file"]))
    names = z.namelist()
    data_name = next(n for n in names if n.startswith(f"Turbine_Data_{spec['turbine']}_"))
    status_name = next(n for n in names if n.startswith(f"Status_{spec['turbine']}_"))
    d = pd.read_csv(z.open(data_name), skiprows=9, low_memory=False, encoding="latin-1")
    d.columns = [c.lstrip("# ") for c in d.columns]
    cols = {k: next(c for c in d.columns if c.startswith(p)) for k, p in spec["variables"].items()}
    t = pd.to_datetime(d["Date and time"])
    X = d[list(cols.values())].astype(float)
    X.columns = list(cols)
    info = {"source": f"zenodo record {spec['record']}, {spec['file']}, {data_name}", "records": int(len(X))}

    # 1. stoppages, alarms, communication loss: drop records overlapping such an event
    s = pd.read_csv(z.open(status_name), skiprows=9)
    s = s[s["Status"].isin(["Stop", "Warning", "Communication"])]
    start = pd.to_datetime(s["Timestamp start"]).to_numpy()
    end = pd.to_datetime(s["Timestamp end"]).to_numpy()
    t0 = t.to_numpy()
    t1 = t0 + np.timedelta64(10, "m")
    bad = np.zeros(len(t0), bool)
    for a, b in zip(start, end):                 # overlap of [t0, t1) with [a, b]
        bad |= (t0 < b) & (t1 > a)
    info["events_used"] = {k: int(v) for k, v in s["Status"].value_counts().items()}
    info["removed_status_events"] = int(bad.sum())
    keep = ~bad
    # 2. not producing
    not_producing = keep & ~(X["power"] > 0)
    info["removed_power_le_0"] = int(not_producing.sum())
    keep &= X["power"] > 0
    # 3. missing values
    missing = keep & X.isna().any(axis=1)
    info["removed_missing"] = int(missing.sum())
    keep &= ~X.isna().any(axis=1)
    info["records_after_filtering"] = int(keep.sum())
    return X[keep].assign(time=t[keep].to_numpy()), info


def load_hill_of_towie():
    raise NotImplementedError("Hill of Towie is screened only if Kelmarsh fails the criterion (README)")


CANDIDATES = {"kelmarsh": load_kelmarsh, "hill_of_towie": load_hill_of_towie}


def sample(df, info, keep_time=False):
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(df), size=N_SAMPLE, replace=False))
    out = df.iloc[idx].sort_values("time")
    info.update({"sample_size": N_SAMPLE, "sample_seed": f"numpy.random.default_rng({SEED})",
                 "sample_first": str(out["time"].iloc[0]), "sample_last": str(out["time"].iloc[-1])})
    out = out.reset_index(drop=True)
    return out if keep_time else out.drop(columns="time")


# ----------------------------------------------------------------------------- fits
def fit(X, n_knots, p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return QLPCA(n_components=p, n_knots=n_knots, random_state=0).fit(X)


def oos_vaf(X, model, X_test, X_train):
    """Same criterion as comparison/princals/knot_selection.py."""
    Xt = np.clip(X_test, X_train.min(0), X_train.max(0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        F = model.transform_variables(Xt)
    V = model.loadings_ / np.linalg.norm(model.loadings_, axis=0)
    return 100 * np.linalg.norm(F @ V) ** 2 / np.linalg.norm(F) ** 2


def non_monotone(model):
    out = []
    for j in range(model.n_features_in_):
        d = np.diff(model.variable_transformation(j, n_points=400)[1])
        out.append(not (np.all(d >= -1e-12) or np.all(d <= 1e-12)))
    return np.array(out)


def screen(name):
    df, info = CANDIDATES[name]()
    X = sample(df, info)
    cols = list(X.columns)
    A = X.to_numpy()
    blocks = np.array_split(np.arange(len(A)), N_BLOCKS)     # contiguous time blocks
    rows, folds = [], []
    for label, r in MODELS.items():
        for p in PS:
            ins = fit(A, r, p).vaf_.sum()
            v = []
            for b, te in enumerate(blocks):
                tr = np.setdiff1d(np.arange(len(A)), te)
                v.append(oos_vaf(A, fit(A[tr], r, p), A[te], A[tr]))
                folds.append({"model": label, "n_knots": r, "p": p, "block": b + 1, "vaf_oos": v[-1]})
            v = np.array(v)
            rows.append({"candidate": name, "model": label, "n_knots": r, "p": p, "vaf_in_sample": ins,
                         "vaf_oos_mean": v.mean(), "vaf_oos_se": v.std(ddof=1) / np.sqrt(len(v))})
            print(f"{name} {label:14s} p={p}: in-sample {ins:6.2f}  out-of-sample {v.mean():6.2f} (se {rows[-1]['vaf_oos_se']:.2f})",
                  flush=True)
    summary = pd.DataFrame(rows)
    pca = summary[(summary.n_knots == 0) & (summary.p == CRITERION_P)].vaf_oos_mean.iloc[0]
    summary["gain_oos_over_pca"] = summary.vaf_oos_mean - summary.groupby("p").vaf_oos_mean.transform(
        lambda s: s.iloc[0])
    best = summary[(summary.n_knots > 0) & (summary.p == CRITERION_P)].vaf_oos_mean.max()
    passed = bool(best - pca >= CRITERION_GAIN)

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(folds).to_csv(RESULTS / f"{name}_vaf.csv", index=False, float_format="%.4f", lineterminator="\n")

    # whole-sample qlPCA, 2 knots, p = 2: transformations and piecewise loadings
    import matplotlib
    matplotlib.use("Agg")
    from qlpca.plots import transformation_plots
    model = fit(X, 2, 2)
    fig = transformation_plots(model, X, names=cols, ncols=3)
    for ext in ("pdf", "png"):
        fig.savefig(RESULTS / f"{name}_transformations.{ext}", bbox_inches="tight", dpi=150,
                    metadata={"CreationDate": None} if ext == "pdf" else None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pl = piecewise_loadings(model, X)
    prow = []
    for j, col in enumerate(cols):
        for s in range(len(pl[j]["slope_sign"])):
            prow.append({"variable": col, "segment_from": pl[j]["edges"][s], "segment_to": pl[j]["edges"][s + 1],
                         "slope_sign": pl[j]["slope_sign"][s], "n": pl[j]["n"][s],
                         "loading_1": pl[j]["loadings"][s][0], "loading_2": pl[j]["loadings"][s][1]})
    pd.DataFrame(prow).to_csv(RESULTS / f"{name}_piecewise_loadings.csv", index=False, float_format="%.4f",
                              lineterminator="\n")
    mono = pd.DataFrame({"variable": cols, **{f"non_monotone_{r}_knots": non_monotone(fit(X, r, 2)) for r in (1, 2)}})
    mono.to_csv(RESULTS / f"{name}_monotonicity.csv", index=False, lineterminator="\n")
    info.update({"criterion": f"p = {CRITERION_P}: out-of-sample VAF of qlPCA (1 or 2 knots) - PCA >= {CRITERION_GAIN}",
                 "pca_oos": pca, "best_qlpca_oos": best, "gain": best - pca, "passed": passed,
                 "non_monotone_1_knot": int(mono.non_monotone_1_knots.sum()),
                 "non_monotone_2_knots": int(mono.non_monotone_2_knots.sum()), "variables": cols,
                 "correlations": X.corr().round(4).to_dict()})
    with open(RESULTS / f"{name}_sample_info.json", "w", encoding="utf8", newline="\n") as fh:
        json.dump(info, fh, indent=2, default=str)
        fh.write("\n")
    return summary, passed, mono


def main(argv):
    order = argv or ["kelmarsh", "hill_of_towie"]
    allrows = []
    for name in order:
        summary, passed, mono = screen(name)
        allrows.append(summary)
        print(f"{name}: criterion {'PASSED' if passed else 'NOT PASSED'}")
        print(mono.to_string(index=False))
        if passed and not argv:
            break
    out = RESULTS / "screening_summary.csv"
    if out.is_file() and argv:
        old = pd.read_csv(out)
        allrows.insert(0, old[~old.candidate.isin(order)])
    pd.concat(allrows).to_csv(out, index=False, float_format="%.4f", lineterminator="\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
