"""Compare qlpca with princals (pyGifi in Python, Gifi in R) on the data and settings of
manifest.json.

Usage (from this folder or the repository root):

    python comparison/princals/compare.py manifest   # write knots + input checksums into manifest.json
                                                     # and export the input CSVs for R (results/inputs/)
    python comparison/princals/compare.py inputs     # only export the input CSVs (used by the R workflow)
    python comparison/princals/compare.py            # run qlpca and pyGifi, compare with R if available,
                                                     # write results/summary.csv and results/figures/
    python comparison/princals/compare.py --refresh  # same, recomputing the cached pyGifi fits

pyGifi and Gifi are GPL-3. This file only imports and calls pyGifi; no code of either
package is copied here or into qlpca.
"""
import hashlib
import io
import json
import sys
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from qlpca import QLPCA
from qlpca.datasets import load_st_gender, make_cylinder
from qlpca.splines import interior_knots

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
CACHE = HERE / ".cache"
RESULTS = HERE / "results"
INPUTS = RESULTS / "inputs"
R_GROUND_TRUTH = RESULTS / "r_ground_truth"
PYGIFI_RESULTS = RESULTS / "pygifi"
FIGURES = RESULTS / "figures"

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
PACKAGES = ("qlpca", "pygifi", "gifi_r")
LABELS = {"qlpca": "qlpca", "pygifi": "pyGifi princals", "gifi_r": "Gifi (R) princals"}


# ----------------------------------------------------------------------------- data
def load_manifest():
    with open(MANIFEST, encoding="utf8") as fh:
        return json.load(fh)


def _gas_turbine_zip(spec):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "gas_turbine_co_nox.zip"
    if not path.is_file():
        with urllib.request.urlopen(spec["url"], timeout=120) as r:
            path.write_bytes(r.read())
    return path


def gas_turbine_year(year, spec=None):
    """All hourly records of one year of the UCI Gas Turbine data (downloaded once into .cache/)."""
    spec = spec or load_manifest()["datasets"]["gas_turbine"]
    with zipfile.ZipFile(_gas_turbine_zip(spec)) as z:
        with z.open(f"gt_{year}.csv") as fh:
            return pd.read_csv(fh)


def load_dataset(name, manifest=None):
    """The data set `name` of the manifest as a DataFrame (one column per variable)."""
    spec = (manifest or load_manifest())["datasets"][name]
    src = spec["source"]
    if src == "qlpca.datasets.make_cylinder":
        X, names = make_cylinder(n=spec["n"], noise=spec["noise"], random_state=spec["random_state"])
        # log/exp/arctan may differ in the last bit between platforms: round so that the input
        # (and its md5 in the manifest) is the same on Windows, Linux and in the R workflow
        X = np.vectorize(lambda v: float(f"{v:.{spec['significant_digits']}g}"))(X)
        return pd.DataFrame(X, columns=names)
    if src == "qlpca.datasets.load_st_gender":
        X, names, countries = load_st_gender()
        return pd.DataFrame(X, columns=names, index=countries)
    if name == "neumann":
        from pygifi import get_dataset           # data loaded from pyGifi at run time
        return get_dataset("neumann").reset_index().astype(float)
    if name == "gas_turbine":
        d = gas_turbine_year(spec["year"], spec)
        d = d.sample(n=spec["n"], random_state=spec["random_state"]).sort_index()
        return d[spec["variables"]].reset_index(drop=True).astype(float)
    raise KeyError(name)


def quantile_knots(x, n_knots):
    """qlpca's default interior knots (MATLAB prctile percentiles), in the units of x;
    knots equal to the minimum or maximum, or to each other, are dropped."""
    k = np.unique(interior_knots(np.asarray(x, float), n_knots))
    return k[(k > np.min(x)) & (k < np.max(x))]


def _csv_bytes(df):
    return df.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf8")


def export_inputs(manifest=None):
    """Write every data set as CSV for R (results/inputs/, git-ignored); returns md5 per file."""
    manifest = manifest or load_manifest()
    INPUTS.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in manifest["datasets"]:
        data = _csv_bytes(load_dataset(name, manifest))
        (INPUTS / f"{name}.csv").write_bytes(data)
        out[name] = hashlib.md5(data).hexdigest()
    return out


def write_manifest():
    """Fill the 'knots' and 'inputs' sections of manifest.json from the data."""
    manifest = load_manifest()
    md5 = export_inputs(manifest)
    n_knots = sorted({r["n_knots"] for r in manifest["runs"].values() if "n_knots" in r})
    manifest["knots"] = {}
    manifest["inputs"] = {}
    for name in manifest["datasets"]:
        df = load_dataset(name, manifest)
        manifest["knots"][name] = {
            str(r): {col: quantile_knots(df[col].to_numpy(), r).tolist() for col in df.columns}
            for r in n_knots}
        manifest["inputs"][name] = {"file": f"results/inputs/{name}.csv", "md5": md5[name],
                                    "n": int(df.shape[0]), "columns": list(df.columns)}
    with open(MANIFEST, "w", encoding="utf8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    return manifest


def knots_for(manifest, name, run):
    k = manifest["knots"][name][str(manifest["runs"][run]["n_knots"])]
    return [np.asarray(k[c], float) for c in manifest["inputs"][name]["columns"]]


# ----------------------------------------------------------------------------- fits
def _result(evals, objectscores, transform, loadings, ntel, f, params):
    T = np.asarray(transform, float)
    T = T / np.linalg.norm(T, axis=0)             # unit norm per column, as in Gifi
    return dict(evals=np.asarray(evals, float), objectscores=np.asarray(objectscores, float),
                transform=T, loadings=np.asarray(loadings, float), ntel=int(ntel), f=float(f),
                params=np.asarray(params, int))


def run_qlpca(df, run, manifest, name, **kw):
    """Fit qlpca for one run of the manifest. Returns a dict with the Gifi field names."""
    spec, fit, ndim = manifest["runs"][run], manifest["fit"], manifest["ndim"]
    if spec.get("package_defaults"):
        model = QLPCA(n_components=ndim, random_state=0, **kw)
    else:
        model = QLPCA(n_components=ndim, degree=spec["degree"], knots=knots_for(manifest, name, run),
                      monotone=spec["ordinal"], tol=spec.get("eps", fit["eps"]), max_iter=fit["itmax"],
                      random_state=0, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # SmallSampleWarning etc. are reported in the notebook
        model.fit(df)
    params = [len(np.unique(k)) - 2 + d for k, d in zip(model.knots_, model._degrees)]
    res = _result(model.all_eigenvalues_, model.scores_, model.transformed_, model.loadings_,
                  model.n_iter_, model.loss_, params)
    res["model"] = model
    return res


def run_pygifi(df, run, manifest, name, init_x=None, r_seed=None, **kw):
    """Fit pyGifi's Princals for one run of the manifest."""
    from pygifi import Princals
    spec, fit, ndim = manifest["runs"][run], manifest["fit"], manifest["ndim"]
    if spec.get("package_defaults"):
        p = Princals(ndim=ndim, init_x=init_x, r_seed=r_seed, **kw)
    else:
        p = Princals(ndim=ndim, knots=[k.tolist() for k in knots_for(manifest, name, run)],
                     degrees=spec["degree"], ordinal=spec["ordinal"], eps=spec.get("eps", fit["eps"]),
                     itmax=fit["itmax"], init_x=init_x, r_seed=r_seed, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p.fit(df.reset_index(drop=True))
    r = p.result_
    degs = np.broadcast_to(np.asarray(r["degrees"]), (df.shape[1],))
    params = [basis_rank(df.iloc[:, j].to_numpy(), int(degs[j]), r["knots"][j]) for j in range(df.shape[1])]
    res = _result(r["evals"], r["objectscores"], r["transform"], np.asarray(r["loadings"]).reshape(df.shape[1], ndim),
                  r["ntel"], r["f"], params)
    res["model"] = p
    return res


def basis_rank(x, degree, knots):
    """Number of free parameters of a princals transformation: rank of the centred B-spline
    basis that pyGifi builds for this variable (computed by calling pyGifi)."""
    from pygifi.utils.splines import bspline_basis
    B = np.asarray(bspline_basis(np.asarray(x, float), degree, np.asarray(knots, float)))
    return int(np.linalg.matrix_rank(B - B.mean(0)))


def _to_json(res):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items() if k != "model"}


def runs_for(manifest, name):
    return [run for run, spec in manifest["runs"].items() if name in spec.get("datasets", [name])]


def pygifi_cached(df, run, manifest, name, refresh=False):
    """pyGifi fit for (name, run), read from results/pygifi/<name>.json when it was computed on
    the same input data and pyGifi version; otherwise computed and stored there."""
    path = PYGIFI_RESULTS / f"{name}.json"
    meta = {"input_md5": manifest["inputs"][name]["md5"], "pygifi": manifest["packages"]["pyGifi"]}
    store = {"meta": meta, "runs": {}}
    if path.is_file():
        with open(path, encoding="utf8") as fh:
            old = json.load(fh)
        if old.get("meta") == meta:
            store = old
    if run in store["runs"] and not refresh:
        r = store["runs"][run]
        return _result(r["evals"], r["objectscores"], r["transform"], r["loadings"], r["ntel"], r["f"], r["params"])
    res = run_pygifi(df, run, manifest, name)
    store["runs"][run] = _to_json(res)
    PYGIFI_RESULTS.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8", newline="\n") as fh:
        json.dump(store, fh)
    return res


def load_r_ground_truth(name):
    """Gifi (R) results for one data set, or None if the JSON has not been generated yet."""
    path = R_GROUND_TRUTH / f"{name}.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf8") as fh:
        js = json.load(fh)
    out = {"meta": js["meta"]}
    for run, r in js["runs"].items():
        # generate_r_ground_truth.R writes the loadings as (ndim, m); qlpca and pyGifi use (m, ndim)
        loadings = np.asarray(r["loadings"], float).T
        out[run] = _result(r["evals"], r["objectscores"], r["transform"], loadings, r["ntel"], r["f"],
                           r.get("params", []))
    return out


# ----------------------------------------------------------------------------- comparison
def compare(a, b, ndim):
    """Differences between two fits, insensitive to sign changes of components and of
    transformed variables. Returns a dict of maximum absolute differences."""
    ev = np.max(np.abs(a["evals"][:ndim] - b["evals"][:ndim]))
    ev_all = np.max(np.abs(a["evals"] - b["evals"]))
    corr = np.array([np.corrcoef(a["transform"][:, j], b["transform"][:, j])[0, 1]
                     for j in range(a["transform"].shape[1])])
    s_var = np.where(corr < 0, -1.0, 1.0)
    s_comp = np.sign(np.sum(a["objectscores"] * b["objectscores"], axis=0))
    s_comp[s_comp == 0] = 1.0
    xs = np.max(np.abs(a["objectscores"] - b["objectscores"] * s_comp))
    ld = np.max(np.abs(a["loadings"] - b["loadings"] * s_var[:, None] * s_comp))
    return dict(evals=float(ev), evals_all=float(ev_all), transform_corr=float(1 - np.min(np.abs(corr))),
                objectscores=float(xs), loadings=float(ld))


def tolerance_key(manifest, run):
    spec = manifest["runs"][run]
    if spec.get("package_defaults"):
        return "defaults"
    return "monotone" if spec["ordinal"] else "unrestricted"


def within(diff, tol):
    return all(diff[k] <= tol[k] for k in ("evals", "transform_corr", "objectscores", "loadings"))


def run_all(datasets=None, runs=None, verbose=True, refresh=False):
    """Fit every (data set, run) with qlpca and pyGifi (cached in results/pygifi/ unless
    refresh=True); add Gifi (R) when its JSON exists."""
    manifest = load_manifest()
    ndim = manifest["ndim"]
    fits, rows = {}, []
    for name in datasets or manifest["datasets"]:
        df = load_dataset(name, manifest)
        m = df.shape[1]
        rgt = load_r_ground_truth(name)
        if rgt is not None and rgt["meta"].get("input_md5") != manifest["inputs"][name]["md5"]:
            raise ValueError(f"R ground truth for {name} was computed on different input data")
        for run in runs_for(manifest, name):
            if runs is not None and run not in runs:
                continue
            res = {"qlpca": run_qlpca(df, run, manifest, name),
                   "pygifi": pygifi_cached(df, run, manifest, name, refresh=refresh)}
            if rgt is not None and run in rgt:
                res["gifi_r"] = rgt[run]
            fits[name, run] = res
            tol = manifest["tolerances"][tolerance_key(manifest, run)]
            row = {"dataset": name, "run": run, "n": df.shape[0], "m": m}
            for pkg in PACKAGES:
                r = res.get(pkg)
                row[f"{pkg}_ev1"] = r["evals"][0] if r else np.nan
                row[f"{pkg}_ev2"] = r["evals"][1] if r else np.nan
                row[f"{pkg}_vaf2"] = 100 * r["evals"][:ndim].sum() / m if r else np.nan
                row[f"{pkg}_ntel"] = r["ntel"] if r else np.nan
                row[f"{pkg}_params_mean"] = r["params"].mean() if (r and r["params"].size) else np.nan
            for a, b in (("qlpca", "pygifi"), ("qlpca", "gifi_r"), ("pygifi", "gifi_r")):
                if a in res and b in res:
                    d = compare(res[a], res[b], ndim)
                    row.update({f"{a}_vs_{b}_{k}": v for k, v in d.items()})
                    same = manifest["runs"][run].get("same_model", True) or a != "qlpca"
                    row[f"{a}_vs_{b}_within_tol"] = within(d, tol) if same else np.nan
            row["r_ground_truth"] = "yes" if "gifi_r" in res else "pending"
            rows.append(row)
            if verbose:
                print(f"{name:12s} {run:14s} VAF2 qlpca {row['qlpca_vaf2']:7.3f}  pyGifi {row['pygifi_vaf2']:7.3f}"
                      f"  R {row['gifi_r_vaf2']:7.3f}", flush=True)
    return fits, pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def save_figure(fig, stem):
    """Save under results/figures/<stem>.pdf and .png (stable names)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{ext}", bbox_inches="tight", dpi=150,
                    metadata={"CreationDate": None} if ext == "pdf" else None)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_transformations(df, res, title=None, ncols=4):
    """Optimal transformations of each variable, one panel per variable, every package overlaid
    (signs aligned to qlpca)."""
    import matplotlib.pyplot as plt
    m = df.shape[1]
    nrows = int(np.ceil(m / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.1 * nrows), squeeze=False)
    n = df.shape[0]
    ref = res["qlpca"]["transform"]
    styles = {"qlpca": dict(color=BLUE, lw=2.0), "pygifi": dict(color=ORANGE, lw=1.2, ls="--"),
              "gifi_r": dict(color=INK, ls="none", marker="o", ms=2.5)}
    for j, ax in enumerate(axes.flat):
        if j >= m:
            ax.axis("off")
            continue
        x = df.iloc[:, j].to_numpy()
        o = np.argsort(x, kind="stable")
        for pkg, st in styles.items():
            if pkg not in res:
                continue
            t = res[pkg]["transform"][:, j]
            s = np.sign(np.corrcoef(t, ref[:, j])[0, 1]) or 1.0
            ax.plot(x[o], s * t[o] * np.sqrt(n), label=LABELS[pkg], **st)
        ax.set_title(str(df.columns[j]), fontsize=8, loc="left", color=INK)
        _style(ax)
    axes.flat[0].legend(fontsize=7, frameon=False)
    if title:
        fig.suptitle(title, fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    return fig


def plot_vaf_summary(summary):
    import matplotlib.pyplot as plt
    runs = ["knots1", "knots2", "defaults"]
    data = summary[summary.run.isin(runs)]
    names = list(dict.fromkeys(data.dataset))
    fig, axes = plt.subplots(1, len(runs), figsize=(4.0 * len(runs), 3.2), sharey=True)
    w = 0.27
    for ax, run in zip(axes, runs):
        d = data[data.run == run].set_index("dataset").loc[names]
        y = np.arange(len(names))
        for i, (pkg, col) in enumerate((("qlpca", BLUE), ("pygifi", ORANGE), ("gifi_r", INK2))):
            ax.barh(y + (i - 1) * w, d[f"{pkg}_vaf2"], height=w, color=col, label=LABELS[pkg])
        ax.set_yticks(y, names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_title(run, fontsize=9, loc="left", color=INK)
        ax.set_xlabel("% variance, 2 dimensions", fontsize=8, color=INK2)
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def main(argv):
    if argv[:1] == ["manifest"]:
        write_manifest()
        print("manifest.json updated; inputs exported to", INPUTS)
        return 0
    if argv[:1] == ["inputs"]:
        print(json.dumps(export_inputs(), indent=1))
        return 0
    import matplotlib
    matplotlib.use("Agg")
    fits, summary = run_all(refresh="--refresh" in argv)
    RESULTS.mkdir(exist_ok=True)
    summary.to_csv(RESULTS / "summary.csv", index=False, float_format="%.10g", lineterminator="\n")
    manifest = load_manifest()
    for name in manifest["datasets"]:
        fig = plot_transformations(load_dataset(name, manifest), fits[name, "knots2"],
                                   title=f"{name}: same knots (2), degree 1, no monotonicity")
        save_figure(fig, f"compare_transformations_{name}_knots2")
    save_figure(plot_vaf_summary(summary), "compare_vaf_summary")
    print("wrote", RESULTS / "summary.csv", "and figures in", FIGURES)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
