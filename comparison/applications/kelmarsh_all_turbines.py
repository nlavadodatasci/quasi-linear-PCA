"""Kelmarsh case study repeated on the six turbines (protocol in README.md, "All turbines: protocol",
fixed before the run).

    python comparison/applications/kelmarsh_all_turbines.py

Writes to comparison/applications/results/ (prefix all_):
    all_inclusion.csv        records after filtering (2018, 2019), boundary estimation, exclusion and reason
    all_boundaries.csv       B1-B4 per turbine (2018): thresholds, wind speeds, medians at the boundaries
    all_vaf.csv              per turbine and model: parameters, VAF in/out of sample, paired gain over PCA
    all_differences.csv      (a) control regions minus percentiles with the same number of knots, free and monotone,
                             with all records and without warning_active records
    all_turbine_facts.csv    rotor-generator correlation, start-up/idle records below B1, gain without them, slope signs
    all_summary.json         (b) and (c), and the reproduction check of Kelmarsh 1
    all_differences.pdf/png, all_operating_curves_2018.pdf/png
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import screening as s                                    # noqa: E402
from qlpca import QLPCA                                  # noqa: E402
from qlpca.diagnostics import piecewise_loadings        # noqa: E402
from qlpca.splines import interior_knots                 # noqa: E402

RES = HERE / "results"
TURBINES = [f"Kelmarsh_{i}" for i in range(1, 7)]
P = 2
MIN_RECORDS = 20000
CLASS_WIDTH, MIN_CLASS_N, THRESHOLD = 0.5, 30, 0.05
CHECK = {"control_gain": 10.97, "difference": 1.98}       # Kelmarsh 1 case study (rounded to 2 decimals)
MODELS = ["PCA", "percentile knots", "percentiles, same number of knots", "percentiles, same number of knots, monotone",
          "control regions (2018)", "control regions (2018), monotone"]


# ----------------------------------------------------------------------------- boundaries (2018)
def class_medians(d):
    edges = np.arange(0, 26.0 + CLASS_WIDTH, CLASS_WIDTH)
    g = d.assign(cls=pd.cut(d.wind_speed, edges, right=False)).groupby("cls", observed=True)
    med = g.agg(n=("power", "size"), power=("power", "median"), rotor_speed=("rotor_speed", "median"),
                pitch_angle=("pitch_angle", "median"))
    med = med[med.n >= MIN_CLASS_N]
    med.index = [iv.left + CLASS_WIDTH / 2 for iv in med.index]
    return med


def crossing(y, thr, start=None):
    w, v = y.index.to_numpy(), y.to_numpy()
    i0 = 0 if start is None else int(np.searchsorted(w, start))
    for i in range(max(i0, 1), len(v)):
        if v[i - 1] < thr <= v[i]:
            return w[i - 1] + (thr - v[i - 1]) / (v[i] - v[i - 1]) * (w[i] - w[i - 1])
    raise ValueError("no crossing")


def boundaries(med):
    r, pt, pw = med.rotor_speed, med.pitch_angle, med.power
    thr = {"B1": r.min() + THRESHOLD * (r.max() - r.min()), "B2": r.max() - THRESHOLD * (r.max() - r.min()),
           "B3": pt.min() + THRESHOLD * (pt.max() - pt.min()), "B4": pw.max() - THRESHOLD * (pw.max() - pw.min())}
    B = {"B1": crossing(r, thr["B1"]), "B2": crossing(r, thr["B2"]),
         "B3": crossing(pt, thr["B3"], start=pt.idxmin()), "B4": crossing(pw, thr["B4"])}
    at = lambda col, w: float(np.interp(w, med.index, med[col]))
    control = {"wind_speed": sorted(B.values()), "power": sorted(at("power", B[k]) for k in B),
               "rotor_speed": [thr["B1"], thr["B2"]], "pitch_angle": [thr["B3"], at("pitch_angle", B["B4"])]}
    return B, thr, at, control


# ----------------------------------------------------------------------------- models
def make_specs(control, VARS, X):
    def knots_from(table, Xtr):
        Xtr = pd.DataFrame(np.asarray(Xtr), columns=VARS)
        return [np.asarray(table[v], float) if v in table else interior_knots(Xtr[v].to_numpy(), 2) for v in VARS]

    specs = {"PCA": lambda Xtr: dict(n_knots=0),
             "percentile knots": lambda Xtr: dict(n_knots=2),
             "control regions (2018)": lambda Xtr: dict(knots=knots_from(control, Xtr)),
             "control regions (2018), monotone": lambda Xtr: dict(knots=knots_from(control, Xtr), monotone=True)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m0 = QLPCA(n_components=P, random_state=0, **specs["control regions (2018)"](X)).fit(X)
    nk = [len(np.unique(k)) - 2 for k in m0.knots_]
    specs["percentiles, same number of knots"] = lambda Xtr: dict(n_knots=nk)
    specs["percentiles, same number of knots, monotone"] = lambda Xtr: dict(n_knots=nk, monotone=True)
    return {k: specs[k] for k in MODELS}, nk


def fit(spec, Xtr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return QLPCA(n_components=P, random_state=0, **spec(Xtr)).fit(Xtr)


def vaf_table(specs, Xd):
    A = Xd.to_numpy()
    blocks = np.array_split(np.arange(len(A)), 5)
    oos = {name: [] for name in specs}
    for te in blocks:
        tr = np.setdiff1d(np.arange(len(A)), te)
        for name, spec in specs.items():
            oos[name].append(s.oos_vaf(A, fit(spec, A[tr]), A[te], A[tr]))
    oos = pd.DataFrame({k: np.array(v) for k, v in oos.items()})
    rows = []
    for name, spec in specs.items():
        m = fit(spec, Xd)
        gain = oos[name] - oos["PCA"]
        rows.append({"model": name, "parameters": int(sum(len(np.unique(k)) - 2 + d for k, d in zip(m.knots_, m._degrees))),
                     "VAF in sample": m.vaf_.sum(), "VAF out of sample": oos[name].mean(),
                     "se": oos[name].std(ddof=1) / np.sqrt(5),
                     "paired gain": np.nan if name == "PCA" else gain.mean(),
                     "se (gain)": np.nan if name == "PCA" else gain.std(ddof=1) / np.sqrt(5)})
    return pd.DataFrame(rows), oos


def paired(oos, a, b):
    d = oos[a].to_numpy() - oos[b].to_numpy()
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def slope_signs_below(m, X, VARS, b1_values):
    """Slope sign of the lowest segment of wind, power and rotor (the segment below the first knot)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pl = piecewise_loadings(m, X)
    out = {}
    for v in ("wind_speed", "power", "rotor_speed"):
        j = VARS.index(v)
        out[v] = int(pl[j]["slope_sign"][0])
        out[f"{v}_lowest_segment"] = f"[{pl[j]['edges'][0]:.3f}, {pl[j]['edges'][1]:.3f}]"
        out[f"{v}_lowest_segment_n"] = int(pl[j]["n"][0])
    return out


# ----------------------------------------------------------------------------- per turbine
def run_turbine(turbine):
    d19, i19 = s.load_kelmarsh(2019, turbine)
    d18, i18 = s.load_kelmarsh(2018, turbine)
    inc = {"turbine": turbine, "records_2019": i19["records_after_filtering"], "records_2018": i18["records_after_filtering"],
           "warning_active_2019": i19["records_after_filtering_with_warning_active"]}
    reason = []
    if inc["records_2019"] < MIN_RECORDS:
        reason.append(f"2019: {inc['records_2019']} < {MIN_RECORDS} records")
    if inc["records_2018"] < MIN_RECORDS:
        reason.append(f"2018: {inc['records_2018']} < {MIN_RECORDS} records")
    med = class_medians(d18)
    try:
        B, thr, at, control = boundaries(med)
        inc["boundaries_estimated"] = True
    except ValueError as err:
        B = None
        inc["boundaries_estimated"] = False
        reason.append(f"boundary not estimable ({err})")
    inc["excluded"] = bool(reason)
    inc["reason"] = "; ".join(reason)
    if inc["excluded"]:
        return inc, None
    bnd = {"turbine": turbine, **{f"{k} wind (m/s)": w for k, w in B.items()},
           **{f"{k} threshold": thr[k] for k in B},
           **{f"{k} power median (kW)": at("power", w) for k, w in B.items()},
           **{f"{k} rotor median (rpm)": at("rotor_speed", w) for k, w in B.items()},
           **{f"{k} pitch median (deg)": at("pitch_angle", w) for k, w in B.items()}}

    data = s.sample(d19, i19, keep_time=True)
    corr = data["rotor_speed"].corr(data["generator_speed"])
    data.pop("time")
    warn = data.pop("warning_active").to_numpy()
    X = data.drop(columns="generator_speed")
    VARS = list(X.columns)
    specs, nk = make_specs(control, VARS, X)

    vaf, oos = vaf_table(specs, X)
    vaf.insert(0, "turbine", turbine)
    diffs = []
    for version, a, b in (("unrestricted", "control regions (2018)", "percentiles, same number of knots"),
                          ("monotone", "control regions (2018), monotone", "percentiles, same number of knots, monotone")):
        dm, se = paired(oos, a, b)
        diffs.append({"turbine": turbine, "version": version, "data": "all records", "difference": dm, "se": se})
    # sensitivity: without warning_active records
    if warn.sum() > 0:
        vaf_nw, oos_nw = vaf_table(specs, X[~warn].reset_index(drop=True))
    else:
        oos_nw = oos
    for version, a, b in (("unrestricted", "control regions (2018)", "percentiles, same number of knots"),
                          ("monotone", "control regions (2018), monotone", "percentiles, same number of knots, monotone")):
        dm, se = paired(oos_nw, a, b)
        diffs.append({"turbine": turbine, "version": version, "data": "without warning_active", "difference": dm, "se": se})

    # start-up / idle below B1, and lowest segments
    low = (X.wind_speed < B["B1"]).to_numpy()
    idle = low & ((X.pitch_angle > 5.0) | (X.rotor_speed < 8.0)).to_numpy()
    mc = fit(specs["control regions (2018)"], X)
    facts = {"turbine": turbine, "corr_rotor_generator": corr, "warning_active_in_sample": int(warn.sum()),
             "records_below_B1": int(low.sum()), "startup_idle_below_B1": int(idle.sum()),
             "interior_knots_control": dict(zip(VARS, nk))}
    facts.update({f"{k}_slope_below_B1": v for k, v in slope_signs_below(mc, X, VARS, B["B1"]).items()})
    sub = {k: specs[k] for k in ("PCA", "control regions (2018)")}
    vaf_ni, _ = vaf_table(sub, X[~idle].reset_index(drop=True))
    r = vaf_ni.set_index("model").loc["control regions (2018)"]
    facts.update({"control_gain_without_idle": r["paired gain"], "control_gain_without_idle_se": r["se (gain)"]})
    return inc, dict(boundaries=bnd, vaf=vaf, diffs=pd.DataFrame(diffs), facts=facts, med=med, d18=d18, B=B)


# ----------------------------------------------------------------------------- figures
def figures(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from qlpca.plots import set_style, BLUE, ORANGE, INK2, GRID
    set_style()

    diffs = pd.concat([r["diffs"] for r in results.values()])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, version in zip(axes, ("unrestricted", "monotone")):
        for k, (data, col, off) in enumerate((("all records", BLUE, -0.15), ("without warning_active", ORANGE, 0.15))):
            d = diffs[(diffs.version == version) & (diffs.data == data)]
            x = np.array([TURBINES.index(t) for t in d.turbine]) + 1 + off
            ax.errorbar(x, d.difference, yerr=d.se, fmt="o", color=col, capsize=3, label=data)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(1, 7), [t.replace("Kelmarsh_", "K") for t in TURBINES])
        ax.set_title(f"{version}", fontsize=9, loc="left")
        ax.set_xlabel("turbine")
    axes[0].set_ylabel("control regions minus percentiles,\nsame number of knots (VAF points, ± se)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(RES / f"all_differences.{ext}", bbox_inches="tight", dpi=150,
                    metadata={"CreationDate": None} if ext == "pdf" else None)

    fig, axes = plt.subplots(3, 6, figsize=(16, 7.5), sharex=True, sharey="row")
    for c, (t, r) in enumerate(results.items()):
        med, d18, B = r["med"], r["d18"], r["B"]
        for rr, (col, lab) in enumerate((("power", "power (kW)"), ("rotor_speed", "rotor speed (rpm)"),
                                         ("pitch_angle", "pitch (deg)"))):
            ax = axes[rr, c]
            ax.scatter(d18.wind_speed, d18[col], s=0.5, color=GRID, alpha=0.4, rasterized=True)
            ax.plot(med.index, med[col], color=BLUE, lw=1.3)
            for k, w in B.items():
                ax.axvline(w, color=ORANGE, lw=0.8, ls="--")
            ax.set_xlim(0, 20)
            if c == 0:
                ax.set_ylabel(lab)
            if rr == 0:
                ax.set_title(t.replace("_", " ") + " (2018)", fontsize=8, loc="left")
            if rr == 2:
                ax.set_xlabel("wind speed (m/s)")
    axes[2, 0].set_ylim(-2, 30)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(RES / f"all_operating_curves_2018.{ext}", bbox_inches="tight", dpi=150,
                    metadata={"CreationDate": None} if ext == "pdf" else None)


# ----------------------------------------------------------------------------- main
def main():
    RES.mkdir(exist_ok=True)
    inclusion, results = [], {}
    for t in TURBINES:
        inc, res = run_turbine(t)
        inclusion.append(inc)
        print(t, "excluded: " + inc["reason"] if inc["excluded"] else "included", flush=True)
        if res is None:
            continue
        results[t] = res
        if t == "Kelmarsh_1":
            v = res["vaf"].set_index("model")
            d = res["diffs"].set_index(["version", "data"]).loc[("unrestricted", "all records")]
            got = {"control_gain": round(v.loc["control regions (2018)", "paired gain"], 2),
                   "difference": round(d["difference"], 2)}
            print("Kelmarsh 1 check:", got, "expected", CHECK, flush=True)
            if got != CHECK:
                raise SystemExit(f"Kelmarsh 1 does not reproduce the case study: {got} vs {CHECK}")
    pd.DataFrame(inclusion).to_csv(RES / "all_inclusion.csv", index=False, lineterminator="\n")
    pd.DataFrame([r["boundaries"] for r in results.values()]).to_csv(RES / "all_boundaries.csv", index=False,
                                                                     float_format="%.4f", lineterminator="\n")
    vaf = pd.concat([r["vaf"] for r in results.values()])
    vaf.to_csv(RES / "all_vaf.csv", index=False, float_format="%.4f", lineterminator="\n")
    diffs = pd.concat([r["diffs"] for r in results.values()])
    diffs.to_csv(RES / "all_differences.csv", index=False, float_format="%.4f", lineterminator="\n")
    facts = pd.DataFrame([r["facts"] for r in results.values()])
    facts.to_csv(RES / "all_turbine_facts.csv", index=False, float_format="%.6f", lineterminator="\n")

    summary = {"turbines_included": list(results), "excluded": {i["turbine"]: i["reason"] for i in inclusion if i["excluded"]},
               "kelmarsh_1_check": CHECK}
    for (version, data), g in diffs.groupby(["version", "data"]):
        n = len(g)
        summary[f"{version}, {data}"] = {
            "positive": int((g.difference > 0).sum()), "turbines": n, "mean": float(g.difference.mean()),
            "se_between_turbines": float(g.difference.std(ddof=1) / np.sqrt(n)) if n > 1 else None}
    with open(RES / "all_summary.json", "w", encoding="utf8", newline="\n") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    figures(results)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
