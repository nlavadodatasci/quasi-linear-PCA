"""Plots for qlPCA: scree plot, transformation plots, biplot (matplotlib, imported lazily).

Style: light surface, thin lines, recessive grid, blue for the main series and orange for
the comparison series; text stays in neutral ink.
"""
import numpy as np

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _style(ax, grid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9, width=0.8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    return ax


def scree_plot(eigenvalues, ax=None, label=None, reference=None, reference_label=None, kaiser=True, title=None):
    """Scree plot: eigenvalues of the correlation matrix of the (transformed) variables.

    ``eigenvalues`` is an array or a fitted QLPCA (its ``all_eigenvalues_`` are used).
    ``reference`` optionally adds a second series (e.g. the PCA eigenvalues). Solutions of
    different dimension are not nested, so compare the scree plots of p-1, p and p+1.
    """
    ev = np.asarray(getattr(eigenvalues, "all_eigenvalues_", eigenvalues), float)
    if ax is None:
        ax = _plt().subplots(figsize=(5.2, 3.4))[1]
    k = np.arange(1, len(ev) + 1)
    ax.plot(k, ev, "-o", color=BLUE, lw=1.6, ms=5, mec="white", mew=1, label=label)
    if reference is not None:
        ref = np.asarray(getattr(reference, "all_eigenvalues_", reference), float)
        ax.plot(np.arange(1, len(ref) + 1), ref, "-o", color=ORANGE, lw=1.6, ms=5, mec="white", mew=1,
                label=reference_label)
    if kaiser:
        ax.axhline(1, color=INK2, lw=0.8, ls=(0, (4, 3)))
        ax.text(len(ev), 1, " eigenvalue = 1", va="bottom", ha="right", fontsize=8, color=INK2)
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_xticks(k)
    ax.set_ylim(bottom=0)
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    if label or reference_label:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    return _style(ax)


def transformation_plots(model, X, names=None, ncols=4, figsize=None, n_points=200):
    """Small multiples of the optimal transformations: f_j against the original variable.

    Points are the observations, the line is the fitted spline, dashed verticals are the
    interior knots. A flat or reversed segment shows where the relation with the other
    variables is not linear (or not monotone).
    """
    X = np.asarray(X, float)
    m = X.shape[1]
    names = names or [f"x{j + 1}" for j in range(m)]
    nrows = int(np.ceil(m / ncols))
    plt = _plt()
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (2.6 * ncols, 2.2 * nrows), squeeze=False)
    for j in range(nrows * ncols):
        ax = axes.flat[j]
        if j >= m:
            ax.axis("off")
            continue
        x, f = model.variable_transformation(j, n_points)
        for kn in np.unique(model.knots_[j])[1:-1]:
            ax.axvline(kn * model.std_[j] + model.mean_[j], color=GRID, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.plot(x, f, color=BLUE, lw=1.6, zorder=3)
        ax.scatter(X[:, j], model.transformed_[:, j], s=14, color=BLUE, alpha=0.55, edgecolor="white",
                   linewidth=0.5, zorder=2)
        ax.set_title(names[j], fontsize=9, loc="left")
        _style(ax, grid=False)
    fig.supxlabel("original variable", fontsize=9, color=INK2)
    fig.supylabel("transformed variable", fontsize=9, color=INK2)
    fig.tight_layout()
    return fig


def biplot(model, X=None, names=None, labels=None, components=(0, 1), ax=None, arrow_scale=None):
    """Biplot: object scores as points (optionally labelled) and loadings as arrows.

    Arrows are rescaled by one common factor to the range of the scores. In qlPCA the
    loadings refer to the *transformed* variables; a variable with a V-shaped
    transformation has an arbitrary sign (see ``QLPCA`` and docs/DESIGN.md D6).
    """
    s, t = components
    S, A = model.scores_[:, [s, t]], model.loadings_[:, [s, t]]
    if ax is None:
        ax = _plt().subplots(figsize=(5.6, 5.2))[1]
    ax.scatter(S[:, 0], S[:, 1], s=26, color=BLUE, alpha=0.8, edgecolor="white", linewidth=0.6, zorder=3)
    if labels is not None:
        for (x, y), lab in zip(S, labels):
            ax.annotate(lab, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8, color=INK2)
    k = arrow_scale or 0.85 * np.abs(S).max() / max(np.abs(A).max(), 1e-12)
    names = names or [f"x{j + 1}" for j in range(A.shape[0])]
    for (x, y), nm in zip(A * k, names):
        ax.annotate("", (x, y), (0, 0), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2), zorder=2)
        ax.annotate(nm, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=8, color=INK, zorder=4)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    ax.set_xlabel(f"Component {s + 1}")
    ax.set_ylabel(f"Component {t + 1}")
    ax.set_aspect("equal", adjustable="datalim")
    return _style(ax, grid=False)


def set_style():
    """Matplotlib defaults used by the notebooks (white surface, neutral ink, thin lines)."""
    plt = _plt()
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "figure.dpi": 110, "axes.titlesize": 10, "axes.labelsize": 9,
        "font.size": 9, "text.color": INK, "legend.frameon": False,
    })


def algorithm_diagram(ax=None):
    """Flow diagram of the qlPCA algorithm (thesis Figure 3.1), with the ALS loop shaded."""
    from matplotlib.patches import FancyBboxPatch
    plt = _plt()
    if ax is None:
        ax = plt.subplots(figsize=(7.2, 6.2))[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.4)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((1.2, 2.45), 7.6, 3.95, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="#f3f2ee", ec=INK2, lw=0.8, zorder=0))
    ax.text(8.15, 4.6, "Alternating\nleast squares", ha="center", va="center", fontsize=9, color=INK2,
            fontweight="bold")
    boxes = {
        "start": (4.4, 10.0, "Start"), "read": (4.4, 9.0, "Read H, p, r, v"),
        "init": (7.7, 8.0, "Initialise X"), "G": (1.9, 7.4, "Compute G (pseudo-indicator)"),
        "load": (7.7, 6.9, "Compute loadings"), "quant": (4.0, 5.7, "Update quantifications"),
        "scores": (4.0, 3.4, "Estimate object scores"), "conv": (4.0, 1.5, "Convergence test"),
        "end": (4.0, 0.4, "End"),
    }
    for key, (x, y, label) in boxes.items():
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5, color=INK, zorder=3,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=INK2, lw=0.9))
    arrow = dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=11, shrinkB=14)
    for a, b in [("start", "read"), ("read", "init"), ("read", "G"), ("init", "load"), ("G", "quant"),
                 ("load", "quant"), ("quant", "scores"), ("scores", "conv"), ("conv", "end")]:
        ax.annotate("", xy=boxes[b][:2], xytext=boxes[a][:2], arrowprops=arrow, zorder=2)
    xr = 6.3  # return path of the loop: convergence test -> right -> up -> update quantifications
    ax.plot([5.5, xr, xr], [1.5, 1.5, 5.7], color=INK2, lw=0.9, zorder=2)
    ax.annotate("", xy=(5.55, 5.7), xytext=(xr, 5.7), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9,
                                                                       shrinkA=0, shrinkB=0), zorder=2)
    ax.text(xr + 0.15, 3.0, "not converged", fontsize=8, color=INK2, va="center")
    return ax
