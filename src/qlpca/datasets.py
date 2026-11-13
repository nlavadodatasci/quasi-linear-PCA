"""Data loaders. The chapter 4 data set lives in the repository's ``data/`` folder."""
import csv
import os
from pathlib import Path

import numpy as np


def data_dir():
    """Repository data folder (override with the QLPCA_DATA environment variable)."""
    return Path(os.environ.get("QLPCA_DATA", Path(__file__).resolve().parents[2] / "data"))


def load_st_gender():
    """Science & Technology gender data, 15 EU countries (thesis Appendix A.1).

    Returns ``(X, variables, countries)`` with X of shape (15, 11).
    """
    with open(data_dir() / "st_gender_eu15.csv", encoding="utf8", newline="") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    X = np.array([[float(v) for v in r[2:]] for r in body])
    return X, header[2:], [r[0] for r in body]


CYLINDER_VARIABLES = [
    "a", "b", "2*sqrt(b*pi)", "2*a*sqrt(b*pi)", "a*b", "a*b^2/(2*pi)", "a/sqrt(2*b*pi)",
    "arctan(a*sqrt(pi)/(2*sqrt(b)))", "arccot(a*sqrt(pi)/(2*sqrt(b)))", "a/b", "b/a", "2*a*pi/b^2",
]


def make_cylinder(n=51, noise=0.0, random_state=None):
    """Simulated "cylinder problem" (thesis section 3.3): 12 variables derived from the
    height a and base area b of n cylinders.

    Parameters
    ----------
    n : number of cylinders (thesis: 51).
    noise : 0, 0.10 or 0.25 in the thesis. Every variable is mapped to its linearising scale
        (log; log(tan) for the arctan variable and log(cot) for the arccot variable), normal
        noise with standard deviation ``noise`` times the sample standard deviation of the
        variable *on that linearised scale* is added, and the result is mapped back.
    random_state : seed or ``numpy.random.Generator``.

    Returns ``(X, names)`` with X of shape (n, 12).

    Choices that the thesis does not document (docs/SPEC.md sections 7 and 9):
    a and b are drawn from U(0, 1) (stated in the MSc thesis, chapter 6, for its own version of
    the problem; the ranges printed in the PhD figures are consistent with it); and the noise
    standard deviation is measured **on the linearised scale**, the reading that makes sense
    (on the natural scale it would be hundreds of log-units for variable 12).
    """
    rng = np.random.default_rng(random_state)
    a, b = rng.uniform(size=n), rng.uniform(size=n)
    u = a * np.sqrt(np.pi) / (2 * np.sqrt(b))
    X = np.column_stack([
        a, b, 2 * np.sqrt(b * np.pi), 2 * a * np.sqrt(b * np.pi), a * b, a * b ** 2 / (2 * np.pi),
        a / np.sqrt(2 * b * np.pi), np.arctan(u), np.arctan(1 / u), a / b, b / a, 2 * a * np.pi / b ** 2,
    ])
    if noise:
        Z = np.log(X)
        Z[:, 7] = np.log(np.tan(X[:, 7]))       # log(tan) of the angle = log(u)
        Z[:, 8] = np.log(1 / np.tan(X[:, 8]))   # log(cot) of the angle = log(u)
        Z = Z + rng.normal(size=Z.shape) * noise * Z.std(0, ddof=1)
        X = np.exp(Z)
        X[:, 7] = np.arctan(np.exp(Z[:, 7]))
        X[:, 8] = np.arctan(np.exp(-Z[:, 8]))   # arccot(e^z) = arctan(e^-z)
    return X, list(CYLINDER_VARIABLES)
