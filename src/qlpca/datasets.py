"""Data loaders. The chapter 4 data set is shipped inside the package (``qlpca/data``)."""
import csv
import os
from importlib import resources
from pathlib import Path

import numpy as np


def data_dir():
    """Folder that holds the data files.

    Returns
    -------
    pathlib.Path
        The folder given by the ``QLPCA_DATA`` environment variable if set, otherwise the package's own
        ``data`` folder.
    """
    env = os.environ.get("QLPCA_DATA")
    return Path(env) if env else Path(str(resources.files("qlpca") / "data"))


def _open_data(name):
    env = os.environ.get("QLPCA_DATA")
    if env:
        path = Path(env) / name
        if not path.is_file():
            raise FileNotFoundError(
                f"{name} was not found in the folder given by QLPCA_DATA ({env}). Unset QLPCA_DATA "
                "to use the copy shipped with the package, or point it to a folder containing the file.")
        return open(path, encoding="utf8", newline="")
    return (resources.files("qlpca") / "data" / name).open(encoding="utf8", newline="")


def load_st_gender():
    """Science & Technology gender data, 15 EU countries (thesis chapter 4, Appendix A.1).

    The file is shipped with the package and read with :mod:`importlib.resources`, so this works after
    ``pip install``. Set the environment variable ``QLPCA_DATA`` to a folder containing
    ``st_gender_eu15.csv`` to read another copy. The data are not covered by the MIT licence of the code
    (see ``qlpca/data/README.md``).

    Returns
    -------
    X : ndarray of shape (15, 11)
        The 11 indicators.
    variables : list of str
        Variable names.
    countries : list of str
        Country codes.
    """
    with _open_data("st_gender_eu15.csv") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    X = np.array([[float(v) for v in r[2:]] for r in body])
    return X, header[2:], [r[0] for r in body]


CYLINDER_VARIABLES = [
    "a", "b", "2*sqrt(b*pi)", "2*a*sqrt(b*pi)", "a*b", "a*b^2/(2*pi)", "a/sqrt(2*b*pi)",
    "arctan(a*sqrt(pi)/(2*sqrt(b)))", "arccot(a*sqrt(pi)/(2*sqrt(b)))", "a/b", "b/a", "2*a*pi/b^2",
]


def make_cylinder(n=51, noise=0.0, random_state=None):
    """Simulated "cylinder problem" (thesis section 3.3).

    Twelve variables derived from the height a and base area b of n cylinders, with a, b ~ U(0, 1) (stated
    in the MSc thesis, chapter 6). With noise, every variable is mapped to its linearising scale (log;
    log(tan) for the arctan variable and log(cot) for the arccot variable), normal noise with standard
    deviation ``noise`` times the sample standard deviation of the variable on that scale is added, and
    the result is mapped back (docs/SPEC.md sections 7 and 9).

    Parameters
    ----------
    n : int, default=51
        Number of cylinders (thesis: 51).
    noise : float, default=0.0
        0, 0.10 or 0.25 in the thesis.
    random_state : None, int or numpy.random.Generator, default=None
        Seed.

    Returns
    -------
    X : ndarray of shape (n, 12)
        The simulated variables.
    names : list of str
        Variable names.
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
