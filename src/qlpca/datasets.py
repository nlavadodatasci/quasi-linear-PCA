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
