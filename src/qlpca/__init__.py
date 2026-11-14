"""qlPCA: quasi-linear principal component analysis."""
from importlib.metadata import PackageNotFoundError, version

from .core import QLPCA, ConvergenceWarning, SmallSampleWarning
from .splines import bspline_basis, interior_knots, ispline_basis, pseudo_indicator

__all__ = ["QLPCA", "ConvergenceWarning", "SmallSampleWarning", "bspline_basis", "interior_knots", "ispline_basis", "pseudo_indicator"]
try:
    __version__ = version("qlpca")     # single source: pyproject.toml, via the package metadata
except PackageNotFoundError:           # running from a source tree that was never installed
    __version__ = "0+unknown"
