"""qlPCA: quasi-linear principal component analysis."""
from .core import QLPCA
from .splines import bspline_basis, interior_knots, ispline_basis, pseudo_indicator

__all__ = ["QLPCA", "bspline_basis", "interior_knots", "ispline_basis", "pseudo_indicator"]
__version__ = "0.1.0"
