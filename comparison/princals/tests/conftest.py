import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # import compare.py


def pytest_configure(config):
    config.addinivalue_line("markers", "comparison: optional comparison with princals (needs pyGifi)")
