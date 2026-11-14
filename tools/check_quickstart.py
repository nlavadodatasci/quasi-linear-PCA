"""Run the README quick start and check that the packaged data load (wheel-install check).

Usage:  python tools/check_quickstart.py [path/to/README.md]

The first python block after the "Quick start" heading is executed as is; then
load_st_gender() must return the chapter 4 data (15 x 11) from the installed package.
"""
import os
import pathlib
import re
import sys


def main(argv):
    readme = pathlib.Path(argv[0] if argv else pathlib.Path(__file__).resolve().parents[1] / "README.md")
    m = re.search(r"## Quick start\s+```python\n(.*?)```", readme.read_text(encoding="utf8"), re.S)
    if not m:
        print("quick start block not found in", readme)
        return 1
    exec(compile(m.group(1), "README quick start", "exec"), {"__name__": "__quickstart__"})
    print("quick start ok")

    os.environ.pop("QLPCA_DATA", None)
    import qlpca
    from qlpca.datasets import load_st_gender
    X, variables, countries = load_st_gender()
    if X.shape != (15, 11) or len(variables) != 11 or len(countries) != 15:
        print("load_st_gender returned unexpected shapes:", X.shape)
        return 1
    print(f"load_st_gender ok: {X.shape} from {pathlib.Path(qlpca.__file__).parent}")
    print("qlpca", qlpca.__version__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
