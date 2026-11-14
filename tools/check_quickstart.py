"""Run the README quick start, and check the wheel-install behaviour of the data loader.

Usage:  python tools/check_quickstart.py [path/to/README.md] [--expect-no-data]

The first python block after the "Quick start" heading is executed as is. With --expect-no-data
(an installation outside a clone) load_st_gender must raise a FileNotFoundError.
"""
import pathlib
import re
import sys


def main(argv):
    flags = [a for a in argv if a.startswith("--")]
    args = [a for a in argv if not a.startswith("--")]
    readme = pathlib.Path(args[0] if args else pathlib.Path(__file__).resolve().parents[1] / "README.md")
    m = re.search(r"## Quick start\s+```python\n(.*?)```", readme.read_text(encoding="utf8"), re.S)
    if not m:
        print("quick start block not found in", readme)
        return 1
    exec(compile(m.group(1), "README quick start", "exec"), {"__name__": "__quickstart__"})
    print("quick start ok")
    if "--expect-no-data" in flags:
        from qlpca.datasets import load_st_gender
        try:
            load_st_gender()
        except FileNotFoundError as err:
            print("load_st_gender raises FileNotFoundError, as expected:\n ", str(err)[:120], "...")
        else:
            print("load_st_gender found data outside a clone: unexpected")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
