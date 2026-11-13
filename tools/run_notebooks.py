"""Execute every notebook in notebooks/ from top to bottom; exit with status 1 if any cell fails.

Usage (from the repository root):  python tools/run_notebooks.py [--write]

Without --write the notebooks are executed in memory and left unchanged on disk (CI check).
With --write the executed notebooks, outputs included, are saved back in place.
"""
import pathlib
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def main(write=False):
    failed = []
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        t0 = time.time()
        try:
            NotebookClient(nb, timeout=1800, kernel_name="python3",
                           resources={"metadata": {"path": str(NOTEBOOKS)}}).execute()
        except CellExecutionError as err:
            failed.append(path.name)
            print(f"FAILED  {path.name}\n{err}", flush=True)
            continue
        if write:
            nbformat.write(nb, path)
        print(f"ok      {path.name}  ({time.time() - t0:.0f} s)", flush=True)
    if failed:
        print("failed notebooks:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(write="--write" in sys.argv[1:]))
