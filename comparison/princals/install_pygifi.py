"""Install pyGifi at the pinned commit together with its C extension pygifi_rng.

    python install_pygifi.py

1. clones https://github.com/vam-math/pyGifi into .cache/pyGifi and checks out 7aa81ed;
2. installs it with pip;
3. builds pygifi/rng/pygifi_rng.c in place (needs a C compiler: gcc/clang, or MSVC on Windows);
4. copies the compiled module into site-packages so that `import pygifi_rng` works.

pyGifi is GPL-3. Nothing from it is copied into this repository.
"""
import glob
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

URL = "https://github.com/vam-math/pyGifi"
COMMIT = "7aa81ed"
SRC = Path(__file__).resolve().parent / ".cache" / "pyGifi"


def run(*cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    if not SRC.exists():
        SRC.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--quiet", URL, str(SRC))
    run("git", "fetch", "--quiet", "origin", cwd=SRC)
    run("git", "checkout", "--quiet", COMMIT, cwd=SRC)
    run(sys.executable, "-m", "pip", "install", "--quiet", str(SRC))
    rng = SRC / "pygifi" / "rng"
    run(sys.executable, "-m", "pip", "install", "--quiet", "setuptools", "numpy")
    run(sys.executable, "setup_rng.py", "build_ext", "--inplace", cwd=rng)
    built = glob.glob(str(rng / "pygifi_rng*.so")) + glob.glob(str(rng / "pygifi_rng*.pyd"))
    if not built:
        sys.exit("pygifi_rng was not built")
    dest = Path(sysconfig.get_paths()["platlib"])
    for f in built:
        shutil.copy2(f, dest)
        print("copied", Path(f).name, "to", dest)
    run(sys.executable, "-c", "import pygifi, pygifi_rng; print('pygifi', pygifi.__file__); print('pygifi_rng ok')")


if __name__ == "__main__":
    main()
