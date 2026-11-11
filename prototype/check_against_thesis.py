"""Run from the repo root:  python prototype/check_against_thesis.py
Compares the numpy port of the thesis MATLAB listings with thesis Tables 4.3, 4.5, 4.7, 4.8."""
import csv, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qlpca_prototype import qlpca

root = pathlib.Path(__file__).resolve().parents[1]
rows = list(csv.reader(open(root / "data" / "st_gender_eu15.csv", encoding="utf8")))
X = np.array([[float(v) for v in r[2:]] for r in rows[1:]])
ref = json.load(open(root / "tests" / "reference_values.json", encoding="utf8"))

ev = np.linalg.eigvalsh(np.corrcoef(X.T))[::-1]
print("PCA eigenvalues  max|diff| vs Table 4.3:", np.abs(ev - ref["table_4_3_pca"]["eigenvalues"]).max().round(4))
for key, r in (("qlpca1", 1), ("qlpca2", 2)):
    o = qlpca(X, p=2, degree=1, r=r)
    t = ref["table_4_5_qlpca"][key]
    comm = (o["A"] ** 2).sum(1)
    print(key, "eig", o["eig"].round(3), "thesis", t["eigenvalues"],
          "| communalities max|diff|:", np.abs(comm - ref["communalities"][key]).max().round(3),
          "| |loadings| max|diff|:", np.abs(np.abs(o["A"]) - np.abs(np.array(ref["table_4_6_loadings"][key]))).max().round(3))
