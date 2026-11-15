"""MkDocs hook: prepare the material the site takes from the rest of the repository.

Runs before every build (``mkdocs build`` / ``mkdocs serve``), so the build needs no extra step:

* copies the executed notebooks (``notebooks/*.ipynb``; not re-executed) and the result figures of
  ``comparison/princals/results/figures`` and ``comparison/applications/results`` into
  ``docs/site/generated/`` (git-ignored);
* writes Markdown fragments to ``build/docs_fragments/`` (git-ignored), included in the pages with
  ``--8<--``: sections of the READMEs, DESIGN.md, PACKAGING.md, CHANGELOG.md and LICENSE-docs.md as they
  are (relative links rewritten to the GitHub repository), and tables built from result CSV files.

Nothing is taken from comparison/princals/notes, comparison/princals/paper, the thesis PDFs, data caches
or matlab_original.
"""
import json
import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs" / "site"
GEN = SITE / "generated"
FRAG = ROOT / "build" / "docs_fragments"
REPO = "https://github.com/nlavadodatasci/quasi-linear-PCA/blob/main"


def _rewrite_links(text, source):
    """Relative Markdown links of `source` -> absolute links to the file on GitHub."""
    base = source.parent.relative_to(ROOT)

    def fix(m):
        label, target = m.group(1), m.group(2)
        if re.match(r"^(https?:|mailto:|#)", target):
            return m.group(0)
        path, _, anchor = target.partition("#")
        resolved = (ROOT / base / path).resolve().relative_to(ROOT).as_posix()
        return f"[{label}]({REPO}/{resolved}{'#' + anchor if anchor else ''})"

    return re.sub(r"\[([^\]]*)\]\(([^)\s]+)\)", fix, text)


def _section(path, heading, drop_heading=True):
    """Text of the Markdown section that starts with `heading` (up to the next heading of the same level)."""
    text = path.read_text(encoding="utf8")
    level = len(heading) - len(heading.lstrip("#"))
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == heading)
    end = next((i for i in range(start + 1, len(lines))
                if re.match(rf"^#{{1,{level}}} ", lines[i])), len(lines))
    body = lines[start + (1 if drop_heading else 0):end]
    return _rewrite_links("\n".join(body).strip() + "\n", path)


def _whole(path, shift=0):
    text = _rewrite_links(path.read_text(encoding="utf8"), path)
    if shift:
        text = re.sub(r"^(#+) ", lambda m: "#" * (len(m.group(1)) + shift) + " ", text, flags=re.M)
    return text


def _write(name, text):
    FRAG.mkdir(parents=True, exist_ok=True)
    (FRAG / name).write_text(text, encoding="utf8")


def _copy_assets():
    if GEN.exists():
        shutil.rmtree(GEN)
    (GEN / "notebooks").mkdir(parents=True)
    for nb in sorted((ROOT / "notebooks").glob("*.ipynb")):
        shutil.copy2(nb, GEN / "notebooks" / nb.name)
    for src, dest in ((ROOT / "comparison/princals/results/figures", GEN / "princals"),
                      (ROOT / "comparison/applications/results", GEN / "kelmarsh")):
        dest.mkdir(parents=True)
        for png in sorted(src.glob("*.png")):
            shutil.copy2(png, dest / png.name)


def _knot_selection_table():
    d = pd.read_csv(ROOT / "comparison/princals/results/knot_selection.csv")
    d["cell"] = d.apply(lambda r: f"{r.vaf_oos_mean:.1f} ({r.vaf_oos_se:.1f})", axis=1)
    t = d.pivot(index="dataset", columns="n_knots", values="cell").reindex(list(dict.fromkeys(d.dataset)))
    head = "| data | " + " | ".join(f"{k} knots" for k in t.columns) + " |\n"
    head += "|---|" + "---|" * len(t.columns) + "\n"
    return head + "".join(f"| {i} | " + " | ".join(t.loc[i]) + " |\n" for i in t.index)


def _princals_tables():
    s = pd.read_csv(ROOT / "comparison/princals/results/summary.csv")
    out = ["| data | run | qlpca VAF | pyGifi VAF | Gifi (R) VAF | max abs diff. eigenvalues qlpca - R |",
           "|---|---|---|---|---|---|"]
    for r in s[s.run.isin(["knots1", "knots2", "monotone_deg1", "monotone_deg2", "defaults"])].itertuples():
        out.append(f"| {r.dataset} | {r.run} | {r.qlpca_vaf2:.3f} | {r.pygifi_vaf2:.3f} | {r.gifi_r_vaf2:.3f} | "
                   f"{r.qlpca_vs_gifi_r_evals:.1e} |")
    return "\n".join(out) + "\n"


def _notebook_conclusions(path, marker="lines = []"):
    nb = json.loads(path.read_text(encoding="utf8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and marker in "".join(cell["source"]):
            text = "".join("".join(o.get("text", "")) for o in cell.get("outputs", []) if o.get("output_type") == "stream")
            return "```text\n" + text.strip() + "\n```\n"
    return "*(conclusions cell not found)*\n"


def on_pre_build(config, **kwargs):
    _copy_assets()
    readme = ROOT / "README.md"
    _write("readme_results.md", _section(readme, "## Results compared with the thesis"))
    _write("readme_notebooks.md", _section(readme, "## Notebooks"))
    _write("readme_cite.md", _section(readme, "## How to cite"))
    _write("readme_related.md", _section(readme, "## Related work and software"))
    _write("design.md", _whole(ROOT / "docs/DESIGN.md"))
    _write("packaging.md", _whole(ROOT / "docs/PACKAGING.md"))
    _write("changelog.md", _whole(ROOT / "CHANGELOG.md"))
    _write("licence_docs.md", _whole(ROOT / "LICENSE-docs.md", shift=1))
    _write("knot_selection_table.md", _knot_selection_table())
    _write("princals_tables.md", _princals_tables())
    _write("princals_conclusions.md", _notebook_conclusions(ROOT / "comparison/princals/qlpca_vs_princals.ipynb"))
    apps = ROOT / "comparison/applications/README.md"
    _write("kelmarsh_protocol.md", _section(apps, "## All turbines: protocol"))
    _write("kelmarsh_results.md", _section(apps, "## All turbines: results (run on 2026-11-15)"))
    _write("kelmarsh_screening_criterion.md", _section(apps, "## Approval criterion (fixed before running)"))
