# Licences

* **Code** (`src/`, `tests/`, `prototype/`, `tools/`, the comparison scripts and the code cells of the notebooks):
  [MIT](https://github.com/nlavadodatasci/quasi-linear-PCA/blob/main/LICENSE).
* **Text and figures** (notebook prose and outputs, `docs/`, this site, the READMEs):
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
* **Data** keep the terms of their origin and are not covered by the licences above:
    * thesis chapter 4 data (`qlpca.datasets.load_st_gender`): compiled by Oliveira & Carvalho (2009) from Eurostat
      and *She Figures 2006*, provided to the author for the thesis;
    * UCI Gas Turbine CO and NOx Emission Data Set: doi:10.24432/C5WC95, CC BY 4.0 (downloaded at run time);
    * Kelmarsh wind farm data: doi:10.5281/zenodo.16807551, CC-BY-4.0 (downloaded at run time);
    * `neumann`: loaded from pyGifi at run time (GPL-3 package), not stored in the repository.
* **Third-party software** used only in the comparison: Gifi (R) and pyGifi are GPL-3; they are called or imported,
  never copied into `qlpca`.

The licence file for text and notebooks, as it is in the repository:

--8<-- "licence_docs.md"
