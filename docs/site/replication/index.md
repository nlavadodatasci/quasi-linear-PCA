# Thesis replication

Chapter-by-chapter replication of Lavado (2012), *Extensões via splines da Análise em Componentes Principais*
(PhD thesis, ISCTE-IUL, <http://hdl.handle.net/10071/6202>). The original MATLAB files were lost; the listings
printed in the thesis were transcribed and the package was tested against them. The notebooks are shown as they
are stored in the repository, already executed; the site does not re-execute them.

| notebook | source | content |
|---|---|---|
| [01 Spline transformations](../generated/notebooks/01_splines.ipynb) | thesis §1.1 | B-, M- and I-splines, knots, pseudo-indicator matrix; thesis Fig. 1.1 |
| [02 PCA as loss minimisation](../generated/notebooks/02_pca_loss_function.ipynb) | thesis §1.2–1.3 | the common loss function; PCA, MCA and qlPCA as spline classes |
| [03 From CATPCA to qlPCA](../generated/notebooks/03_from_catpca_to_qlpca.ipynb) | thesis ch. 2 | HOMALS and CATPCA in brief; discretisation of continuous variables (Table 2.2) |
| [04 The qlPCA algorithm](../generated/notebooks/04_qlpca_algorithm.ipynb) | thesis §3.1–3.2 | corrected pseudo-code, legacy vs default, properties; Figs 3.1, 3.2 |
| [05 The cylinder problem](../generated/notebooks/05_cylinder_problem.ipynb) | thesis §3.3 | simulation study; Table 3.3, Figs 3.4–3.7 (qualitative) |
| [06 Science & technology: gender differences](../generated/notebooks/06_science_technology_gender.ipynb) | thesis ch. 4 | the application; Tables 4.1–4.8, Figs 4.1–4.10 (exact) |
| [07 Eurostat economic indicators](../generated/notebooks/07_wce2011_eurostat.ipynb) | WCE 2011 | introduction only (the data values and reference year are not in the paper) |

## Results compared with the thesis

The table of the repository README, included as it is.

--8<-- "readme_results.md"
