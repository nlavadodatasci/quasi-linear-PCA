# Data

## `st_gender_eu15.csv` — Science & Technology gender differences, 15 EU countries
Transcribed from Appendix A.1 (p. 112) of the PhD thesis and checked against the PDF
page. It reproduces the summary statistics of thesis Table 4.1 and the PCA
eigenvalues of Table 4.3.

- Origin: Eurostat and *She Figures 2006* (reference years 2003-2004), as selected
  by Oliveira & Carvalho (2009). The thesis states the data were kindly provided
  by those authors. **Check with them before redistributing under an open licence.**
- 12 of the 27 EU countries were dropped for missing values.
- Six values carry a third decimal (e.g. 2.301, 21.101, 4.201). These were added
  in the thesis on purpose so that every value within a variable is unique, a
  requirement of the 2012 implementation (thesis p. 82, footnote 5). Keep them.

| column | thesis label | meaning |
|---|---|---|
| researchers | Investig. | researchers per 1000 active population (2003) |
| scientists_engineers | Cient. e Eng. | scientists and engineers, share of active population (2004) |
| investment | Investimento | S&T investment per researcher, PPS |
| mf_funds | H-M Fundos | men minus women, success rate in research funding (2004) |
| w_top | M top | women at top grade, share of female academic staff (2004) |
| m_top | H top | men at top grade, share of male academic staff (2004) |
| glass_ceiling | Glass Ceiling | Glass Ceiling Index (2004) |
| mf_phd | H-M PhD | men minus women, % of PhD (ISCED 6) graduates (2003) |
| mf_researchers | H-M investig. | men minus women, % of researchers (2003) |
| mf_teaching | H-M docentes | men minus women, % of higher-education teaching staff (2004) |
| mf_boards | H-M CC | men minus women, % on scientific boards (2004) |

## `st_gender_eu15_ranks.csv`
Appendix A.2 (p. 113): the same data discretised by ranking (used for CATPCA).
Six cells differ from ranks recomputed from A.1; all six are the pairs that were
tied before the third decimal was added (the thesis ranks break those ties the
other way round).

## `wce2011_eurostat_indicators.csv`
The 12 Eurostat indicator codes used in the WCE 2011 paper (26 countries).
**The data values are not printed in the paper and the reference year is not
stated**, so that example can only be re-created approximately from Eurostat.
Paper results to compare with: explained variance in two dimensions 58% (qlPCA),
64% (CATPCA), 54% (PCA).

## Cylinder problem (thesis section 3.3)
Simulated, not stored: see `docs/SPEC.md` section 7.
