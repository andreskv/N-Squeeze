# paper/data/

The data extract for the paper. With it, the results can be rebuilt without the point-in-time archive.

| File | Content |
|---|---|
| `nitrogen_cost_panel.csv` | Monthly, 2015-04 to 2026-09: TTF and Henry Hub ($/MMBtu), urea f.o.b. ($/t), USD per EUR |
| `comext_2814_imports.csv` | Monthly, 2015-04 to 2026-07: EU-27 imports of ammonia (CN 2814) from outside the EU, tonnes |
| `comext_310210_border.csv` | Monthly, 2015-04 to 2026-07: EU-27 imports of urea (HS 310210) from outside the EU, unit value (€/t at the border) and tonnes |
| `comext_cbam_partners_2026.csv` | Monthly, 2026-01 to 2026-07: EU-27 imports of ammonia and urea in tonnes. Extra-EU total, and each partner that the CBAM cost check prices on its own |
| `nitrogen_cost_panel.meta.json` | Archive vintages (retrieval timestamps), seal date, series definitions |

Rebuild from the archive with `python scripts/paper_data.py`. While `config/freeze.yaml` has `frozen: false`, the script writes no data after the end of the calibration window.

**EU allowance prices are not included.** The EEX terms for redistribution are not checked. To rebuild, put `eua_monthly.csv` (columns `month`, `eua_eur_t`) in this folder. Use the monthly volume-weighted clearing price of the EEX primary auctions, from the auction results that EEX publishes.

**Sources and licences.** World Bank Pink Sheet data are CC BY 4.0. FX rates are from the ECB, and trade data are from Eurostat Comext. ECB and Eurostat data are free to reuse with attribution.
