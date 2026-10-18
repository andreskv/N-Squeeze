"""Rebuild the working paper's data extract (paper/data/) from the point-in-time archive.

    .venv/Scripts/python scripts/paper_data.py

monthly_panel already drops data after the seal date in config/freeze.yaml, and the
Comext series are cut at the same date, so no hold-out month reaches the extract while
the pre-registration is unfrozen.
"""

import json

import pandas as pd

from nsqueeze import case_nitrogen as cn
from nsqueeze import validation as v
from nsqueeze.config import REPO_ROOT
from nsqueeze.freeze import sealed_after
from nsqueeze.import_parity import ORIGINS
from nsqueeze.io import PITStore
from nsqueeze.params import ParameterSet

SOURCES = {
    "ttf_usd_mmbtu": "World Bank Pink Sheet, 'Natural gas, Europe' (TTF from 2015-04)",
    "hh_usd_mmbtu": "World Bank Pink Sheet, 'Natural gas, US' (Henry Hub)",
    "urea_usd_t": "World Bank Pink Sheet, 'Urea' (fob Black Sea to 2022-02; Middle East)",
    "eua_eur_t": "EEX primary auctions, monthly volume-weighted clearing price",
    "fx_usd_per_eur": "ECB euro reference rate, monthly mean of daily fixings",
}


def main() -> None:
    store, out = PITStore(), REPO_ROOT / "paper" / "data"
    p = cn.monthly_panel(store)
    # EUA goes in its own file, kept local: the EEX terms for redistribution are unchecked.
    p[[c for c in SOURCES if c != "eua_eur_t"]].to_csv(
        out / "nitrogen_cost_panel.csv", lineterminator="\n"
    )
    p[["eua_eur_t"]].to_csv(out / "eua_monthly.csv", lineterminator="\n")
    meta = {
        "rows": len(p),
        "first_month": p.index.min().date().isoformat(),
        "last_month": p.index.max().date().isoformat(),
        "sealed_after": p.attrs["sealed_after"],
        "vintage_pink_sheet": p.attrs["vintage_pink_sheet"],
        "vintage_eex": p.attrs["vintage_eex"],
        "sources": SOURCES,
    }
    (out / "nitrogen_cost_panel.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    seal = sealed_after()
    imp = v.comext_imports_t(store).loc[p.index.min() : seal or p.index.max()]
    imp.rename("nh3_imports_t").rename_axis("month").to_csv(
        out / "comext_2814_imports.csv", float_format="%.0f", lineterminator="\n"
    )
    end = seal or p.index.max()
    uv = cn.comext_unit_value(store).loc[p.index.min() : end]
    uv.to_csv(out / "comext_310210_border.csv", float_format="%.2f", lineterminator="\n")
    if seal is None:  # the CBAM cost check uses 2026 only
        keep = ["EXT_EU27_2020", *ORIGINS, *ParameterSet.load()["cbam.exempt_origins"].value]
        rows = []
        for hs in ("2814", "310210"):
            t = cn.comext_partner_tonnes(store, hs).loc["2026-01-01":]
            t = t[[c for c in keep if c in t]].stack().rename("tonnes").reset_index()
            rows.append(t.assign(product=hs))
        part = pd.concat(rows)[["month", "product", "partner", "tonnes"]]
        part.to_csv(out / "comext_cbam_partners_2026.csv", index=False, lineterminator="\n")
    print(f"wrote {len(p)} panel rows, {len(imp)} import rows and {len(uv)} border rows to {out}")


if __name__ == "__main__":
    main()
