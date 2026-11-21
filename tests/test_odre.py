"""French industrial gas (ODRÉ): one row per region and day, the final status wins.

No network. The values are invented.
"""

import datetime as dt
import json

import httpx

from nsqueeze.io.sources.odre import NatranIndustrial, canonical


def rec(day, status, value, operator="NaTran", code=28, region="Normandie"):
    return {
        "date": day,
        "operateur": operator,
        "region": region,
        "code_region": code,
        "statut": status,
        "consommation_journaliere_mwh_pcs": value,
    }


RECORDS = [
    rec("2023-03-02", "Meilleur statut", 100),
    rec("2023-03-02", "Définitif", 110),
    rec("2023-03-03", "Meilleur statut", 120),
    rec("2023-03-02", "Définitives", 50, "Teréga", 76, "Occitanie"),
]


def test_final_row_wins_and_ids_are_ascii():
    df = canonical(RECORDS)
    assert df.series_id.tolist() == [
        "ODRE.IND.NATRAN.28",
        "ODRE.IND.NATRAN.28",
        "ODRE.IND.TEREGA.76",
    ]
    assert df.value.tolist() == [110.0, 120.0, 50.0]
    assert df.observed_at[0] == dt.datetime(2023, 3, 2)
    assert json.loads(df.meta[1]) == {"region": "Normandie", "status": "Meilleur statut"}


def test_fetch_asks_for_the_window(tmp_path, monkeypatch):
    monkeypatch.setenv("NSQ_DATA_DIR", str(tmp_path))
    asked = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.params["where"])
        return httpx.Response(200, json=RECORDS[:1])

    src = NatranIndustrial(client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert len(src.fetch(dt.date(2023, 3, 1), dt.date(2023, 3, 31))) == 1
    assert asked == ["date >= '2023-03-01' AND date <= '2023-03-31'"]
