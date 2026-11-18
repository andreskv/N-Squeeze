"""THE consumption: API rows, the two archive formats, archives read only before October 2021.

No network. The values are invented.
"""

import datetime as dt
import io
import json
import zipfile

import httpx

from nsqueeze.io.sources.the import THEConsumption, api_canonical, gpl_canonical, ncg_canonical

API_ROW = {
    "gasXType": "clearing allocation",
    "statusEN": "final",
    "gastag": "2022-09-05T06:00:00",
    "slPsyn_H_Gas": 1,
    "slPana_H_Gas": 2,
    "slPsyn_L_Gas": 3,
    "slPana_L_Gas": 4,
    "rlMmT_H_Gas": 5,
    "rlMmT_L_Gas": 6,
    "rlMoT_H_Gas": 7,
    "rlMoT_L_Gas": 8,
}
NCG = (
    "DayOfUse;HGasRLMmT;UnitHGasRLMmT;LGasSLPsyn;UnitLGasSLPsyn;Status\n"
    "30.09.2021;10;kWh;20;kWh;final\n"
)
GPL = "Datum;RLMmT_H [MWh];SLPsyn_L [MWh]\n30.09.2021;1;2\n"


def zipped(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("data.csv", text.encode("latin-1"))
    return buf.getvalue()


def test_api_rows_take_the_gas_day_and_status():
    df = api_canonical([API_ROW])
    assert len(df) == 8 and set(df.unit) == {"kWh"}
    row = df[df.series_id == "THE.RLMmT_H"].iloc[0]
    assert row.observed_at == dt.datetime(2022, 9, 5) and row.value == 5.0
    assert json.loads(row.meta)["status"] == "final"


def test_archive_formats():
    ncg = ncg_canonical(NCG)
    assert dict(zip(ncg.series_id, ncg.value, strict=True)) == {
        "NCG.RLMmT_H": 10.0,
        "NCG.SLPsyn_L": 20.0,
    }
    gpl = gpl_canonical(GPL)
    assert gpl.series_id.tolist() == ["GPL.RLMmT_H", "GPL.SLPsyn_L"]
    assert set(gpl.unit) == {"MWh"} and gpl.observed_at[0] == dt.datetime(2021, 9, 30)


def test_fetch_reads_the_archives_only_for_days_before_THE(tmp_path, monkeypatch):
    monkeypatch.setenv("NSQ_DATA_DIR", str(tmp_path))
    asked = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path.rsplit("/", 1)[-1])
        if "NCG" in str(request.url):
            return httpx.Response(200, content=zipped(NCG))
        if "verbrauchsdaten" in str(request.url):
            return httpx.Response(200, content=zipped(GPL))
        return httpx.Response(200, json=[API_ROW])

    src = THEConsumption(client=httpx.Client(transport=httpx.MockTransport(handler)))
    late = src.fetch(dt.date(2022, 9, 1), dt.date(2022, 9, 10))
    assert len(asked) == 1 and late.series_id.str.startswith("THE.").all()
    both = src.fetch(dt.date(2021, 9, 30), dt.date(2022, 9, 10))
    assert set(both.series_id.str[:3]) == {"NCG", "GPL", "THE"}
