"""``nsq`` command line interface.

nsq params                 evidence summary: grades, open items, unfitted values
nsq sources                registered data sources and their status
nsq archive <source> …     fetch into the point-in-time store
nsq jobs …                 pausable, resumable back-fills (new | run | pause | resume | status)
nsq figures                render README figures into docs/assets/
nsq fan-chart [--dry-run]  weekly end-of-winter storage forecast into data/logs/fan_chart/
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import Counter

from nsqueeze.config import REPO_ROOT


def _params(_: argparse.Namespace) -> int:
    from nsqueeze.params import ParameterSet
    from nsqueeze.provenance import Grade

    P = ParameterSet.load()
    counts = Counter(p.grade for p in P.values())
    print(f"{len(P)} parameters · " + " · ".join(f"{g}: {counts[g]}" for g in Grade))
    print("\nGrade C (flag every output that uses these):")
    for p in P.by_grade(Grade.C):
        status = "UNFITTED" if p.is_fitted and p.value is None else str(p.value)
        print(f"  ⚑ {p.key:<42} {status:<10} {p.basis or p.status or ''}")
    print("\nOpen items referenced by parameters:")
    for oi, keys in P.open_items().items():
        print(f"  {oi}: {', '.join(keys)}")
    return 0


def _sources(_: argparse.Namespace) -> int:
    from nsqueeze.io.sources import REGISTRY
    from nsqueeze.io.sources.base import NotYetImplemented

    for name, cls in sorted(REGISTRY.items()):
        impl = "stub" if issubclass(cls, NotYetImplemented) else "ready"
        ver = "verified" if cls.verified else "unverified"
        key = cls.env_key or "—"
        print(f"  {name:<22} {impl:<6} {ver:<11} key: {key}")
    return 0


def _archive(a: argparse.Namespace) -> int:
    from nsqueeze.io import PITStore
    from nsqueeze.io.sources import get

    end = dt.date.today()
    start = dt.date.fromisoformat(a.since) if a.since else end - dt.timedelta(days=a.days)
    n = get(a.source).archive(PITStore(), start, end)
    print(f"{a.source}: archived {n} rows ({start} → {end})")
    return 0


def _jobs(a: argparse.Namespace) -> int:
    from nsqueeze.jobs import Job, all_jobs

    if a.action == "status":
        jobs = [Job.load(a.name)] if a.name else all_jobs()
        for j in jobs:
            err = f"  error: {j.last_error}" if j.last_error else ""
            print(
                f"  {j.name:<24} {j.source:<12} {j.status:<8} {j.progress:<22} "
                f"{j.rows:>10} rows  updated {j.updated_at}{err}"
            )
        if not jobs:
            print("  no jobs")
        return 0
    if not a.name:
        raise SystemExit(f"nsq jobs {a.action}: give a job name")
    if a.action == "new":
        if not (a.source and a.since):
            raise SystemExit("nsq jobs new NAME --source SOURCE --since YYYY-MM-DD [--until …]")
        until = dt.date.fromisoformat(a.until) if a.until else dt.date.today()
        j = Job.new(a.name, a.source, dt.date.fromisoformat(a.since), until)
        print(f"created {j.name}: {len(j.chunks)} chunks. Start it with `nsq jobs run {j.name}`")
        return 0
    j = Job.load(a.name)
    if a.action == "pause":
        j.pause_flag.touch()
        print(f"{j.name}: will pause after the current chunk")
        return 0
    if a.action == "resume":
        j.pause_flag.unlink(missing_ok=True)
    status = j.run()
    return 0 if status in ("done", "paused") else 1


def _figures(a: argparse.Namespace) -> int:
    """README figures, from the local store and the simulation."""
    import matplotlib

    matplotlib.use("Agg")
    from nsqueeze import case_nitrogen as cn
    from nsqueeze import simulation as sim
    from nsqueeze.params import ParameterSet
    from nsqueeze.viz import paper

    P = ParameterSet.load()
    panel = cn.monthly_panel()
    band = cn.parity_band(panel, P)
    cost = {t: cn.urea_cost(panel, cn.CostInputs.from_params(P, t)) for t in cn.TIERS}
    onsets = cn.episode_onsets().groupby("cluster").month.min()
    out = REPO_ROOT / "docs" / "assets"
    paper.cost_parity(panel, band, cost, onsets).savefig(out / "fig-cost-parity.png", dpi=200)
    s_ = sim.run_all()
    paper.monte_carlo(
        s_["nulls_under_persistence"],
        s_["specification_search_null"],
        s_["specification_search_power"],
        s_["few_independent_events"],
    ).savefig(out / "fig-monte-carlo.png", dpi=200)
    print("wrote docs/assets/fig-cost-parity.png, docs/assets/fig-monte-carlo.png")
    return 0


LABEL = "Model failed back-cast calibration (G2: MAE 7.03 pp, limit 5 pp)."


def _fan_chart(a: argparse.Namespace) -> int:
    import json
    import subprocess

    from nsqueeze import gas_balance as gb
    from nsqueeze.freeze import require_frozen
    from nsqueeze.io import PITStore
    from nsqueeze.params import ParameterSet

    frozen = require_frozen()
    P = ParameterSet.load()
    first, until = (dt.date.fromisoformat(d) for d in P.value("fan_chart.log_window"))
    now = dt.datetime.now(dt.UTC)
    if not a.dry_run and not first <= now.date() < until:
        raise SystemExit(f"nsq fan-chart: outside the logging window {first} to {until}")
    store = PITStore()
    # The model is fitted once: on data to 2025 as known on the freeze date, every week.
    fit_as_of = f"{frozen.frozen_at}T23:59:59"
    frame = gb.layer1_inputs(store, as_of=fit_as_of)
    frame = frame[frame.index < "2026-01-01"].copy()
    frame["dS"] = frame.storage.diff()
    model = gb.BalanceModel().fit(frame)
    hdd = gb.daily_hdd(
        store.read("weather", as_of=fit_as_of), store.read("eurostat_population", as_of=fit_as_of)
    )
    st = gb.eu_storage(store.read("agsi")).dropna()
    start = st.index[-1].date()
    thresholds = P.value("fan_chart.storage_thresholds_pct")
    supply = P.value("fan_chart.supply_scenarios_twh_d")
    draws = P.value("fan_chart.draws_per_weather_year")
    years = list(range(2000, 2026))
    p = gb.fan_chart(
        model,
        float(st.storage.iloc[-1]),
        start,
        until,
        hdd,
        supply,
        weather_years=years,
        draws=draws,
        wgv_twh=float(st.wgv.iloc[-1]),
        thresholds_pct=thresholds,
        seed=0,
    )

    def git(*c: str) -> str:
        return subprocess.run(
            ["git", *c], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip()

    entry = {
        "issued_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": git("rev-parse", "HEAD"),
        "tracked_changes": bool(git("status", "--porcelain", "--untracked-files=no")),
        "label": LABEL,
        "model": f"G2 (trial 1), fitted on data to 2025-12-31 as known on {frozen.frozen_at}",
        "start": {
            "gas_day": str(start),
            "storage_twh": float(st.storage.iloc[-1]),
            "wgv_twh": float(st.wgv.iloc[-1]),
        },
        "until": str(until),
        "thresholds_pct": thresholds,
        "supply_twh_d": supply,
        "weather_years": [years[0], years[-1]],
        "draws_per_weather_year": draws,
        "seed": 0,
        "p_below": p,
    }
    text = json.dumps(entry, indent=2)
    if a.dry_run:
        print(text)
        return 0
    out = REPO_ROOT / "data" / "logs" / "fan_chart" / f"{now.date()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:  # never overwrite an entry
        fh.write(text + "\n")
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252, which cannot print ⚑ ≈ · → used in reports.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="nsq", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("params", help="evidence summary").set_defaults(fn=_params)
    sub.add_parser("sources", help="list data sources").set_defaults(fn=_sources)
    arc = sub.add_parser("archive", help="fetch a source into the PIT store")
    arc.add_argument("source")
    arc.add_argument("--days", type=int, default=7, help="look-back window (default 7)")
    arc.add_argument("--since", help="back-fill from this date (YYYY-MM-DD); overrides --days")
    arc.set_defaults(fn=_archive)
    jb = sub.add_parser("jobs", help="pausable, resumable back-fills")
    jb.add_argument("action", choices=["new", "run", "pause", "resume", "status"])
    jb.add_argument("name", nargs="?")
    jb.add_argument("--source")
    jb.add_argument("--since")
    jb.add_argument("--until")
    jb.set_defaults(fn=_jobs)
    sub.add_parser("figures", help="render docs/assets figures").set_defaults(fn=_figures)
    fc = sub.add_parser("fan-chart", help="weekly end-of-winter storage forecast")
    fc.add_argument("--dry-run", action="store_true", help="print the entry, write nothing")
    fc.set_defaults(fn=_fan_chart)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
