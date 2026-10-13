"""``nsq`` command line interface.

nsq params                 evidence summary: grades, open items, unfitted values
nsq sources                registered data sources and their status
nsq archive <source> …     fetch into the point-in-time store
nsq jobs …                 pausable, resumable back-fills (new | run | pause | resume | status)
nsq figures                render README figures into docs/assets/
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
    paper.monte_carlo(s_["e2"], s_["e3_null"], s_["e3_power"], s_["e4"]).savefig(
        out / "fig-monte-carlo.png", dpi=200
    )
    print("wrote docs/assets/fig-cost-parity.png, docs/assets/fig-monte-carlo.png")
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
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
