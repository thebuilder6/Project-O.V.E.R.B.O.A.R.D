"""Risk against speed: how much margin does each candidate carry? (Q14)

The gate says pass or fail, so a plan with 0.1 mm to spare and one with 8 mm
to spare look the same, and the selection takes the fastest passing one. This
script measures the other half of that trade: for every candidate of a
mission, the margin to each limit, and the duration it costs to keep it.

Two steps, because the second needs no solving::

    python research/risk_margins.py run --per-profile 8 --propagation plant \\
        --out research/results/q14_plant.jsonl
    python research/risk_margins.py analyse research/results/q14_plant.jsonl

``run`` shells the CLI once per mission with ``--candidates 3 --validate`` and
keeps its ``##OVERBOARD-VALIDATION##`` report, so nothing here re-implements a
gate. ``analyse`` turns those reports into margins and compares selection
rules: fastest passing (today), a margin floor, and a balanced pick.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
MARKER = "##OVERBOARD-VALIDATION## "
# The tracking gate's threshold, as main._immrax_failure_reasons applies it.
TRACKING_LIMIT_M = 0.005


def _pick_missions(root: Path, per_profile: int) -> List[Dict[str, Any]]:
    """The first N missions of every profile, so each kind of path is present."""
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    taken: Dict[str, int] = {}
    chosen = []
    for entry in index["missions"]:
        profile = entry.get("profile") or "mixed"
        if taken.get(profile, 0) >= per_profile:
            continue
        taken[profile] = taken.get(profile, 0) + 1
        chosen.append(entry)
    return chosen


def _cli_report(mission: Path, propagation: str) -> Dict[str, Any]:
    """The CLI's validation report, or what the run said instead of one."""
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, "-m", "overboard.main", "-w", str(mission),
             "--candidates", "3", "--validate",
             "--propagation", propagation,
             "-o", str(Path(tmp) / "out.traj")],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=3600)
    for line in result.stdout.splitlines():
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
    # A run that printed no report must not look like a mission with no
    # candidates: keep why it stopped.
    tail = (result.stderr or result.stdout or "").strip().splitlines()[-6:]
    return {"error": f"exit {result.returncode}", "output": tail}


def _is_report(report: Any) -> bool:
    return isinstance(report, dict) and "candidates" in report


def run(args: argparse.Namespace) -> None:
    missions_root = REPO_ROOT / args.missions
    chosen = _pick_missions(missions_root, args.per_profile)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # The tree is edited while a long run is going, so every row says which
    # commit produced it.
    tree = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          cwd=str(REPO_ROOT), capture_output=True, text=True
                          ).stdout.strip()
    start = time.time()
    # A run of this size outlives the tree it started on, so it resumes: a
    # mission that already has a report is not solved again.
    done = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if _is_report(row.get("report")):
                done.add(row.get("mission"))
    with out_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for number, entry in enumerate(chosen, 1):
            name = entry.get("id") or entry["file"]
            if name in done:
                print(f"[{number}/{len(chosen)}] {entry['file']} already done",
                      flush=True)
                continue
            mission = missions_root / entry["file"]
            report = _cli_report(mission, args.propagation)
            handle.write(json.dumps({
                "mission": name,
                "profile": entry.get("profile"),
                "propagation": args.propagation,
                "tree": tree,
                "report": report,
            }) + "\n")
            handle.flush()
            if _is_report(report):
                status = f"{len(report['candidates'])} candidate(s)"
            else:
                status = "NO REPORT: " + " / ".join(report.get("output") or [])
            print(f"[{number}/{len(chosen)}] {entry['file']} {status}  "
                  f"{time.time() - start:.0f}s", flush=True)
    print(f"wrote {out_path} in {time.time() - start:.0f}s")


def margins(candidate: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """The margins one candidate carries, each as a fraction of its own limit.

    Positive is inside the limit, negative is past it. The envelope margin is
    the worst segment, so a limit that tightens along the path is respected.
    A truncated envelope is a lower bound (KI-049), so its margin is an upper
    bound and the candidate is flagged by the caller.

    ``accuracy`` is the envelope and tracking pair alone: the wheel-force
    margin is not a risk dial, because a time-optimal plan saturates the motor
    by design, so it sits near 1.0 - force_margin in almost every candidate
    and would mask the margin that does vary.
    """
    segments = candidate.get("segments") or []
    relative = [(segment["envelope_m"] - segment["limit_m"])
                / max(segment["limit_m"], 1e-9) for segment in segments]
    envelope = -max(relative) if relative else None
    tracking_cm = candidate.get("tracking_cm")
    tracking = (None if tracking_cm is None
                else (TRACKING_LIMIT_M - tracking_cm / 100.0) / TRACKING_LIMIT_M)
    peaks = [value for value in (candidate.get("peak_force_util"),
                                 candidate.get("peak_traction_util"))
             if value is not None]
    force = 1.0 - max(peaks) if peaks else None
    accuracy = [value for value in (envelope, tracking) if value is not None]
    parts = accuracy + ([force] if force is not None else [])
    return {"envelope": envelope, "tracking": tracking, "force": force,
            "accuracy": min(accuracy) if accuracy else None,
            "min": min(parts) if parts else None}


def _rule_fastest(pool, floor=None, balanced=False):
    """The shortest passing candidate, optionally above an accuracy floor."""
    if floor is not None:
        pool = [row for row in pool if row["margin"]["accuracy"] >= floor]
        if not pool:
            return None
    if not balanced:
        return min(pool, key=lambda row: row["duration_s"])
    durations = [row["duration_s"] for row in pool]
    margin_norms = [row["margin"]["accuracy"] for row in pool]
    span_d = max(durations) - min(durations)
    span_m = max(margin_norms) - min(margin_norms)

    def score(row):
        duration = 0.0 if span_d <= 0 else (row["duration_s"] - min(durations)) / span_d
        margin = 0.0 if span_m <= 0 else (max(margin_norms) - row["margin"]["accuracy"]) / span_m
        return duration + margin

    return min(pool, key=score)


def _pearson(pairs) -> Optional[float]:
    if len(pairs) < 3:
        return None
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx < 1e-12 or sy < 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / (sx * sy)


def analyse(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in Path(args.path).read_text(encoding="utf-8").splitlines() if line.strip()]
    rules = {"fastest": {}, "floor 0.1": {"floor": 0.1}, "floor 0.2": {"floor": 0.2},
             "floor 0.3": {"floor": 0.3}, "balanced": {"balanced": True}}
    per_rule: Dict[str, List[Dict[str, Any]]] = {name: [] for name in rules}
    pooled: Dict[str, List[List[Any]]] = {"accuracy": [], "envelope": [],
                                          "tracking": [], "force": []}
    usable = 0
    truncated = 0
    for row in rows:
        report = row.get("report")
        if not _is_report(report):
            continue
        passing = []
        founds = []
        for candidate in report["candidates"]:
            if not candidate.get("ok"):
                continue
            found = margins(candidate)
            if found["accuracy"] is None:
                continue
            passing.append({"duration_s": candidate["duration_s"],
                            "margin": found,
                            "style": candidate.get("style"),
                            "truncated": bool(candidate.get("envelope_truncated"))})
            founds.append((candidate["duration_s"], found))
            truncated += int(bool(candidate.get("envelope_truncated")))
        for key in pooled:
            pooled[key].append([(duration, found[key]) for duration, found in founds
                                if found[key] is not None])
        if not passing:
            continue
        usable += 1
        fastest = _rule_fastest(passing)
        for name, spec in rules.items():
            pick = _rule_fastest(passing, **spec)
            if pick is None:
                continue
            per_rule[name].append({
                "mission": row["mission"], "profile": row.get("profile"),
                "duration_s": pick["duration_s"],
                "delta_s": pick["duration_s"] - fastest["duration_s"],
                "margin": pick["margin"]["accuracy"],
                "fastest_margin": fastest["margin"]["accuracy"],
                "changed": pick is not fastest,
            })

    def _centred(groups):
        """Durations against margins inside each mission, then pooled.

        Pooling raw numbers across missions would measure the missions, not
        the trade: two missions with different limits and different lengths
        differ in both axes for reasons of their own.
        """
        pairs = []
        for group in groups:
            if len(group) < 2:
                continue
            mean_d = statistics.fmean(d for d, _ in group)
            mean_m = statistics.fmean(m for _, m in group)
            pairs.extend((d - mean_d, m - mean_m) for d, m in group)
        return pairs

    print(f"{len(rows)} mission(s) measured, {usable} with a passing candidate")
    print(f"{truncated} passing candidate(s) had a truncated envelope "
          "(lower bound, KI-049)")
    for key, groups in pooled.items():
        pairs = _centred(groups)
        r = _pearson(pairs)
        if pairs:
            print(f"  duration vs {key:<9} margin, within mission, "
                  f"{len(pairs):>3} candidate(s): "
                  f"r = {'n/a' if r is None else f'{r:+.2f}'}")
    print()
    print(f"{'rule':<12}{'picks':>6}{'changed':>9}{'mean ds':>9}{'max ds':>8}"
          f"{'mean margin':>13}")
    for name, picks in per_rule.items():
        if not picks:
            print(f"{name:<12}{'0':>6}")
            continue
        deltas = [pick["delta_s"] for pick in picks]
        print(f"{name:<12}{len(picks):>6}"
              f"{sum(1 for p in picks if p['changed']):>9}"
              f"{statistics.fmean(deltas):>+9.3f}{max(deltas):>+8.3f}"
              f"{statistics.fmean(p['margin'] for p in picks):>13.3f}")
    if args.verbose:
        print()
        for name, picks in per_rule.items():
            for pick in picks:
                if pick["changed"]:
                    print(f"  {name:<10} {pick['mission']:<6} "
                          f"{pick['profile']:<7} +{pick['delta_s']:.3f}s "
                          f"margin {pick['fastest_margin']:+.2f} -> "
                          f"{pick['margin']:+.2f}")

    # The other half of the risk question: the plans that fail, and how far
    # past the limit they run. A needed-limit factor of 2 means the local
    # threshold would have to double for that plan to fit.
    overshoot = []
    tracking_failures = 0
    for row in rows:
        report = row.get("report")
        if not _is_report(report):
            continue
        for candidate in report["candidates"]:
            if candidate.get("ok"):
                continue
            segments = candidate.get("segments") or []
            if segments:
                overshoot.append(max(segment["envelope_m"] / max(segment["limit_m"], 1e-9)
                                     for segment in segments))
            if (candidate.get("tracking_cm") or 0.0) / 100.0 > TRACKING_LIMIT_M:
                tracking_failures += 1
    if overshoot:
        overshoot.sort()
        def pct(share):
            return overshoot[min(len(overshoot) - 1, int(share * len(overshoot)))]
        print()
        print(f"failing candidates: {len(overshoot)} with a segment report; "
              f"the limit would have to grow by")
        print(f"  x{pct(0.5):.1f} (median)  x{pct(0.9):.1f} (p90)  "
              f"x{pct(1.0):.1f} (worst) to hold the plan")
        print(f"  {tracking_failures} candidate(s) also exceeded the "
              f"{TRACKING_LIMIT_M * 1000:.0f} mm tracking gate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run", help="Solve the missions through the CLI.")
    runner.add_argument("--missions", default="research/missions")
    runner.add_argument("--per-profile", type=int, default=8)
    runner.add_argument("--propagation", default="plant", choices=("plant", "tick"))
    runner.add_argument("--out", required=True)
    runner.add_argument("--resume", action="store_true",
                        help="Skip the missions the output file already has.")
    runner.set_defaults(func=run)
    analysis = sub.add_parser("analyse", help="Compare the selection rules.")
    analysis.add_argument("path")
    analysis.add_argument("--verbose", action="store_true")
    analysis.set_defaults(func=analyse)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
