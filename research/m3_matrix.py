"""Measure M3/Q17 arms on the shipping planner's own candidate set.

`research/cli_matrix.py` drives the CLI, which cannot switch the new
optimizer options on (the CLI flags belong to another session), and
`research/harness.py` drives the retired stock spec sets. This drives the
library exactly the way `main.py`'s candidate loop does - the planner's
mesh, the planner's frontier candidates, the production
`_record_candidate` / `_score_attempt` gate chain.

Q17 (docs/RESEARCH_AGENDA.md 24) runs this driver twice on the same 32
missions: `--candidates 3` (the baseline arm, the triad) against
`--candidates 6` (the expanded arm, the triad plus the Q17 tail). The
measurement scoring rules:

* Research missions have no scoring budgets. Per mission T_ref is the
  fastest CONVERGED duration in the baseline arm (the first three
  candidates), falling back to the fastest converged candidate in the run;
  when nothing converges the mission scores None and counts as a miss in
  both arms. `target_time_s = T_ref`, `max_time_s = 2 * T_ref`, jerk
  defaults 10/20.
* The mission record reports BOTH selections: `selected` is the production
  score-based pick (what ships) and `selected_fastest` the fastest passing
  pick (what Q13-Q15 measured). `duration_s` is the production pick's.
* The collapse diagnostic is the mean per-sample Euclidean position
  distance over every (new candidate x triad) pair, mesh-aligned
  (`pairwise_deviation_cm`); above 2 cm the new candidates are distinct.
* A candidate that fails to converge is counted as infeasible, never
  dropped (Q15 discipline). Its score fields stay None.

One JSON record per mission, written as the run finishes. Every candidate
carries converged/duration/iterations/wall/infeasible and the gate verdict,
plus score/categories/eligibility.

Usage::

    python research/m3_matrix.py research/missions --out research/results/q17_base.jsonl \
        --candidates 3 --propagation plant
    python research/m3_matrix.py research/missions --out research/results/q17_expanded.jsonl \
        --candidates 6 --propagation plant
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import overboard.main as cli  # noqa: E402  (needs ROOT on sys.path)
from overboard.core.optimizer import TrajectoryOptimizer  # noqa: E402
from overboard.core.unified_planner import UnifiedPathEngine  # noqa: E402
from overboard.models.robot_model import RobotConfig  # noqa: E402

DEFAULT_CONFIG = ROOT / "examples" / "fll_choreo.json"

# The baseline arm of the candidate list is the frontier triad.
BASELINE_CANDIDATES = 3


def load_mission(path: Path):
    wps, events, thresholds, json_stops = cli._load_waypoints(str(path))
    stops = sorted(set(json_stops))
    return wps, events, thresholds, stops


def peak_wheel_accel(samples):
    peak = 0.0
    for k in range(len(samples) - 1):
        dt = float(samples[k + 1]["t"]) - float(samples[k]["t"])
        if dt <= 0:
            continue
        peak = max(peak,
                   abs(float(samples[k + 1]["vl"]) - float(samples[k]["vl"])) / dt,
                   abs(float(samples[k + 1]["vr"]) - float(samples[k]["vr"])) / dt)
    return peak


def _num(value, digits=3):
    """JSON-safe number: Infinity and NaN become None (Infinity breaks JSON)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits) if digits is not None else number


def derive_budget(solved):
    """The mission's measured scoring budget (the M1 rule).

    T_ref = fastest converged duration in the baseline arm (the triad);
    fallback to the fastest converged candidate in either arm of the run;
    when nothing converges, (None, "none") and the mission is a miss.

    Returns (T_ref, source).
    """
    def duration(entry):
        _idx, _spec, samples, stats, _wall = entry
        if samples and stats.get("converged"):
            return _num(samples[-1]["t"], digits=None)
        return None

    baseline = [d for d in (duration(e) for e in solved[:BASELINE_CANDIDATES])
                if d is not None]
    if baseline:
        return min(baseline), "baseline"
    any_arm = [d for d in (duration(e) for e in solved) if d is not None]
    if any_arm:
        return min(any_arm), "fallback"
    return None, "none"


def pairwise_deviation(solved):
    """The collapse diagnostic: mean over (new x triad) pairs of the mean
    per-sample Euclidean position distance (the candidates share one mesh,
    so samples align by index).

    Returns (mean_distance_m, pair_count).
    """
    def positions(entry):
        _idx, _spec, samples, _stats, _wall = entry
        if not samples:
            return None
        return np.array([[float(s["x"]), float(s["y"])] for s in samples],
                        dtype=float)

    triad = [positions(e) for e in solved[:BASELINE_CANDIDATES]]
    fresh = [positions(e) for e in solved[BASELINE_CANDIDATES:]]
    distances = []
    for new in fresh:
        if new is None:
            continue
        for base in triad:
            if base is None or len(base) != len(new):
                continue
            distances.append(float(np.mean(np.hypot(
                new[:, 0] - base[:, 0], new[:, 1] - base[:, 1]))))
    if not distances:
        return None, 0
    return _num(sum(distances) / len(distances), digits=4), len(distances)


def run_mission(mission: Path, args, robot_cfg, digital_twin) -> dict:
    wps, events, thresholds, stops = load_mission(mission)
    planner = UnifiedPathEngine(robot_cfg)
    mesh = planner.compute_invariant_mesh_samples(wps)
    specs = planner.candidate_specs(count=args.candidates)

    # Solve every candidate first: the scoring budget is measured, not
    # guessed, so it needs the arm's own converged durations.
    started = time.time()
    solved = []
    for idx, spec in enumerate(specs):
        guess = planner.generate_seed(wps, mesh, spec["guess_style"], stops)
        t0 = time.time()
        try:
            samples, stats = TrajectoryOptimizer(robot_cfg).solve(
                wps,
                num_samples_per_segment=mesh,
                accuracy_weight=spec["accuracy_weight"],
                stop_waypoint_indices=stops,
                waypoint_events=events,
                initial_guess=guess,
                per_segment_dt=True,
                cost_metric="discretization",
                epigraph_constraints=True,
                anchor_weight=args.anchor_weight,
                verbose=False,
            )
        except Exception as exc:  # a solve that raises is a failed candidate
            samples, stats = [], {"converged": False, "error": str(exc)}
        solved.append((idx, spec, samples, stats, time.time() - t0))

    t_ref, budget_source = derive_budget(solved)
    score_ctx = cli._ScoreContext(
        robot_cfg=robot_cfg,
        samples=mesh,
        stop_indices=stops,
        waypoint_thresholds=thresholds,
        validate=True,
        use_validation=True,
        digital_twin=digital_twin,
        propagation=args.propagation,
        control_dt=0.02,
        quiet=True,
        target_time_s=t_ref,
        max_time_s=(2.0 * t_ref) if t_ref is not None else None,
    )

    attempts = []
    candidates = []
    for idx, spec, samples, stats, wall in solved:
        peak = peak_wheel_accel(samples) if samples else 0.0
        attempt = cli._record_candidate(spec, (samples, stats), idx,
                                        len(specs), [], quiet=True)
        cli._score_attempt(attempt, score_ctx)
        attempts.append(attempt)
        candidates.append({
            "index": idx + 1,
            "style": spec["guess_style"],
            "converged": bool(stats.get("converged", False)),
            # A solve that did not converge is the answer for Q15, not a
            # failure of the matrix: counted, never dropped.
            "infeasible": not bool(stats.get("converged", False)),
            "duration_s": _num(samples[-1]["t"]) if samples else None,
            "iterations": int(stats.get("iter_count") or 0),
            "wall_s": round(wall, 2),
            "peak_accel_m_s2": round(peak, 3),
            "over_ramp_limit": bool(peak > 7.8 + 1e-6),
            "ok": bool(attempt.get("ok", False)),
            "envelope_cm": _num(float(attempt.get("tube", 0.0)) * 100.0),
            "tracking_cm": _num(float(attempt.get("tracking_m", 0.0)) * 100.0),
            "reasons": list(attempt.get("reasons", [])),
            "score": _num(attempt.get("score"), digits=2),
            "categories": (None if attempt.get("categories") is None
                           else {name: _num(value, digits=2)
                                 for name, value in attempt["categories"].items()}),
            "eligibility": bool(cli._attempt_eligible(attempt, None)),
            "error": stats.get("error"),
        })

    # Both selections: the production score-based pick (what ships) and the
    # fastest passing pick (the Q13-Q15 rule). The success criterion reads
    # the production duration.
    production, production_ok = cli._select_attempt_scored(attempts)
    passing = [a for a in attempts if a.get("ok")]
    fastest = min(passing, key=lambda a: a["duration"], default=None)
    deviation_cm, deviation_pairs = pairwise_deviation(solved)

    return {
        "mission": mission.stem,
        "mission_path": str(mission),
        "accel_bound": bool(args.accel),
        "propagation": args.propagation,
        "anchor_weight": float(args.anchor_weight),
        "candidates": args.candidates,
        "mesh": mesh,
        "target_time_s": _num(t_ref, digits=None),
        "max_time_s": _num(2.0 * t_ref, digits=None) if t_ref is not None else None,
        "budget_source": budget_source,
        "passed": bool(passing),
        "selected": production["guess_style"] if (production_ok and production) else None,
        "duration_s": _num(production["duration"]) if (production_ok and production) else None,
        "selected_fastest": fastest["guess_style"] if fastest else None,
        "duration_fastest_s": _num(fastest["duration"]) if fastest else None,
        "pairwise_deviation_cm": deviation_cm,
        "pairwise_pairs": deviation_pairs,
        "candidates_total": len(candidates),
        "candidates_passing": len(passing),
        "infeasible": sum(1 for c in candidates if c["infeasible"]),
        "over_ramp": sum(1 for c in candidates if c["over_ramp_limit"]),
        "wall_s": round(time.time() - started, 1),
        "results": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("missions", nargs="+", help="Mission files or a directory of them.")
    parser.add_argument("--out", required=True, help="Result JSONL (resumed, not replaced).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--propagation", choices=("plant", "tick"), default="plant")
    parser.add_argument("--anchor-weight", type=float, default=0.0,
                        help="Q17 stage 2: w_anchor of the geometric anchor "
                             "term, 0 (default) is off.")
    parser.add_argument("--accel", dest="accel", action="store_true", default=True,
                        help="Q11: keep the declared wheel-acceleration bound (default).")
    parser.add_argument("--no-accel", dest="accel", action="store_false",
                        help="Clear the declared limit, reproducing the pre-change planner.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    missions = []
    for entry in args.missions:
        path = Path(entry)
        if path.is_dir():
            # The suite writes an index.json alongside the missions; a file
            # that is not a waypoint list is not a mission.
            for file in sorted(path.glob("*.json")):
                if isinstance(json.loads(file.read_text(encoding="utf-8")), list):
                    missions.append(file)
        else:
            missions.append(path)
    if args.limit:
        missions = missions[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(line)["mission"]
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()}
    todo = [m for m in missions if m.stem not in done]
    print(f"{len(missions)} missions, {len(todo)} to run | "
          f"accel={args.accel} prop={args.propagation} candidates={args.candidates}",
          flush=True)

    config_data = json.loads(Path(args.config).read_text(encoding="utf-8"))
    with out_path.open("a", encoding="utf-8") as handle:
        for number, mission in enumerate(todo, 1):
            robot_cfg = RobotConfig(config_data)
            if not args.accel:
                robot_cfg.accel_limit_m_s2 = None
            digital_twin = cli._config_digital_twin(config_data)
            record = run_mission(mission, args, robot_cfg, digital_twin)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(f"[{number}/{len(todo)}] {mission.stem} passed={record['passed']} "
                  f"selected={record['selected']} dur={record['duration_s']} "
                  f"fastest={record['selected_fastest']} "
                  f"dev={record['pairwise_deviation_cm']}cm "
                  f"infeasible={record['infeasible']} over_ramp={record['over_ramp']} "
                  f"wall={record['wall_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
