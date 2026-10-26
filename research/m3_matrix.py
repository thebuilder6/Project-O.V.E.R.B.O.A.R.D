"""Measure the M3 constraint arms on the shipping planner's own candidate set.

`research/cli_matrix.py` drives the CLI, which cannot switch the new
optimizer options on (the CLI flags belong to another session), and
`research/harness.py` drives the retired stock spec sets. This drives the
library exactly the way `main.py`'s candidate loop does - the planner's
mesh, the planner's three frontier candidates, the production
`_record_candidate` / `_score_attempt` gate chain - with the Q11 arm as a
switch:

* `--accel` / `--no-accel`: Q11 step 2's wheel-acceleration bound. The
  shipping config declares the 7.8 m/s^2 limit, so `--accel` is the tree as
  it now ships and `--no-accel` clears the declared limit to reproduce the
  pre-change planner.

Q15's `disc_constraints` arm was measured with this driver and **retired**
after the measurement (zero verdict flips on the six examples and the
32-mission suite; see docs/RESEARCH_AGENDA.md Q15). The records stand in
`research/results/m3_*_disc_accel.jsonl`; the constraint rows were removed
from the optimizer with the verdict, so `--disc` is gone with them.

One JSON record per mission, written as the run finishes. Every candidate
carries converged/duration/iterations/wall/infeasible and the gate verdict.

Usage::

    python research/m3_matrix.py examples/example_straight.json examples/example_s_curve.json \
        --out research/results/m3_examples_accel.jsonl --accel
    python research/m3_matrix.py research/missions --out research/results/m3_suite_accel.jsonl \
        --accel --limit 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import overboard.main as cli  # noqa: E402  (needs ROOT on sys.path)
from overboard.core.optimizer import TrajectoryOptimizer  # noqa: E402
from overboard.core.unified_planner import UnifiedPathEngine  # noqa: E402
from overboard.models.robot_model import RobotConfig  # noqa: E402

DEFAULT_CONFIG = ROOT / "examples" / "fll_choreo.json"


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


def run_mission(mission: Path, args, robot_cfg, digital_twin) -> dict:
    wps, events, thresholds, stops = load_mission(mission)
    planner = UnifiedPathEngine(robot_cfg)
    mesh = planner.compute_invariant_mesh_samples(wps)
    specs = planner.candidate_specs(count=args.candidates)

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
    )

    started = time.time()
    candidates = []
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
                verbose=False,
            )
        except Exception as exc:  # a solve that raises is a failed candidate
            samples, stats = [], {"converged": False, "error": str(exc)}
        wall = time.time() - t0

        peak = peak_wheel_accel(samples) if samples else 0.0
        attempt = cli._record_candidate(spec, (samples, stats), idx,
                                        len(specs), [], quiet=True)
        cli._score_attempt(attempt, score_ctx)
        candidates.append({
            "index": idx + 1,
            "style": spec["guess_style"],
            "converged": bool(stats.get("converged", False)),
            # A solve that did not converge is the answer for Q15, not a
            # failure of the matrix: counted, never dropped.
            "infeasible": not bool(stats.get("converged", False)),
            "duration_s": float(samples[-1]["t"]) if samples else None,
            "iterations": int(stats.get("iter_count") or 0),
            "wall_s": round(wall, 2),
            "peak_accel_m_s2": round(peak, 3),
            "over_ramp_limit": bool(peak > 7.8 + 1e-6),
            "ok": bool(attempt.get("ok", False)),
            "envelope_cm": round(float(attempt.get("tube", 0.0)) * 100.0, 3),
            "tracking_cm": round(float(attempt.get("tracking_m", 0.0)) * 100.0, 3),
            "reasons": list(attempt.get("reasons", [])),
            "error": stats.get("error"),
        })

    passing = [c for c in candidates if c["ok"]]
    selected = min(passing, key=lambda c: c["duration_s"], default=None)
    return {
        "mission": mission.stem,
        "mission_path": str(mission),
        "accel_bound": bool(args.accel),
        "propagation": args.propagation,
        "candidates": args.candidates,
        "mesh": mesh,
        "passed": bool(selected is not None),
        "selected": selected["style"] if selected else None,
        "duration_s": selected["duration_s"] if selected else None,
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
          f"accel={args.accel} prop={args.propagation}",
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
                  f"infeasible={record['infeasible']} over_ramp={record['over_ramp']} "
                  f"wall={record['wall_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
