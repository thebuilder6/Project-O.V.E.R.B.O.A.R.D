"""Measurement harness for the path-planning research agenda.

Runs a (mission x candidate) matrix, scores every candidate with the
production gate chain, and writes one JSON record per candidate.

The scoring below mirrors the candidate loop in ``main.py`` (the
context-aware envelope check, the Immrax check and the CPU audit) and calls
the production helpers for every gate, so a record says the same thing the
CLI would say. Only the bookkeeping is new: this harness adds no planning
logic and changes no production module.

Usage::

    python research/harness.py --missions research/missions --out research/results/matrix.jsonl \\
        --specs matrix --samples 10 --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The measured system. OVERBOARD_TREE points at a pinned copy of the planner,
# the same convention docs/research/rq_a uses, so a measurement is tied to a
# tree state even while the working tree is being edited.
TOOL_ROOT = Path(__file__).resolve().parent.parent
MEASURE_ROOT = Path(os.environ.get("OVERBOARD_TREE") or TOOL_ROOT)
for _root in (TOOL_ROOT, MEASURE_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import numpy as np

import main as cli
from optimizer import GUESS_STYLES, TrajectoryOptimizer
from robot_model import RobotConfig
from validator import check_final_waypoint, compute_metrics, validate_samples

from research import robots as robot_lib
from research import seeds as seed_lib

# The weight grid the CLI sweeps with (main.SWEEP_WEIGHTS).
CLI_SWEEP_WEIGHTS = (0.0, 0.001, 0.002, 0.005, 0.01)
# A wider grid for the Q6 study: is the weight a dial or a lottery?
WIDE_WEIGHTS = (0.0, 0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)

# The files whose content decides what a measurement means.
PROVENANCE_FILES = ("optimizer.py", "main.py", "immrax_validator.py",
                     "validator.py", "robot_model.py", "multiverse_optimizer.py")


def _hash_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest().upper()


def provenance() -> Dict[str, Any]:
    """Hashes of the files that decide what a measurement means."""
    hashes = {}
    for name in PROVENANCE_FILES:
        path = MEASURE_ROOT / name
        if path.exists():
            hashes[name] = _hash_file(path)
    return hashes


def tree_state() -> str:
    """One short token for the current tree state.

    The tree is edited while a measurement runs, so every record carries the
    state it was produced under. A record whose token differs from the
    others was produced by a different solver.
    """
    digest = hashlib.md5()
    for name in PROVENANCE_FILES:
        path = MEASURE_ROOT / name
        if path.exists():
            digest.update(_hash_file(path).encode("ascii"))
    return digest.hexdigest().upper()[:12]


def _resolve(path: str) -> str:
    """Resolve a path against the measured tree, not the working directory."""
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)
    return str(MEASURE_ROOT / path)


# --------------------------------------------------------------------------
# Specs: what to solve
# --------------------------------------------------------------------------

def _spec(label: str, style: str, timing: str, weight: float, samples: int,
          guess: str = "stock", engine: str = "simple") -> Dict[str, Any]:
    return {
        "label": label,
        "style": style,
        "timing": timing,
        "weight": float(weight),
        "samples": int(samples),
        "guess": guess,
        "engine": engine,
    }


def build_specs(name: str, base_weight: float, samples: int) -> List[Dict[str, Any]]:
    """The candidate sets the harness can run."""
    timings = ("equal-duration", "per-segment")
    specs: List[Dict[str, Any]] = []
    seen = set()

    def add(spec: Dict[str, Any]) -> None:
        key = (spec["style"], spec["timing"], spec["weight"], spec["samples"],
               spec["guess"])
        if key in seen:
            return
        seen.add(key)
        specs.append(spec)

    if name in ("default", "matrix"):
        # The CLI's own default pair: the baseline seed on both timing
        # models. Two candidates, so a stress run over many missions and
        # many robots stays affordable.
        for timing in timings:
            add(_spec(f"baseline/{timing}/a={base_weight:g}", "baseline", timing,
                      base_weight, samples))

    if name in ("seeds", "matrix"):
        for style in GUESS_STYLES:
            for timing in timings:
                add(_spec(f"{style}/{timing}/a={base_weight:g}", style, timing,
                          base_weight, samples))

    if name in ("weights", "matrix"):
        for weight in CLI_SWEEP_WEIGHTS:
            for timing in timings:
                add(_spec(f"baseline/{timing}/a={weight:g}", "baseline", timing,
                          weight, samples))

    if name == "wide-weights":
        for weight in WIDE_WEIGHTS:
            for timing in timings:
                add(_spec(f"baseline/{timing}/a={weight:g}", "baseline", timing,
                          weight, samples))

    if name in ("spline", "matrix"):
        # Q1: a real path family against the straight-line seed.
        for timing in timings:
            add(_spec(f"spline/{timing}/a={base_weight:g}", "baseline", timing,
                      base_weight, samples, guess="spline"))

    if name in ("reeds", "matrix"):
        # Q12: the Multi-Verse bootstrap, with its rsplan API fixed.
        for timing in timings:
            add(_spec(f"reeds-shepp/{timing}/a={base_weight:g}", "baseline", timing,
                      base_weight, samples, guess="reeds_shepp"))

    if name in ("refined", "matrix"):
        # Q12: the repaired Multi-Verse pipeline as a seed provider. The
        # refined trajectory carries a per-segment interval, so it only
        # makes sense under the per-segment timing model.
        add(_spec(f"refined/per-segment/a={base_weight:g}", "baseline",
                  "per-segment", base_weight, samples, guess="refined"))

    if name == "refined-weights":
        # The refined seed against the accuracy-weight grid: the weight is
        # what buys the tracking error (Q6), so the seed has to be judged
        # with it.
        for weight in CLI_SWEEP_WEIGHTS:
            add(_spec(f"refined/per-segment/a={weight:g}", "baseline",
                      "per-segment", weight, samples, guess="refined"))

    if name in ("proportional", "matrix"):
        # Q2: a length-proportional time allocation.
        add(_spec(f"proportional-dt/per-segment/a={base_weight:g}", "rolling",
                  "per-segment", base_weight, samples, guess="proportional_dt"))

    if name == "density":
        # Q7: sample density against timing flexibility.
        for density in (5, 10, 20, 40):
            for timing in timings:
                add(_spec(f"baseline/{timing}/a={base_weight:g}/n={density}",
                          "baseline", timing, base_weight, density))

    if name == "multiverse":
        # Q12: the experimental pipeline against the simple solver.
        for weight in (base_weight, base_weight + 2.0):
            add(_spec(f"multiverse/equal-duration/a={weight:g}", "baseline",
                      "equal-duration", weight, samples, engine="multiverse"))

    if not specs:
        raise SystemExit(f"Unknown spec set {name!r}")
    return specs


# --------------------------------------------------------------------------
# Scoring: the production gates, in the production order
# --------------------------------------------------------------------------

def _scaled_range(values: Sequence[float], spread: float) -> Tuple[float, float]:
    """Widen or narrow a declared uncertainty range around its nominal value.

    ``spread = 1`` is the range as declared, ``0`` collapses it to the
    nominal value, ``2`` doubles its width. The declared ranges are test
    assumptions (KI-020), so this measures how much of a PASS depends on
    them.
    """
    low, high = float(values[0]), float(values[1])
    nominal = 0.5 * (low + high)
    half = 0.5 * (high - low) * float(spread)
    return (max(1e-9, nominal - half), nominal + half)


def _score(samples_data: List[Dict[str, Any]], stats: Dict[str, Any],
           robot_cfg, validator_cls, robot_config_cls,
           thresholds: Dict[int, float], samples_per_segment: int,
           waypoints: Sequence[Tuple[float, float, Optional[float]]],
           critic=None, profiles: bool = False, spread: float = 1.0,
           propagation: str = "plant") -> Dict[str, Any]:
    """Run the full gate chain on one candidate and return the measurements."""
    record: Dict[str, Any] = {
        "converged": bool(stats.get("converged", False)),
        "duration_s": float(samples_data[-1]["t"]) if samples_data else float("inf"),
        "cost": float(stats.get("final_cost", float("inf"))),
        "solve_s": float(stats.get("total_time", float("nan"))),
        "iter_count": int(stats.get("iter_count", -1)),
        "segment_dt": [float(d) for d in (stats.get("segment_dt") or [])],
        # The model the envelope came from. The tube is a property of the
        # propagation, so a record without it cannot be compared with a
        # record from another propagation (KI-049).
        "propagation": propagation,
    }
    probe = stats.get("refine_probe")
    if isinstance(probe, dict):
        record["refine_probe"] = {k: (float(v) if isinstance(v, (int, float)) else v)
                                  for k, v in probe.items()}
    if stats.get("error"):
        # The solver caught an exception. Keep the message: a stress run
        # counts crashes, and "no samples" alone says nothing about why.
        record["error"] = str(stats["error"])

    if not samples_data:
        record.update({"ok": False, "reasons": ["no samples"],
                       "tube_m": float("inf"), "tracking_m": float("inf")})
        return record

    try:
        jax_cfg = robot_config_cls(robot_cfg)
        immrax_val = validator_cls(jax_cfg, propagation=propagation)
        uncertainties = robot_cfg.uncertainties
        report = immrax_val.validate_trajectory(
            samples_data,
            _scaled_range(uncertainties["cof"], spread),
            _scaled_range(uncertainties["torque_headroom"], spread),
            _scaled_range(uncertainties["gear_backlash_m"], spread),
            _scaled_range(uncertainties["track_width_range"], spread),
            _scaled_range(uncertainties["wheel_radius_range"], spread),
            _scaled_range(uncertainties["inertia_kg_m2"], spread),
        )
    except Exception as exc:  # a validator crash is not a pass (KI-002)
        record.update({"ok": False, "reasons": [f"Immrax error: {exc}"],
                       "tube_m": float("inf"), "tracking_m": float("inf")})
        return record

    for i, env in enumerate(report["reachability"]["envelope"]):
        if i < len(samples_data):
            samples_data[i]["reachability_envelope"] = env

    failed_threshold = False
    max_violation = 0.0
    max_tube = 0.0
    for i in range(len(samples_data)):
        limit = thresholds.get(i // samples_per_segment, 0.02)
        env = samples_data[i].get("reachability_envelope", {})
        if not env:
            continue
        err_x = (env["x_max"] - env["x_min"]) / 2.0
        err_y = (env["y_max"] - env["y_min"]) / 2.0
        radius = math.sqrt(err_x ** 2 + err_y ** 2)
        max_tube = max(max_tube, radius)
        if env.get("unknown") or radius > limit:
            failed_threshold = True
            max_violation = max(max_violation, radius)

    cpu_summary = validate_samples(samples_data, robot_cfg)
    waypoint_check = check_final_waypoint(samples_data, waypoints[-1])

    reasons = cli._immrax_failure_reasons(report, failed_threshold, max_violation)
    # The same rule main.py uses for a candidate: converged, inside the local
    # envelope threshold, physically safe by Immrax, and clean in the CPU
    # audit. The final-waypoint check is a separate gate in main.py (it runs
    # on the selected candidate), so it is reported here rather than folded
    # into the pass flag.
    ok = (record["converged"] and not failed_threshold and bool(report["passed"]))
    if not cpu_summary["passed"]:
        ok = False
        reasons = list(reasons) + cli._cpu_failure_reasons(cpu_summary, waypoint_check)
    if not waypoint_check["passed"]:
        reasons = list(reasons) + [
            f"final waypoint error {waypoint_check['pos_error_m'] * 100:.2f} cm",
        ]

    metrics = compute_metrics(samples_data, robot_cfg)
    # The heading change between consecutive samples. RQ-A (docs/research/
    # RQ-A_tracking_error.md) identifies this as the variable that explains
    # the closed-loop tracking error: the plan, the controller's reference and
    # the plant's integration are three different discretisations, and they
    # disagree by roughly this much per sample.
    steps = []
    for i in range(len(samples_data) - 1):
        delta = (samples_data[i + 1]["heading"] - samples_data[i]["heading"]
                 + math.pi) % (2 * math.pi) - math.pi
        steps.append(abs(delta))
    # Peak wheel acceleration. The hub's DriveBase ramps its trajectories at
    # about 7.8 m/s^2 for this robot's geometry (KI-021), and the optimizer
    # does not model that limit, so it is measured here.
    peak_accel = 0.0
    for sample in samples_data:
        peak_accel = max(peak_accel, abs(float(sample.get("al", 0.0))),
                         abs(float(sample.get("ar", 0.0))))
    record.update({
        "ok": bool(ok),
        "tube_m": float(max_tube),
        "tracking_m": float(report.get("max_tracking_error_m", float("inf"))),
        "max_dtheta_rad": float(max(steps)) if steps else 0.0,
        "mean_dtheta_rad": float(sum(steps) / len(steps)) if steps else 0.0,
        "max_accel_m_s2": float(peak_accel),
        "num_samples": len(samples_data),
        "motor_violation_N": float(report.get("max_motor_violation_N", 0.0)),
        "slip_l_N": float(report.get("max_slip_l_N", 0.0)),
        "slip_r_N": float(report.get("max_slip_r_N", 0.0)),
        "speed_violation_m_s": float(report.get("speed_violation_m_s", 0.0)),
        "diverged": bool(report["reachability"].get("diverged", False)),
        "integrate_max_m": float(cpu_summary["errors"]["max_pos_error_m"]),
        "integrate_final_m": float(cpu_summary["errors"]["final_pos_error_m"]),
        "violating_samples": int(cpu_summary["audit"]["num_violating_samples"]),
        "slip_points": int(cpu_summary["audit"]["num_slip_points"]),
        "final_wp_pos_m": float(waypoint_check["pos_error_m"]),
        "final_wp_heading_rad": float(waypoint_check["heading_error_rad"]),
        "path_length_m": float(metrics.get("path_length_m", float("nan"))),
        "max_jerk_m_s3": float(metrics.get("max_jerk_m_s3", float("nan"))),
        "max_wheel_speed_m_s": float(metrics.get("max_wheel_speed_m_s", float("nan"))),
        "reasons": reasons,
    })

    if critic is not None:
        try:
            windows = critic.evaluate(samples_data, samples_per_segment)
            record["critic_windows"] = len(windows)
        except Exception as exc:
            record["critic_windows"] = -1
            record["critic_error"] = str(exc)

    if profiles:
        # Per-sample envelope radius and forward drift, for the Q4 proxy study.
        radii = []
        for sample in samples_data:
            env = sample.get("reachability_envelope", {})
            if not env:
                radii.append(None)
                continue
            err_x = (env["x_max"] - env["x_min"]) / 2.0
            err_y = (env["y_max"] - env["y_min"]) / 2.0
            radii.append(round(math.sqrt(err_x ** 2 + err_y ** 2), 6))
        record["envelope_profile_m"] = radii
        record["distance_profile_m"] = [
            round(float(sample.get("t", 0.0)), 6) for sample in samples_data
        ]
    return record


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------

def load_mission(path: Path) -> Dict[str, Any]:
    """Load a mission file the way the CLI does."""
    wps, events, thresholds, stop_indices = cli._load_waypoints(str(path))
    meta = {"file": path.name, "profile": None}
    index_path = path.parent / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for entry in index.get("missions", []):
            if entry.get("file") == path.name:
                meta["profile"] = entry.get("profile")
                meta["id"] = entry.get("id")
                break
    return {
        "waypoints": wps,
        "events": events,
        "thresholds": thresholds,
        "stop_indices": stop_indices,
        "meta": meta,
    }


def _initial_guess(spec: Dict[str, Any], mission: Dict[str, Any], robot_cfg):
    """Build the initial guess for a spec, when it is not a stock seed."""
    guess_kind = spec["guess"]
    if guess_kind == "stock":
        return None
    samples = spec["samples"]
    waypoints = mission["waypoints"]
    stops = mission["stop_indices"]
    if guess_kind == "spline":
        return seed_lib.spline_guess(waypoints, samples, robot_cfg, stops)
    if guess_kind == "proportional_dt":
        return seed_lib.proportional_dt_guess(waypoints, samples, robot_cfg, stops)
    if guess_kind == "reeds_shepp":
        return seed_lib.reeds_shepp_guess(waypoints, samples, robot_cfg, stops)
    if guess_kind == "refined":
        return seed_lib.refined_guess(waypoints, samples, robot_cfg, stops,
                                      spec.get("weight", 0.0))
    raise ValueError(f"Unknown guess kind {guess_kind!r}")


def _payloads(mission: Dict[str, Any], specs: List[Dict[str, Any]], robot_cfg,
              self_consistent: bool = True) -> List[Any]:
    """The solve payloads for one mission on one robot."""
    waypoints = mission["waypoints"]
    stop_indices = mission["stop_indices"]

    payloads = []
    for spec in specs:
        solve_kwargs: Dict[str, Any] = {
            "num_samples_per_segment": spec["samples"],
            "accuracy_weight": spec["weight"],
            "stop_waypoint_indices": stop_indices,
            "waypoint_events": {},
            "verbose": False,
            "capture_iterations": False,
        }
        engine = spec.get("engine", "simple")
        if engine == "simple":
            solve_kwargs["guess_style"] = spec["style"]
            solve_kwargs["per_segment_dt"] = spec["timing"] == "per-segment"
            solve_kwargs["self_consistent"] = bool(self_consistent)
            guess = _initial_guess(spec, mission, robot_cfg)
            if guess is not None:
                solve_kwargs["initial_guess"] = guess
        payloads.append((robot_cfg, waypoints, spec, solve_kwargs,
                         engine == "multiverse"))
    return payloads


def _score_jobs(jobs: List[Any], solved: List[Any], validator_cls,
                robot_config_cls, spreads: Sequence[float] = (1.0,),
                profiles: bool = False, self_consistent: bool = True,
                propagation: str = "plant"
                ) -> List[Dict[str, Any]]:
    """Score a solved batch. One job per solved payload, in order.

    A job is (mission, spec, robot config, critic, robot label, robot
    factors): the mission and the spec say what was solved, the robot says
    what it was solved for.
    """
    records = []
    state = tree_state()
    for job, result in zip(jobs, solved):
        mission, spec, robot_cfg, critic, robot_label, factors = job
        samples_data, stats = result
        for spread in spreads:
            record = _score(samples_data, stats, robot_cfg, validator_cls,
                            robot_config_cls, mission["thresholds"],
                            spec["samples"], mission["waypoints"],
                            critic=critic, profiles=profiles, spread=spread,
                            propagation=propagation)
            record.update({
                "mission": mission["meta"].get("id") or mission["meta"]["file"],
                "mission_file": mission["meta"]["file"],
                "profile": mission["meta"].get("profile"),
                "tree": state,
                "spread": spread,
                "label": spec["label"],
                "style": spec["style"],
                "guess": spec["guess"],
                "engine": spec.get("engine", "simple"),
                "self_consistent": bool(self_consistent),
                "timing": spec["timing"],
                "weight": spec["weight"],
                "samples": spec["samples"],
                "num_waypoints": len(mission["waypoints"]),
                "num_segments": len(mission["waypoints"]) - 1,
                "robot": robot_label,
                "robot_factors": factors,
            })
            records.append(record)
    return records


def run_mission(mission: Dict[str, Any], specs: List[Dict[str, Any]], robot_cfg,
                validator_cls, robot_config_cls, critic=None,
                workers: int = 8, profiles: bool = False,
                spreads: Sequence[float] = (1.0,),
                self_consistent: bool = True,
                propagation: str = "plant") -> List[Dict[str, Any]]:
    """Solve every spec for one mission and score each result."""
    payloads = _payloads(mission, specs, robot_cfg, self_consistent)
    solved, _histories = cli._solve_candidates(payloads, workers, quiet=True,
                                                parallel=workers > 1)
    jobs = [(mission, spec, robot_cfg, critic, None, None) for spec in specs]
    return _score_jobs(jobs, solved, validator_cls, robot_config_cls,
                       spreads=spreads, profiles=profiles,
                       self_consistent=self_consistent,
                       propagation=propagation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--missions", nargs="+", default=["research/missions"],
                        help="Mission files or directories.")
    parser.add_argument("--out", required=True, help="Output JSONL file.")
    parser.add_argument("--specs", default="matrix",
                        help="Spec set: seeds, weights, wide-weights, spline, "
                             "proportional, density, matrix.")
    parser.add_argument("--samples", type=int, default=10,
                        help="Samples per segment for the non-density spec sets.")
    parser.add_argument("--weight", type=float, default=0.0,
                        help="Base accuracy weight.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Worker processes used to solve the candidates.")
    parser.add_argument("--config", default="examples/fll_choreo.json",
                        help="Robot config.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only run the first N missions (0 = all).")
    parser.add_argument("--no-critic", action="store_true",
                        help="Skip the Multi-Verse critic metrics (Q5).")
    parser.add_argument("--profiles", action="store_true",
                        help="Store the per-sample envelope profile (Q4).")
    parser.add_argument("--spread", type=float, default=1.0,
                        help="Scale the declared uncertainty ranges: 1 = as "
                             "declared, 0 = collapsed to nominal, 2 = twice as "
                             "wide (KI-020 sensitivity).")
    parser.add_argument("--spreads", type=float, nargs="*", default=None,
                        help="Validate every candidate at these spread "
                             "factors. Overrides --spread.")
    parser.add_argument("--self-consistent", dest="self_consistent", action="store_true", default=True,
                        help="Plan self-consistent trajectories (KI-026): the "
                             "samples lie on the path the plan's own profile "
                             "traces (default: True).")
    parser.add_argument("--no-self-consistent", dest="self_consistent", action="store_false",
                        help="Disable self-consistent trajectories (use trapezoidal collocation).")
    parser.add_argument("--robot-jitter", type=float, default=0.0,
                        help="Move the robot's constants away from the "
                             "declared values: 0 = the declared robot, 1 = the "
                             "full range research/robots.py lists.")
    parser.add_argument("--robot-seed", type=int, default=0,
                        help="Seed for the robot draws. The same seed gives "
                             "the same robots.")
    parser.add_argument("--robots", type=int, default=1,
                        help="How many robots to draw per mission.")
    parser.add_argument("--chunk", type=int, default=40,
                        help="How many (mission, robot) pairs one worker pool "
                             "solves. The pool is not rebuilt between chunks.")
    parser.add_argument("--append", action="store_true",
                        help="Append to the output file instead of replacing it.")
    parser.add_argument("--propagation", default="plant",
                        choices=("plant", "tick"),
                        help="Which reachability propagation the envelope gate "
                             "uses: 'plant' (the shipped open-loop tube) or "
                             "'tick' (the closed-loop tick map). The record "
                             "carries which one produced it.")
    args = parser.parse_args()

    # Collect the mission files.
    files: List[Path] = []
    for entry in args.missions:
        path = Path(entry)
        if path.is_dir():
            files.extend(sorted(p for p in path.glob("*.json")
                                if p.name != "index.json"))
        else:
            files.append(path)
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit("No mission files found")

    config_data, _declared_robot = cli._load_config(_resolve(args.config))
    validator_cls, robot_config_cls = cli._ensure_jax_classes()

    critic_cls = None
    if not args.no_critic:
        try:
            from multiverse_optimizer import TrajectoryCritic
            critic_cls = TrajectoryCritic
        except Exception as exc:
            print(f"critic unavailable: {exc}")

    specs = build_specs(args.specs, args.weight, args.samples)
    spreads = tuple(args.spreads) if args.spreads else (args.spread,)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    start = time.time()

    print(f"{len(files)} missions x {len(specs)} candidates "
          f"x {len(spreads)} spread(s) x {max(1, args.robots)} robot(s) = "
          f"{len(files) * len(specs) * len(spreads) * max(1, args.robots)} "
          f"validations", flush=True)
    with out_path.open(mode, encoding="utf-8") as handle:
        if not args.append:
            handle.write(json.dumps({
                "record": "meta",
                "specs": args.specs,
                "samples": args.samples,
                "base_weight": args.weight,
                "config": args.config,
                "spread": list(spreads),
                "self_consistent": bool(args.self_consistent),
                "robot_jitter": args.robot_jitter,
                "robot_seed": args.robot_seed,
                "robots": max(1, args.robots),
                "provenance": provenance(),
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            }) + "\n")
        # One job per (mission, robot, candidate). The jobs are solved in
        # chunks: one worker pool per chunk, so the pool is not rebuilt for
        # every pair. On Windows a pool costs a process start per worker,
        # which is seconds each time.
        jobs = []
        for path in files:
            mission = load_mission(path)
            mission_key = mission["meta"].get("id") or path.stem
            for draw in range(max(1, args.robots)):
                # Every mission is planned for a robot of its own, so a suite
                # of paths is a suite of paths on different machines.
                robot_data, factors = robot_lib.robot_for(
                    config_data, args.robot_seed, f"{mission_key}#{draw}",
                    args.robot_jitter)
                robot_cfg = RobotConfig(robot_data)
                critic = critic_cls(robot_cfg) if critic_cls is not None else None
                for spec in specs:
                    jobs.append((mission, spec, robot_cfg, critic,
                                 f"{mission_key}#{draw}", factors))

        chunk = max(1, int(args.chunk)) * max(1, len(specs))
        print(f"{len(jobs)} candidates in chunks of {chunk}", flush=True)
        for start_index in range(0, len(jobs), chunk):
            batch = jobs[start_index:start_index + chunk]
            payloads = []
            for mission, spec, robot_cfg, critic, label, factors in batch:
                payloads.extend(_payloads(mission, [spec], robot_cfg,
                                         args.self_consistent))
            solved, _histories = cli._solve_candidates(
                payloads, args.workers, quiet=True, parallel=args.workers > 1)
            records = _score_jobs(batch, solved, validator_cls, robot_config_cls,
                                  spreads=spreads, profiles=args.profiles,
                                  self_consistent=args.self_consistent,
                                  propagation=args.propagation)
            for record in records:
                handle.write(json.dumps(record) + "\n")
            handle.flush()
            passing = sum(1 for r in records if r.get("ok"))
            errors = sum(1 for r in records if r.get("error"))
            done = min(start_index + chunk, len(jobs))
            print(f"[{done}/{len(jobs)}] {passing}/{len(records)} pass  "
                  f"{errors} error(s)  {time.time() - start:.1f}s", flush=True)

    print(f"wrote {out_path} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()

