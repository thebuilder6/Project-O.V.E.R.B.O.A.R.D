"""Q19: one-knob robot sensitivity (registered 2026-10-26, report-only).

Which single robot constant, mis-measured within the range ``research/robots.py``
declares for it, moves the product's verdicts and its model residuals the most?
The registration (docs/RESEARCH_AGENDA.md, Q19) is binding: eight knobs at two
levels each plus one nominal arm (17 arms), exactly ONE constant moved per arm
at exactly the range's low/high multiplier (the ``_factor`` semantics at spread
1: ``1.0 + (bound - 1.0)`` applied to the declared value), and every other
constant and every declared uncertainty range stays as ``examples/fll_choreo.json``
declares it. ``gear_backlash_m`` is the knob whose declared range itself moves
(it has no scalar constant - ``robots.FREE_RANGES``). No draws, no seeds: the
levels are exact and the run is reproducible.

Protocol: the 32-mission suite (``research/suite.py --count 32 --seed 7``), the
shipped default candidate pair (harness ``--specs default`` at its defaults:
2 candidates per (mission, robot), the 2026-10-26 joint-jitter arithmetic),
``--propagation plant``, default limits, no field clearance, one robot per arm
per mission, workers 8. ``research/harness.py`` is not modified: this driver
orchestrates the per-arm configs and calls harness's scoring (``_score_jobs``,
the production gate chain). Each arm is equivalent to one ``harness.py``
invocation with ``--config`` pointed at the arm's config file; the driver
prints those exact per-arm invocations (its own records label the robot with
the arm name and its single factor instead of the harness's per-draw label -
the gate fields are identical).

Metrics (fixed before the run, both rankings side by side):

* pass rates per (mission, robot) pair ("any of its candidates passes") and
  per candidate, each level against the nominal arm;
* continuous residuals: ``max_tracking_error_m`` (the record's ``tracking_m``),
  the tube radius (``tube_m``) and the envelope excess over the local limit -
  the worst sample's envelope radius minus the gate's own local limit
  ``thresholds.get(i // samples_per_segment, 0.02)`` (harness._score), so a
  negative number is margin - and solver errors/crashes per arm.

Each knob's effect is reported per unit of mis-measurement: the delta from the
nominal arm divided by (|factor - 1| / 0.1). Residual means are taken over the
candidates with a scored rollout; solver errors are counted separately. The
pass-rate ranking keys on the worst level's |pair-pass delta per 0.1|, the
residual ranking on the worst level's |mean tracking delta per 0.1| (the
registration's own first residual); the other residuals stay columns, so the
priorities stay arguable.

Usage::

    python research/suite.py --count 32 --seed 7 --out research/missions
    python research/q19_oneknob.py research/missions --config examples/fll_choreo.json \\
        --out-dir research/results --propagation plant --workers 8
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research import robots as robot_lib  # noqa: E402  (needs ROOT on sys.path)
from overboard.core.optimizer import classify_solver_error  # noqa: E402

# The registered knobs and levels: the range's (low, high) multipliers, in the
# registration's order. The values are research/robots.py's own; the registered
# fixture pins them against the registration table.
KNOBS: Dict[str, Tuple[float, float]] = {}
KNOBS.update(robot_lib.PARAMETERS)
KNOBS.update(robot_lib.FREE_RANGES)

# (arm name, the one knob it moves or None, the factor it moves it at).
ARMS: List[Tuple[str, Optional[str], float]] = [("nominal", None, 1.0)]
for _name, (_low, _high) in KNOBS.items():
    ARMS.append((f"{_name}_low", _name, _low))
    ARMS.append((f"{_name}_high", _name, _high))

# The candidate protocol: the shipped default pair (baseline seed on both
# timing models) at the harness's own defaults.
SPEC_SET = "default"
SAMPLES_PER_SEGMENT = 10
BASE_WEIGHT = 0.0


def _move_knob(config_data: Dict[str, Any], knob: str,
               factor: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The declared config with exactly one knob scaled by ``factor``.

    A scalar knob scales its declared value; ``gear_backlash_m`` scales both
    endpoints of its declared range. Nothing else moves: the other constants
    and every declared uncertainty range stay as declared (the Q19
    registration). Returns the copy and a description of the moved leaf.
    """
    data = copy.deepcopy(config_data)
    block = robot_lib._robot_block(data)
    if knob in robot_lib.PARAMETERS:
        value = robot_lib._read(block, knob)
        if value is None:
            raise ValueError(f"{knob} is not declared in this config")
        moved = float(value) * factor
        robot_lib._write(block, knob, moved)
        info = {"knob": knob, "factor": factor,
                "declared": float(value), "value": moved}
    elif knob in robot_lib.FREE_RANGES:
        ranges = robot_lib._uncertainty_block(block)
        holder, declared = robot_lib._read_range(block, ranges, knob)
        if declared is None or holder is None:
            raise ValueError(f"{knob} is not declared in this config")
        moved = [declared[0] * factor, declared[1] * factor]
        robot_lib._write(holder, knob, moved)
        info = {"knob": knob, "factor": factor,
                "declared": list(declared), "value": list(moved)}
    else:
        raise ValueError(f"unknown knob {knob!r}")
    return data, info


def arm_config(config_data: Dict[str, Any], arm: str) -> Dict[str, Any]:
    """The config for one arm, by name (``nominal``, ``mass_low``, ...)."""
    for name, knob, factor in ARMS:
        if name == arm:
            if knob is None:
                return copy.deepcopy(config_data)
            return _move_knob(config_data, knob, factor)[0]
    raise KeyError(f"unknown arm {arm!r}")


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------

def _harness_invocation(missions_arg: str, out_dir: Path, arm: str,
                        propagation: str, workers: int) -> str:
    """The exact harness command whose records this arm's records mirror."""
    return (
        f"python research/harness.py --missions {missions_arg} "
        f"--out {out_dir / f'q19_{arm}.jsonl'} "
        f"--specs {SPEC_SET} --samples {SAMPLES_PER_SEGMENT} "
        f"--weight {BASE_WEIGHT:g} --config {out_dir / f'q19_config_{arm}.json'} "
        f"--propagation {propagation} --workers {workers} --profiles "
        f"--robot-jitter 0 --robots 1 --spread 1"
    )


def _envelope_excess_m(profile: Optional[List[Any]], thresholds: Dict[int, float],
                       samples_per_segment: int) -> Optional[float]:
    """Worst sample envelope radius minus the gate's local limit (meters).

    The limit is the production gate's own (harness._score):
    ``thresholds.get(i // samples_per_segment, 0.02)``. Negative = margin.
    """
    worst = None
    for i, radius in enumerate(profile or []):
        if radius is None:
            continue
        excess = float(radius) - thresholds.get(i // samples_per_segment, 0.02)
        if worst is None or excess > worst:
            worst = excess
    return worst


def _collect_missions(entries: List[str]) -> List[Path]:
    files: List[Path] = []
    for entry in entries:
        path = Path(entry)
        if path.is_dir():
            files.extend(sorted(p for p in path.glob("*.json")
                                if p.name != "index.json"))
        else:
            files.append(path)
    if not files:
        raise SystemExit("No mission files found")
    return files


def run_arm(arm: str, knob: Optional[str], factor: float,
            config_data: Dict[str, Any], mission_files: List[Path],
            out_dir: Path, propagation: str, workers: int) -> Path:
    """One arm: 32 missions x 2 candidates on one robot, scored and written."""
    from research import harness
    import overboard.main as cli
    from overboard.models.robot_model import RobotConfig

    out_file = out_dir / f"q19_{arm}.jsonl"
    if out_file.exists():
        print(f"[{arm}] already recorded in {out_file}, skipping", flush=True)
        return out_file

    config_file = out_dir / f"q19_config_{arm}.json"
    robot_data, info = ((copy.deepcopy(config_data),
                         {"knob": None, "factor": 1.0, "declared": None,
                          "value": None})
                        if knob is None else _move_knob(config_data, knob, factor))
    config_file.write_text(json.dumps(robot_data, indent=2) + "\n", encoding="utf-8")

    specs = harness.build_specs(SPEC_SET, BASE_WEIGHT, SAMPLES_PER_SEGMENT)
    robot_cfg = RobotConfig(robot_data)
    factors = {knob: factor} if knob is not None else {}
    jobs = []
    payloads = []
    for path in mission_files:
        mission = harness.load_mission(path)
        for spec in specs:
            payloads.extend(harness._payloads(mission, [spec], robot_cfg))
            jobs.append((mission, spec, robot_cfg, arm, factors))

    print(f"[{arm}] {len(jobs)} candidates ({len(mission_files)} missions x "
          f"{len(specs)} candidates), knob={knob} factor={factor}", flush=True)
    start = time.time()
    validator_cls, robot_config_cls = cli._ensure_jax_classes()
    solved, _histories = cli._solve_candidates(payloads, workers, quiet=True,
                                               parallel=workers > 1)
    records = harness._score_jobs(jobs, solved, validator_cls, robot_config_cls,
                                  spreads=(1.0,), profiles=True,
                                  propagation=propagation)
    for record, (mission, spec, _cfg, _label, _factors) in zip(records, jobs):
        excess = _envelope_excess_m(record.get("envelope_profile_m"),
                                    mission["thresholds"], spec["samples"])
        if excess is not None:
            record["envelope_excess_m"] = excess
    wall = time.time() - start

    meta = {
        "record": "meta",
        "arm": arm, "knob": knob, "factor": factor,
        "moved": info,
        "config": str(config_file),
        "specs": SPEC_SET, "samples": SAMPLES_PER_SEGMENT,
        "base_weight": BASE_WEIGHT, "propagation": propagation,
        "missions": len(mission_files),
        "provenance": harness.provenance(),
        "tree": harness.tree_state(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_s": round(wall, 1),
    }
    with out_file.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(meta) + "\n")
        for record in records:
            handle.write(json.dumps(record) + "\n")
    kinds = {}
    for record in records:
        if record.get("error"):
            kind = classify_solver_error(record["error"])
            kinds[kind] = kinds.get(kind, 0) + 1
    kinds_text = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())) or "none"
    print(f"[{arm}] {sum(1 for r in records if r.get('ok'))}/{len(records)} pass, "
          f"solver failures by kind: {kinds_text}, "
          f"{wall:.1f}s -> {out_file}", flush=True)
    return out_file


# --------------------------------------------------------------------------
# Aggregation (the two rankings, side by side)
# --------------------------------------------------------------------------

def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _residuals(records: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    values = [float(r[key]) for r in records
              if r.get(key) is not None and r.get(key) == r.get(key)
              and float(r[key]) != float("inf")]
    return {"n": len(values), "mean": _mean(values),
            "max": max(values) if values else None}


def summarize_arm(records: List[Dict[str, Any]], meta: Dict[str, Any],
                  pairs: int) -> Dict[str, Any]:
    by_mission: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        by_mission.setdefault(str(record.get("mission")), []).append(record)
    pair_pass = sum(1 for group in by_mission.values()
                    if any(r.get("ok") for r in group))
    return {
        "knob": meta.get("knob"),
        "factor": meta.get("factor"),
        "propagation": meta.get("propagation"),
        "moved": meta.get("moved"),
        "wall_s": meta.get("wall_s"),
        "candidates": len(records),
        "pairs": pairs,
        "candidate_pass": sum(1 for r in records if r.get("ok")),
        "pair_pass": pair_pass,
        "solver_errors": sum(1 for r in records if r.get("error")),
        "solver_errors_by_kind": {
            kind: sum(1 for r in records
                      if r.get("error") and classify_solver_error(r["error"]) == kind)
            for kind in sorted({classify_solver_error(r["error"])
                                for r in records if r.get("error")})
        },
        "not_converged": sum(1 for r in records if not r.get("converged")),
        "validator_crashes": sum(1 for r in records
                                 if any("Immrax error" in str(x)
                                        for x in (r.get("reasons") or []))),
        "tracking_m": _residuals(records, "tracking_m"),
        "tube_m": _residuals(records, "tube_m"),
        "envelope_excess_m": _residuals(records, "envelope_excess_m"),
    }


def _per_0_1(delta: Optional[float], factor: float) -> Optional[float]:
    """The delta per 0.1 of factor error (the mis-measurement unit)."""
    if delta is None:
        return None
    error = abs(factor - 1.0)
    return None if error == 0 else delta / (error / 0.1)


def summarize(out_dir: Path) -> Dict[str, Any]:
    """Read every arm's records back and build q19_summary.json's content."""
    arms: Dict[str, Any] = {}
    for arm, knob, factor in ARMS:
        out_file = out_dir / f"q19_{arm}.jsonl"
        if not out_file.exists():
            raise SystemExit(f"{out_file} is missing; run the arm first")
        meta = {}
        records = []
        with out_file.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if item.get("record") == "meta":
                    meta = item
                else:
                    records.append(item)
        arms[arm] = summarize_arm(records, meta, pairs=meta.get("missions", 0))

    nominal = arms["nominal"]
    knobs: Dict[str, Any] = {}
    for name, (low, high) in KNOBS.items():
        levels = {}
        for level, factor in (("low", low), ("high", high)):
            arm = arms[f"{name}_{level}"]
            delta = {
                "pair_pass": arm["pair_pass"] - nominal["pair_pass"],
                "candidate_pass": arm["candidate_pass"] - nominal["candidate_pass"],
                "tracking_m_mean": ((arm["tracking_m"]["mean"] or 0)
                                    - (nominal["tracking_m"]["mean"] or 0)),
                "tube_m_mean": ((arm["tube_m"]["mean"] or 0)
                                - (nominal["tube_m"]["mean"] or 0)),
                "envelope_excess_m_mean": ((arm["envelope_excess_m"]["mean"] or 0)
                                           - (nominal["envelope_excess_m"]["mean"] or 0)),
            }
            levels[level] = {
                "arm": f"{name}_{level}",
                "factor": factor,
                "factor_error": abs(factor - 1.0),
                "delta": delta,
                "per_0_1": {key: _per_0_1(value, factor)
                            for key, value in delta.items()},
            }
        knobs[name] = levels

    return {
        "experiment": "Q19 - one-knob robot sensitivity (report-only)",
        "config": "examples/fll_choreo.json",
        "specs": f"{SPEC_SET} ({SAMPLES_PER_SEGMENT} samples/segment, "
                 f"weight {BASE_WEIGHT:g}, 2 candidates)",
        "propagation": arms["nominal"].get("propagation"),
        "excess_formula": "max over samples of (envelope radius - local limit); "
                          "local limit = thresholds.get(i // samples_per_segment, "
                          "0.02), the production gate's own limit",
        "residual_means": "over the candidates with a scored rollout; solver "
                          "errors are counted separately",
        "ranking_keys": {
            "pass_rate": "worst level |pair_pass delta per 0.1| "
                         "(pair = a mission whose candidates include a pass)",
            "continuous_residuals": "worst level |mean max_tracking_error_m "
                                    "delta per 0.1|",
        },
        "arms": arms,
        "knobs": knobs,
    }


def _fmt(value: Optional[float], scale: float = 1.0, digits: int = 3) -> str:
    return "-" if value is None else f"{value * scale:.{digits}f}"


def print_tables(summary: Dict[str, Any]) -> None:
    arms = summary["arms"]
    nominal = arms["nominal"]
    print()
    pairs = nominal["pairs"]
    candidates = nominal["candidates"]
    print(f"Per-arm results (baseline = nominal arm; {pairs} missions x 2 "
          f"candidates = {candidates})")
    print(f"{'arm':<20}{'knob':<16}{'factor':>7}{'pair':>8}{'cand':>8}"
          f"{'track_mm':>10}{'tube_mm':>10}{'excess_mm':>11}{'errors':>8}{'wall_s':>8}")
    for arm, data in arms.items():
        pair = f"{data['pair_pass']}/{data['pairs']}"
        cand = f"{data['candidate_pass']}/{data['candidates']}"
        print(f"{arm:<20}{str(data['knob']):<16}{data['factor']:>7.2f}"
              f"{pair:>8}{cand:>8}"
              f"{_fmt(data['tracking_m']['mean'], 1000):>10}"
              f"{_fmt(data['tube_m']['mean'], 1000):>10}"
              f"{_fmt(data['envelope_excess_m']['mean'], 1000):>11}"
              f"{data['solver_errors']:>8}{_fmt(data['wall_s'], 1, 1):>8}")

    for title, key, unit_scale in (
        ("PASS-RATE RANKING (pair = any-of-2 per mission; delta per 0.1 of "
         "factor error vs nominal)", "pair_pass", 1.0),
        ("CONTINUOUS-RESIDUAL RANKING (mean max_tracking_error_m, mm; delta "
         "per 0.1 of factor error vs nominal)", "tracking_m_mean", 1000.0),
    ):
        rows = []
        for name, levels in summary["knobs"].items():
            lo, hi = levels["low"], levels["high"]
            worst = max(abs(lo["per_0_1"][key] or 0.0),
                        abs(hi["per_0_1"][key] or 0.0))
            rows.append((worst, name, lo, hi))
        rows.sort(key=lambda row: row[0], reverse=True)
        print()
        print(title)
        print(f"{'rank':<5}{'knob':<15}"
              f"{'low f':>6}{'delta':>9}{'/0.1':>9}"
              f"{'high f':>7}{'delta':>9}{'/0.1':>9}{'worst |/0.1|':>14}")
        for rank, (worst, name, lo, hi) in enumerate(rows, 1):
            print(f"{rank:<5}{name:<15}"
                  f"{lo['factor']:>6.2f}"
                  f"{_fmt(lo['delta'][key], unit_scale):>9}"
                  f"{_fmt(lo['per_0_1'][key], unit_scale):>9}"
                  f"{hi['factor']:>7.2f}"
                  f"{_fmt(hi['delta'][key], unit_scale):>9}"
                  f"{_fmt(hi['per_0_1'][key], unit_scale):>9}"
                  f"{_fmt(worst, unit_scale):>14}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("missions", nargs="+",
                        help="Mission files or directories (the 32-mission suite).")
    parser.add_argument("--config", default="examples/fll_choreo.json",
                        help="The declared robot config.")
    parser.add_argument("--out-dir", default="research/results",
                        help="Where the per-arm records and the summary go.")
    parser.add_argument("--propagation", default="plant",
                        choices=("plant", "tick"),
                        help="The reachability propagation the envelope gate "
                             "uses (the registration's is plant).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Worker processes used to solve the candidates.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = ROOT / args.config
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    mission_files = _collect_missions(args.missions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missions_arg = " ".join(args.missions)
    print("Exact per-arm harness invocations (each arm = one of these):")
    for arm, knob, factor in ARMS:
        print("  " + _harness_invocation(missions_arg, out_dir, arm,
                                         args.propagation, args.workers))
    print(flush=True)

    for arm, knob, factor in ARMS:
        run_arm(arm, knob, factor, config_data, mission_files, out_dir,
                args.propagation, args.workers)

    summary = summarize(out_dir)
    summary_path = out_dir / "q19_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print_tables(summary)
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
