#!/usr/bin/env python3
"""Q20 probe: classify the solver deaths of the tightened-limit arms.

The registered corpus is the failing (mission, arm) candidates of the Q19
`wheel_radius_high` and `t_max_nm_low` arms and the joint-jitter harness run,
plus the nominal arm for comparison. Each failed iterate was already scored by
the harness that ran it, so the probe reads the registered solves rather than
re-solving them: same solves as the population counts, zero drift.

Per death it records the classification from the product's own
``classify_solver_error`` (labels cannot drift from the product), the solver
evidence (status text, iterations, wall time) and the failed iterate's worst
constraint violations (wheel force over bound, wheel speed over limit, accel
ramp against the declared ``accel_limit_m_s2``), and whether the nominal arm
converges the same mission. Mechanism rule (registered, Q20):

  max_iter + nominal converges  -> limit_tightening_budget_death
  max_iter + nominal fails      -> already_hard
  infeasible                    -> declared_infeasible
  numeric                       -> numeric
  anything else                 -> unknown (reported, never papered over)

Usage:
  python research/q20_probe.py [--out-dir research/results]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from overboard.core.optimizer import classify_solver_error  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MEASUREMENTS = REPO / "docs" / "measurements" / "2026-10-26-benchmarks"

SOURCES = (
    ("wheel_radius_high", REPO / "research" / "results" / "q19_wheel_radius_high.jsonl"),
    ("t_max_nm_low", REPO / "research" / "results" / "q19_t_max_nm_low.jsonl"),
    ("joint_jitter", MEASUREMENTS / "bench_2026-10-26_harness_robots.jsonl"),
)
NOMINAL = REPO / "research" / "results" / "q19_nominal.jsonl"


def _rows(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _split_rows(path):
    """Meta rows carry the config; candidate rows carry converged/error."""
    meta, candidates = {}, []
    for row in _rows(path):
        if "converged" in row:
            candidates.append(row)
        else:
            meta.update(row)
    return meta, candidates


def _accel_limit(meta):
    config = meta.get("config") or {}
    if isinstance(config, str):
        config_path = REPO / config
        with open(config_path, encoding="utf-8") as fh:
            config = json.load(fh)
    twin = config.get("digital_twin") or {}
    actuator = twin.get("actuator") or {}
    straight = actuator.get("straight") or {}
    value = straight.get("accel_limit")
    return float(value) if value is not None else None


def _mechanism(error_kind, nominal_converged):
    if error_kind == "max_iter":
        return ("limit_tightening_budget_death" if nominal_converged
                else "already_hard")
    if error_kind == "infeasible":
        return "declared_infeasible"
    if error_kind == "numeric":
        return "numeric"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="research/results")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nominal_ok = {}
    for row in _rows(NOMINAL):
        mission = row.get("mission")
        if mission:
            nominal_ok[mission] = nominal_ok.get(mission, False) or bool(
                row.get("converged"))

    deaths = []
    for arm, path in SOURCES:
        if not path.exists():
            raise SystemExit(f"missing registered corpus file: {path}")
        meta, candidates = _split_rows(path)
        accel_limit = _accel_limit(meta)
        for row in candidates:
            if not row.get("error"):
                continue
            kind = classify_solver_error(row["error"])
            mission = row.get("mission")
            nom = nominal_ok.get(mission, None)
            accel = row.get("max_accel_m_s2")
            deaths.append({
                "arm": arm,
                "mission": mission,
                "style": row.get("style"),
                "label": row.get("label"),
                "error_kind": kind,
                "error_status": row["error"].split("return_status is ")[-1].strip().strip("'")
                if "return_status is " in row["error"] else None,
                "iter_count": row.get("iter_count"),
                "solve_s": row.get("solve_s"),
                "motor_violation_N": row.get("motor_violation_N"),
                "speed_violation_m_s": row.get("speed_violation_m_s"),
                "max_accel_m_s2": accel,
                "accel_limit_m_s2": accel_limit,
                "accel_violation_m_s2": (max(0.0, accel - accel_limit)
                                          if accel is not None and accel_limit is not None
                                          else None),
                "diverged": row.get("diverged"),
                "tube_m": row.get("tube_m"),
                "tracking_m": row.get("tracking_m"),
                "nominal_converged": nom,
                "mechanism": _mechanism(kind, nom),
            })

    by_kind, by_arm, by_mechanism = {}, {}, {}
    for death in deaths:
        by_kind[death["error_kind"]] = by_kind.get(death["error_kind"], 0) + 1
        by_arm[death["arm"]] = by_arm.get(death["arm"], 0) + 1
        by_mechanism[death["mechanism"]] = by_mechanism.get(death["mechanism"], 0) + 1

    summary = {
        "total_deaths": len(deaths),
        "by_error_kind": by_kind,
        "by_arm": by_arm,
        "by_mechanism": by_mechanism,
        "unknown_count": by_kind.get("unknown", 0),
        "nominal_converged_missions": sorted(
            m for m, ok in nominal_ok.items() if ok),
        "nominal_failed_missions": sorted(
            m for m, ok in nominal_ok.items() if not ok),
        "accel_limit_m_s2": {d["arm"]: d["accel_limit_m_s2"] for d in deaths},
    }

    with open(out_dir / "q20_probe.jsonl", "w", encoding="utf-8") as fh:
        for death in deaths:
            fh.write(json.dumps(death, sort_keys=True) + "\n")
    with open(out_dir / "q20_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)

    header = (f"{'arm':<18}{'mission':<8}{'kind':<10}{'iters':>6}"
              f"{'solve_s':>9}{'mot_viol_N':>11}{'spd_viol':>9}"
              f"{'accel':>7}{'nominal':>8}  mechanism")
    print(header)
    print("-" * len(header))
    for d in deaths:
        print(f"{d['arm']:<18}{str(d['mission']):<8}{d['error_kind']:<10}"
              f"{d['iter_count']:>6}{d['solve_s']:>9.1f}"
              f"{d['motor_violation_N']:>11.4f}{d['speed_violation_m_s']:>9.4f}"
              f"{d['max_accel_m_s2']:>7.2f}"
              f"{('yes' if d['nominal_converged'] else 'NO'):>8}  {d['mechanism']}")
    print()
    print(f"total {summary['total_deaths']} deaths | by kind: {by_kind} "
          f"| by mechanism: {by_mechanism}")
    print(f"records: {out_dir / 'q20_probe.jsonl'} + q20_summary.json")


if __name__ == "__main__":
    main()
