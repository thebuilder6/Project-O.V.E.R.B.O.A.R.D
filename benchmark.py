"""
FLL Trajectory Optimizer – Unified Benchmark Suite
===================================================

Combines three previous benchmark scripts into a single entry-point:

    python benchmark.py --mode cli        # compare simple vs MV using waypoint files
    python benchmark.py --mode pipeline   # quick in-process speed comparison
    python benchmark.py --mode random     # comprehensive randomised stress test

Common options
--------------
    --config <path>   Robot config file (default: examples/fll_choreo.json)
    --output <path>   Where to write JSON results (default: benchmark_results.json)
    --runs  <int>     How many runs / iterations  (default depends on mode)

CLI-mode extra options
----------------------
    --samples <int>   Samples per segment (default: 10)
    --validate        Enable trajectory validation (default: off for speed)

Random-mode extra options
-------------------------
    --profiles        Comma-separated list from: short_sprint,complex,stress_test
    --report          Generate a Markdown report alongside the JSON results
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

from robot_model import RobotConfig
from optimizer import TrajectoryOptimizer
from validator import validate_trajectory

# MV optimizer is optional (JAX may not be installed)
try:
    from multiverse_optimizer import MasterTrajectoryOptimizer
    HAS_MULTIVERSE = True
except ImportError:
    HAS_MULTIVERSE = False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

STANDARD_CONFIG = {
    "mass": 0.723,
    "inertia": 0.0024,
    "track_width": 0.0965,
    "wheel_radius": 0.028,
    "v_max_rad_s": 15.7,
    "t_max_nm": 0.04,
    "gearing": 1.0,
    "cof": 0.40,
    "gravity": 9.81,
    "torque_headroom": 0.85,
    "speed_headroom": 0.90,
}


def _load_config(config_path: str) -> tuple[dict, RobotConfig]:
    """Load robot config from file, falling back to STANDARD_CONFIG."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            data = json.load(f)
        return data, RobotConfig(data)
    print(f"[warn] '{config_path}' not found – using built-in standard config.")
    return STANDARD_CONFIG, RobotConfig(STANDARD_CONFIG)


def _save_results(results, output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


# ---------------------------------------------------------------------------
# Mode 1 – CLI  (subprocess-based, validates .traj files)
# ---------------------------------------------------------------------------

class CLIBenchmarkRunner:
    """
    Benchmark by invoking main.py as a subprocess so that wall-clock times
    and .traj output match exactly what end-users experience.
    """

    def __init__(self, config_path: str = "examples/fll_choreo.json"):
        self.config_path = config_path
        self.results: list[dict] = []

    # ------------------------------------------------------------------
    def run_single(
        self,
        waypoint_file: str,
        optimizer_type: str = "simple",
        samples: int = 10,
        runs: int = 3,
        validate: bool = False,
    ) -> dict:
        import subprocess

        times, total_times, validation_results = [], [], []

        for run in range(runs):
            out_file = f"benchmark_{optimizer_type}_{Path(waypoint_file).stem}_run{run}.traj"

            cmd = [
                "py", "main.py",
                "-c", self.config_path,
                "-w", waypoint_file,
                "-o", out_file,
                "-n", str(samples),
            ]
            if optimizer_type == "simple":
                cmd.append("--simple")
            else:
                # The simple solver is the default now, so the Multi-Verse arm
                # has to ask for it explicitly.
                cmd.extend(["--multiverse", "--no-parallel"])

            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            elapsed = time.time() - t0

            total_time = self._extract_total_time(proc.stdout)
            times.append(elapsed)
            if total_time is not None:
                total_times.append(total_time)

            if validate and os.path.exists(out_file):
                try:
                    metrics, audit, errors = validate_trajectory(
                        out_file, self.config_path, apply_headroom=False
                    )
                    validation_results.append({"metrics": metrics, "audit": audit, "errors": errors})
                except Exception as exc:
                    print(f"  [warn] Validation failed for run {run}: {exc}")
                    validation_results.append(None)

            if os.path.exists(out_file):
                os.remove(out_file)

        return {
            "waypoint_file": waypoint_file,
            "optimizer_type": optimizer_type,
            "samples": samples,
            "runs": runs,
            "avg_wall_time": sum(times) / len(times),
            "avg_trajectory_time": sum(total_times) / len(total_times) if total_times else None,
            "min_wall_time": min(times),
            "max_wall_time": max(times),
            "validation": self._aggregate_validation(validation_results) if validation_results else None,
        }

    # ------------------------------------------------------------------
    def run_suite(
        self,
        waypoint_files: list[str],
        samples: int = 10,
        runs: int = 3,
        validate: bool = False,
    ) -> None:
        print("=== CLI Benchmark Suite ===")
        print(f"Config : {self.config_path}")
        print(f"Samples: {samples}  Runs: {runs}  Validate: {validate}")
        print("-" * 60)

        for wp in waypoint_files:
            print(f"\n  {wp}")
            r_simple = self.run_single(wp, "simple", samples, runs, validate)
            self.results.append(r_simple)
            r_mv = self.run_single(wp, "multiverse", samples, runs, validate)
            self.results.append(r_mv)

            speedup = r_simple["avg_wall_time"] / r_mv["avg_wall_time"]
            print(f"    simple={r_simple['avg_wall_time']:.2f}s  mv={r_mv['avg_wall_time']:.2f}s  speedup={speedup:.2f}x")

    # ------------------------------------------------------------------
    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("CLI BENCHMARK SUMMARY")
        print("=" * 60)
        files = sorted({r["waypoint_file"] for r in self.results})
        for wp in files:
            s = next((r for r in self.results if r["waypoint_file"] == wp and r["optimizer_type"] == "simple"), None)
            m = next((r for r in self.results if r["waypoint_file"] == wp and r["optimizer_type"] == "multiverse"), None)
            if s and m:
                print(f"\n{Path(wp).name}:")
                print(f"  Simple:      {s['avg_wall_time']:.2f}s (range {s['max_wall_time'] - s['min_wall_time']:.2f}s)")
                print(f"  Multi-Verse: {m['avg_wall_time']:.2f}s (range {m['max_wall_time'] - m['min_wall_time']:.2f}s)")
                print(f"  Speedup:     {s['avg_wall_time'] / m['avg_wall_time']:.2f}x")
                if s.get("validation") and m.get("validation"):
                    vs, vm = s["validation"], m["validation"]
                    print(f"  Val  simple: pos_err={vs['avg_max_pos_error_m']:.6f}m  pass={vs['pass_rate']:.0%}")
                    print(f"  Val  mv:     pos_err={vm['avg_max_pos_error_m']:.6f}m  pass={vm['pass_rate']:.0%}")

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_total_time(output: str):
        for line in output.splitlines():
            if "Total time:" in line:
                try:
                    return float(line.split("Total time:")[1].strip().rstrip("s"))
                except (IndexError, ValueError):
                    pass
        return None

    @staticmethod
    def _aggregate_validation(results: list) -> dict | None:
        valid = [v for v in results if v is not None]
        if not valid:
            return None
        n = len(valid)
        agg = {k: 0.0 for k in [
            "avg_max_pos_error_m", "avg_rms_pos_error_m", "avg_max_heading_error_rad",
            "avg_num_violating_samples", "avg_num_slip_points",
            "avg_max_left_wheel_slip", "avg_max_right_wheel_slip",
        ]}
        pass_count = 0
        for v in valid:
            e, a = v["errors"], v["audit"]
            agg["avg_max_pos_error_m"]         += e["max_pos_error_m"]
            agg["avg_rms_pos_error_m"]         += e["rms_pos_error_m"]
            agg["avg_max_heading_error_rad"]   += e["max_heading_error_rad"]
            agg["avg_num_violating_samples"]   += a["num_violating_samples"]
            agg["avg_num_slip_points"]         += a["num_slip_points"]
            agg["avg_max_left_wheel_slip"]     += a["left_wheel_slip"]
            agg["avg_max_right_wheel_slip"]    += a["right_wheel_slip"]
            if e["max_pos_error_m"] < 0.01 and e["final_pos_error_m"] < 0.01 \
                    and a["num_violating_samples"] == 0 and a["num_slip_points"] == 0:
                pass_count += 1
        for k in agg:
            agg[k] /= n
        agg["pass_rate"] = pass_count / n
        return agg


# ---------------------------------------------------------------------------
# Mode 2 – Pipeline  (quick in-process comparison, no subprocess)
# ---------------------------------------------------------------------------

def run_pipeline_benchmark(config_data: dict, cfg: RobotConfig) -> dict:
    """
    Fast, in-process comparison of Simple vs JAX/MV solver.
    Returns a summary dict.
    """
    wps = [(0, 0, 0), (1, 0, 0), (2, 1, np.pi / 2)]
    print(f"\n=== Pipeline Benchmark  ({len(wps)} waypoints) ===")

    result: dict = {"waypoints": len(wps)}

    # --- Simple -----------------------------------------------------------
    opt_simple = TrajectoryOptimizer(cfg)
    t0 = time.time()
    samples_s, stats_s = opt_simple.solve(wps, verbose=False)
    t_simple = time.time() - t0
    cost_s = samples_s[-1]["t"]
    # A solve that did not converge is not a result (KI-001). Report it as a
    # failure instead of averaging its debug values into the summary.
    converged_s = bool(stats_s.get("converged", False))
    print(f"  Simple CasADi  solve={t_simple:.4f}s  traj_time={cost_s:.4f}s"
          f"  {'converged' if converged_s else 'FAILED'}")
    result["simple"] = {"solve_time": t_simple, "traj_time": cost_s,
                         "converged": converged_s}

    # --- Multi-Verse (if available) ---------------------------------------
    if HAS_MULTIVERSE:
        opt_mv = MasterTrajectoryOptimizer(cfg, enable_parallel=True)
        # warmup pass
        opt_mv.solve(wps, verbose=False)
        t0 = time.time()
        samples_mv, stats_mv = opt_mv.solve(wps, verbose=False)
        t_mv = time.time() - t0
        cost_mv = samples_mv[-1]["t"]
        converged_mv = bool(stats_mv.get("converged", False))
        print(f"  JAX Multi-Verse solve={t_mv:.4f}s  traj_time={cost_mv:.4f}s"
              f"  {'converged' if converged_mv else 'FAILED'}")
        improvement = (cost_s - cost_mv) / cost_s * 100
        print(f"  Cost improvement: {improvement:.2f}%")
        if stats_mv.get("phase_times", {}).get("refinement") is not None:
            print(f"  JAX refine phase: {stats_mv['phase_times']['refinement']:.4f}s")
        result["multiverse"] = {"solve_time": t_mv, "traj_time": cost_mv,
                                "improvement_pct": improvement,
                                "converged": converged_mv}
    else:
        print("  Multi-Verse skipped (JAX not installed).")
        result["multiverse"] = None

    return result


# ---------------------------------------------------------------------------
# Mode 4 - Solver  (per-mission solve cost, iterations and pass/fail)
# ---------------------------------------------------------------------------

SOLVER_BENCHMARK_MISSIONS = [
    "examples/example_straight.json",
    "examples/example_s_curve.json",
    "examples/example_mission_first4.json",
    "examples/example_with_stops.json",
    "examples/example_with_events.json",
    "examples/example_complex_segment.json",
    "examples/example_complete.json",
    "examples/example_complex_mission.json",
]


def _load_mission(path: str):
    """Return (waypoints, stop_indices) for a waypoint file."""
    with open(path, "r") as handle:
        raw = json.load(handle)
    waypoints = raw["waypoints"] if isinstance(raw, dict) and "waypoints" in raw else raw
    wps, stops = [], []
    for i, item in enumerate(waypoints):
        if isinstance(item, dict):
            wps.append((item["x"], item["y"], item.get("heading")))
            if item.get("stop"):
                stops.append(i)
        else:
            wps.append((item[0], item[1], item[2] if len(item) > 2 else None))
    return wps, stops


def run_solver_benchmark(config_data: dict, cfg: RobotConfig,
                         missions: list[str] | None = None,
                         fast_polish: bool = False,
                         samples: int = 10) -> list[dict]:
    """Solve every mission and report the cost of the solve itself.

    Reports the wall time, the IPOPT iteration count and the trajectory
    duration for each mission, so a solver change can be compared on the
    missions that matter instead of on a three-waypoint toy path.
    """
    paths = missions or SOLVER_BENCHMARK_MISSIONS
    results: list[dict] = []
    print(f"\n=== Solver Benchmark  ({len(paths)} missions, "
          f"fast_polish={fast_polish}) ===")
    header = f"  {'mission':38s} {'wall':>8s} {'iters':>7s} {'traj':>9s}  result"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for path in paths:
        if not os.path.exists(path):
            print(f"  {os.path.basename(path):38s} skipped (file not found)")
            continue
        wps, stops = _load_mission(path)
        if len(wps) < 2:
            continue
        optimizer = TrajectoryOptimizer(cfg)
        t0 = time.time()
        samples_out, stats = optimizer.solve(
            wps, num_samples_per_segment=samples,
            stop_waypoint_indices=stops, verbose=False,
            fast_polish=fast_polish)
        wall = time.time() - t0
        iterations = stats.get("iter_count")
        if fast_polish and stats.get("stage_iterations"):
            iterations = sum(i or 0 for i in stats["stage_iterations"])
        row = {
            "mission": os.path.basename(path),
            "waypoints": len(wps),
            "solve_time": wall,
            "iterations": iterations,
            "traj_time": stats.get("final_cost"),
            "converged": bool(stats.get("converged", False)),
            "fast_polish": fast_polish,
        }
        results.append(row)
        print(f"  {row['mission']:38s} {wall:7.2f}s {str(iterations):>7s} "
              f"{row['traj_time']:8.4f}s  {'converged' if row['converged'] else 'FAILED'}")

    total = sum(r["solve_time"] for r in results)
    print(f"  {'total':38s} {total:7.2f}s")
    return results


# ---------------------------------------------------------------------------
# Mode 3 – Random / Comprehensive stress test
# ---------------------------------------------------------------------------

class RandomBenchmark:
    """
    Generates random mission profiles, solves with both optimisers,
    validates, and produces a Markdown report.
    """

    PROFILES = {
        "short_sprint": {"n": (3, 3), "x": (0, 1.5), "y": (0, 0.5)},
        "complex":      {"n": (5, 8), "x": (0, 2.0), "y": (0, 1.5)},
        "stress_test":  {"n": (4, 6), "x": (0, 1.0), "y": (0, 1.0)},
    }

    def __init__(self, config_data: dict, cfg: RobotConfig, output_dir: str = "benchmarks"):
        self.config_data = config_data
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: list[dict] = []
        # Keep the config that was actually used, so validation reads it back instead of
        # parsing robot parameters out of the trajectory file.
        self.config_file = self.output_dir / "config_used.json"

    # ------------------------------------------------------------------
    def generate_waypoints(self, profile: str) -> tuple[list, list]:
        """Return (wps_list, stop_indices) for random mission profile.

        The waypoints are a random walk inside the profile's field, not
        independent uniform draws. Uniform draws put two waypoints on top of
        each other often enough to make the mission unsolvable for reasons
        that have nothing to do with the planner, which is exactly the kind
        of artefact a benchmark must not have.
        """
        p = self.PROFILES[profile]
        n = random.randint(*p["n"])
        wps_json, wps_list, stops = [], [], []
        x = random.uniform(*p["x"])
        y = random.uniform(*p["y"])
        direction = random.uniform(-np.pi, np.pi)
        for i in range(n):
            if i > 0:
                length = random.uniform(0.2, 0.8)
                direction += random.uniform(-0.6, 0.6)
                nx = x + length * math.cos(direction)
                ny = y + length * math.sin(direction)
                if not (p["x"][0] <= nx <= p["x"][1]):
                    direction = math.pi - direction
                    nx = x + length * math.cos(direction)
                if not (p["y"][0] <= ny <= p["y"][1]):
                    direction = -direction
                    ny = y + length * math.sin(direction)
                x, y = nx, ny
            heading = None
            if i == 0:
                heading = direction
            elif profile == "stress_test" and random.random() < 0.5:
                heading = direction + math.pi
            elif random.random() < 0.4:
                heading = direction
            stop = (i == 0) or (i == n - 1) or (random.random() < 0.3)
            wp = {
                "x": round(x, 4),
                "y": round(y, 4),
                "heading": heading,
                "stop": stop,
            }
            wps_json.append(wp)
            wps_list.append((wp["x"], wp["y"], wp["heading"]))
            if stop:
                stops.append(i)
        return wps_json, wps_list, stops

    # ------------------------------------------------------------------
    def _validate(self, samples, output_file: Path):
        if not samples:
            return None
        data = {"trajectory": {"config": self.config_data.get("robot", {}), "samples": samples}}
        with open(output_file, "w") as f:
            json.dump(data, f)
        try:
            metrics, audit, errors = validate_trajectory(
                str(output_file), str(self.config_file), apply_headroom=False
            )
            return {"metrics": metrics, "audit": audit, "errors": errors}
        except Exception as exc:
            print(f"    [warn] Validation error: {exc}")
            return None

    # ------------------------------------------------------------------
    def run(self, num_runs: int = 10, profiles: list[str] | None = None) -> None:
        profiles = profiles or list(self.PROFILES)
        print(f"\n=== Random Benchmark  ({num_runs} runs, profiles={profiles}) ===")

        with open(self.config_file, "w") as f:
            json.dump(self.config_data, f, indent=2)

        for i in range(num_runs):
            profile = random.choice(profiles)
            print(f"\n--- Run {i+1}/{num_runs}  profile={profile} ---")

            wps_json, wps_list, stops = self.generate_waypoints(profile)
            run_id = f"run_{i:03d}_{profile}"
            run_dir = self.output_dir / run_id
            run_dir.mkdir(exist_ok=True)

            with open(run_dir / "waypoints.json", "w") as f:
                json.dump(wps_json, f, indent=2)

            # Simple
            print("  Simple optimizer…")
            simple_opt = TrajectoryOptimizer(self.cfg)
            try:
                samples_s, stats_s = simple_opt.solve(wps_list, stop_waypoint_indices=stops, verbose=False)
                val_s = self._validate(samples_s, run_dir / "traj_simple.traj")
            except Exception as exc:
                print(f"    simple failed: {exc}")
                samples_s, stats_s, val_s = None, None, None

            # Multi-Verse
            if HAS_MULTIVERSE:
                print("  Multi-Verse optimizer…")
                mv_opt = MasterTrajectoryOptimizer(self.cfg, enable_parallel=True, num_workers=4, verbose=False)
                try:
                    samples_mv, stats_mv = mv_opt.solve(wps_list, stop_waypoint_indices=stops, verbose=False)
                    val_mv = self._validate(samples_mv, run_dir / "traj_mv.traj")
                except Exception as exc:
                    print(f"    multiverse failed: {exc}")
                    samples_mv, stats_mv, val_mv = None, None, None
            else:
                samples_mv, stats_mv, val_mv = None, None, None

            self.results.append({
                "id": run_id,
                "profile": profile,
                "waypoints": wps_json,
                "simple":    {"stats": stats_s,  "validation": val_s},
                "multiverse": {"stats": stats_mv, "validation": val_mv},
            })
            self._save_incremental()

    # ------------------------------------------------------------------
    def _save_incremental(self) -> None:
        with open(self.output_dir / "benchmark_results.json", "w") as f:
            json.dump(self.results, f, indent=2)

    # ------------------------------------------------------------------
    def generate_report(self) -> None:
        report_path = self.output_dir / "benchmark_report.md"
        lines = [
            "# Comprehensive Optimizer Benchmark Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary Statistics",
            "| Metric | Simple | Multi-Verse | Delta / Speedup |",
            "| :--- | :--- | :--- | :--- |",
        ]

        simple_times = [r["simple"]["stats"]["total_time"] for r in self.results if r["simple"]["stats"]]
        mv_times     = [r["multiverse"]["stats"]["total_time"] for r in self.results if r["multiverse"]["stats"]]
        avg_s = np.mean(simple_times) if simple_times else 0
        avg_mv = np.mean(mv_times) if mv_times else 0
        speedup = avg_s / avg_mv if avg_mv > 0 else 0
        lines.append(f"| Avg Solve Time | {avg_s:.3f}s | {avg_mv:.3f}s | {speedup:.2f}x |")

        simple_costs = [r["simple"]["stats"]["final_cost"] for r in self.results if r["simple"]["stats"]]
        mv_costs     = [r["multiverse"]["stats"]["final_cost"] for r in self.results if r["multiverse"]["stats"]]
        avg_cs = np.mean(simple_costs) if simple_costs else 0
        avg_cmv = np.mean(mv_costs) if mv_costs else 0
        improvement = (1 - avg_cmv / avg_cs) * 100 if avg_cs > 0 else 0
        lines.append(f"| Avg Traj Duration | {avg_cs:.3f}s | {avg_cmv:.3f}s | {improvement:.1f}% better |")

        ok_s  = sum(1 for r in self.results if r["simple"]["stats"])
        ok_mv = sum(1 for r in self.results if r["multiverse"]["stats"])
        lines.append(f"| Success Rate | {ok_s}/{len(self.results)} | {ok_mv}/{len(self.results)} | |")

        lines += [
            "\n## Heuristic Effectiveness",
            "| Run | Window | Heuristic | Improvement |",
            "| :--- | :--- | :--- | :--- |",
        ]
        for run in self.results:
            mv_stats = run["multiverse"]["stats"]
            if mv_stats and mv_stats.get("heuristic_wins"):
                for win in mv_stats["heuristic_wins"]:
                    lines.append(f"| {run['id']} | {win['window']} | {win['heuristic']} | {win['improvement_pct']:.1f}% |")

        lines += [
            "\n## Physical Validation (Multi-Verse)",
            "| Run | Max Pos Error | Max Slip | Status |",
            "| :--- | :--- | :--- | :--- |",
        ]
        for run in self.results:
            v = run["multiverse"]["validation"]
            if v:
                err  = v["errors"]["max_pos_error_m"]
                slip = v["audit"]["left_wheel_slip"] + v["audit"]["right_wheel_slip"]
                status = "✅ PASS" if err < 0.01 and slip < 1e-3 else "❌ FAIL"
                lines.append(f"| {run['id']} | {err:.6f}m | {slip:.6f}N | {status} |")
            else:
                lines.append(f"| {run['id']} | N/A | N/A | N/A |")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nMarkdown report -> {report_path}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="FLL Trajectory Optimizer – Unified Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["cli", "pipeline", "random", "solver"], default="pipeline",
                        help="Benchmark mode (default: pipeline)")
    parser.add_argument("--config", default="examples/fll_choreo.json",
                        help="Robot config file")
    parser.add_argument("--output", default="benchmark_results.json",
                        help="Output JSON path")
    parser.add_argument("--runs", type=int, default=None,
                        help="Number of runs / iterations")

    # Solver-mode
    parser.add_argument("--fast-polish", action="store_true",
                        help="[solver] Use the fast solve path (budgeted warm pass + cold finish)")
    parser.add_argument("--missions", nargs="+",
                        help="[solver] Waypoint JSON files to solve (default: the example missions)")

    # CLI-mode
    parser.add_argument("--samples", type=int, default=10,
                        help="[cli] Samples per segment")
    parser.add_argument("--validate", action="store_true",
                        help="[cli] Validate trajectories")
    parser.add_argument("--waypoints", nargs="+",
                        help="[cli] Waypoint JSON files to benchmark")

    # Random-mode
    parser.add_argument("--profiles", default="short_sprint,complex,stress_test",
                        help="[random] Comma-separated mission profiles")
    parser.add_argument("--report", action="store_true",
                        help="[random] Generate Markdown report")

    return parser.parse_args()


def main():
    args = _parse_args()
    config_data, cfg = _load_config(args.config)

    if args.mode == "pipeline":
        runs = args.runs or 1
        all_results = []
        for _ in range(runs):
            all_results.append(run_pipeline_benchmark(config_data, cfg))
        _save_results(all_results, args.output)

    elif args.mode == "cli":
        wp_files = args.waypoints or [
            "examples/example_straight.json",
            "examples/example_s_curve.json",
            "examples/example_complex_mission.json",
        ]
        existing = [f for f in wp_files if os.path.exists(f)]
        if not existing:
            print("No waypoint files found. Pass --waypoints or add example files.")
            return
        runner = CLIBenchmarkRunner(config_path=args.config)
        runner.run_suite(existing, samples=args.samples, runs=args.runs or 3, validate=args.validate)
        runner.print_summary()
        _save_results(runner.results, args.output)

    elif args.mode == "solver":
        results = run_solver_benchmark(config_data, cfg, missions=args.missions,
                                       fast_polish=args.fast_polish,
                                       samples=args.samples)
        _save_results(results, args.output)

    elif args.mode == "random":
        profiles = [p.strip() for p in args.profiles.split(",")]
        rb = RandomBenchmark(config_data, cfg)
        rb.run(num_runs=args.runs or 10, profiles=profiles)
        if args.report:
            rb.generate_report()
        _save_results(rb.results, args.output)


if __name__ == "__main__":
    main()
