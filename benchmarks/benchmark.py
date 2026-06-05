"""
FLL Trajectory Optimizer – Unified Benchmark Suite
===================================================

Combines three previous benchmark scripts into a single entry-point:

    python benchmark.py --mode cli        # compare simple vs MV using waypoint files
    python benchmark.py --mode pipeline   # quick in-process speed comparison
    python benchmark.py --mode random     # comprehensive randomised stress test

Common options
--------------
    --config <path>   Robot config file (default: robot_config.json)
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
import os
import random
import time
from pathlib import Path

import numpy as np

from overboard.core.robot_model import RobotConfig
from overboard.core.optimizer import TrajectoryOptimizer
from overboard.validation.validator import validate_trajectory

# MV optimizer is optional (JAX may not be installed)
try:
    from overboard.core.multiverse_optimizer import MasterTrajectoryOptimizer
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

    def __init__(self, config_path: str = "robot_config.json"):
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
            cmd.append("--simple" if optimizer_type == "simple" else "--no-parallel")

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
    samples_s, _ = opt_simple.solve(wps, verbose=False)
    t_simple = time.time() - t0
    cost_s = samples_s[-1]["t"]
    print(f"  Simple CasADi  solve={t_simple:.4f}s  traj_time={cost_s:.4f}s")
    result["simple"] = {"solve_time": t_simple, "traj_time": cost_s}

    # --- Multi-Verse (if available) ---------------------------------------
    if HAS_MULTIVERSE:
        opt_mv = MasterTrajectoryOptimizer(cfg, enable_parallel=True)
        # warmup pass
        opt_mv.solve(wps, verbose=False)
        t0 = time.time()
        samples_mv, stats_mv = opt_mv.solve(wps, verbose=False)
        t_mv = time.time() - t0
        cost_mv = samples_mv[-1]["t"]
        print(f"  JAX Multi-Verse solve={t_mv:.4f}s  traj_time={cost_mv:.4f}s")
        improvement = (cost_s - cost_mv) / cost_s * 100
        print(f"  Cost improvement: {improvement:.2f}%")
        if stats_mv.get("phase_times", {}).get("refinement") is not None:
            print(f"  JAX refine phase: {stats_mv['phase_times']['refinement']:.4f}s")
        result["multiverse"] = {"solve_time": t_mv, "traj_time": cost_mv,
                                "improvement_pct": improvement}
    else:
        print("  Multi-Verse skipped (JAX not installed).")
        result["multiverse"] = None

    return result


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

    # ------------------------------------------------------------------
    def generate_waypoints(self, profile: str) -> tuple[list, list]:
        """Return (wps_list, stop_indices) for random mission profile."""
        p = self.PROFILES[profile]
        n = random.randint(*p["n"])
        wps_json, wps_list, stops = [], [], []
        for i in range(n):
            heading = random.choice([None, random.uniform(-np.pi, np.pi)])
            stop = (i == 0) or (i == n - 1) or (random.random() < 0.3)
            wp = {
                "x": random.uniform(*p["x"]),
                "y": random.uniform(*p["y"]),
                "heading": heading,
                "stop": stop,
            }
            if profile == "stress_test" and i > 0 and random.random() < 0.5:
                wp["heading"] = (wps_json[i - 1].get("heading") or 0) + np.pi
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
                str(output_file), str(output_file), apply_headroom=False
            )
            return {"metrics": metrics, "audit": audit, "errors": errors}
        except Exception as exc:
            print(f"    [warn] Validation error: {exc}")
            return None

    # ------------------------------------------------------------------
    def run(self, num_runs: int = 10, profiles: list[str] | None = None) -> None:
        profiles = profiles or list(self.PROFILES)
        print(f"\n=== Random Benchmark  ({num_runs} runs, profiles={profiles}) ===")

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
        print(f"\nMarkdown report → {report_path}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="FLL Trajectory Optimizer – Unified Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["cli", "pipeline", "random"], default="pipeline",
                        help="Benchmark mode (default: pipeline)")
    parser.add_argument("--config", default="robot_config.json",
                        help="Robot config file")
    parser.add_argument("--output", default="benchmark_results.json",
                        help="Output JSON path")
    parser.add_argument("--runs", type=int, default=None,
                        help="Number of runs / iterations")

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

    elif args.mode == "random":
        profiles = [p.strip() for p in args.profiles.split(",")]
        rb = RandomBenchmark(config_data, cfg)
        rb.run(num_runs=args.runs or 10, profiles=profiles)
        if args.report:
            rb.generate_report()
        _save_results(rb.results, args.output)


if __name__ == "__main__":
    main()
