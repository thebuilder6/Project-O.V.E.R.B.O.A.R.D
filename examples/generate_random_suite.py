"""Generate a random mission suite for review.

This script used to be broken in two ways (KI-015): it read a
``robot_config.json`` that does not exist in the repository, and it treated
``solve()`` as returning the samples when it returns a ``(samples, stats)``
tuple, so it wrote a malformed trajectory. Both are fixed here.

The mission generator itself lives in ``research/suite.py``: it is stratified
over four path profiles (even, uneven, turny, mixed) and deterministic from a
seed, so a suite can be regenerated exactly. ``research/harness.py`` runs the
same missions through the full gate chain; this script only solves and writes
them out for a human to look at.

Usage::

    python examples/generate_random_suite.py --count 10 --seed 7 --out random_suite
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from overboard.core.optimizer import TrajectoryOptimizer
from overboard.models.robot_model import RobotConfig
from research.suite import build_suite

DEFAULT_CONFIG = "examples/fll_choreo.json"


def run_suite(num_runs: int = 10, output_dir: str = "random_suite",
              config_path: str = DEFAULT_CONFIG, seed: int = 7,
              samples: int = 10, plot: bool = False):
    """Generate, solve and write ``num_runs`` random missions."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(config_path, "r", encoding="utf-8") as handle:
        config_data = json.load(handle)
    robot_cfg = RobotConfig(config_data)

    missions = build_suite(num_runs, seed)
    suite_data = []

    for index, mission in enumerate(missions):
        run_name = mission["id"]
        run_dir = os.path.join(output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        with open(os.path.join(run_dir, "waypoints.json"), "w", encoding="utf-8") as handle:
            json.dump(mission["waypoints"], handle, indent=2)

        waypoints = [(wp["x"], wp["y"], wp.get("heading")) for wp in mission["waypoints"]]
        stop_indices = [i for i, wp in enumerate(mission["waypoints"]) if wp.get("stop")]

        optimizer = TrajectoryOptimizer(robot_cfg)
        try:
            # solve() returns (samples, stats). Treating the tuple as the
            # samples is what made the old script write garbage.
            #
            # The result is not called ``samples``: that name is the
            # samples-per-segment count, and reusing it shadowed the
            # parameter, so the second mission solved with the first
            # mission's sample *list* as the sample count and died in the
            # optimizer with "can only concatenate list (not int) to list".
            trajectory, stats = optimizer.solve(
                waypoints,
                num_samples_per_segment=samples,
                stop_waypoint_indices=stop_indices,
                verbose=False,
            )
            converged = bool(stats.get("converged", False))
            with open(os.path.join(run_dir, "trajectory.json"), "w", encoding="utf-8") as handle:
                json.dump({"trajectory": {"samples": trajectory}}, handle, indent=2)

            if plot:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from overboard.viz.plotter import plot_trajectory
                figure = plot_trajectory(
                    trajectory, waypoints=waypoints,
                    title=f"{run_name} ({mission['profile']})", show=False)
                figure.savefig(os.path.join(run_dir, "plot.png"))
                plt.close(figure)

            suite_data.append({
                "id": run_name,
                "profile": mission["profile"],
                "status": "converged" if converged else "failed",
                "converged": converged,
                "duration_s": trajectory[-1]["t"] if trajectory else None,
                "iterations": stats.get("iter_count"),
                "review": {"is_wrong": False, "explanation": ""},
            })
            if trajectory:
                print(f"--- {run_name} ({mission['profile']}): "
                      f"{'converged' if converged else 'FAILED'} "
                      f"{trajectory[-1]['t']:.3f}s")
            else:
                print(f"--- {run_name} ({mission['profile']}): FAILED")
        except Exception as exc:  # a failed solve is a result, not a crash
            suite_data.append({
                "id": run_name,
                "profile": mission["profile"],
                "status": "failed",
                "error": str(exc),
            })
            print(f"--- {run_name} ({mission['profile']}): failed: {exc}")

        with open(os.path.join(output_dir, "suite_metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(suite_data, handle, indent=2)

    print(f"\nSuite generation complete. Review folder: {output_dir}")
    print("Please check suite_metadata.json to log your review comments.")
    return suite_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10, help="How many missions.")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed.")
    parser.add_argument("--out", default="random_suite", help="Output folder.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Robot config.")
    parser.add_argument("--samples", type=int, default=10,
                        help="Samples per segment.")
    parser.add_argument("--plot", action="store_true",
                        help="Also write a plot.png per mission.")
    args = parser.parse_args()
    run_suite(num_runs=args.count, output_dir=args.out, config_path=args.config,
              seed=args.seed, samples=args.samples, plot=args.plot)
